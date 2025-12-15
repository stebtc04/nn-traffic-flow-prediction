from _config import TFTConfig, GlobalConfig

from typing import Any
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)

import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer, seed_everything
from pytorch_forecasting.models.base import Prediction
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss, MultivariateNormalDistributionLoss
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
    optimize_hyperparameters,
)

import optuna


warnings.filterwarnings("ignore")  # avoid printing out absolute paths

SEED = 100
seed_everything(SEED)


def train(train_dataset: TimeSeriesDataSet,
          train_dataloader,
          val_dataloader,
          learning_rate: float = TFTConfig.LEARNING_RATE,
          max_epochs: int = TFTConfig.MAX_EPOCHS,
          model_dir: str = TFTConfig.MODEL_DIR,
          device: str = GlobalConfig.DEVICE
          ) -> tuple[TemporalFusionTransformer | Trainer]:


    model = TemporalFusionTransformer.from_dataset(
        train_dataset,
        learning_rate=learning_rate,
        hidden_size=16, #NOTE 16
        attention_head_size=8, #NOTE 8
        dropout=0.01,
        hidden_continuous_size=16,
        loss=QuantileLoss(quantiles=[0.5]), #Using the median (0.5 quantile)
        reduce_on_plateau_patience=8, #NOTE 8
    ).to(device) #NOTE 12

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        filename="tft-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=8, mode="min"),
            checkpoint_callback,
            LearningRateMonitor()
        ],
        default_root_dir=model_dir,
        gradient_clip_val=0.1
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )

    # Loading the best model
    if best_path := checkpoint_callback.best_model_path:
        best_model = TemporalFusionTransformer.load_from_checkpoint(best_path)
    else:
        warnings.warn("Checkpoint not found — returning last model")
        best_model = model

    return best_model, trainer


def evaluate(
    model: TemporalFusionTransformer,
    test_dataloader
) -> tuple[dict[str, Any], Prediction]:

    preds_data = model.predict(
        test_dataloader,
        return_index=True,
        trainer_kwargs=dict(accelerator=GlobalConfig.DEVICE)
    ) # TFT built-in prediction

    actuals = []
    preds = []

    device = model.device
    model.to(device)
    model.eval()

    with torch.no_grad():
        for x, y in test_dataloader:
            x = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}

            out = model(x)
            #print(out)

            pred = out["prediction"]
            if isinstance(pred, list):
                pred = torch.cat(pred, dim=-1)
            batch_preds = pred.cpu().numpy()

            if isinstance(y, tuple):
                y = y[0]  # drop weights

            if isinstance(y, list):
                y = torch.stack(y, dim=-1)

            batch_target = y.cpu().numpy()

            preds.append(batch_preds)
            actuals.append(batch_target)

    if preds and actuals:
        preds = np.concatenate(preds, axis=0)
        actuals = np.concatenate(actuals, axis=0)

        metrics = {
            "MAE": [],
            "MSE": [],
            "RMSE": [],
            "MAPE": []
        }

        for i in range(actuals.shape[-1]):
            y_true = actuals[..., i].ravel()
            y_pred = preds[..., i].ravel()

            metrics["MAE"].append(mean_absolute_error(y_pred=y_pred, y_true=y_true))
            metrics["MSE"].append(mean_squared_error(y_pred=y_pred, y_true=y_true))
            metrics["RMSE"].append(root_mean_squared_error(y_pred=y_pred, y_true=y_true))
            metrics["MAPE"].append(mean_absolute_percentage_error(y_pred=y_pred, y_true=y_true))

        metrics_summary = {
            **{
                f"MAE_{TFTConfig.TARGET_COLS[i]}": metrics["MAE"][i]
                for i in range(len(TFTConfig.TARGET_COLS))
            },
            **{
                f"MSE_{TFTConfig.TARGET_COLS[i]}": metrics["MSE"][i]
                for i in range(len(TFTConfig.TARGET_COLS))
            },
            **{
                f"RMSE_{TFTConfig.TARGET_COLS[i]}": metrics["RMSE"][i]
                for i in range(len(TFTConfig.TARGET_COLS))
            },
            **{
                f"MAPE_{TFTConfig.TARGET_COLS[i]}": metrics["MAPE"][i]
                for i in range(len(TFTConfig.TARGET_COLS))
            },
        }
    else:
        metrics_summary = {}

    return metrics_summary, preds_data


def plot_training(trainer: Trainer) -> None:
    try:
        metrics = trainer.callback_metrics
        print("Training metrics:", metrics)
    except:
        print("No training metrics available to plot")
    return None


def plot_predictions(preds_df: pd.DataFrame, actuals: pd.DataFrame, target: str) -> None:
    plt.figure(figsize=(12, 6))
    if target in actuals.columns:
        plt.plot(actuals['timestamp'], actuals[target], label='actual')
    if f"prediction_{target}" in preds_df.columns:
        plt.plot(preds_df['timestamp'], preds_df[f"prediction_{target}"], label='prediction')
    plt.title(f"Actual vs Prediction for {target}")
    plt.legend()
    plt.show()
    return None


def tune_hyperparameters(
    train_dataloader,
    val_dataloader,
    model_dir: str = TFTConfig.MODEL_DIR,
    n_trials: int = 20,
    max_epochs: int = TFTConfig.MAX_EPOCHS,
    timeout: float | None = None,
    trainer_kwargs: dict | None = None,
    verbose: bool = True,
    pruner = None,
    **tft_kwargs
) -> dict[str, Any]:
    """
    Runs hyperparameter optimization using pytorch_forecasting.optimize_hyperparameters
    Returns a dictionary with:
      - best_params
      - best_value
      - best_trial_number
      - trial_history (list of per-trial dicts)
      - study (the optuna.Study object) which can be useful for deeper inspection

    Prints a readable dictionary of the best result and prints trial-by-trial steps
    (uses study.trials_dataframe()) so you can see exactly what happened.
    """
    if trainer_kwargs is None:
        trainer_kwargs = {}

    # Default timeout if not specified (8 hours like the function default)
    if timeout is None:
        timeout = 3600 * 8.0

    study = optimize_hyperparameters(
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        model_path=model_dir,
        max_epochs=max_epochs,
        n_trials=n_trials,
        timeout=timeout,
        trainer_kwargs=trainer_kwargs,
        verbose=verbose,
        pruner=pruner,
        **tft_kwargs,
    )

    # Extraction of the best parameters and info
    best_trial = study.best_trial
    best_params = study.best_params if hasattr(study, "best_params") else best_trial.params
    best_value = study.best_value if hasattr(study, "best_value") else best_trial.value
    best_trial_number = best_trial.number if best_trial is not None else None

    try:
        trials_df = study.trials_dataframe() # Getting full trial history as a DataFrame
    except:
        # As a fallback we can convert single trials from dictionaries to dataframe records
        trials = study.trials
        trials_df = pd.DataFrame(({
            "number": t.number,
            "value": t.value,
            "state": str(t.state),
            "params": t.params,
            "datetime_start": getattr(t, "datetime_start", None),
            "datetime_complete": getattr(t, "datetime_complete", None),
            "duration": (getattr(t, "datetime_complete", None) - getattr(t, "datetime_start", None)).total_seconds() if getattr(t, "datetime_start", None) and getattr(t, "datetime_complete", None) else None,
            "intermediate_values": t.intermediate_values if hasattr(t, "intermediate_values") else {},
        } for t in trials)) # If it wasn't possible to get a full trial history then build the dataframe from scratch

    # Convert DataFrame into a list of dicts for easy printing / saving
    trial_history = trials_df.to_dict(orient="records")
    results = {
        "best_params": best_params,
        "best_value": best_value,
        "best_trial_number": best_trial_number,
        "n_trials_ran": len(trial_history),
        "trial_history": trial_history,
        # include the study object for any deeper post-inspection if desired
        "study": study,
    }

    print("******** Hyperparameter tuning summary ********")
    print(results)

    print("\n=== Trial-by-trial details ===")
    print(trials_df)

    return results


















