from _config import TFTConfig, GlobalConfig

from typing import Any
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)

import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer, seed_everything
from pytorch_forecasting.models.base import Prediction
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
    optimize_hyperparameters,
)

import optuna


warnings.filterwarnings("ignore")  # avoid printing out absolute paths

SEED = 100
seed_everything(SEED)


def train(train_dataset: TimeSeriesDataSet,
          train_dataloader: DataLoader,
          val_dataloader: DataLoader,
          learning_rate: float = TFTConfig.LEARNING_RATE,
          max_epochs: int = TFTConfig.MAX_EPOCHS,
          model_dir: str = TFTConfig.MODEL_DIR,
          device: str = GlobalConfig.DEVICE
          ) -> tuple[TemporalFusionTransformer | Trainer]:


    model = TemporalFusionTransformer.from_dataset(
        train_dataset,
        learning_rate=learning_rate,
        hidden_size=45, #NOTE 16
        attention_head_size=3, #NOTE 8
        dropout=0.11013129302802382, #NOTE 0.01,
        hidden_continuous_size=26, #NOTE 16,
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
            EarlyStopping(monitor="val_loss", patience=8, mode="min"), #TODO 16
            checkpoint_callback,
            LearningRateMonitor()
        ],
        default_root_dir=model_dir,
        gradient_clip_val=0.8762948027181958, #NOTE 0.1
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
    test_dataloader: DataLoader
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


def plot_training(
        model: TemporalFusionTransformer,
        val_dl: DataLoader,
) -> None:

    model.to(GlobalConfig.DEVICE)
    model.eval()

    pred = model.predict(
        val_dl,
        mode="raw",
        return_x=True,
    )

    x = pred.x
    pred_tensor = pred.output["prediction"]

    # Predictions can be:
    # - Tensor [batch, time, target]
    # - Tensor [batch, time, target, quantile]
    # - List[Tensor] (one per target)
    if isinstance(pred_tensor, list):
        # stack targets -> [batch, time, target, (quantile?)]
        pred_tensor = torch.stack(pred_tensor, dim=2)

    # ---- Quantile handling ----
    quantiles = getattr(model.loss, "quantiles", None)
    use_quantiles = False
    median_idx = None

    if quantiles is not None:
        try:
            q = [float(v) for v in quantiles]
        except:
            q = []

        if len(q) > 1:
            use_quantiles = True
            if 0.5 in q:
                median_idx = q.index(0.5)
            else:
                median_idx = int(np.argmin(np.abs(np.array(q) - 0.5)))

    MAX_PLOTS = 3 # One plot for each target feature
    batch_size = pred_tensor.shape[0]

    decoder_target = x["decoder_target"] # Extracting the decoded target features

    # decoder_target is, generally speaking, a List[Tensor], one per target
    if isinstance(decoder_target, list): # Ensuring that the shape is [batch, time] for each target
        gt_targets = []
        for t in decoder_target:
            if t.dim() == 3 and t.shape[-1] == 1:
                t = t.squeeze(-1)
            gt_targets.append(t)
        n_targets = len(gt_targets)
    else:
        # Tensor case, where the shape is: [batch, time, target] or [batch, time]
        if decoder_target.dim() == 2:
            gt_targets = [decoder_target]
            n_targets = 1
        else:
            gt_targets = [decoder_target[:, :, i] for i in range(decoder_target.shape[2])]
            n_targets = decoder_target.shape[2]

    # Plotting section
    for idx in range(min(MAX_PLOTS, batch_size)):

        fig, axes = plt.subplots(
            n_targets, 1,
            figsize=(12, 4 * n_targets),
            sharex=True
        )

        if n_targets == 1:
            axes = [axes]

        for t, ax in enumerate(axes):

            # Ground truth
            gt = gt_targets[t][idx].detach().cpu().numpy()
            ax.plot(gt, label="Ground truth", color="black")

            # Prediction
            if use_quantiles and pred_tensor.ndim == 4:
                preds = pred_tensor[idx, :, t, median_idx]
            else:
                preds = pred_tensor[idx, :, t]

            preds = preds.detach().cpu().numpy()
            ax.plot(preds, label="Prediction (P50)")

            # Prediction interval
            if use_quantiles and pred_tensor.ndim == 4:
                lower = pred_tensor[idx, :, t, 0].detach().cpu().numpy()
                upper = pred_tensor[idx, :, t, -1].detach().cpu().numpy()

                ax.fill_between(
                    range(len(preds)),
                    lower,
                    upper,
                    alpha=0.3,
                    label="Prediction Interval"
                )

            ax.set_title(f"{TFTConfig.TARGET_COLS[t]} – Forecast")
            ax.set_ylabel("Value")
            ax.grid(True)
            ax.legend()

        axes[-1].set_xlabel("Prediction Time Step")
        fig.suptitle(f"TFT Forecast – Sample {idx + 1}", fontsize=14)
        plt.tight_layout()
        plt.show()

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
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    model_dir: str = TFTConfig.MODEL_DIR,
    n_trials: int = 20,
    max_epochs: int = TFTConfig.MAX_EPOCHS,
    timeout: float | None = None,
    trainer_kwargs: dict | None = None,
    verbose: bool = True,
    pruner: Any | None = None,
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



def get_future_dataloader_from_train(
    train_dataset: TimeSeriesDataSet,
    data: pd.DataFrame,
    batch_size: int = 64,
    num_workers: int = 0
) -> DataLoader:
    """
    Creates a future dataloader compatible with a pretrained trained TFT model.
    """

    predict_dataset = TimeSeriesDataSet.from_dataset(
        train_dataset,
        data,
        predict=True,
        stop_randomization=True
    )

    future_dataloader = predict_dataset.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=num_workers
    )

    return future_dataloader



def predict_future(
    model: TemporalFusionTransformer,
    future_dataloader: DataLoader,
    device: str = GlobalConfig.DEVICE
) -> dict[str, Any]:
    """
    Predict future values for each target feature.
    """

    model.to(device)
    model.eval()

    pred = model.predict(
        future_dataloader,
        mode="raw",
        return_x=True,
        return_index=True,
        trainer_kwargs=dict(accelerator=device)
    )

    raw_preds = pred.output["prediction"]

    if isinstance(raw_preds, list):
        preds = torch.stack(raw_preds, dim=2) # Prediction output shape normalization
    else:
        preds = raw_preds

    if preds.dim() == 4:
        quantiles = model.loss.quantiles
        q = [float(v) if isinstance(v, (int, float)) else float(v[0]) for v in quantiles] # Quantile handling (median)
        median_idx = q.index(0.5) if 0.5 in q else int(np.argmin(np.abs(np.array(q) - 0.5)))
        preds = preds[..., median_idx]

    preds = preds.detach().cpu().numpy()
    predictions_index = pred.index.copy() # Index of the predictions' dataframe

    # Determine horizon and number of sequences
    if preds.ndim == 3:
        n_sequences, horizon, n_targets = preds.shape # Expected preds shape: (n_sequences, horizon, n_targets)
    elif preds.ndim == 2:
        n_sequences = preds.shape[0]
        horizon = 1
        n_targets = preds.shape[1]
        preds = preds.reshape(n_sequences, horizon, n_targets)
    else:
        raise ValueError(f"Unexpected preds ndim: {preds.ndim}. Expected 2 or 3.")

    flat_preds = preds.reshape(-1, n_targets)  # (n_sequences * horizon, n_targets)

    len_index = len(predictions_index)
    expected_flat_len = n_sequences * horizon

    rows = []

    if len_index == expected_flat_len: # The index has one entry per predicted step (original assumption) case
        for i in range(n_sequences):
            for t in range(horizon):
                idx = i * horizon + t
                row = predictions_index.iloc[idx].to_dict()
                for j, name in enumerate(TFTConfig.TARGET_COLS):
                    row[name] = preds[i, t, j]
                rows.append(row)

    elif len_index == n_sequences: # The index has one entry per sequence (repeat for each horizon step) case
        for i in range(n_sequences):
            base_row = predictions_index.iloc[i].to_dict()
            for t in range(horizon):
                row = base_row.copy()
                for j, name in enumerate(TFTConfig.TARGET_COLS):
                    row[name] = preds[i, t, j]
                rows.append(row)

    elif len_index == flat_preds.shape[0]: # The index already matches flattened predictions: map one-to-one case
        for idx in range(flat_preds.shape[0]):
            row = predictions_index.iloc[idx].to_dict()
            for j, name in enumerate(TFTConfig.TARGET_COLS):
                row[name] = flat_preds[idx, j]
            rows.append(row)

    else: # Fallback: map up to the minimum length and warn the user
        n_map = min(len_index, expected_flat_len)
        for idx in range(n_map):
            row = predictions_index.iloc[idx].to_dict()
            for j, name in enumerate(TFTConfig.TARGET_COLS):
                row[name] = flat_preds[idx, j]
            rows.append(row)

    return {
        "predictions": preds,
        "dataframe": pd.DataFrame(rows),
        "raw": pred
    }














