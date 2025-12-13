from _config import TFTConfig, GlobalConfig

from typing import Any
from metrics import WMAPE, RMSSE
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer, seed_everything
from pytorch_forecasting.models.base import Prediction
from pytorch_forecasting import BaseModel, TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss, MultiLoss
from pytorch_forecasting.metrics import MAE, RMSE, MAPE
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
    optimize_hyperparameters,
)

warnings.filterwarnings("ignore")  # avoid printing out absolute paths

SEED = 100
seed_everything(SEED)


def train(train_dataset: TimeSeriesDataSet,
          val_dataset: TimeSeriesDataSet,
          train_dataloader,
          val_dataloader,
          max_epochs: int = TFTConfig.MAX_EPOCHS,
          learning_rate: float = TFTConfig.LEARNING_RATE,
          model_dir: str = TFTConfig.MODEL_DIR,
          ) -> tuple[TemporalFusionTransformer | Trainer]:


    model = TemporalFusionTransformer.from_dataset(
        train_dataset,
        learning_rate=TFTConfig.LEARNING_RATE,
        hidden_size=32,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=16,
        loss=QuantileLoss(quantiles=[0.5]), #Using the median (0.5 quantile)
        reduce_on_plateau_patience=4
    )

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
        accelerator=GlobalConfig.GPU,
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

    # TFT built-in prediction (kept as you had it)
    preds_data = model.predict(
        test_dataloader,
        return_index=True,
        trainer_kwargs=dict(accelerator="cpu")
    )

    actuals = []
    preds = []

    device = model.device
    model.to(device)
    model.eval()

    with torch.no_grad():
        for x, y in test_dataloader:
            # move inputs to device
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

        metrics = {"MAE": [], "RMSE": [], "MAPE": []}

        for i in range(actuals.shape[-1]):
            y_true = actuals[..., i].ravel()
            y_pred = preds[..., i].ravel()

            mae = np.mean(np.abs(y_true - y_pred))
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            mape = (
                np.mean(
                    np.abs(
                        (y_true - y_pred)
                        / np.where(y_true == 0, 1e-8, y_true)
                    )
                ) * 100
            )

            metrics["MAE"].append(mae)
            metrics["RMSE"].append(rmse)
            metrics["MAPE"].append(mape)

        metrics_summary = {
            **{
                f"MAE_{TFTConfig.TARGET_COLS[i]}": metrics["MAE"][i]
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





















