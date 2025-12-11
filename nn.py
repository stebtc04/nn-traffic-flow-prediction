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
from pytorch_lightning import Trainer, seed_everything
from pytorch_forecasting.models.base import Prediction
from pytorch_forecasting import Baseline, TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
from pytorch_forecasting.metrics import MAE, RMSE, MAPE
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
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
        learning_rate=learning_rate,
        hidden_size=256,  # The size of LSTM layers
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=64,
        output_size=len(TFTConfig.TARGET_COLS) * 3,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        filename="tft-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
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
        gradient_clip_val=0.1,
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # Loading the best model
    if best_path := checkpoint_callback.best_model_path:
        best_model = TemporalFusionTransformer.load_from_checkpoint(best_path)
    else:
        warnings.warn("Checkpoint not found — returning last model")
        best_model = model

    return best_model, trainer


def evaluate(model: TemporalFusionTransformer, test_dataloader) -> tuple[dict[str, Any], Prediction]:

    preds_data = model.predict(test_dataloader, return_index=True, trainer_kwargs=dict(accelerator="cpu")) #The predict() method returns a Prediction object

    # Evaluating target-wise metrics by comparing decoder horizon predictions to actuals
    actuals = []
    preds = []

    # Iterating through test_dataloader to gather the ground truth and the predictions
    device = model.device
    model.to(device)

    for batch in test_dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()} # Device is a specific dtype used in PyTorch-Forecasting
        with torch.no_grad():
            out = model(batch) # out has shape: (batch_size, prediction_length, output_size)

        if isinstance(out, dict) and "prediction" in out:
            batch_preds = out["prediction"].cpu().numpy()
        elif hasattr(out, "detach"):
            batch_preds = out.detach().cpu().numpy()
        else:
            batch_preds = np.array(out)

        # Targets' values usually lie in batch["decoder_target"]
        if "decoder_target" in batch:
            batch_target = batch["decoder_target"].cpu().numpy()
        elif "target" in batch:
            batch_target = batch["target"].cpu().numpy()
        else:
            batch_target = None

        if batch_target is not None:
            preds.append(batch_preds)
            actuals.append(batch_target)

    if preds and actuals:
        preds = np.concatenate(preds, axis=0) # preds shape: (N, prediction_length, output_size)
        actuals = np.concatenate(actuals, axis=0) # actuals shape: (N, prediction_length, number_of_targets)

        # Computing metrics target-wise
        metrics = {"MAE": [], "RMSE": [], "MAPE": []}
        for i in range(actuals.shape[-1]):
            y_true = actuals[..., i].ravel()
            y_pred = preds[..., i].ravel()
            mae = np.mean(np.abs(y_true - y_pred))
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-8, y_true))) * 100 # MAPE: avoiding division by zero with np.where()

            metrics["MAE"].append(mae)
            metrics["RMSE"].append(rmse)
            metrics["MAPE"].append(mape)

        # Aggregate metrics
        metrics_summary = {
            *{f"MAE_{TFTConfig.TARGET_COLS[i]}": metrics["MAE"][i] for i in range(len(TFTConfig.TARGET_COLS))},
            *{f"RMSE_{TFTConfig.TARGET_COLS[i]}": metrics["RMSE"][i] for i in range(len(TFTConfig.TARGET_COLS))},
            *{f"MAPE_{TFTConfig.TARGET_COLS[i]}": metrics["MAPE"][i] for i in range(len(TFTConfig.TARGET_COLS))}
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





















