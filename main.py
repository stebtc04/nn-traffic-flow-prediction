"""
Temporal Fusion Transformer (TFT) - Multivariate Time Series Forecasting
=======================================================================

Implementing a multivariate forecasting
pipeline with PyTorch Forecasting's TemporalFusionTransformer.

Targets (multivariate):
 - traffic_volume
 - mean_speed
 - percentile_85

Known time-varying features (provided to the model):
 - hour, day, month, year

"""

from _config import TFTConfig, GlobalConfig
from prep import *
from nn import *

import os
import datetime
from typing import Literal
from pathlib import Path
import pandas as pd

def init() -> None:
    os.makedirs(TFTConfig.MODEL_DIR, exist_ok=True)
    return None

def load(fp: os.PathLike | Path) -> pd.DataFrame:
    return pd.read_csv(fp, sep=";", encoding="utf-8")


if __name__ == "__main__":

    prep = Preprocessor(data=load(Path.cwd() / "47408V625213_speeds.csv"))

    prep.standard_preprocess()
    train_ds, val_ds, test_ds, train_dl, val_dl, test_dl = prep.nn_preprocess()

    model, trainer = train(
       train_dataset=train_ds,
       train_dataloader=train_dl,
       val_dataloader=val_dl
    )

    metrics_summary, preds_df = evaluate(model, test_dl)
    print("Evaluation metrics:")
    for k, v in metrics_summary.items():
        print(f"  {k}: {v:.4f}")

    plot_training(trainer=trainer)

    tuning_results = tune_hyperparameters(
        train_dataloader=train_dl,
        val_dataloader=val_dl
    )

    print("Model saved to:", trainer.checkpoint_callback.best_model_path) # Save best model checkpoint path

    preds_out_path = Path(TFTConfig.MODEL_DIR) / "predictions.csv"
    try:
        preds_df.to_csv(preds_out_path, index=False)
        print(f"Predictions exported to {preds_out_path}")
    except Exception:
        print("Failed to export predictions DataFrame (format may not be pandas).")























