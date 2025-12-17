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

from prep import *
from nn import *

import os
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
    std_preprocessed_data = prep.data
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

    try:
        metrics = trainer.callback_metrics
        print("Training metrics:", metrics)
    except:
        print("No training metrics available")

    plot_training(
        model=model,
        val_dl=val_dl
    )

    #tuning_results = tune_hyperparameters(
    #    train_dataloader=train_dl,
    #    val_dataloader=val_dl
    #)

    future_dl = get_future_dataloader_from_train(
        train_dataset=val_ds,
        data=std_preprocessed_data,
        batch_size=128
    )

    future_results = predict_future(
        model=model,
        future_dataloader=future_dl
    )

    print(future_results["dataframe"])
    print("Model saved to:", trainer.checkpoint_callback.best_model_path) # Save best model checkpoint path

    preds_out_path = Path(TFTConfig.MODEL_DIR) / "predictions.csv"
    try:
        preds_df.to_csv(preds_out_path, index=False)
        print(f"Predictions exported to {preds_out_path}")
    except:
        print("Failed to export predictions DataFrame (format may not be pandas).")























