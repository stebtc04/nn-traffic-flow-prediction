from pydantic import BaseModel
import torch

class TFTConfig(BaseModel):
    MODEL_DIR: str = "./tft_checkpoints"
    ENCODER_LENGTH: int = 168  # lookback (e.g., last 168 hours = 7 days hourly)
    DECODER_LENGTH: int = 24  # forecast horizon (e.g., next 24 hours)
    BATCH_SIZE: int = 64
    MAX_EPOCHS: int = 30
    LEARNING_RATE: float = 3e-4
    DATE_COL: str
    DATE_COL = "date"  # YYYY-MM-DD or similar
    TIME_COL = "hour_start"  # hour as padded int
    TARGET_COLS = ["volume", "mean_speed", "percentile_85"]

    class Config:
        frozen=True


class Targets(BaseModel):
    VOLUME: str = "volume"
    MEAN_SPEED: str = "mean_speed"
    PERCENTILE_85: str = "percentile_85"

    class Config:
        frozen=True


class GlobalConfig(BaseModel):
    SERIES_ID: str = "series_0"
    GPU: str = "gpu" if torch.cuda.is_available() else "cpu"

    class Config:
        frozen=True