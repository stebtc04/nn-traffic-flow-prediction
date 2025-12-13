from typing import ClassVar
from pydantic import BaseModel
import torch

class TFTConfig(BaseModel):
    MODEL_DIR: ClassVar[str] = "./tft_checkpoints"
    ENCODER_LENGTH: ClassVar[int] = 168  # lookback (e.g., last 168 hours = 7 days hourly)
    DECODER_LENGTH: ClassVar[int] = 24  # forecast horizon (e.g., next 24 hours)
    BATCH_SIZE: ClassVar[int] = 128 #TODO 128
    MAX_EPOCHS: ClassVar[int] = 30
    LEARNING_RATE: ClassVar[float] = 3e-4
    DATE_COL: ClassVar[str] = "date"  # YYYY-MM-DD or similar
    TIME_COL: ClassVar[str] = "hour_start"  # hour as padded int
    TARGET_COLS: ClassVar[list[str]] = ["volume", "mean_speed", "percentile_85"]
    KNOWN_REALS: ClassVar[list] = ["hour", "day", "month", "year"] # KNOWN_REALS are features known for both encoder + decoder (like hour, day, month, year)

    class Config:
        frozen=True


class Targets(BaseModel):
    VOLUME: ClassVar[str] = "volume"
    MEAN_SPEED: ClassVar[str] = "mean_speed"
    PERCENTILE_85: ClassVar[str] = "percentile_85"

    class Config:
        frozen=True


class GlobalConfig(BaseModel):
    SERIES_ID: ClassVar[str] = "series_0"
    DEVICE: ClassVar[str] = "cuda" if torch.cuda.is_available() else "cpu"

    class Config:
        frozen=True

print(torch.version.cuda)
print(torch.cuda.current_device())
print(torch.cuda.is_available())