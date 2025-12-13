from typing import ClassVar
from pydantic import BaseModel
import pandas as pd
from holidays import HolidayBase
import holidays
from astral import LocationInfo


import torch

class TFTConfig(BaseModel):
    MODEL_DIR: ClassVar[str] = "./tft_checkpoints"
    ENCODER_LENGTH: ClassVar[int] = 168  # lookback (e.g., last 168 hours = 7 days hourly)
    DECODER_LENGTH: ClassVar[int] = 24  # forecast horizon (e.g., next 24 hours)
    BATCH_SIZE: ClassVar[int] = 128 #TODO 128
    MAX_EPOCHS: ClassVar[int] = 7 #TODO 30
    LEARNING_RATE: ClassVar[float] = 3e-4 #3e-4
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
    COUNTRY_HOLIDAYS: ClassVar[HolidayBase] = holidays.country_holidays("NO", years=[2018, 2019, 2020])
    FALLBACK_YEAR: ClassVar[int] = 2000
    SEASONS: ClassVar[dict[str, pd.Timestamp]] = {
        "winter": (pd.Timestamp(f"{FALLBACK_YEAR}-12-21"),
                   pd.Timestamp(f"{FALLBACK_YEAR + 1}-03-19")),
        "spring": (pd.Timestamp(f"{FALLBACK_YEAR}-03-20"),
                   pd.Timestamp(f"{FALLBACK_YEAR}-06-20")),
        "summer": (pd.Timestamp(f"{FALLBACK_YEAR}-06-21"),
                   pd.Timestamp(f"{FALLBACK_YEAR}-09-22")),
        "autumn": (pd.Timestamp(f"{FALLBACK_YEAR}-09-23"),
                   pd.Timestamp(f"{FALLBACK_YEAR}-12-20")),
    }
    REFERENCE_CITY: ClassVar[LocationInfo] = LocationInfo(
        name="Trondheim",
        region="Norway",
        timezone="Europe/Oslo",
        latitude=63.4305,
        longitude=10.3951
    )
    FESTIVITIES: ClassVar[dict[str, str]] = {
        "Christmas": "12-25",
        "NewYear": "01-01",
        "Halloween": "10-31",
        "IndependenceDay": "07-04"
    }

    class Config:
        frozen=True


print(torch.version.cuda)
print(torch.cuda.current_device())
print(torch.cuda.is_available())