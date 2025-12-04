import datetime
import numpy as np
import pandas as pd
from pydantic import BaseModel

class Preprocessor(BaseModel):
    data: pd.DataFrame
    verbose: bool

    def _encode_cyclical(self, columns: list[str]) -> None:
        ...

    def _decode_cyclical(self, columns: list[str]) -> None:
        ...

    def standard_preprocess(self) -> pd.DataFrame:
        if self.data.empty:
            raise Exception("Empty file detected! Try with another one")

        self.data["coverage"] = self.data["coverage"].replace(",", ".", regex=True).astype("float") * 100
        self.data["mean_speed"] = self.data["mean_speed"].replace(",", ".", regex=True).astype("float")
        self.data["percentile_85"] = self.data["percentile_85"].replace(",", ".", regex=True).astype("float")
        self.data["date"] = pd.to_datetime(self.data["date"] + "T" + self.data["hour_start"])

        self.data["day"] = self.data["date"].dt.day.astype(np.int16)
        self.data["month"] = self.data["date"].dt.month.astype(np.int16)
        self.data["year"] = self.data["date"].dt.year.astype(np.int16)

        self.data = self.data.drop(columns=["date", "hour_start"])
        self.data = self.data.rename(columns={"traffic_volume": "volume", "hour_start": "hour"})

        self.data = (
            self.data
            .groupby(['year', 'month', 'day', 'hour'])
            .agg({
                'mean_speed': 'median', # Average across lanes
                'percentile_85': 'median',  # Average of 85th percentile speeds
                'volume': 'sum'  # Total hourly volume across lanes. We aren't aggregating by the mean or the median because that wouldn't make any sense
                                 # since lanes of one direction may have more traffic than the lanes towards the opposite direction
            })
            .reset_index()
        )

        return self.data


    def nn_preprocess(self) -> pd.DataFrame:
        #TODO CREATING CYCLICAL ENCODED FEATURES

        # TODO SCALE FEATURES SAVE THE SCALER FOR LATER TO DECODE THEM
        ...  # TODO TimeSeriesDataset and so on

    def decode(self) -> pd.DataFrame:
        ... #TODO DECODE CYCLICAL VARIABLES, ETC.




































