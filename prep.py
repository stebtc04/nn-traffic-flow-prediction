from _config import GlobalConfig, TFTConfig
import os
import math
import datetime
import warnings
from typing import cast
import numpy as np
import pandas as pd
from pydantic import BaseModel
from astral.sun import sun

from sklearn.linear_model import Lasso, GammaRegressor, QuantileRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeClassifier

from sklego.meta import ZeroInflatedRegressor

from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from pytorch_forecasting import TimeSeriesDataSet

pd.set_option("display.max_columns", None)



class RegressorTypes(BaseModel):
    lasso: str = "lasso"
    gamma: str = "gamma"
    quantile: str = "quantile"

    class Config:
        frozen=True


class Preprocessor(BaseModel):
    data: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True


    @staticmethod
    def get_season(date: pd.Timestamp) -> str:
        return next(
            (season for season, (start, end) in GlobalConfig.SEASONS.items()
             if start <= pd.Timestamp(GlobalConfig.FALLBACK_YEAR, date.month, date.day) <= end),
            "winter"  # Fallback season set as default
        )

    @staticmethod
    def get_daylight_info(date: pd.Timestamp) -> pd.Series: # pd.Series[dict[str, float | bool]]
        try:
            s = sun(GlobalConfig.REFERENCE_CITY.observer, date=date)
            return pd.Series({
                "daylight_hours": (s["sunset"] - s["sunrise"]).total_seconds() / 3600,
                "polar_day": False,
                "polar_night": False,
            })

        except ValueError as e:
            msg = str(e).lower()

            if "never sets" in msg or "never reaches" in msg:
                return pd.Series({
                    "daylight_hours": 24.0,
                    "polar_day": True,
                    "polar_night": False,
                })

            if "never rises" in msg:
                return pd.Series({
                    "daylight_hours": 0.0,
                    "polar_day": False,
                    "polar_night": True,
                })

            raise #Raising unexpected errors instead of silently hiding them

    @staticmethod
    def get_festivity(date: pd.Timestamp) -> str:
        return next(
            (name for name, d in GlobalConfig.FESTIVITIES.items() if d == date.strftime("%m-%d")),
            None
        )

    @staticmethod
    def get_time_of_day(date: pd.Timestamp) -> pd.Series: # pd.Series[dict[str, bool | str]]:
        try:
            s = sun(
                GlobalConfig.REFERENCE_CITY.observer,  # This could change in case of TRPs in different cities or towns
                date=date.date(),
                tzinfo=GlobalConfig.REFERENCE_CITY.timezone
                # The same as above applies here, maybe not in Norway, but yes in bigger countries cases
            )

            # Computing the actual sunrise/sunset/dawn/dusk times to decimals to obtain a usable measure of comparison to determine the time of day (TOD)
            # Example: suppose that sunrise is at 03:37 in the morning, we want to convert that exact point of the day into a decimal value.
            # We'll say that since one hour is equal to 60 minutes, we can just compute the number of minutes divided by 60 and obtain the decimal part of the time of day.
            # In the example case this will be: 3.62 (3 hours + 0.62 hours) (0.62 hours = 37 minutes / 60 minutes)
            sunrise = s["sunrise"].hour + s["sunrise"].minute / 60
            sunset = s["sunset"].hour + s["sunset"].minute / 60
            dawn = s["dawn"].hour + s["dawn"].minute / 60
            dusk = s["dusk"].hour + s["dusk"].minute / 60

            hour = date.hour + date.minute / 60

            is_morning_peak = 6.5 <= hour < 9.5
            is_evening_peak = 16 <= hour < 19

            return pd.Series({
                "is_day": sunrise <= hour < sunset,
                "time_of_day": (
                    "night" if hour < dawn or hour >= dusk else
                    "morning" if sunrise <= hour < 10 else
                    "mid-day" if 10 <= hour < 14 else
                    "afternoon" if 14 <= hour < sunset else
                    "evening"
                ),  # A.K.A. TOD (Time Of Day)
                "is_peak": bool(is_morning_peak or is_evening_peak),
            })
            # The modelling of the time of day based on simple hour-of-the-day and not on the sun's position is due to the fact that traffic,
            # which is a human-related phenomena, is mainly determined by human customs and not by the natural context

        except ValueError as e:
            msg = str(e).lower()

            if any(case in msg for case in [
                "never reaches",
                "never sets",
                "unable to find a dusk time on the date specified",
            ]):
                return pd.Series({
                    "is_day": True,
                    "time_of_day": "day"
                })  # Midnight sun case

            if any(case in msg for case in
            [
                "never rises",
                "unable to find a dawn time on the date specified"
            ]):
                return pd.Series({
                    "is_day": False,
                    "time_of_day": "night"
                })  # Polar night case

            raise

    @staticmethod
    def _impute_missing_values(cols: list[str], data: pd.DataFrame, r: str = "gamma") -> pd.DataFrame:
        if r not in cast(dict, RegressorTypes.model_fields).keys():  # cast is used to silence static type checker warnings
            raise ValueError(
                f"Regressor type '{r}' is not supported. Must be one of: {cast(dict, RegressorTypes.model_fields).keys()}")

        # Using Gamma regression to address for the zeros present in the data (which will need to be predicted as well)
        mice_imputer = IterativeImputer(
            estimator=ZeroInflatedRegressor(
                regressor=GammaRegressor(fit_intercept=True, verbose=0),
                classifier=DecisionTreeClassifier(random_state=100)
            ),
            random_state=100,
            verbose=0,
            imputation_order="roman",
            initial_strategy="mean"
        )  # Imputation order is set to arabic so that the imputations start from the right (so from the traffic volume columns)

        return pd.DataFrame(mice_imputer.fit_transform(data), columns=cols)  # Fitting the imputer and processing all the data columns except the date one #TODO BOTTLENECK. MAYBE USE POLARS LazyFrame or PyArrow?


    def standard_preprocess(self, series_id: str = "series_0") -> pd.DataFrame:
        if self.data.empty:
            raise Exception("Empty file detected! Try with another one")

        imputation_columns = [*TFTConfig.TARGET_COLS, *TFTConfig.KNOWN_REALS]

        self.data["coverage"] = self.data["coverage"].replace(",", ".", regex=True).astype("float") * 100
        self.data["mean_speed"] = self.data["mean_speed"].replace(",", ".", regex=True).astype("float")
        self.data["percentile_85"] = self.data["percentile_85"].replace(",", ".", regex=True).astype("float")
        self.data["date"] = pd.to_datetime(self.data["date"] + "T" + self.data["hour_start"])

        self.data["day"] = self.data["date"].dt.day.astype(np.int16)
        self.data["month"] = self.data["date"].dt.month.astype(np.int16)
        self.data["year"] = self.data["date"].dt.year.astype(np.int16)
        self.data["hour_start"] = self.data["date"].dt.hour.astype(np.int16) #Overwriting the old date format with just hour integer values

        self.data = self.data.rename(columns={"traffic_volume": "volume", "hour_start": "hour"})

        self.data = (
            self.data
            .groupby(['date', 'year', 'month', 'day', 'hour'])
            .agg({
                'mean_speed': 'median', # Average across lanes
                'percentile_85': 'median',  # Average of 85th percentile speeds
                'volume': 'sum'  # Total hourly volume across lanes. We aren't aggregating by the mean or the median because that wouldn't make any sense
                                 # since lanes of one direction may have more traffic than the lanes towards the opposite direction
            })
            .reset_index()
        )

        self.data[imputation_columns] = self._impute_missing_values(data=self.data[imputation_columns], r="gamma", cols=imputation_columns)

        self.data["series_id"] = series_id

        # Sort chronologically first
        self.data = self.data.sort_values(["series_id", "date"]).reset_index(drop=True)

        # Global sequential time index per series (required by TFT)
        self.data["time_idx"] = self.data.groupby("series_id").cumcount()

        # -------- Time-series related features --------
        self.data["is_weekend"] = self.data["date"].dt.weekday >= 5
        self.data["is_working_day"] = ~self.data["is_weekend"]
        self.data["season"] = self.data["date"].apply(self.get_season)
        self.data["season"] = pd.Categorical(
            self.data["season"],
            categories=["winter", "spring", "summer", "autumn"],
            ordered=True
        )

        self.data["is_holiday"] = self.data["date"].isin(GlobalConfig.COUNTRY_HOLIDAYS)
        self.data["holiday_name"] = self.data["date"].map(GlobalConfig.COUNTRY_HOLIDAYS)

        self.data["rain_mm"] = np.random.gamma(2, 1, len(self.data))
        self.data["is_rainy_day"] = self.data["rain_mm"] > 1

        rain_by_season = (
            self.data.groupby("season")["is_rainy_day"]
            .sum()
        )
        rainiest_season = rain_by_season.idxmax()

        self.data["is_rainiest_season"] = self.data["season"] == rainiest_season
        self.data = pd.concat([self.data, self.data["date"].apply(self.get_daylight_info)], axis=1) #Daylight info columns get concatenated to the main dataframe

        self.data["festivity"] = self.data["date"].apply(self.get_festivity)
        self.data["is_festivity"] = self.data["festivity"].notna()

        self.data = pd.concat([self.data, self.data["date"].apply(self.get_time_of_day)], axis=1) #Time of day info columns get concatenated to the main dataframe
        self.data["time_of_day"] = pd.Categorical(
            self.data["time_of_day"],
            categories=["night", "morning", "day", "mid-day", "afternoon", "evening"],
            ordered=True
        )

        self.data.drop(columns=["holiday_name", "festivity"], inplace=True)

        self.data["year"] = self.data["year"].astype("int")
        self.data["month"] = self.data["month"].astype("int")
        self.data["day"] = self.data["day"].astype("int")
        self.data["hour"] = self.data["hour"].astype("int")
        self.data["volume"] = self.data["volume"].astype("int")

        print(self.data.columns)
        print(self.data.head(5))
        print(self.data.dtypes)
        print(self.data.isna().sum())

        return self.data


    def nn_preprocess(self):

        ENC = TFTConfig.ENCODER_LENGTH  # 168
        DEC = TFTConfig.DECODER_LENGTH  # 24

        # Normalizer definition
        try:
            target_normalizer = MultiNormalizer(
                [GroupNormalizer(groups=["series_id"]) for _ in TFTConfig.TARGET_COLS]
            )
        except:
            warnings.warn("MultiNormalizer not available. Using GroupNormalizer.")
            target_normalizer = GroupNormalizer(groups=["series_id"])

        horizon = ENC + DEC

        # The test set should have at least numer of row equal to ENC + DEC
        max_time = self.data["time_idx"].max()
        test_cutoff = max_time - horizon

        test_data = self.data[self.data["time_idx"] > test_cutoff].copy()

        if len(test_data) < horizon:
            raise RuntimeError(f"The test set is too small. At least {horizon} rows are needed, but only {len(test_data)} are available")

        # The validation set should also have at least number rows equal to ENC + DEC
        val_cutoff = test_cutoff - horizon
        val_data = self.data[(self.data["time_idx"] > val_cutoff) & (self.data["time_idx"] <= test_cutoff)].copy()

        if len(val_data) < horizon:
            raise RuntimeError(f"The validation set is too small. At least {horizon} rows are needed, but only {len(val_data)} are available")

        # Everything before validation cutoff becomes train
        train_data = self.data[self.data["time_idx"] <= val_cutoff].copy()

        print(f"Train rows: {len(train_data)}")
        print(f"Val rows:   {len(val_data)}")
        print(f"Test rows:  {len(test_data)}")

        train_tsds = TimeSeriesDataSet(
            train_data,
            time_idx="time_idx",
            target=TFTConfig.TARGET_COLS,
            group_ids=["series_id"],
            min_encoder_length=ENC,
            max_encoder_length=ENC,
            min_prediction_length=DEC,
            max_prediction_length=DEC,
            static_reals=[],
            time_varying_known_reals=TFTConfig.KNOWN_REALS,
            time_varying_unknown_reals=TFTConfig.TARGET_COLS,
            target_normalizer=target_normalizer,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            time_varying_known_categoricals=TFTConfig.KNOWN_CATEGORICALS
        )

        val_tsds = TimeSeriesDataSet.from_dataset(
            train_tsds,
            val_data,
            predict=True,
            stop_randomization=True
        )

        test_tsds = TimeSeriesDataSet.from_dataset(
            train_tsds,
            test_data,
            predict=True,
            stop_randomization=True
        )

        train_dl = train_tsds.to_dataloader(
            train=True,
            batch_size=TFTConfig.BATCH_SIZE,
            num_workers=0,
        )
        val_dl = val_tsds.to_dataloader(
            train=False,
            batch_size=TFTConfig.BATCH_SIZE,
            num_workers=0,
        )
        test_dl = test_tsds.to_dataloader(
            train=False,
            batch_size=TFTConfig.BATCH_SIZE,
            num_workers=0,
        )

        return train_tsds, val_tsds, test_tsds, train_dl, val_dl, test_dl





































