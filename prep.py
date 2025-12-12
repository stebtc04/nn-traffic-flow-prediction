from _config import TFTConfig
import datetime
import warnings
from typing import cast
import numpy as np
import pandas as pd
from pydantic import BaseModel

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


    def _impute_missing_values(self, data: pd.DataFrame, r: str = "gamma") -> pd.DataFrame:
        if r not in cast(dict, RegressorTypes.model_fields).keys():  # cast is used to silence static type checker warnings
            raise ValueError(
                f"Regressor type '{r}' is not supported. Must be one of: {cast(dict, RegressorTypes.model_fields).keys()}")

        reg = None
        if r == "lasso":
            reg = ZeroInflatedRegressor(
                regressor=Lasso(random_state=100, fit_intercept=True),
                classifier=DecisionTreeClassifier(random_state=100)
            )  # Using Lasso regression (L1 Penalization) to get better results in case of non-informative columns present in the data (coverage data, because their values all the same)
        elif r == "gamma":
            reg = ZeroInflatedRegressor(
                regressor=GammaRegressor(fit_intercept=True, verbose=0),
                classifier=DecisionTreeClassifier(random_state=100)
            )  # Using Gamma regression to address for the zeros present in the data (which will need to be predicted as well)
        elif r == "quantile":
            reg = ZeroInflatedRegressor(
                regressor=QuantileRegressor(fit_intercept=True),
                classifier=DecisionTreeClassifier(random_state=100)
            )

        mice_imputer = IterativeImputer(
            estimator=reg,
            random_state=100,
            verbose=0,
            imputation_order="roman",
            initial_strategy="mean"
        )  # Imputation order is set to arabic so that the imputations start from the right (so from the traffic volume columns)

        return pd.DataFrame(mice_imputer.fit_transform(data), columns=self.data.columns)  # Fitting the imputer and processing all the data columns except the date one #TODO BOTTLENECK. MAYBE USE POLARS LazyFrame or PyArrow?


    def standard_preprocess(self, series_id: str = "series_0") -> pd.DataFrame:
        if self.data.empty:
            raise Exception("Empty file detected! Try with another one")

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

        print(self.data.columns)
        print([[*TFTConfig.TARGET_COLS, *TFTConfig.KNOWN_REALS]]) #TODO date is missing

        self.data[[*TFTConfig.TARGET_COLS, *TFTConfig.KNOWN_REALS]] = self._impute_missing_values(self.data[[*TFTConfig.TARGET_COLS, *TFTConfig.KNOWN_REALS]], r="gamma")

        print(self.data.columns)
        print(self.data.head(5))
        print(self.data.dtypes)
        print(self.data.isna().sum())

        # Set or create group id (series identifier) - single series by default
        self.data["series_id"] = series_id
        self.data["time_idx"] = (self.data.groupby("series_id")["date"].rank(method="dense").astype(int) - 1) # Monotonically increasing time_idx per series

        return self.data


    def nn_preprocess(self):

        unknown_reals: list = TFTConfig.TARGET_COLS  # PyTorch Forecasting handles targets separately
        # These two variables are needed for model to learn from past target values (unknown reals) and predict multiple targets

        max_encoder_length: int = TFTConfig.ENCODER_LENGTH
        max_prediction_length: int = TFTConfig.DECODER_LENGTH

        # Use MultiNormalizer for multi-target normalization (targets are a list)
        # Each group's target distribution is normalized separately via GroupNormalizer inside MultiNormalizer
        try:
            target_normalizer = MultiNormalizer([GroupNormalizer(groups=["series_id"]) for _ in TFTConfig.TARGET_COLS])
        except:
            # Fallback
            warnings.warn("MultiNormalizer not available; falling back to single GroupNormalizer on first target")
            target_normalizer = GroupNormalizer(groups=["series_id"])

        training_cutoff = self.data["time_idx"].max() - max_prediction_length - 365  # Leaving about 1 year for validation and testing if possible
        if training_cutoff < max_encoder_length + max_prediction_length:
            training_cutoff = int(self.data["time_idx"].max() * 0.75) #In case there's particularly few data, we're leaving a little bit of data for the validation set by taking less of it for training

        test_cutoff = self.data["time_idx"].max() - max_prediction_length


        # TimeSeriesDataSet is needed for PyTorch-Forecasting models
        data = TimeSeriesDataSet(
            self.data,
            time_idx="time_idx",
            target=TFTConfig.TARGET_COLS,  # For multi-targets, TimeSeriesDataSet supports passing target as list
            group_ids=["series_id"],
            min_encoder_length=max_encoder_length,  # Allowing variable encoder length
            max_encoder_length=max_encoder_length,
            min_prediction_length=max_prediction_length,
            max_prediction_length=max_prediction_length,
            static_reals=[],
            time_varying_known_reals=TFTConfig.KNOWN_REALS,
            time_varying_unknown_reals=TFTConfig.TARGET_COLS,  # We declare targets as "unknown reals"
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            target_normalizer=target_normalizer
        )

        # Split data into training, testing and validation
        train_data = data.filter(lambda x: x["time_idx"].max() <= training_cutoff)
        test_data = data.filter(lambda x: x["time_idx"].max() > test_cutoff - max_encoder_length)
        val_data = data.filter(lambda x: x["time_idx"].max() > training_cutoff)

        # Dataloaders wrap an iterable around the dataset to enable easy access to the samples when training models
        train_dataloader = train_data.to_dataloader(train=True, batch_size=TFTConfig.BATCH_SIZE, num_workers=0)
        test_dataloader = test_data.to_dataloader(train=False, batch_size=TFTConfig.BATCH_SIZE, num_workers=0)
        val_dataloader = val_data.to_dataloader(train=False, batch_size=TFTConfig.BATCH_SIZE, num_workers=0)

        return train_data, test_data, val_data, train_dataloader, test_dataloader, val_dataloader





































