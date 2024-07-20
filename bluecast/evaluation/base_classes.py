"""Base classes for evaluation purposes"""

from abc import ABC, abstractmethod
from typing import Union

import pandas as pd
import polars as pl

from bluecast.blueprints.cast import BlueCast
from bluecast.blueprints.cast_cv import BlueCastCV
from bluecast.blueprints.cast_cv_regression import BlueCastCVRegression
from bluecast.blueprints.cast_regression import BlueCastRegression


class DataReader(ABC):
    """Abstract class to define error reading out of fold datasets from BlueCast pipelines."""

    @abstractmethod
    def read_data_from_bluecast_instance(
        self, bluecast_instance: Union[BlueCast, BlueCastRegression]
    ) -> pl.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def read_data_from_bluecast_cv_instance(
        self, bluecast_instance: Union[BlueCastCV, BlueCastCVRegression]
    ) -> pl.DataFrame:
        raise NotImplementedError


class ErrorAnalyser(ABC):
    """Abstract class to define analysing prediction errors on out of fold datasets"""

    @abstractmethod
    def analyse_errors(self, df) -> None:
        raise NotImplementedError

    @abstractmethod
    def show_leaderboard(self) -> pd.DataFrame:
        raise NotImplementedError
