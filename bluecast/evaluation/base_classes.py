"""Base classes for evaluation purposes"""

from abc import ABC, abstractmethod

import polars as pl


class DataReader(ABC):
    """Abstract class to define error reading out of fold datasets from BlueCast pipelines."""

    @abstractmethod
    def read_data_from_bluecast_instance(self) -> pl.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def read_data_from_bluecast_cv_instance(self) -> pl.DataFrame:
        raise NotImplementedError


class ErrorPreprocessor(ABC):
    """Abstract class to define analysing prediction errors on out of fold datasets"""

    @abstractmethod
    def stack_predictions_by_class(self, df: pl.DataFrame) -> pl.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def calculate_errors(self, df) -> pl.DataFrame:
        raise NotImplementedError


class ErrorAnalyser(ABC):
    """Abstract class to define the analysis of prediction errors on out of fold datasets"""

    @abstractmethod
    def analyse_errors(self, df, descending: bool = True) -> None:
        raise NotImplementedError


class ErrorDistributionPlotter(ABC):
    """Abstract class to define the plots for error analysis"""

    @abstractmethod
    def plot_error_distributions(
        self,
        df: pl.DataFrame,
        target_column: str,
    ):
        raise NotImplementedError
