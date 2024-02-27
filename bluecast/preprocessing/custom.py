"""This module contains the CustomPreprocessing class. This is an entry point for last mile computations before model
training or tuning."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import pandas as pd


class CustomPreprocessing(ABC):
    """This class is an entry point for last mile computations before model training or tuning. It is an abstract class
    and must be extended by the user. For fit_transform x_train and y_train are passed. For transform x_test and y_test
    are passed in the BlueCast pipeline. Use prediction_mode = False to skip processing the missing targets.
    :param trained_patterns: Optional. Can we used to save anything from training to be loaded and used in transform.
    If more placeholders are needed, use a dictionary.
    """

    def __init__(self, trained_patterns: Optional[Any] = None):
        self.trained_patterns = trained_patterns

    @abstractmethod
    def fit_transform(
        self, df: pd.DataFrame, target: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        return df, target

    @abstractmethod
    def transform(
        self,
        df: pd.DataFrame,
        target: Optional[pd.Series] = None,
        predicton_mode: bool = False,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Use prediction mode to not process the missing target during inference."""
        return df, target
