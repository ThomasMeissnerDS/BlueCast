"""Base classes for all ML models."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, TypeVar

import numpy as np
import pandas as pd

PredictedProbas = TypeVar("PredictedProbas", np.ndarray, pd.Series)
PredictedClasses = TypeVar("PredictedClasses", np.ndarray, pd.Series)


class BaseClassMlModel(ABC):
    """Base class for all ML models.

    Enforces the implementation of the fit and predict methods.
    If hyperparameter tuning is required, then the fit method should implement the tuning.
    """

    @abstractmethod
    def fit(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> Optional[Any]:
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> Tuple[PredictedProbas, PredictedClasses]:
        """
        Predict on unseen data.

        :return tuple of predicted probabilities and predicted classes
        """
        pass


class BaseClassMlRegressionModel(ABC):
    """Base class for all ML models.

    Enforces the implementation of the fit and predict methods.
    If hyperparameter tuning is required, then the fit method should implement the tuning.
    """

    @abstractmethod
    def fit(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> Optional[Any]:
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict on unseen data.

        :return numpy array of predictions
        """
        pass
