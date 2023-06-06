from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd


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
    ):
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        pass
