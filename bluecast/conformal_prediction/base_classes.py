from abc import ABC, abstractmethod
from typing import Any, Union

import numpy as np
import pandas as pd


class ConformalPredictionWrapper(ABC):
    """Base class for the experiment tracker.

    Enforces the implementation of the add_results and retrieve_results_as_df methods.
    """

    @abstractmethod
    def calibrate(
        self,
        model: Any,
        x_calibration: pd.DataFrame,
        y_calibration: Union[pd.Series, np.ndarray],
    ) -> None:
        """
        Calibrate a model instance given a calibration set.
        :param model: An already fitted model instance of any type
        :param x_calibration: Calibration set features. Must be unseen data for the model
        :param y_calibration: Calibration set labels or values
        """
        pass
