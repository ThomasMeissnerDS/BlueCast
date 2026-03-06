from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd


class ConformalPredictionWrapperBaseClass(ABC):
    """Base class for conformal prediction wrappers.

    Enforces the implementation of the calibrate method.
    """

    @abstractmethod
    def calibrate(
        self,
        x_calibration: pd.DataFrame,
        y_calibration: Union[pd.Series, np.ndarray],
    ) -> None:
        """
        Calibrate a model instance given a calibration set.

        :param x_calibration: Calibration set features. Must be unseen data for the model
        :param y_calibration: Calibration set labels or values
        """
        pass
