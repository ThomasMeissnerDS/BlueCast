from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
import pandas as pd


class ConformalPredictionWrapperBaseClass(ABC):
    """Base class for conformal prediction wrappers.

    Enforces the implementation of the calibrate method. Supports optional
    group-conditional calibration via the group_columns parameter.
    """

    @abstractmethod
    def calibrate(
        self,
        x_calibration: pd.DataFrame,
        y_calibration: Union[pd.Series, np.ndarray],
        group_columns: Optional[List[str]] = None,
    ) -> None:
        """
        Calibrate a model instance given a calibration set.

        :param x_calibration: Calibration set features. Must be unseen data for the model
        :param y_calibration: Calibration set labels or values
        :param group_columns: Optional list of column names to use for group-conditional
            calibration. When provided, nonconformity scores are stored per group, producing
            group-specific prediction intervals or sets.
        """
        pass
