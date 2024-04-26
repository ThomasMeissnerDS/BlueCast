from typing import Union

import numpy as np
import pandas as pd


def absolute_error(
    y_true: Union[pd.Series, np.ndarray], y_hat: np.ndarray
) -> np.ndarray:
    """
    Calculate the absolute error per row.

    It calculates the absolute error between the predictions and the actual results.
    :param y_true: True values
    :param y_hat: Predicted values
    :return: Absolute error per row
    """
    if isinstance(y_true, pd.Series):
        return np.abs(y_hat - y_true.values)
    else:
        return np.abs(y_hat - y_true)


def normalized_error(
    y_true: Union[pd.Series, np.ndarray], y_hat: np.ndarray, scale: float
) -> np.ndarray:
    """
    Calculate the normalized error per row.

    The normalized error nonconformity measure is the absolute error divided
    by an estimate of the prediction error’s scale, such as the mean absolute error (MAE) or the
    standard deviation of the residuals. This measure can be used with any regression model and
    helps account for heteroscedasticity in the data.
    :param y_true: True values
    :param y_hat: Predicted values
    :param scale: Scale to normalize the error
    :return: Absolute error per row
    """
    if isinstance(y_true, pd.Series):
        return np.abs(y_hat - y_true.values) / scale
    else:
        return np.abs(y_hat - y_true) / scale
