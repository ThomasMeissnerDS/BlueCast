from typing import Tuple

import numpy as np
import pandas as pd

from bluecast.conformal_prediction.nonconformity_measures_regression import (
    absolute_error,
    normalized_error,
)


def create_synthetic_regression_data() -> Tuple[np.ndarray, pd.Series]:
    y_true_regression = pd.Series([15.0, 25.0, 40.0])
    synthetic_results_regression = np.asarray([5.0, 25.0, 45.0])
    return synthetic_results_regression, y_true_regression


def test_absolute_error_loss():
    synthetic_results_regression, y_true_regression = create_synthetic_regression_data()
    assert (
        absolute_error(y_true_regression, synthetic_results_regression)
        == np.asarray([10.0, 0.0, 5.0])
    ).all()


def test_normalized_error_error_loss():
    synthetic_results_regression, y_true_regression = create_synthetic_regression_data()
    assert (
        normalized_error(y_true_regression, synthetic_results_regression, 5)
        == np.asarray([2.0, 0.0, 1.0])
    ).all()
