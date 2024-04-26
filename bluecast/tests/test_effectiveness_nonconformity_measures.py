from typing import Tuple

import numpy as np
import pandas as pd

from bluecast.conformal_prediction.effectiveness_nonconformity_measures import (
    avg_c,
    one_c,
)


def create_synthetic_binary_data() -> Tuple[np.ndarray, pd.Series]:
    y_true_binary = pd.Series([0, 1, 0])
    synthetic_results_binary = np.asarray([0.7, 0.3, 0.0])
    return synthetic_results_binary, y_true_binary


def create_synthetic_prediction_set() -> np.ndarray:
    synthetic_results_sets = pd.DataFrame(
        {
            "prediction_set": [
                {0},
                {0, 1, 2},
                {0},
            ]
        }
    )
    return synthetic_results_sets.values


def test_one_c():
    synthetic_results_sets = create_synthetic_prediction_set()
    assert one_c(synthetic_results_sets) == 2 / 3


def test_avg_c():
    synthetic_results_sets = create_synthetic_prediction_set()
    assert avg_c(synthetic_results_sets) == 5 / 3
