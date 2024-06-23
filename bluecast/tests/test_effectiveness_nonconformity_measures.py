from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from bluecast.conformal_prediction.effectiveness_nonconformity_measures import (
    avg_c,
    convert_expected_effectiveness_nonconformity_input_types,
    one_c,
)


def create_synthetic_binary_data() -> Tuple[np.ndarray, pd.Series]:
    y_true_binary = pd.Series([0, 1, 0])
    synthetic_results_binary = np.asarray([0.7, 0.3, 0.0])
    return synthetic_results_binary, y_true_binary


def create_synthetic_prediction_set() -> pd.DataFrame:
    synthetic_results_sets = pd.DataFrame(
        {
            "prediction_set": [
                {0},
                {0, 1, 2},
                {0},
            ]
        }
    )
    return synthetic_results_sets


def test_one_c():
    synthetic_results_sets = create_synthetic_prediction_set()
    assert one_c(synthetic_results_sets) == 2 / 3
    assert one_c(pd.DataFrame(synthetic_results_sets)) == 2 / 3
    assert one_c(pd.Series(synthetic_results_sets.values.tolist())) == 2 / 3


def test_avg_c():
    synthetic_results_sets = create_synthetic_prediction_set()
    assert avg_c(synthetic_results_sets) == 5 / 3
    assert avg_c(pd.DataFrame(synthetic_results_sets)) == 5 / 3
    assert avg_c(pd.Series(synthetic_results_sets.values.tolist())) == 5 / 3


def test_convert_expected_effectiveness_nonconformity_input_types():
    synthetic_results_sets = create_synthetic_prediction_set()
    trans_synthetic_results_sets = (
        convert_expected_effectiveness_nonconformity_input_types(
            synthetic_results_sets["prediction_set"]
        )
    )
    assert isinstance(trans_synthetic_results_sets, np.ndarray)

    trans_synthetic_results_sets = (
        convert_expected_effectiveness_nonconformity_input_types(
            synthetic_results_sets["prediction_set"].values
        )
    )
    assert isinstance(trans_synthetic_results_sets, np.ndarray)

    with pytest.raises(ValueError):
        convert_expected_effectiveness_nonconformity_input_types(["a", "b", "c"])
