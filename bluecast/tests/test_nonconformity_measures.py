from decimal import Decimal
from typing import Tuple
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from bluecast.conformal_prediction.nonconformity_measures import (
    brier_score,
    hinge_loss,
    margin_nonconformity_measure,
)


def create_synthetic_binary_data() -> Tuple[np.ndarray, pd.Series]:
    y_true_binary = pd.Series([0, 1, 0])
    synthetic_results_binary = np.asarray([0.7, 0.3, 0.0])
    return synthetic_results_binary, y_true_binary


def create_synthetic_multiclass_data() -> Tuple[np.ndarray, pd.Series]:
    y_true_multiclass = pd.Series([0, 1, 2])
    synthetic_results_multiclass = np.asarray(
        [[0.7, 0.3, 0.0], [0.7, 0.3, 0.0], [0.7, 0.3, 0.0]]
    )
    return synthetic_results_multiclass, y_true_multiclass


def test_hinge_loss():
    synthetic_results_binary, y_true_binary = create_synthetic_binary_data()
    synthetic_results_multiclass, y_true_multiclass = create_synthetic_multiclass_data()
    assert np.allclose(
        hinge_loss(y_true_binary, synthetic_results_binary), np.asarray([0.7, 0.7, 0.0])
    )
    assert np.allclose(
        hinge_loss(y_true_multiclass, synthetic_results_multiclass),
        np.asarray([0.3, 0.7, 1.0]),
    )


def test_margin_nonconformity_measure():
    synthetic_results_binary, y_true_binary = create_synthetic_binary_data()
    synthetic_results_multiclass, y_true_multiclass = create_synthetic_multiclass_data()
    result_binary = margin_nonconformity_measure(
        y_true_binary, synthetic_results_binary
    )
    result_multiclass = margin_nonconformity_measure(
        y_true_multiclass, synthetic_results_multiclass
    )
    assert np.allclose(result_binary, np.asarray([-0.4, -0.4, 1.0]))
    assert np.allclose(result_multiclass, np.asarray([0.4, -0.4, -0.7]))


def test_brier_score():
    synthetic_results_binary, y_true_binary = create_synthetic_binary_data()
    synthetic_results_multiclass, y_true_multiclass = create_synthetic_multiclass_data()
    assert np.allclose(
        brier_score(y_true_binary, synthetic_results_binary),
        np.asarray([0.49, 0.49, 0.0]),
    )
    assert np.allclose(
        brier_score(y_true_multiclass, synthetic_results_multiclass),
        np.asarray([0.09, 0.49, 1.0]),
    )


def test_hinge_loss_value_error():
    # Create synthetic data where conversion to float64 would raise a ValueError
    y_true = pd.Series([0, 1, 0])
    # Use Decimal objects that would cause an issue when converting to float64
    y_hat = np.asarray(
        [
            [Decimal("0.7"), Decimal("0.3")],
            [Decimal("0.3"), Decimal("0.7")],
            [Decimal("0.0"), Decimal("1.0")],
        ]
    )

    result = hinge_loss(y_true, y_hat)

    # The result should not be float64 due to the ValueError, so we expect dtype 'object'
    assert result.dtype == float


def test_margin_nonconformity_measure_value_error():
    # Create synthetic data where conversion to float64 would raise a ValueError
    y_true = pd.Series([0, 1, 0])
    # Use Decimal objects that would cause an issue when converting to float64
    y_hat = np.asarray(
        [
            [Decimal("0.7"), Decimal("0.3")],
            [Decimal("0.3"), Decimal("0.7")],
            [Decimal("0.0"), Decimal("1.0")],
        ]
    )

    result = margin_nonconformity_measure(y_true, y_hat)

    # The result should not be float64 due to the ValueError, so we expect dtype 'object'
    assert result.dtype == float


def test_brier_score_value_error():
    # Create synthetic data where conversion to float64 would raise a ValueError
    y_true = pd.Series([0, 1, 0])
    # Use Decimal objects that would cause an issue when converting to float64
    y_hat = np.asarray(
        [
            [Decimal("0.7"), Decimal("0.3")],
            [Decimal("0.3"), Decimal("0.7")],
            [Decimal("0.0"), Decimal("1.0")],
        ]
    )

    result = brier_score(y_true, y_hat)

    # The result should not be float64 due to the ValueError, so we expect dtype 'object'
    assert result.dtype == float


# Test data setup
@pytest.fixture
def test_data():
    y_true = pd.Series([0, 1, 1, 0])
    y_hat = np.array([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6], [0.9, 0.1]])
    return y_true, y_hat


def test_hinge_loss_value_error_obj(test_data):
    y_true, y_hat = test_data

    # Patch np.asarray to raise ValueError only at specific points
    with patch("numpy.asarray", side_effect=ValueError) as mock_asarray:
        with pytest.raises(ValueError):
            _ = hinge_loss(y_true, y_hat)
        mock_asarray.assert_called()  # Check that np.asarray was indeed called


def test_margin_nonconformity_measure_value_error_obj(test_data):
    y_true, y_hat = test_data

    with patch("numpy.asarray", side_effect=ValueError):
        with pytest.raises(ValueError):
            _ = margin_nonconformity_measure(y_true, y_hat)


def test_brier_score_value_error_obj(test_data):
    y_true, y_hat = test_data

    with patch("numpy.asarray", side_effect=ValueError):
        with pytest.raises(ValueError):
            _ = brier_score(y_true, y_hat)
