from typing import Tuple

import numpy as np
import pandas as pd

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
    assert hinge_loss(y_true_binary, synthetic_results_binary) == np.asarray(
        [0.7, 0.7, 0.0]
    )
    assert (
        hinge_loss(y_true_multiclass, synthetic_results_multiclass)
        == np.asarray([0.3, 0.7, 1.0])
    ).all()


def test_margin_nonconformity_measure():
    synthetic_results_binary, y_true_binary = create_synthetic_binary_data()
    synthetic_results_multiclass, y_true_multiclass = create_synthetic_multiclass_data()
    assert margin_nonconformity_measure(
        y_true_binary, synthetic_results_binary
    ) == np.asarray([-0.4, -0.4, 1.0])
    assert (
        margin_nonconformity_measure(y_true_multiclass, synthetic_results_multiclass)
        == np.asarray([0.4, -0.4, -0.7])
    ).all()


def test_brier_score():
    synthetic_results_binary, y_true_binary = create_synthetic_binary_data()
    synthetic_results_multiclass, y_true_multiclass = create_synthetic_multiclass_data()
    assert (
        brier_score(y_true_binary, synthetic_results_binary)
        == np.asarray([0.09, 0.09, 1.0])
    ).all()
    assert (
        brier_score(y_true_multiclass, synthetic_results_multiclass)
        == np.asarray([0.09, 0.09, 1.0])
    ).all()
