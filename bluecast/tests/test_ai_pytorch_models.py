"""Tests for PyTorch MLP and SO1DCNN models used in BlueCastAI."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

import torch

from bluecast.ml_modelling.pytorch_models import (
    PyTorchMLPRegressor,
    PyTorchSO1DCNNRegressor,
    PyTorchMLPClassifier,
    PyTorchSO1DCNNClassifier,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_regression_data():
    rng = np.random.default_rng(42)
    n = 50
    X = pd.DataFrame(
        {
            "num1": rng.normal(0, 1, n),
            "num2": rng.normal(5, 2, n),
        }
    )
    y = pd.Series(X["num1"] * 2 + rng.normal(0, 0.1, n))
    return X, y


@pytest.fixture
def synthetic_classification_data():
    rng = np.random.default_rng(42)
    n = 50
    X = pd.DataFrame(
        {
            "num1": rng.normal(0, 1, n),
            "num2": rng.normal(5, 2, n),
        }
    )
    y = pd.Series(rng.choice([0, 1], n))
    return X, y


# ---------------------------------------------------------------------------
# MLP Regression
# ---------------------------------------------------------------------------


class TestPyTorchMLPRegressor:
    def test_fit_predict(self, synthetic_regression_data):
        X, y = synthetic_regression_data

        model = PyTorchMLPRegressor(
            hidden_sizes=[16, 8],
            lr=0.01,
            max_iter=2,
            batch_size=16,
            early_stopping_rounds=1,
            random_state=42,
        )

        model.fit(X, y)
        preds = model.predict(X)

        assert len(preds) == len(y)
        assert not np.isnan(preds).any()

    def test_early_stopping(self, synthetic_regression_data):
        X, y = synthetic_regression_data

        # Will trigger early stopping quickly
        model = PyTorchMLPRegressor(
            max_iter=100, early_stopping_rounds=0, validation_fraction=0.5
        )
        model.fit(X, y)

        # The loop should break before 100 epochs
        assert model.convergence_info_["epochs_run"] < 100

    def test_budget_exhausted(self, synthetic_regression_data):
        X, y = synthetic_regression_data

        import time

        model = PyTorchMLPRegressor(
            max_iter=1000, training_deadline=time.time() + 0.001
        )

        model.fit(X, y)
        # Should stop after very few epochs due to timeout
        assert model.convergence_info_["epochs_run"] < 1000


# ---------------------------------------------------------------------------
# MLP Classification
# ---------------------------------------------------------------------------


class TestPyTorchMLPClassifier:
    def test_fit_predict(self, synthetic_classification_data):
        X, y = synthetic_classification_data

        model = PyTorchMLPClassifier(
            hidden_sizes=[16],
            max_iter=2,
            batch_size=16,
        )

        model.fit(X, y)

        # predict returns class labels
        preds = model.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset({0, 1})

        # predict_proba returns probabilities
        probas = model.predict_proba(X)
        assert probas.shape == (len(y), 2)
        assert ((probas >= 0) & (probas <= 1)).all()


# ---------------------------------------------------------------------------
# SO1DCNN Regression
# ---------------------------------------------------------------------------


class TestPyTorchSO1DCNNRegressor:
    def test_fit_predict(self, synthetic_regression_data):
        X, y = synthetic_regression_data

        # Pad data with zeros to ensure width >= kernel_size + padding logic
        X_wide = pd.concat([X, pd.DataFrame(np.zeros((len(X), 10)))], axis=1)
        X_wide.columns = [str(c) for c in X_wide.columns]

        model = PyTorchSO1DCNNRegressor(
            out_channels=8,
            kernel_size=3,
            max_iter=2,
            batch_size=16,
        )

        model.fit(X_wide, y)
        preds = model.predict(X_wide)

        assert len(preds) == len(y)
        assert not np.isnan(preds).any()


# ---------------------------------------------------------------------------
# SO1DCNN Classification
# ---------------------------------------------------------------------------


class TestPyTorchSO1DCNNClassifier:
    def test_fit_predict(self, synthetic_classification_data):
        X, y = synthetic_classification_data

        # Pad data
        X_wide = pd.concat([X, pd.DataFrame(np.zeros((len(X), 10)))], axis=1)
        X_wide.columns = [str(c) for c in X_wide.columns]

        model = PyTorchSO1DCNNClassifier(
            out_channels=8,
            kernel_size=3,
            max_iter=2,
        )

        model.fit(X_wide, y)
        preds = model.predict(X_wide)
        assert len(preds) == len(y)
        assert set(preds).issubset({0, 1})

        probas = model.predict_proba(X_wide)
        assert probas.shape == (len(y), 2)


# ---------------------------------------------------------------------------
# Helper logic
# ---------------------------------------------------------------------------


class TestPyTorchHelpers:
    def test_l1_regularization(self, synthetic_regression_data):
        X, y = synthetic_regression_data

        model = PyTorchMLPRegressor(max_iter=2, l1_ratio=0.5)
        # Should not crash, just applies penalty
        model.fit(X, y)

    def test_activations(self, synthetic_regression_data):
        X, y = synthetic_regression_data

        for act in ["relu", "gelu", "swish"]:
            model = PyTorchMLPRegressor(max_iter=1, activation=act)
            model.fit(X, y)
