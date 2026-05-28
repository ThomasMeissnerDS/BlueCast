"""Tests for architecture factories and model registry."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from bluecast.ai.architectures import get_architectures_for_problem


@pytest.fixture
def tiny_regression_data():
    rng = np.random.default_rng(42)
    n = 50
    X = pd.DataFrame(
        {
            "num1": rng.normal(0, 1, n),
            "num2": rng.normal(5, 2, n),
            "cat1": rng.choice(["a", "b", "c"], n),
        }
    )
    y = pd.Series(X["num1"] * 2 + rng.normal(0, 0.1, n))
    return X, y


@pytest.fixture
def tiny_classification_data():
    rng = np.random.default_rng(42)
    n = 50
    X = pd.DataFrame(
        {
            "num1": rng.normal(0, 1, n),
            "num2": rng.normal(5, 2, n),
            "cat1": rng.choice(["a", "b", "c"], n),
        }
    )
    y = pd.Series(rng.choice([0, 1], n))
    return X, y


class TestArchitecturesRegistry:
    def test_get_architectures_for_problem_binary(self):
        archs = get_architectures_for_problem("binary")
        assert "catboost" in archs
        assert "xgboost" in archs
        assert "histgb" in archs
        assert "linear" in archs

        # Test factory execution
        cb = archs["catboost"]["factory"]("binary")
        assert cb is None  # CatBoost uses default pipeline

    def test_get_architectures_for_problem_multiclass(self):
        archs = get_architectures_for_problem("multiclass")
        assert "catboost" in archs
        assert "xgboost" in archs
        assert "histgb" in archs
        assert "linear" in archs

    def test_get_architectures_for_problem_regression(self):
        archs = get_architectures_for_problem("regression")
        assert "catboost" in archs
        assert "xgboost" in archs
        assert "linear" in archs
        assert "mlp" in archs
        assert "randomforest" in archs


class TestHistGBArchitecture:
    def test_factory_regression(self):
        factory = get_architectures_for_problem("regression")["histgb"]["factory"]
        model = factory("regression")
        assert model is not None

    @patch("optuna.create_study")
    def test_autotune(self, mock_optuna, tiny_regression_data):
        X, y = tiny_regression_data
        X_num = __import__("pandas").DataFrame(
            {col: X[col].values for col in X.columns if col != "cat1"}
        )

        factory = get_architectures_for_problem("regression")["histgb"]["factory"]
        model = factory("regression")

        mock_study = MagicMock()
        mock_study.best_params = {"max_depth": 5, "min_samples_leaf": 5}
        mock_study.best_value = 0.5
        mock_optuna.return_value = mock_study

        model.autotune(X_num, X_num, y, y)
        assert model.model is not None

    def test_fit_predict(self, tiny_regression_data):
        X, y = tiny_regression_data
        X_num = __import__("pandas").DataFrame(
            {col: X[col].values for col in X.columns if col != "cat1"}
        )

        factory = get_architectures_for_problem("regression")["histgb"]["factory"]
        model = factory("regression")
        with patch.object(model.__class__, "autotune"):
            model.model = MagicMock()
            model.model.predict.return_value = np.zeros(len(y))
            model.imputer = MagicMock()
            model.imputer.transform.return_value = np.zeros((len(y), X_num.shape[1]))
            model.scaler = MagicMock()
            model.scaler.transform.return_value = np.zeros((len(y), X_num.shape[1]))

            model.fit(X_num, X_num, y, y)
            preds = model.predict(X_num)
        assert len(preds) == len(y)


class TestRandomForestArchitecture:
    def test_factory_regression(self):
        factory = get_architectures_for_problem("regression")["randomforest"]["factory"]
        model = factory("regression")
        assert model is not None

    @patch("optuna.create_study")
    def test_autotune(self, mock_optuna, tiny_regression_data):
        X, y = tiny_regression_data
        X_num = __import__("pandas").DataFrame(
            {col: X[col].values for col in X.columns if col != "cat1"}
        )

        factory = get_architectures_for_problem("regression")["randomforest"]["factory"]
        model = factory("regression")

        mock_study = MagicMock()
        mock_study.best_params = {"max_depth": 10, "n_estimators": 50}
        mock_optuna.return_value = mock_study

        model.autotune(X_num, X_num, y, y)
        assert model.model is not None

    def test_fit_predict(self, tiny_regression_data):
        X, y = tiny_regression_data
        X_num = __import__("pandas").DataFrame(
            {col: X[col].values for col in X.columns if col != "cat1"}
        )

        factory = get_architectures_for_problem("regression")["randomforest"]["factory"]
        model = factory("regression")
        with patch.object(model.__class__, "autotune"):
            model.model = MagicMock()
            model.model.predict.return_value = np.zeros(len(y))
            model.imputer = MagicMock()
            model.imputer.transform.return_value = np.zeros((len(y), X_num.shape[1]))
            model.scaler = MagicMock()
            model.scaler.transform.return_value = np.zeros((len(y), X_num.shape[1]))

            model.fit(X_num, X_num, y, y)
            preds = model.predict(X_num)
        assert len(preds) == len(y)


class TestLinearArchitecture:
    def test_factory_regression(self):
        factory = get_architectures_for_problem("regression")["linear"]["factory"]
        model = factory("regression")
        assert model is not None

    @patch("optuna.create_study")
    def test_autotune(self, mock_optuna, tiny_regression_data):
        X, y = tiny_regression_data
        X_num = __import__("pandas").DataFrame(
            {col: X[col].values for col in X.columns if col != "cat1"}
        )
        factory = get_architectures_for_problem("regression")["linear"]["factory"]
        model = factory("regression")

        mock_study = MagicMock()
        mock_study.best_params = {
            "model_type": "ridge",
            "alpha": 1.0,
            "target_transformer_type": "standard",
            "imputer_strategy": "median",
            "learning_rate_init": 0.05,
            "batch_size": 256,
            "dropout_rate": 0.2,
        }
        mock_study.best_value = 0.5
        mock_optuna.return_value = mock_study

        model.autotune(X_num, X_num, y, y)
        assert model.model is not None

    @patch("bluecast.blueprints.custom_model_recipes.PyTorchMLPRegressor.fit")
    def test_fit_predict(self, mock_fit, tiny_regression_data):
        X, y = tiny_regression_data
        X_num = __import__("pandas").DataFrame(
            {col: X[col].values for col in X.columns if col != "cat1"}
        )

        factory = get_architectures_for_problem("regression")["linear"]["factory"]
        model = factory("regression")

        with patch.object(model.__class__, "autotune"):
            # Fake the internal model and preprocessing
            model.model = MagicMock()
            model.model.predict.return_value = np.zeros(len(y))
            model.scaler = MagicMock()
            model.imputer = MagicMock()

            model.fit(X_num, X_num, y, y)
            preds = model.predict(X_num)

        assert len(preds) == len(y)


class TestMLPArchitecture:
    def test_factory_regression(self):
        factory = get_architectures_for_problem("regression")["mlp"]["factory"]
        model = factory("regression")
        assert model is not None

    @patch("optuna.create_study")
    def test_autotune(self, mock_optuna, tiny_regression_data):
        X, y = tiny_regression_data
        X_num = __import__("pandas").DataFrame(
            {col: X[col].values for col in X.columns if col != "cat1"}
        )

        factory = get_architectures_for_problem("regression")["mlp"]["factory"]
        model = factory("regression")

        mock_study = MagicMock()
        mock_study.best_params = {
            "hidden_layer_sizes": (50,),
            "learning_rate_init": 0.01,
        }
        mock_optuna.return_value = mock_study

        model.autotune(X_num, X_num, y, y)
        assert model.model is not None

    @patch("bluecast.ml_modelling.pytorch_models.PyTorchMLPRegressor.fit")
    def test_fit_predict(self, mock_fit, tiny_regression_data):
        X, y = tiny_regression_data
        X_num = __import__("pandas").DataFrame(
            {col: X[col].values for col in X.columns if col != "cat1"}
        )

        factory = get_architectures_for_problem("regression")["mlp"]["factory"]
        model = factory("regression")

        with patch.object(model.__class__, "autotune") as mock_autotune:
            model.model = MagicMock()
            model.model.predict.return_value = np.zeros(len(y))
            model.imputer = MagicMock()
            model.imputer.transform.return_value = np.zeros((len(y), X_num.shape[1]))
            model.scaler = MagicMock()
            model.scaler.transform.return_value = np.zeros((len(y), X_num.shape[1]))

            model.fit(X_num, X_num, y, y)
            preds = model.predict(X_num)

        assert mock_autotune.called
        assert len(preds) == len(y)
