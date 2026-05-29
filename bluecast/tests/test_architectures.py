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


class TestCustomArchitectures:
    @patch("sklearn.model_selection.cross_val_score")
    @patch("bluecast.ml_modelling.pytorch_models.PyTorchMLPClassifier.fit")
    @patch("bluecast.ml_modelling.pytorch_models.PyTorchMLPClassifier.predict_proba")
    @patch("bluecast.ml_modelling.pytorch_models.PyTorchMLPClassifier.predict")
    @patch("bluecast.ml_modelling.pytorch_models.PyTorchSO1DCNNClassifier.fit")
    @patch("bluecast.ml_modelling.pytorch_models.PyTorchSO1DCNNClassifier.predict_proba")
    @patch("bluecast.ml_modelling.pytorch_models.PyTorchSO1DCNNClassifier.predict")
    def test_custom_models_classification(self, mock_cnn_pred, mock_cnn_proba, mock_cnn_fit, mock_mlp_pred, mock_mlp_proba, mock_mlp_fit, mock_cv_score, tiny_classification_data):
        mock_cv_score.return_value = np.array([0.5, 0.6])
        mock_mlp_proba.return_value = np.zeros((10, 2))
        mock_mlp_pred.return_value = np.zeros(10)
        mock_cnn_proba.return_value = np.zeros((10, 2))
        mock_cnn_pred.return_value = np.zeros(10)
        X, y = tiny_classification_data
        X_num = __import__("pandas").DataFrame(
            {col: X[col].values for col in X.columns if col != "cat1"}
        )

        from bluecast.ai.architectures import (
            MLPClassificationModel, HistGBClassificationModel, SO1DCNNClassificationModel, RandomForestClassificationModel
        )
        
        models = [
            MLPClassificationModel(scoring="roc_auc"),
            HistGBClassificationModel(scoring="roc_auc"),
            SO1DCNNClassificationModel(scoring="roc_auc"),
            RandomForestClassificationModel(scoring="roc_auc")
        ]
        
        import traceback
        for model in models:
            model.conf_tuning = {"tuning_rounds": 1, "tuning_max_runtime": 10}
            try:
                model.autotune(X_num, X_num, y, y)
            except Exception as e:
                print(f"Exception in {model.__class__.__name__}:")
                traceback.print_exc()
                raise
            
            # Mock the internal sklearn/PyTorch model to test architectures predict logic
            model.model = MagicMock()
            model.model.predict.return_value = np.zeros(len(y))
            if hasattr(model, "scoring") and "roc_auc" in getattr(model, "scoring", ""):
                model.model.predict_proba.return_value = np.zeros((len(y), 2))
            
            model.predict(X_num)
            
    @patch("sklearn.model_selection.cross_val_score")
    @patch("bluecast.ml_modelling.pytorch_models.PyTorchMLPRegressor.fit")
    @patch("bluecast.ml_modelling.pytorch_models.PyTorchMLPRegressor.predict")
    @patch("bluecast.ml_modelling.pytorch_models.PyTorchSO1DCNNRegressor.fit")
    @patch("bluecast.ml_modelling.pytorch_models.PyTorchSO1DCNNRegressor.predict")
    def test_custom_models_regression(self, mock_cnn_pred, mock_cnn_fit, mock_mlp_pred, mock_mlp_fit, mock_cv_score, tiny_regression_data):
        mock_cv_score.return_value = np.array([-0.5, -0.6])
        mock_mlp_pred.return_value = np.zeros(10)
        mock_cnn_pred.return_value = np.zeros(10)
        X, y = tiny_regression_data
        X_num = __import__("pandas").DataFrame(
            {col: X[col].values for col in X.columns if col != "cat1"}
        )

        from bluecast.ai.architectures import (
            MLPRegressionModel, HistGBRegressionModel, SO1DCNNRegressionModel, RandomForestRegressionModel
        )
        
        models = [
            MLPRegressionModel(scoring="neg_mean_absolute_error"),
            HistGBRegressionModel(scoring="neg_mean_absolute_error"),
            SO1DCNNRegressionModel(scoring="neg_mean_absolute_error"),
            RandomForestRegressionModel(scoring="neg_mean_absolute_error")
        ]
        
        for model in models:
            model.conf_tuning = {"tuning_rounds": 1, "tuning_max_runtime": 10}
            model.autotune(X_num, X_num, y, y)
            
            model.model = MagicMock()
            model.model.predict.return_value = np.zeros(len(y))
            
            model.predict(X_num)


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
    @patch("bluecast.ml_modelling.pytorch_models.PyTorchMLPRegressor.fit")
    def test_autotune(self, mock_fit, mock_optuna, tiny_regression_data):
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
