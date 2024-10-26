from unittest.mock import MagicMock, patch

import numpy as np
import optuna
import pandas as pd
import pytest
from optuna import Trial
from optuna.pruners import HyperbandPruner
from optuna.samplers import CmaEsSampler

from bluecast.blueprints.cast import BlueCast
from bluecast.blueprints.cast_cv import BlueCastCV
from bluecast.blueprints.cast_cv_regression import BlueCastCVRegression
from bluecast.blueprints.cast_regression import BlueCastRegression
from bluecast.blueprints.orchestration import ModelMatchMaker, OptunaWeights
from bluecast.monitoring.data_monitoring import DataDrift


@pytest.fixture
def setup_model_match_maker():
    mm = ModelMatchMaker()

    # Creating mock instances of each model type
    model1 = MagicMock(BlueCast)
    model2 = MagicMock(BlueCastCV)
    model3 = MagicMock(BlueCastRegression)
    model4 = MagicMock(BlueCastCVRegression)

    # Creating sample DataFrames
    df1 = pd.DataFrame({"a": [i for i in range(50)], "b": [i for i in range(50)]})
    df2 = pd.DataFrame(
        {"a": [i for i in range(100, 150)], "b": [i for i in range(150, 200)]}
    )
    df3 = pd.DataFrame(
        {"a": [i for i in range(1000, 1050)], "b": [i for i in range(700, 750)]}
    )
    df4 = pd.DataFrame({"a": [i for i in range(-83, -33)], "b": [i for i in range(50)]})

    # Appending models and datasets
    mm.append_model_and_dataset(model1, df1)
    mm.append_model_and_dataset(model2, df2)
    mm.append_model_and_dataset(model3, df3)
    mm.append_model_and_dataset(model4, df4)

    return mm, model1, model2, model3, model4, df1, df2, df3, df4


def test_append_model_and_dataset(setup_model_match_maker):
    mm, model1, model2, model3, model4, df1, df2, df3, df4 = setup_model_match_maker

    assert len(mm.bluecast_instances) == 4
    assert len(mm.training_datasets) == 4
    assert mm.bluecast_instances[0] == model1
    assert mm.bluecast_instances[1] == model2
    assert mm.bluecast_instances[2] == model3
    assert mm.bluecast_instances[3] == model4
    assert mm.training_datasets[0].equals(df1)
    assert mm.training_datasets[1].equals(df2)
    assert mm.training_datasets[2].equals(df3)
    assert mm.training_datasets[3].equals(df4)


def test_find_best_match_no_drift(setup_model_match_maker, mocker):
    mm, _, _, _, _, df1, _, _, _ = setup_model_match_maker
    mocker.patch.object(DataDrift, "adversarial_validation", return_value=np.inf)

    model, dataset = mm.find_best_match(df1, ["a", "b"], cat_columns=[], delta=0.1)

    assert model is None
    assert dataset is None


def test_find_best_match_finds_best_model(setup_model_match_maker, mocker):
    mm, _, model2, _, _, _, df2, _, _ = setup_model_match_maker
    mocker.patch.object(
        DataDrift, "adversarial_validation", side_effect=[0.3, 0.5, 0.4, 0.6]
    )

    model, dataset = mm.find_best_match(df2, ["a", "b"], cat_columns=[], delta=0.1)

    assert model == model2
    assert dataset.equals(mm.training_datasets[1])


def test_find_best_match_no_model_reaches_threshold(setup_model_match_maker, mocker):
    mm, _, _, _, _, df1, _, _, _ = setup_model_match_maker
    mocker.patch.object(
        DataDrift, "adversarial_validation", side_effect=[0.61, 0.7, 0.8, 0.9]
    )

    model, dataset = mm.find_best_match(df1, ["a", "b"], cat_columns=[], delta=0.1)

    assert model is None
    assert dataset is None


def test_find_best_match_with_all_models(setup_model_match_maker, mocker):
    mm, model1, model2, model3, model4, _, df2, _, _ = setup_model_match_maker
    mocker.patch.object(
        DataDrift, "adversarial_validation", side_effect=[0.7, 0.5, 0.4, 0.6]
    )

    model, dataset = mm.find_best_match(df2, ["a", "b"], cat_columns=[], delta=0.1)

    # Verify the model with the best score is chosen
    assert model == model2
    assert dataset.equals(mm.training_datasets[1])


# Mock Trial for testing _objective method
@pytest.fixture
def mock_trial():
    trial = MagicMock(spec=Trial)
    trial.suggest_float.side_effect = lambda name, low, high: 0.5
    return trial


@pytest.fixture
def sample_data():
    # Sample true labels and predictions for testing
    y_true = np.array([1, 0, 1, 0, 1])
    y_preds = [
        np.array([0.6, 0.4, 0.7, 0.2, 0.9]),
        np.array([0.5, 0.3, 0.8, 0.1, 0.85]),
    ]
    return y_true, y_preds


def test_optuna_weights_initialization():
    model = OptunaWeights(random_state=42, n_trials=100)
    assert model.random_state == 42
    assert model.n_trials == 100
    assert model.optimize_direction == "maximize"
    assert model.weights == []


def test_objective_method(mock_trial, sample_data):
    model = OptunaWeights(random_state=42)
    y_true, y_preds = sample_data

    auc_score = model._objective(mock_trial, y_true, y_preds)

    # Check if suggest_float was called correctly
    assert mock_trial.suggest_float.call_count == len(y_preds) - 1
    assert 0 <= auc_score <= 1, "AUC score should be in the range [0, 1]"


def test_fit_method(sample_data):
    y_true, y_preds = sample_data
    model = OptunaWeights(random_state=42, n_trials=10)

    with patch.object(CmaEsSampler, "__init__", lambda self, seed: None):
        with patch.object(HyperbandPruner, "__init__", lambda self: None):
            model.fit(y_true, y_preds)

    assert model.study is not None, "Study should be created during fit"
    assert len(model.weights) == len(
        y_preds
    ), "Weights should be calculated for each model"
    assert abs(sum(model.weights) - 1.0) < 1e-6, "Weights should sum to 1"


def test_fit_with_single_prediction(sample_data):
    y_true, y_preds = sample_data
    model = OptunaWeights(random_state=42, n_trials=10)

    with pytest.raises(
        ValueError, match="`y_preds` must contain predictions from at least two models."
    ):
        model.fit(y_true, [y_preds[0]])  # Only one prediction


def test_predict_without_fitting(sample_data):
    y_true, y_preds = sample_data
    model = OptunaWeights(random_state=42, n_trials=10)

    with pytest.raises(
        ValueError, match="Model weights have not been optimized. Call `fit` first."
    ):
        model.predict(y_preds)


def test_predict_after_fitting(sample_data):
    y_true, y_preds = sample_data
    model = OptunaWeights(random_state=42, n_trials=10)

    with patch.object(CmaEsSampler, "__init__", lambda self, seed: None):
        with patch.object(HyperbandPruner, "__init__", lambda self: None):
            model.fit(y_true, y_preds)

    prediction = model.predict(y_preds)
    assert isinstance(prediction, np.ndarray), "Prediction should be a numpy array"
    assert (
        prediction.shape == y_true.shape
    ), "Prediction shape should match y_true shape"
    assert (0 <= prediction).all() and (
        prediction <= 1
    ).all(), "Predictions should be probabilities"


def test_study_direction_maximize(sample_data):
    y_true, y_preds = sample_data
    model = OptunaWeights(random_state=42, n_trials=10, optimize_direction="maximize")
    with patch.object(CmaEsSampler, "__init__", lambda self, seed: None):
        with patch.object(HyperbandPruner, "__init__", lambda self: None):
            model.fit(y_true, y_preds)
    assert model.study.direction == optuna.study.StudyDirection.MAXIMIZE


def test_study_direction_minimize(sample_data):
    y_true, y_preds = sample_data
    model = OptunaWeights(random_state=42, n_trials=10, optimize_direction="minimize")
    with patch.object(CmaEsSampler, "__init__", lambda self, seed: None):
        with patch.object(HyperbandPruner, "__init__", lambda self: None):
            model.fit(y_true, y_preds)
    assert model.study.direction == optuna.study.StudyDirection.MINIMIZE
