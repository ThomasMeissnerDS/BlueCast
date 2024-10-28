from unittest.mock import MagicMock

import numpy as np
import optuna
import pandas as pd
import pytest
from optuna.trial import create_trial
from sklearn.metrics import roc_auc_score

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


# Mock data for testing
@pytest.fixture
def setup_data():
    np.random.seed(0)
    y_true = np.random.randint(0, 2, 100)  # Binary target
    y_preds = [np.random.rand(100) for _ in range(3)]  # Predictions from 3 models
    return y_true, y_preds


@pytest.fixture
def optuna_weights_instance():
    return OptunaWeights(random_state=42, n_trials=10)


# Test initialization
def test_init(optuna_weights_instance):
    assert optuna_weights_instance.random_state == 42
    assert optuna_weights_instance.n_trials == 10
    assert optuna_weights_instance.objective == roc_auc_score
    assert optuna_weights_instance.optimize_direction == "maximize"
    assert optuna_weights_instance.weights == []


# Test fit method with valid data
def test_fit(setup_data, optuna_weights_instance):
    y_true, y_preds = setup_data
    optuna_weights_instance.fit(y_true, y_preds)
    assert optuna_weights_instance.weights is not None
    assert np.isclose(
        sum(optuna_weights_instance.weights), 1.0
    ), "Weights should sum to 1."


# Test fit method raises error with single prediction
def test_fit_single_prediction(setup_data, optuna_weights_instance):
    y_true, y_preds = setup_data
    with pytest.raises(
        ValueError, match="`y_preds` must contain predictions from at least two models"
    ):
        optuna_weights_instance.fit(y_true, [y_preds[0]])


# Test predict method without calling fit first
def test_predict_without_fit(setup_data, optuna_weights_instance):
    _, y_preds = setup_data
    with pytest.raises(
        ValueError, match="Model weights have not been optimized. Call `fit` first."
    ):
        optuna_weights_instance.predict(y_preds)


# Test predict method with optimized weights
def test_predict(setup_data, optuna_weights_instance):
    y_true, y_preds = setup_data
    optuna_weights_instance.fit(y_true, y_preds)
    weighted_pred = optuna_weights_instance.predict(y_preds)
    assert (
        weighted_pred.shape == y_true.shape
    ), "Prediction output should match shape of y_true"
    assert (0 <= weighted_pred).all() and (
        weighted_pred <= 1
    ).all(), "Predictions should be probabilities between 0 and 1"


# Test objective function
def test_objective_function(setup_data, optuna_weights_instance):
    y_true, y_preds = setup_data

    # Create a mock trial with predefined suggestions
    trial = create_trial(
        params={"weight0": 0.5, "weight1": 0.3, "weight2": 0.2},
        distributions={
            "weight0": optuna.distributions.FloatDistribution(0, 1),
            "weight1": optuna.distributions.FloatDistribution(0, 1),
            "weight2": optuna.distributions.FloatDistribution(0, 1),
        },
        value=0.0,
    )

    score = optuna_weights_instance._objective(trial, y_true, y_preds)
    assert isinstance(
        score, float
    ), "Objective function should return a score as a float."
