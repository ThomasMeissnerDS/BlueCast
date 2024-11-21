from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from bluecast.blueprints.cast import BlueCast
from bluecast.blueprints.cast_cv import BlueCastCV
from bluecast.blueprints.cast_cv_regression import BlueCastCVRegression
from bluecast.blueprints.cast_regression import BlueCastRegression
from bluecast.blueprints.orchestration import ModelMatchMaker
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
