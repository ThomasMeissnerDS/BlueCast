import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pytest

from bluecast.blueprints.cast import BlueCast
from bluecast.config.training_config import CatboostTuneParamsConfig, TrainingConfig
from bluecast.ml_modelling.catboost import CatboostModel
from bluecast.preprocessing.custom import CustomPreprocessing


def test_catboost_no_catboost_config_warnings():
    expected_message = re.escape("conf_catboost or conf_training is None")

    catboost_model = CatboostModel(class_problem="binary")
    del catboost_model.conf_catboost

    dummy_df = pd.DataFrame({"A": [1, 2, 3], "B": [7, 8, 9]})

    with pytest.warns(UserWarning, match=expected_message):
        catboost_model.predict_proba(dummy_df)


def test_catboost_no_conf_training_warnings():
    expected_message = re.escape("conf_catboost or conf_training is None")

    catboost_model = CatboostModel(class_problem="binary")
    del catboost_model.conf_training

    dummy_df = pd.DataFrame({"A": [1, 2, 3], "B": [7, 8, 9]})

    with pytest.warns(UserWarning, match=expected_message):
        catboost_model.predict_proba(dummy_df)


def test_catboost_no_model_warnings():
    expected_message = re.escape("No trained CatBoost model found.")

    catboost_model = CatboostModel(class_problem="binary")
    del catboost_model.model

    dummy_df = pd.DataFrame({"A": [1, 2, 3], "B": [7, 8, 9]})

    with pytest.warns(UserWarning, match=expected_message):
        catboost_model.predict_proba(dummy_df)


def test_catboost_no_conf_params_catboost_warnings():
    expected_message = re.escape("No CatBoost model configuration found.")

    catboost_model = CatboostModel(class_problem="binary")
    del catboost_model.conf_params_catboost

    dummy_df = pd.DataFrame({"A": [1, 2, 3], "B": [7, 8, 9]})

    with pytest.warns(UserWarning, match=expected_message):
        catboost_model.predict_proba(dummy_df)


def test_bluecast_without_hyperparam_tuning():
    train_config = TrainingConfig()
    train_config.hyperparameter_tuning_rounds = 10
    train_config.hypertuning_cv_folds = 2
    train_config.autotune_model = False

    catboost_pram_config = CatboostTuneParamsConfig()

    class MyCustomLastMilePreprocessing(CustomPreprocessing):
        def custom_function(self, df: pd.DataFrame) -> pd.DataFrame:
            df["custom_col"] = 5
            return df

        def fit_transform(
            self, df: pd.DataFrame, target: pd.Series
        ) -> Tuple[pd.DataFrame, pd.Series]:
            df = self.custom_function(df)
            return df, target

        def transform(
            self,
            df: pd.DataFrame,
            target: Optional[pd.Series] = None,
            predicton_mode: bool = False,
        ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
            df = self.custom_function(df)
            return df, target

    # Create an instance of the BlueCast class with the custom model
    bluecast = BlueCast(
        class_problem="binary",
        ml_model=CatboostModel(
            class_problem="binary",
            conf_training=train_config,
            conf_catboost=catboost_pram_config,
        ),
        conf_xgboost=catboost_pram_config,
        conf_training=train_config,
        custom_last_mile_computation=MyCustomLastMilePreprocessing(),
    )

    # Create some sample data for testing
    x_train = pd.DataFrame(
        {
            "feature1": [i for i in range(20)],
            "feature2": [i for i in range(20)],
            "feature3": [i for i in range(20)],
            "feature4": [i for i in range(20)],
            "feature5": [i for i in range(20)],
            "feature6": [i for i in range(20)],
        }
    )
    y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    x_test = pd.DataFrame(
        {
            "feature1": [i for i in range(10)],
            "feature2": [i for i in range(10)],
            "feature3": [i for i in range(10)],
            "feature4": [i for i in range(10)],
            "feature5": [i for i in range(10)],
            "feature6": [i for i in range(10)],
        }
    )

    x_train["target"] = y_train

    # Fit the BlueCast model using the custom model
    bluecast.fit(x_train, "target")

    # Predict on the test data using the custom model
    predicted_probas, predicted_classes = bluecast.predict(x_test)
    _ = bluecast.predict_proba(x_test)

    # Assert the expected results
    assert isinstance(predicted_probas, np.ndarray)
    assert isinstance(predicted_classes, np.ndarray)
    print(bluecast.experiment_tracker.experiment_id)
    assert (
        len(bluecast.experiment_tracker.experiment_id) == 0
    )  # due to custom model and fit method


def test_bluecast_with_hyperparam_tuning():
    train_config = TrainingConfig()
    train_config.hyperparameter_tuning_rounds = 10
    train_config.hypertuning_cv_folds = 2
    train_config.autotune_model = True
    train_config.plot_hyperparameter_tuning_overview = False

    catboost_pram_config = CatboostTuneParamsConfig()

    # Create an instance of the BlueCast class with the custom model
    bluecast = BlueCast(
        class_problem="binary",
        ml_model=CatboostModel(
            class_problem="binary",
            conf_training=train_config,
            conf_catboost=catboost_pram_config,
        ),
        conf_xgboost=catboost_pram_config,
        conf_training=train_config,
    )

    # Create some sample data for testing
    x_train = pd.DataFrame(
        {
            "feature1": [i for i in range(20)],
            "feature2": [i for i in range(20)],
            "feature3": [i for i in range(20)],
            "feature4": [i for i in range(20)],
            "feature5": [i for i in range(20)],
            "feature6": [i for i in range(20)],
        }
    )
    y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    x_test = pd.DataFrame(
        {
            "feature1": [i for i in range(10)],
            "feature2": [i for i in range(10)],
            "feature3": [i for i in range(10)],
            "feature4": [i for i in range(10)],
            "feature5": [i for i in range(10)],
            "feature6": [i for i in range(10)],
        }
    )

    x_train["target"] = y_train

    # Fit the BlueCast model using the custom model
    bluecast.fit(x_train, "target")

    # Predict on the test data using the custom model
    predicted_probas, predicted_classes = bluecast.predict(x_test)
    _ = bluecast.predict_proba(x_test)

    # Assert the expected results
    assert isinstance(predicted_probas, np.ndarray)
    assert isinstance(predicted_classes, np.ndarray)
    print(bluecast.experiment_tracker.experiment_id)
    assert (
        len(bluecast.experiment_tracker.experiment_id) == 0
    )  # due to custom model and fit method

    # TEST with 1 fold
    train_config = TrainingConfig()
    train_config.hyperparameter_tuning_rounds = 10
    train_config.hypertuning_cv_folds = 1
    train_config.autotune_model = True

    catboost_pram_config = CatboostTuneParamsConfig()

    # Create an instance of the BlueCast class with the custom model
    bluecast = BlueCast(
        class_problem="binary",
        ml_model=CatboostModel(
            class_problem="binary",
            conf_training=train_config,
            conf_catboost=catboost_pram_config,
        ),
        conf_xgboost=catboost_pram_config,
        conf_training=train_config,
    )

    # Fit the BlueCast model using the custom model
    bluecast.fit(x_train, "target")

    # Predict on the test data using the custom model
    predicted_probas, predicted_classes = bluecast.predict(x_test)
    _ = bluecast.predict_proba(x_test)

    # Assert the expected results
    assert isinstance(predicted_probas, np.ndarray)
    assert isinstance(predicted_classes, np.ndarray)
    print(bluecast.experiment_tracker.experiment_id)
    assert (
        len(bluecast.experiment_tracker.experiment_id) == 0
    )  # due to custom model and fit method


def test_bluecast_with_fine_tune_hyperparam_tuning():
    train_config = TrainingConfig()
    train_config.hyperparameter_tuning_rounds = 10
    train_config.hypertuning_cv_folds = 2
    train_config.autotune_model = True
    train_config.precise_cv_tuning = True

    catboost_pram_config = CatboostTuneParamsConfig()

    # Create an instance of the BlueCast class with the custom model
    bluecast = BlueCast(
        class_problem="binary",
        ml_model=CatboostModel(
            class_problem="binary",
            conf_training=train_config,
            conf_catboost=catboost_pram_config,
        ),
        conf_xgboost=catboost_pram_config,
        conf_training=train_config,
    )

    # Create some sample data for testing
    x_train = pd.DataFrame(
        {
            "feature1": [i for i in range(20)],
            "feature2": [i for i in range(20)],
            "feature3": [i for i in range(20)],
            "feature4": [i for i in range(20)],
            "feature5": [i for i in range(20)],
            "feature6": [i for i in range(20)],
        }
    )
    y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    x_test = pd.DataFrame(
        {
            "feature1": [i for i in range(10)],
            "feature2": [i for i in range(10)],
            "feature3": [i for i in range(10)],
            "feature4": [i for i in range(10)],
            "feature5": [i for i in range(10)],
            "feature6": [i for i in range(10)],
        }
    )

    x_train["target"] = y_train

    # Fit the BlueCast model using the custom model
    bluecast.fit(x_train, "target")

    # Predict on the test data using the custom model
    predicted_probas, predicted_classes = bluecast.predict(x_test)
    _ = bluecast.predict_proba(x_test)

    # Assert the expected results
    assert isinstance(predicted_probas, np.ndarray)
    assert isinstance(predicted_classes, np.ndarray)
    print(bluecast.experiment_tracker.experiment_id)
    assert (
        len(bluecast.experiment_tracker.experiment_id) == 0
    )  # due to custom model and fit method


def test_bluecast_with_grid_search_tune_hyperparam_tuning():
    train_config = TrainingConfig()
    train_config.hyperparameter_tuning_rounds = 2
    train_config.hypertuning_cv_folds = 2
    train_config.autotune_model = True
    train_config.enable_grid_search_fine_tuning = True
    train_config.gridsearch_nb_parameters_per_grid = 2

    catboost_pram_config = CatboostTuneParamsConfig()

    # Create an instance of the BlueCast class with the custom model
    bluecast = BlueCast(
        class_problem="binary",
        ml_model=CatboostModel(
            class_problem="binary",
            conf_training=train_config,
            conf_catboost=catboost_pram_config,
        ),
        conf_xgboost=catboost_pram_config,
        conf_training=train_config,
    )

    # Create some sample data for testing
    x_train = pd.DataFrame(
        {
            "feature1": [i for i in range(20)],
            "feature2": [i for i in range(20)],
            "feature3": [i for i in range(20)],
            "feature4": [i for i in range(20)],
            "feature5": [i for i in range(20)],
            "feature6": [i for i in range(20)],
        }
    )
    y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    x_test = pd.DataFrame(
        {
            "feature1": [i for i in range(10)],
            "feature2": [i for i in range(10)],
            "feature3": [i for i in range(10)],
            "feature4": [i for i in range(10)],
            "feature5": [i for i in range(10)],
            "feature6": [i for i in range(10)],
        }
    )

    x_train["target"] = y_train

    # Fit the BlueCast model using the custom model
    bluecast.fit(x_train, "target")

    # Predict on the test data using the custom model
    predicted_probas, predicted_classes = bluecast.predict(x_test)
    _ = bluecast.predict_proba(x_test)

    # Assert the expected results
    assert isinstance(predicted_probas, np.ndarray)
    assert isinstance(predicted_classes, np.ndarray)
    print(bluecast.experiment_tracker.experiment_id)
    assert (
        len(bluecast.experiment_tracker.experiment_id) == 0
    )  # due to custom model and fit method
