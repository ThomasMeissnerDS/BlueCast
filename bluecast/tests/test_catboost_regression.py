import numpy as np
import pandas as pd

from bluecast.blueprints.cast_regression import BlueCastRegression
from bluecast.config.training_config import (
    CatboostTuneParamsRegressionConfig,
    TrainingConfig,
)
from bluecast.ml_modelling.catboost_regression import CatboostModelRegression


def test_BlueCastRegression_without_hyperparam_tuning():
    train_config = TrainingConfig()
    train_config.hyperparameter_tuning_rounds = 10
    train_config.hypertuning_cv_folds = 2
    train_config.autotune_model = False

    catboost_pram_config = CatboostTuneParamsRegressionConfig()

    # Create an instance of the BlueCastRegression class with the custom model
    bluecast = BlueCastRegression(
        class_problem="binary",
        ml_model=CatboostModelRegression(
            class_problem="regression",
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

    # Fit the BlueCastRegression model using the custom model
    bluecast.fit(x_train, "target")

    # Predict on the test data using the custom model
    predicted_values = bluecast.predict(x_test)

    # Assert the expected results
    assert isinstance(predicted_values, np.ndarray)
    print(bluecast.experiment_tracker.experiment_id)
    assert (
        len(bluecast.experiment_tracker.experiment_id) == 0
    )  # due to custom model and fit method


def test_BlueCastRegression_with_hyperparam_tuning():
    train_config = TrainingConfig()
    train_config.hyperparameter_tuning_rounds = 10
    train_config.hypertuning_cv_folds = 2
    train_config.autotune_model = True

    catboost_pram_config = CatboostTuneParamsRegressionConfig()

    # Create an instance of the BlueCastRegression class with the custom model
    bluecast = BlueCastRegression(
        class_problem="regression",
        ml_model=CatboostModelRegression(
            class_problem="regression",
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

    # Fit the BlueCastRegression model using the custom model
    bluecast.fit(x_train, "target")

    # Predict on the test data using the custom model
    predicted_values = bluecast.predict(x_test)

    # Assert the expected results
    assert isinstance(predicted_values, np.ndarray)
    print(bluecast.experiment_tracker.experiment_id)
    assert (
        len(bluecast.experiment_tracker.experiment_id) == 0
    )  # due to custom model and fit method
