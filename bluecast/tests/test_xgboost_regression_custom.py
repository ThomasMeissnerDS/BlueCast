import numpy as np
import pandas as pd

from bluecast.blueprints.cast_regression import BlueCastRegression
from bluecast.config.training_config import (
    TrainingConfig,
    XgboostTuneParamsRegressionConfig,
)
from bluecast.ml_modelling.xgboost_regression import XgboostModelRegression
from bluecast.tests.shared_test_helpers import MyCustomLastMilePreprocessing


def test_bluecast_regression_with_custom_xgboost_no_tuning():
    train_config = TrainingConfig()
    train_config.hyperparameter_tuning_rounds = 2
    train_config.hypertuning_cv_folds = 2
    train_config.autotune_model = False

    xgboost_param_config = XgboostTuneParamsRegressionConfig()
    xgboost_param_config.steps_min = 2
    xgboost_param_config.steps_max = 100
    xgboost_param_config.max_depth_max = 3

    bluecast = BlueCastRegression(
        class_problem="regression",
        ml_model=XgboostModelRegression(
            class_problem="regression",
            conf_training=train_config,
            conf_xgboost=xgboost_param_config,
        ),
        conf_tuning=xgboost_param_config,
        conf_training=train_config,
        custom_last_mile_computation=MyCustomLastMilePreprocessing(),
    )

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

    bluecast.fit(x_train, "target")

    predicted_values = bluecast.predict(x_test)

    assert isinstance(predicted_values, np.ndarray)
    assert len(bluecast.experiment_tracker.experiment_id) == 0


def test_bluecast_regression_with_custom_xgboost_with_tuning():
    train_config = TrainingConfig()
    train_config.hyperparameter_tuning_rounds = 2
    train_config.hypertuning_cv_folds = 2
    train_config.autotune_model = True
    train_config.plot_hyperparameter_tuning_overview = False

    xgboost_param_config = XgboostTuneParamsRegressionConfig()
    xgboost_param_config.steps_min = 2
    xgboost_param_config.steps_max = 100
    xgboost_param_config.max_depth_max = 3

    bluecast = BlueCastRegression(
        class_problem="regression",
        ml_model=XgboostModelRegression(
            class_problem="regression",
            conf_training=train_config,
            conf_xgboost=xgboost_param_config,
        ),
        conf_tuning=xgboost_param_config,
        conf_training=train_config,
    )

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

    bluecast.fit(x_train, "target")

    predicted_values = bluecast.predict(x_test)

    assert isinstance(predicted_values, np.ndarray)
    assert len(bluecast.experiment_tracker.experiment_id) == 2
