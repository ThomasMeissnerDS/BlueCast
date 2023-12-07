import pytest

from bluecast.config.training_config import (
    TrainingConfig,
    XgboostFinalParamConfig,
    XgboostTuneParamsConfig,
)
from bluecast.experimentation.tracking import ExperimentTracker
from bluecast.ml_modelling.xgboost import XgboostModel
from bluecast.tests.make_data.create_data import create_synthetic_dataframe


# Create a fixture for the XGBoost model
@pytest.fixture
def xgboost_model():
    return XgboostModel(class_problem="binary")


# Test case to check if fine-tuning runs without errors
def test_fine_tune_runs_without_errors(xgboost_model):
    xgboost_params = XgboostFinalParamConfig()
    xgboost_params.params["num_class"] = 2

    xgboost_model.conf_params_xgboost = xgboost_params
    xgboost_model.conf_training = TrainingConfig()
    xgboost_model.conf_xgboost = XgboostTuneParamsConfig()
    print(xgboost_model.conf_params_xgboost.params)
    xgboost_model.experiment_tracker = ExperimentTracker()
    xgboost_model.conf_training.autotune_model = False
    xgboost_model.conf_training.hypertuning_cv_folds = 3  # enable cross validation
    xgboost_model.conf_training.hyperparameter_tuning_rounds = 5
    xgboost_model.conf_training.gridsearch_nb_parameters_per_grid = 2

    df_train, df_val = create_synthetic_dataframe(
        2000, random_state=20
    ), create_synthetic_dataframe(2000, random_state=200)
    df_train = df_train.drop(
        ["categorical_feature_1", "categorical_feature_2", "datetime_feature"], axis=1
    )
    df_val = df_val.drop(
        ["categorical_feature_1", "categorical_feature_2", "datetime_feature"], axis=1
    )

    x_train = df_train.drop("target", axis=1)
    y_train = df_train["target"]
    x_test = df_val.drop("target", axis=1)
    y_test = df_val["target"]

    xgboost_model.fine_tune(x_train, x_test, y_train, y_test)
    assert (
        (xgboost_model.conf_params_xgboost.params["alpha"] != 0.1)
        or (xgboost_model.conf_params_xgboost.params["lambda"] != 0.1)
        or (xgboost_model.conf_params_xgboost.params["eta"] != 0.1)
    )


def test_fine_tune_runs_without_errors_using_cv(xgboost_model):
    xgboost_params = XgboostFinalParamConfig()
    xgboost_params.params["num_class"] = 2
    xgboost_model.conf_params_xgboost = xgboost_params
    xgboost_model.conf_training = TrainingConfig()
    xgboost_model.conf_xgboost = XgboostTuneParamsConfig()
    print(xgboost_model.conf_params_xgboost.params)
    xgboost_model.experiment_tracker = ExperimentTracker()
    xgboost_model.conf_training.autotune_model = False
    xgboost_model.conf_training.hypertuning_cv_folds = 3  # enable cross validation
    xgboost_model.conf_training.hyperparameter_tuning_rounds = 5
    xgboost_model.conf_training.gridsearch_nb_parameters_per_grid = 2

    df_train, df_val = create_synthetic_dataframe(
        1000, random_state=20
    ), create_synthetic_dataframe(1000, random_state=200)
    df_train = df_train.drop(
        ["categorical_feature_1", "categorical_feature_2", "datetime_feature"], axis=1
    )
    df_val = df_val.drop(
        ["categorical_feature_1", "categorical_feature_2", "datetime_feature"], axis=1
    )

    x_train = df_train.drop("target", axis=1)
    y_train = df_train["target"]
    x_test = df_val.drop("target", axis=1)
    y_test = df_val["target"]

    xgboost_model.fine_tune(x_train, x_test, y_train, y_test)
    assert (
        (xgboost_model.conf_params_xgboost.params["alpha"] != 0.1)
        or (xgboost_model.conf_params_xgboost.params["lambda"] != 0.1)
        or (xgboost_model.conf_params_xgboost.params["eta"] != 0.1)
    )
