from typing import Tuple
from unittest.mock import patch

import pandas as pd
import pytest

from bluecast.config.training_config import (
    TrainingConfig,
    XgboostRegressionFinalParamConfig,
    XgboostTuneParamsRegressionConfig,
)
from bluecast.ml_modelling.xgboost_regression import XgboostModelRegression
from bluecast.tests.make_data.create_data import create_synthetic_dataframe_regression


@pytest.fixture
def default_model():
    return XgboostModelRegression(class_problem="regression")


@pytest.fixture
def synthetic_train_test_data_xgboost() -> (
    Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]
):
    df_train = create_synthetic_dataframe_regression(2000, random_state=20)
    df_val = create_synthetic_dataframe_regression(2000, random_state=200)
    y_train = df_train.pop("target")
    y_test = df_val.pop("target")
    return df_train, y_train, df_val, y_test


def test_check_load_confs_defaults(default_model):
    with patch("bluecast.ml_modelling.xgboost_regression.logging") as mock_logger:
        default_model.check_load_confs()

        # Check if the defaults are loaded
        assert isinstance(default_model.conf_training, TrainingConfig)
        assert isinstance(default_model.conf_xgboost, XgboostTuneParamsRegressionConfig)
        assert isinstance(
            default_model.conf_params_xgboost, XgboostRegressionFinalParamConfig
        )

        # Ensure logger was called with default config loading messages
        assert mock_logger.call_count == 0


def test_check_load_confs_partial():
    conf_training = TrainingConfig()
    model = XgboostModelRegression(
        class_problem="regression", conf_training=conf_training
    )

    with patch("bluecast.ml_modelling.xgboost_regression.logging") as mock_logger:
        model.check_load_confs()

        # Check if the provided config is used and defaults are loaded for the rest
        assert model.conf_training is conf_training
        assert isinstance(model.conf_xgboost, XgboostTuneParamsRegressionConfig)
        assert isinstance(model.conf_params_xgboost, XgboostRegressionFinalParamConfig)

        # Ensure logger was called with appropriate messages
        assert mock_logger.call_count == 0


def test_check_load_confs_all_provided():
    conf_training = TrainingConfig()
    conf_xgboost = XgboostTuneParamsRegressionConfig()
    conf_params_xgboost = XgboostRegressionFinalParamConfig()

    model = XgboostModelRegression(
        class_problem="regression",
        conf_training=conf_training,
        conf_xgboost=conf_xgboost,
        conf_params_xgboost=conf_params_xgboost,
    )

    with patch("bluecast.ml_modelling.xgboost_regression.logging") as mock_logger:
        model.check_load_confs()

        # Check if all provided configs are used
        assert model.conf_training is conf_training
        assert model.conf_xgboost is conf_xgboost
        assert model.conf_params_xgboost is conf_params_xgboost

        # Ensure logger was called with appropriate messages
        assert mock_logger.call_count == 0


def test_xgboost_regression_fit_errors(synthetic_train_test_data_xgboost):
    model = XgboostModelRegression(
        class_problem="regression",
    )
    df_train, y_train, df_val, y_test = synthetic_train_test_data_xgboost
    with pytest.raises(ValueError):
        model.fit(df_train, df_val, y_train, y_test)
