from unittest.mock import patch

import pytest

from bluecast.config.training_config import (
    TrainingConfig,
    XgboostRegressionFinalParamConfig,
    XgboostTuneParamsRegressionConfig,
)
from bluecast.ml_modelling.xgboost_regression import XgboostModelRegression


@pytest.fixture
def default_model():
    return XgboostModelRegression(class_problem="regression")


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
