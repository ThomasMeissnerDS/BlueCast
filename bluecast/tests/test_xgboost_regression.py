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
    # Check if the defaults are loaded
    assert isinstance(default_model.conf_training, TrainingConfig)
    assert isinstance(default_model.conf_xgboost, XgboostTuneParamsRegressionConfig)
    assert isinstance(
        default_model.conf_params_xgboost, XgboostRegressionFinalParamConfig
    )
