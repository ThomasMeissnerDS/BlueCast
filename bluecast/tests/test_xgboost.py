import pytest

from bluecast.config.training_config import (
    TrainingConfig,
    XgboostFinalParamConfig,
    XgboostTuneParamsConfig,
)
from bluecast.ml_modelling.xgboost import XgboostModel


@pytest.fixture
def default_model():
    return XgboostModel(class_problem="binary")


def test_check_load_confs_defaults(default_model):
    # Check if the defaults are loaded
    assert isinstance(default_model.conf_training, TrainingConfig)
    assert isinstance(default_model.conf_xgboost, XgboostTuneParamsConfig)
    assert isinstance(default_model.conf_params_xgboost, XgboostFinalParamConfig)
