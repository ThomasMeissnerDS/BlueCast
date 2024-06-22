from unittest.mock import Mock

import optuna

from bluecast.config.training_config import XgboostTuneParamsConfig
from bluecast.ml_modelling.parameter_tuning_utils import (
    update_params_based_on_tree_method,
)

# Mocking XgboostTuneParamsConfig for testing
mock_xgboost_params = XgboostTuneParamsConfig(
    tree_method=["hist", "gpu_hist"],
    booster=["gbtree", "gblinear"],
    grow_policy=["depthwise", "lossguide"],
)

# Mocking optuna.Trial for testing
mock_trial = Mock(spec=optuna.Trial)


def setup_mock_trial(trial: Mock, booster: str, tree_method: str, grow_policy: str):
    trial.suggest_categorical.side_effect = [tree_method, booster, grow_policy]


def test_update_params_based_on_tree_method_gbtree():
    param = {
        "booster": "gbtree",
        "tree_method": "auto",  # Initial value for testing
        "max_depth": 5,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "colsample_bylevel": 0.8,
        "gamma": 0.2,
    }

    setup_mock_trial(mock_trial, "gbtree", "hist", "depthwise")

    updated_param = update_params_based_on_tree_method(
        param, mock_trial, mock_xgboost_params
    )

    assert updated_param["booster"] == "gbtree"
    assert updated_param["tree_method"] == "hist"
    assert updated_param["grow_policy"] == "depthwise"
    assert "max_depth" in updated_param
    assert "min_child_weight" in updated_param
    assert "subsample" in updated_param
    assert "colsample_bytree" in updated_param
    assert "colsample_bylevel" in updated_param
    assert "gamma" in updated_param


def test_update_params_based_on_tree_method_gblinear():
    param = {
        "booster": "gblinear",
        "max_depth": 5,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "colsample_bylevel": 0.8,
        "gamma": 0.2,
        "tree_method": "auto",
    }

    setup_mock_trial(mock_trial, "gblinear", "hist", "depthwise")

    updated_param = update_params_based_on_tree_method(
        param, mock_trial, mock_xgboost_params
    )

    assert updated_param["booster"] == "gblinear"
    assert "tree_method" not in updated_param
    assert "max_depth" not in updated_param
    assert "min_child_weight" not in updated_param
    assert "subsample" not in updated_param
    assert "colsample_bytree" not in updated_param
    assert "colsample_bylevel" not in updated_param
    assert "gamma" not in updated_param
