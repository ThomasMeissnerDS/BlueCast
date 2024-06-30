from unittest.mock import MagicMock, Mock

import optuna

from bluecast.config.training_config import XgboostTuneParamsConfig
from bluecast.ml_modelling.parameter_tuning_utils import (
    update_hyperparam_space_after_nth_trial,
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


def test_update_hyperparam_space_after_nth_trial():
    # Create a config object
    conf_xgboost = XgboostTuneParamsConfig()

    trial_data = [
        (1, 0.001, 0.3, 0.5, 100.0),
        (25, 0.05, 0.25, 1.0, 10.0),
        (50, 0.001, 0.3, 0.5, 100.0),
        (75, 0.05, 0.25, 1.0, 10.0),
        (100, 0.001, 0.3, 0.5, 100.0),
    ]

    for (
        trial_number,
        expected_eta_min,
        expected_eta_max,
        expected_sub_sample_min,
        expected_min_child_weight_max,
    ) in trial_data:
        trial = MagicMock()
        trial.number = trial_number
        # Call the function
        conf_xgboost = update_hyperparam_space_after_nth_trial(
            trial, conf_xgboost, nth_trial=25
        )
        print(
            trial_number,
            trial_number % 25 * 2,
            trial_number % 25,
            conf_xgboost.eta_min,
            expected_eta_min,
        )

        # Assertions
        assert conf_xgboost.eta_min == expected_eta_min
        assert conf_xgboost.eta_max == expected_eta_max
        assert conf_xgboost.sub_sample_min == expected_sub_sample_min
        assert conf_xgboost.min_child_weight_max == expected_min_child_weight_max
