from unittest.mock import patch

from bluecast.config.training_config import (
    TrainingConfig,
    XgboostFinalParamConfig,
    XgboostTuneParamsConfig,
)
from bluecast.ml_modelling.parameter_tuning_utils import get_params_based_on_device


def test_get_params_based_on_device_auto_gpu():
    conf_training = TrainingConfig(autotune_on_device="auto")
    conf_params_xgboost = XgboostFinalParamConfig()
    conf_xgboost = XgboostTuneParamsConfig(tree_method=["exact"])

    mock_gpu_support = {"device": "cuda"}

    with patch(
        "bluecast.general_utils.general_utils.check_gpu_support",
        return_value=mock_gpu_support,
    ):
        result = get_params_based_on_device(
            conf_training, conf_params_xgboost, conf_xgboost
        )

    assert not result.get("device", None)


def test_get_params_based_on_device_cpu():
    conf_training = TrainingConfig(autotune_on_device="cpu")
    conf_params_xgboost = XgboostFinalParamConfig()
    conf_xgboost = XgboostTuneParamsConfig(tree_method=["exact"])

    with patch("bluecast.general_utils.general_utils.check_gpu_support"):
        result = get_params_based_on_device(
            conf_training, conf_params_xgboost, conf_xgboost
        )

    assert result["device"] == "cpu"
    assert result["tree_method"] == "exact"
    assert "exact" in conf_xgboost.tree_method


def test_get_params_based_on_device_gpu_with_exact_tree_method():
    conf_training = TrainingConfig(autotune_on_device="gpu")
    conf_params_xgboost = XgboostFinalParamConfig()
    conf_xgboost = XgboostTuneParamsConfig(tree_method=["exact", "hist"])

    mock_gpu_support = {"device": "cuda"}

    with patch(
        "bluecast.general_utils.general_utils.check_gpu_support",
        return_value=mock_gpu_support,
    ):
        result = get_params_based_on_device(
            conf_training, conf_params_xgboost, conf_xgboost
        )

    assert result["device"] == "cuda"


def test_get_params_based_on_device_gpu_without_exact_tree_method():
    conf_training = TrainingConfig(autotune_on_device="gpu")
    conf_params_xgboost = XgboostFinalParamConfig()
    conf_xgboost = XgboostTuneParamsConfig(tree_method=["hist"])

    mock_gpu_support = {"device": "cuda"}

    with patch(
        "bluecast.general_utils.general_utils.check_gpu_support",
        return_value=mock_gpu_support,
    ):
        result = get_params_based_on_device(
            conf_training, conf_params_xgboost, conf_xgboost
        )

    assert result["device"] == "cuda"
