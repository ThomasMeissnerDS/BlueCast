import logging
import warnings
from typing import Any, Dict, Tuple, Union

import optuna
import pandas as pd

from bluecast.config.training_config import (
    CatboostFinalParamConfig,
    CatboostRegressionFinalParamConfig,
    CatboostTuneParamsConfig,
    CatboostTuneParamsRegressionConfig,
    TrainingConfig,
    XgboostFinalParamConfig,
    XgboostRegressionFinalParamConfig,
    XgboostTuneParamsConfig,
    XgboostTuneParamsRegressionConfig,
)
from bluecast.general_utils.general_utils import check_gpu_support, log_sampling


def update_params_based_on_tree_method(
    param: Dict[str, Any],
    trial: optuna.Trial,
    xgboost_params: Union[XgboostTuneParamsConfig, XgboostTuneParamsRegressionConfig],
) -> Dict[str, Any]:
    """Update parameters based on tree method."""

    param["tree_method"] = trial.suggest_categorical(
        "tree_method", xgboost_params.tree_method
    )
    if param["tree_method"] in ["hist", "approx", "gpu_hist"]:
        param["max_bin"] = trial.suggest_int(
            "max_bin", xgboost_params.max_bin_min, xgboost_params.max_bin_max
        )

    if param.get("device", None) == "cpu":
        del param["device"]

    param["booster"] = trial.suggest_categorical("booster", xgboost_params.booster)
    if param["booster"] == "gbtree":
        param["grow_policy"] = trial.suggest_categorical(
            "grow_policy", xgboost_params.grow_policy
        )
    elif param["booster"] == "gblinear":
        del param["max_depth"]
        del param["min_child_weight"]
        del param["subsample"]
        del param["colsample_bytree"]
        del param["colsample_bylevel"]
        del param["gamma"]
        del param["tree_method"]
    return param


def get_params_based_on_device_xgboost(
    conf_training: TrainingConfig,
    conf_params_xgboost: Union[
        XgboostFinalParamConfig, XgboostRegressionFinalParamConfig
    ],
    conf_xgboost: Union[XgboostTuneParamsConfig, XgboostTuneParamsRegressionConfig],
) -> Dict[str, Any]:
    """Get parameters based on available or chosen device."""
    if conf_training.autotune_on_device in ["auto"]:
        train_on = check_gpu_support()
        conf_params_xgboost.params["device"] = train_on.get("device", None)
        if "exact" in conf_xgboost.tree_method and conf_params_xgboost.params[
            "device"
        ] in ["gpu", "cuda"]:
            conf_xgboost.tree_method.remove("exact")
    elif conf_training.autotune_on_device == "gpu":
        train_on = {"tree_method": "hist", "device": "cuda"}
        if "exact" in conf_xgboost.tree_method:
            conf_xgboost.tree_method.remove("exact")
    else:
        train_on = {"tree_method": "exact", "device": "cpu"}
    return train_on


def get_params_based_on_device_catboost(
    conf_training: TrainingConfig,
    conf_params_catboost: Union[
        CatboostFinalParamConfig, CatboostRegressionFinalParamConfig
    ],
    conf_catboost: Union[CatboostTuneParamsConfig, CatboostTuneParamsRegressionConfig],
) -> Dict[str, Any]:
    """Get parameters based on available or chosen device."""
    if conf_training.autotune_on_device in ["auto"]:
        warnings.warn(
            """Automatic GPU detection is not supported yet. Falling back to CPU. If a GPU is available, please set autotune_on_device = 'gpu'""",
            stacklevel=2,
        )
        train_on = {"task_type": "CPU"}
    elif conf_training.autotune_on_device == "gpu":
        train_on = {"task_type": "GPU"}
    else:
        train_on = {"task_type": "CPU"}
    return train_on


def update_params_with_best_params(
    param: Dict[str, Any],
    best_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Update parameters based on best parameters after tuning."""

    params_to_check = ["tree_method", "booster", "grow_policy", "max_bin"]
    for param_name in params_to_check:
        if param_name in best_params.keys():
            param[param_name] = best_params[param_name]
    return param


def sample_data(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    conf_training: TrainingConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if conf_training.sample_data_during_tuning:
        original_size = len(x_train.index)

        nb_samples_train = log_sampling(
            len(x_train.index),
            alpha=conf_training.sample_data_during_tuning_alpha,
        )
        nb_samples_test = log_sampling(
            len(x_test.index),
            alpha=conf_training.sample_data_during_tuning_alpha,
        )

        x_train = x_train.sample(
            nb_samples_train, random_state=conf_training.global_random_state
        )
        y_train = y_train.loc[x_train.index]
        x_test = x_test.sample(
            nb_samples_test, random_state=conf_training.global_random_state
        )
        y_test = y_test.loc[x_test.index]

        new_size = len(x_train.index)
        logging.info(f"Down sampling from {original_size} to {new_size} samples.")
    return x_train, x_test, y_train, y_test
