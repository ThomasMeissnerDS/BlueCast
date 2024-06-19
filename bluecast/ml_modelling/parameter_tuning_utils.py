from typing import Any, Dict, Union

import optuna

from bluecast.config.training_config import (
    XgboostTuneParamsConfig,
    XgboostTuneParamsRegressionConfig,
)


def update_params_based_on_tree_method(
    param: Dict[str, Any], trial: optuna.Trial
) -> Dict[str, Any]:
    """Update parameters based on tree method."""

    if param.get("device", "cpu") not in ["cuda"]:
        param["tree_method"] = trial.suggest_categorical(
            "tree_method", ["exact", "approx", "hist"]
        )
    else:
        param["tree_method"] = trial.suggest_categorical(
            "tree_method", ["hist", "approx"]
        )

    param["booster"] = trial.suggest_categorical("booster", ["gbtree", "gblinear"])
    if param["booster"] == "gbtree":
        param["grow_policy"] = trial.suggest_categorical(
            "grow_policy", ["depthwise", "lossguide"]
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


def update_params_with_best_params(
    param: Dict[str, Any],
    best_params: Union[XgboostTuneParamsConfig, XgboostTuneParamsRegressionConfig],
) -> Dict[str, Any]:
    """Update parameters based on best parameters after tuning."""

    params_to_check = ["tree_method", "booster", "grow_policy"]
    for param_name in params_to_check:
        if param_name in best_params:
            param[param_name] = best_params[param_name]
    return param
