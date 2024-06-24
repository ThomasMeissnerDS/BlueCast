from typing import Any, Dict, Tuple, Union

import optuna
import pandas as pd

from bluecast.config.training_config import (
    TrainingConfig,
    XgboostTuneParamsConfig,
    XgboostTuneParamsRegressionConfig,
)
from bluecast.general_utils.general_utils import log_sampling


def update_params_based_on_tree_method(
    param: Dict[str, Any],
    trial: optuna.Trial,
    xgboost_params: Union[XgboostTuneParamsConfig, XgboostTuneParamsRegressionConfig],
) -> Dict[str, Any]:
    """Update parameters based on tree method."""

    param["tree_method"] = trial.suggest_categorical(
        "tree_method", xgboost_params.tree_method
    )

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


def sample_data(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    conf_training: TrainingConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if conf_training.sample_data_during_tuning:
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
    return x_train, x_test, y_train, y_test
