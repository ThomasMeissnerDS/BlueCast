"""Metric registry for mapping text prompts to framework-specific losses."""

from typing import Any, Dict

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_poisson_deviance,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)

from bluecast.evaluation.eval_metrics import RegressionEvalWrapper

REGRESSION_METRICS = {
    "mae": {
        "sklearn_scoring": "neg_mean_absolute_error",
        "tree_criterion": "absolute_error",
        "catboost_loss": "MAE",
        "xgboost_loss": "reg:absoluteerror",
        "higher_is_better": False,
        "func": mean_absolute_error,
        "name": "Mean absolute error",
    },
    "rmse": {
        "sklearn_scoring": "neg_root_mean_squared_error",
        "tree_criterion": "squared_error",
        "catboost_loss": "RMSE",
        "xgboost_loss": "reg:squarederror",
        "higher_is_better": False,
        "func": mean_squared_error,
        "name": "Root mean squared error",  # Handled via squared=False internally if needed, but BlueCast uses MSE and takes root if needed
    },
    "mse": {
        "sklearn_scoring": "neg_mean_squared_error",
        "tree_criterion": "squared_error",
        "catboost_loss": "RMSE",
        "xgboost_loss": "reg:squarederror",
        "higher_is_better": False,
        "func": mean_squared_error,
        "name": "Mean squared error",
    },
    "r2": {
        "sklearn_scoring": "r2",
        "tree_criterion": "squared_error",
        "catboost_loss": "RMSE",  # CatBoost uses RMSE for tree building even for R2
        "xgboost_loss": "reg:squarederror",
        "higher_is_better": True,
        "func": r2_score,
        "name": "R2 score",
    },
    "mape": {
        "sklearn_scoring": "neg_mean_absolute_percentage_error",
        "tree_criterion": "absolute_error",
        "catboost_loss": "MAPE",
        "xgboost_loss": "reg:mean_absolute_percentage_error",
        "higher_is_better": False,
        "func": mean_absolute_percentage_error,
        "name": "Mean absolute percentage error",
    },
    "msle": {
        "sklearn_scoring": "neg_mean_squared_log_error",
        "tree_criterion": "squared_error",
        "catboost_loss": "MSLE",
        "xgboost_loss": "reg:squaredlogerror",
        "higher_is_better": False,
        "func": mean_squared_log_error,
        "name": "Mean squared log error",
    },
    "poisson": {
        "sklearn_scoring": "neg_mean_poisson_deviance",
        "tree_criterion": "poisson",
        "catboost_loss": "Poisson",
        "xgboost_loss": "count:poisson",
        "higher_is_better": False,
        "func": mean_poisson_deviance,
        "name": "Poisson deviance",
    },
}


def get_regression_metric_config(metric_name: str) -> Dict[str, Any]:
    """Get the metric configuration for a given string name."""
    metric_name = metric_name.lower().strip()
    return REGRESSION_METRICS.get(metric_name, REGRESSION_METRICS["rmse"])


def get_bluecast_eval_wrapper(metric_name: str) -> RegressionEvalWrapper:
    """Create a RegressionEvalWrapper for the given metric."""
    config = get_regression_metric_config(metric_name)

    # For RMSE we use mean_squared_error but the wrapper might handle the root,
    # or we just rely on MSE internally for Early Stopping.
    return RegressionEvalWrapper(
        higher_is_better=config["higher_is_better"],
        metric_func=config["func"],
        metric_name=config["name"],
    )


def get_tree_criterion_from_scoring(scoring: str) -> str:
    """Map sklearn scoring string to the appropriate tree criterion/loss."""
    for config in REGRESSION_METRICS.values():
        if config["sklearn_scoring"] == scoring:
            return config["tree_criterion"]

    # Fallback heuristics
    if "absolute" in scoring:
        return "absolute_error"
    elif "poisson" in scoring:
        return "poisson"
    elif "friedman" in scoring:
        return "friedman_mse"
    else:
        return "squared_error"


def get_pytorch_loss_from_scoring(scoring: str):
    import torch.nn as nn

    if "absolute_error" in scoring or "mae" in scoring:
        return nn.L1Loss()
    elif "poisson" in scoring:
        return nn.PoissonNLLLoss(log_input=False)
    else:
        return nn.MSELoss()
