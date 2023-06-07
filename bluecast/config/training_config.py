"""Define training and common configuration parameters.

Pydantic dataclasses are used to define the configuration parameters. This allows for type checking and validation of
the configuration parameters. The configuration parameters are used in the training pipeline and in the evaluation
pipeline. Pydantic dataclasses are used to allow users a pythonic way to define the configuration parameters.
Default configurations can be loaded, adjusted and passed into the blueprints.
"""
from typing import Dict, Optional

from pydantic.dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Define general training parameters."""

    global_random_state: int = 10
    shuffle_during_training: bool = True
    hyperparameter_tuning_rounds: int = 100
    hyperparameter_tuning_max_runtime_secs: int = 3600
    hypertuning_cv_folds: int = 1
    early_stopping_rounds: int = 10
    autotune_model: bool = True
    calculate_shap_values: bool = True


@dataclass
class XgboostTuneParamsConfig:
    """Define hyperparameter tuning search space."""

    max_depth_min: int = 2
    max_depth_max: int = 3
    alpha_min: float = 1.0
    alpha_max: float = 1e3
    lambda_min: float = 1.0
    lambda_max: float = 1e3
    num_leaves_min: int = 2
    num_leaves_max: int = 64
    sub_sample_min: float = 0.3
    sub_sample_max: float = 1.0
    col_sample_by_tree_min: float = 0.3
    col_sample_by_tree_max: float = 1.0
    col_sample_by_level_min: float = 0.3
    col_sample_by_level_max: float = 1.0
    col_sample_by_node_min: float = 0.3
    col_sample_by_node_max: float = 1.0
    min_child_samples_min: int = 2
    min_child_samples_max: int = 1000
    eta: float = 0.1
    steps_min: int = 2
    steps_max: int = 50000
    num_parallel_tree_min: int = 1
    num_parallel_tree_max: int = 3
    model_verbosity: int = 0
    model_objective: str = "multi:softprob"
    model_eval_metric: str = "mlogloss"


@dataclass
class XgboostFinalParamConfig:
    """Define final hyper parameters."""

    params = {
        "objective": "multi:softprob",  # OR  'binary:logistic' #the loss function being used
        "eval_metric": "mlogloss",
        "verbose": 0,
        "tree_method": "exact",  # use GPU for training
        "num_class": 2,
        "max_depth": 3,  # maximum depth of the decision trees being trained
        "alpha": 0.1,
        "lambda": 0.1,
        "num_leaves": 16,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "colsample_bylevel": 0.8,
        "colsample_bynode": 0.8,
        "min_child_samples": 100,
        "eta": 0.1,
        "steps": 1000,
        "num_parallel_tree": 1,
    }
    sample_weight: Optional[Dict[str, float]] = None
    classification_threshold: float = 0.5
