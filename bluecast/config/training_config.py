"""Define training and common configuration parameters.

Pydantic dataclasses are used to define the configuration parameters. This allows for type checking and validation of
the configuration parameters. The configuration parameters are used in the training pipeline and in the evaluation
pipeline. Pydantic dataclasses are used to allow users a pythonic way to define the configuration parameters.
Default configurations can be loaded, adjusted and passed into the blueprints.
"""
from typing import Dict, Optional

from pydantic.dataclasses import dataclass


class Config:
    arbitrary_types_allowed = True


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
    enable_feature_selection: bool = False
    calculate_shap_values: bool = True
    train_size: float = 0.8
    train_split_stratify: bool = True
    use_full_data_for_final_model: bool = True
    min_features_to_select: int = 5
    cat_encoding_via_ml_algorithm: bool = False


@dataclass
class XgboostTuneParamsConfig:
    """Define hyperparameter tuning search space."""

    max_depth_min: int = 2
    max_depth_max: int = 6
    alpha_min: float = 0.0
    alpha_max: float = 10.0
    lambda_min: float = 0.0
    lambda_max: float = 10.0
    num_leaves_min: int = 2
    num_leaves_max: int = 64
    sub_sample_min: float = 0.3
    sub_sample_max: float = 1.0
    col_sample_by_tree_min: float = 0.3
    col_sample_by_tree_max: float = 1.0
    col_sample_by_level_min: float = 0.3
    col_sample_by_level_max: float = 1.0
    min_child_weight_min: float = 0.0
    min_child_weight_max: float = 10.0
    eta_min: float = 0.001
    eta_max: float = 0.3
    steps_min: int = 2
    steps_max: int = 1000
    model_verbosity: int = 0
    model_objective: str = "multi:softprob"
    model_eval_metric: str = "mlogloss"
    booster: str = "gbtree"


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
