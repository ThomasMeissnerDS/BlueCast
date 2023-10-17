"""Define training and common configuration parameters.

Pydantic dataclasses are used to define the configuration parameters. This allows for type checking and validation of
the configuration parameters. The configuration parameters are used in the training pipeline and in the evaluation
pipeline. Pydantic dataclasses are used to allow users a pythonic way to define the configuration parameters.
Default configurations can be loaded, adjusted and passed into the blueprints.
"""
from typing import Dict, Optional

from pydantic import BaseModel
from pydantic.dataclasses import dataclass


class Config:
    arbitrary_types_allowed = True


class TrainingConfig(BaseModel):
    """Define general training parameters.

    :param global_random_state: Global random state to use for reproducibility.
    :param shuffle_during_training: Whether to shuffle the data during training when hypertuning_cv_folds > 1.
    :param hyperparameter_tuning_rounds: Number of hyperparameter tuning rounds. Not used when custom ML model is passed.
    :param hyperparameter_tuning_max_runtime_secs: Maximum runtime in seconds for hyperparameter tuning. Not used when
        custom ML model is passed.
    :param hypertuning_cv_folds: Number of cross-validation folds to use for hyperparameter tuning. Not used when
        custom ML model is passed.
    :param early_stopping_rounds: Number of early stopping rounds. Not used when custom ML model is passed.
    :param autotune_model: Whether to autotune the model. Not used when custom ML model is passed.
    :param enable_feature_selection: Whether to enable recursive feature selection.
    :param calculate_shap_values: Whether to calculate shap values. Also used when custom ML model is passed. Not
        compatible with all ML models. See the SHAP documentation for more details.
    :param train_size: Train size to use for train-test split.
    :param train_split_stratify: Whether to stratify the train-test split. Not used when custom ML model is passed.
    :param use_full_data_for_final_model: Whether to use the full data for the final model. This might cause overfitting.
        Not used when custom ML model is passed.
    :param min_features_to_select: Minimum number of features to select. Only used when enable_feature_selection is
        True.
    :param cat_encoding_via_ml_algorithm: Whether to use an ML algorithm for categorical encoding. If True, the
        categorical encoding is done via a ML algorithm. If False, the categorical encoding is done via a  target
        encoding in the preprocessing steps. See the ReadMe for more details.
    :param show_detailed_tuning_logs: Whether to show detailed tuning logs. Not used when custom ML model is passed.
    :param enable_grid_search_fine_tuning: After hyperparameter tuning run Gridsearch tuning on a fine-grained grid
        based on the previous hyperparameter tuning. Only possible when autotune_model is True.
    :param gridsearch_nb_parameters_per_grid: Decides how many steps the grid shall have per parameter.
    :param gridsearch_tuning_max_runtime_secs: Sets the maximum time in seconds the tuning shall run. This will finish
        the latest trial nd will exceed this limit though.
    :param experiment_name: Name of the experiment. Will be logged inside the ExperimentTracker.
    """

    global_random_state: int = 10
    shuffle_during_training: bool = True
    hyperparameter_tuning_rounds: int = 200
    hyperparameter_tuning_max_runtime_secs: int = 3600
    hypertuning_cv_folds: int = 1
    early_stopping_rounds: int = 10
    autotune_model: bool = True
    enable_feature_selection: bool = False
    calculate_shap_values: bool = True
    train_size: float = 0.8
    train_split_stratify: bool = True
    use_full_data_for_final_model: bool = False
    min_features_to_select: int = 5
    cat_encoding_via_ml_algorithm: bool = False
    show_detailed_tuning_logs: bool = False
    optuna_sampler_n_startup_trials: int = 10
    enable_grid_search_fine_tuning: bool = False
    gridsearch_tuning_max_runtime_secs: int = 3600
    gridsearch_nb_parameters_per_grid: int = 5
    experiment_name: str = "new experiment"


class XgboostTuneParamsConfig(BaseModel):
    """Define hyperparameter tuning search space."""

    max_depth_min: int = 2
    max_depth_max: int = 6
    alpha_min: float = 0.0
    alpha_max: float = 10.0
    lambda_min: float = 0.0
    lambda_max: float = 10.0
    gamma_min: float = 0.0
    gamma_max: float = 10.0
    subsample_min: float = 0.0
    subsample_max: float = 10.0
    max_leaves_min: int = 0
    max_leaves_max: int = 0
    sub_sample_min: float = 0.3
    sub_sample_max: float = 1.0
    col_sample_by_tree_min: float = 0.3
    col_sample_by_tree_max: float = 1.0
    col_sample_by_level_min: float = 0.3
    col_sample_by_level_max: float = 1.0
    eta_min: float = 0.001
    eta_max: float = 0.3
    steps_min: int = 2
    steps_max: int = 1000
    model_verbosity: int = 0
    model_verbosity_during_final_training: int = 0
    model_objective: str = "multi:softprob"
    model_eval_metric: str = "mlogloss"
    booster: str = "gbtree"


@dataclass
class XgboostFinalParamConfig:
    """Define final hyper parameters."""

    params = {
        "objective": "multi:softprob",  # OR  'binary:logistic' #the loss function being used
        "eval_metric": "mlogloss",
        "tree_method": "exact",  # use GPU for training
        "num_class": 2,
        "max_depth": 3,  # maximum depth of the decision trees being trained
        "alpha": 0.1,
        "lambda": 0.1,
        "gamma": 0.0,
        "subsample": 1.0,
        "max_leaves": 16,
        "colsample_bytree": 0.8,
        "colsample_bylevel": 0.8,
        "eta": 0.1,
        "steps": 1000,
        "num_parallel_tree": 1,
    }
    sample_weight: Optional[Dict[str, float]] = None
    classification_threshold: float = 0.5
