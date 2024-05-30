"""Define training and common configuration parameters.

Pydantic dataclasses are used to define the configuration parameters. This allows for type checking and validation of
the configuration parameters. The configuration parameters are used in the training pipeline and in the evaluation
pipeline. Pydantic dataclasses are used to allow users a pythonic way to define the configuration parameters.
Default configurations can be loaded, adjusted and passed into the blueprints.
"""

from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel
from pydantic.dataclasses import dataclass


class Config:
    arbitrary_types_allowed = True


class TrainingConfig(BaseModel):
    """Define general training parameters.

    :param global_random_state: Global random state to use for reproducibility.
    :param increase_random_state_in_bluecast_cv_by: In BlueCastCV multiple models are trained. Define by how much the
        random state changes with each additional model.
    :param shuffle_during_training: Whether to shuffle the data during training when hypertuning_cv_folds > 1.
    :param hyperparameter_tuning_rounds: Number of hyperparameter tuning rounds. Not used when custom ML model is passed.
    :param hyperparameter_tuning_max_runtime_secs: Maximum runtime in seconds for hyperparameter tuning. Not used when
        custom ML model is passed.
    :param hypertuning_cv_folds: Number of cross-validation folds to use for hyperparameter tuning. Not used when
        custom ML model is passed.
    :param sample_data_during_tuning: Whether to sample the data during tuning. Not used when custom ML model is passed.
    :param sample_data_during_tuning_alpha: Alpha value for sampling the data during tuning. The higher alpha the
        fewer samples will be left. Not used when custom ML model is passed.
    :param class_weight_during_dmatrix_creation: Whether to use class weights during DMatrix creation. Not used when
        custom ML model is passed.
    :param early_stopping_rounds: Number of early stopping rounds during final training or when hyperparameter tuning
        follows a single train-test split. Not used when custom ML model is passed.
    :param autotune_model: Whether to autotune the model. Not used when custom ML model is passed.
    :param enable_feature_selection: Whether to enable recursive feature selection.
    :param calculate_shap_values: Whether to calculate shap values. Also used when custom ML model is passed. Not
        compatible with all ML models. See the SHAP documentation for more details.
    :param shap_waterfall_indices: List of sample indices to plot. Each index represents a sample (i.e.: [0, 1, 499]).
    :param show_dependence_plots_of_top_n_features: Maximum number of dependence plots to show. Not used when custom ML
        model is passed.
    :param store_shap_values_in_instance: Whether to store the SHAP values in the BlueCast instance. Not applicable when
        custom ML model is used.
    :param train_size: Train size to use for train-test split.
    :param train_split_stratify: Whether to stratify the train-test split. Not used when custom ML model is passed.
    :param use_full_data_for_final_model: Whether to use the full data for the final model. This might cause overfitting.
        Not used when custom ML model is passed.
    :param min_features_to_select: Minimum number of features to select. Only used when enable_feature_selection is
        True.
    :param cardinality_threshold_for_onehot_encoding: Categorical features with a cardinality of less or equal
        this threshold will be onehot encoded. The rest will be target encoded. Will be ignored if
        cat_encoding_via_ml_algorithm is set to true.
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
    :param logging_file_path: Path to the logging file. If None, the logging will be printed to the Jupyter notebook
        instead.
    """

    global_random_state: int = 33
    increase_random_state_in_bluecast_cv_by: int = 200
    shuffle_during_training: bool = True
    hyperparameter_tuning_rounds: int = 200
    hyperparameter_tuning_max_runtime_secs: int = 3600
    hypertuning_cv_folds: int = 1
    sample_data_during_tuning: bool = False
    sample_data_during_tuning_alpha: float = 2.0
    precise_cv_tuning: bool = False
    early_stopping_rounds: Optional[int] = 20
    autotune_model: bool = True
    autotune_n_random_seeds: int = 1
    enable_feature_selection: bool = False
    calculate_shap_values: bool = True
    shap_waterfall_indices: List[int] = [0]
    show_dependence_plots_of_top_n_features: int = 1
    store_shap_values_in_instance: bool = False
    train_size: float = 0.8
    train_split_stratify: bool = True
    use_full_data_for_final_model: bool = False
    cardinality_threshold_for_onehot_encoding: int = 5
    cat_encoding_via_ml_algorithm: bool = False
    show_detailed_tuning_logs: bool = False
    optuna_sampler_n_startup_trials: int = 10
    enable_grid_search_fine_tuning: bool = False
    gridsearch_tuning_max_runtime_secs: int = 3600
    gridsearch_nb_parameters_per_grid: int = 5
    bluecast_cv_train_n_model: Tuple[int, int] = (5, 1)
    logging_file_path: Optional[str] = None
    experiment_name: str = "new experiment"


class XgboostTuneParamsConfig(BaseModel):
    """Define hyperparameter tuning search space."""

    max_depth_min: int = 3
    max_depth_max: int = 10
    alpha_min: float = 1e-8
    alpha_max: float = 10
    lambda_min: float = 1
    lambda_max: float = 100
    gamma_min: float = 1e-8
    gamma_max: float = 10
    min_child_weight_min: float = 0.1
    min_child_weight_max: float = 100
    sub_sample_min: float = 1.0
    sub_sample_max: float = 1.0
    col_sample_by_tree_min: float = 0.5
    col_sample_by_tree_max: float = 1.0
    col_sample_by_level_min: float = 1.0
    col_sample_by_level_max: float = 1.0
    eta_min: float = 1e-3
    eta_max: float = 0.3
    steps_min: int = 50
    steps_max: int = 1000
    verbosity_during_hyperparameter_tuning: int = 0
    verbosity_during_final_model_training: int = 0
    xgboost_objective: str = "multi:softprob"
    xgboost_eval_metric: str = "mlogloss"
    booster: str = "gbtree"
    tree_method: str = "hist"


class XgboostTuneParamsRegressionConfig(BaseModel):
    """Define hyperparameter tuning search space."""

    max_depth_min: int = 3
    max_depth_max: int = 10
    alpha_min: float = 1e-8
    alpha_max: float = 10
    lambda_min: float = 1e-8
    lambda_max: float = 100
    gamma_min: float = 1e-8
    gamma_max: float = 10
    min_child_weight_min: float = 0.1
    min_child_weight_max: float = 100
    sub_sample_min: float = 1.0
    sub_sample_max: float = 1.0
    col_sample_by_tree_min: float = 0.5
    col_sample_by_tree_max: float = 1.0
    col_sample_by_level_min: float = 1.0
    col_sample_by_level_max: float = 1.0
    eta_min: float = 1e-3
    eta_max: float = 0.3
    steps_min: int = 50
    steps_max: int = 1000
    verbosity_during_hyperparameter_tuning: int = 0
    verbosity_during_final_model_training: int = 0
    xgboost_objective: str = "reg:squarederror"
    xgboost_eval_metric: str = "rmse"
    booster: str = "gbtree"
    tree_method: str = "hist"


@dataclass
class XgboostFinalParamConfig:
    """Define final hyper parameters."""

    params = {
        "booster": "gbtree",
        "max_depth": 10,  # maximum depth of the decision trees being trained
        "alpha": 0.0,
        "lambda": 1.0,
        "gamma": 0.0,
        "subsample": 1.0,
        "min_child_weight": 1.0,
        "colsample_bytree": 1.0,
        "colsample_bylevel": 1.0,
        "eta": 0.05,
        "steps": 1000,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "tree_method": "hist",
    }
    sample_weight: Optional[Dict[str, float]] = None
    classification_threshold: float = 0.5


@dataclass
class XgboostRegressionFinalParamConfig:
    """Define final hyper parameters."""

    params = {
        "booster": "gbtree",
        "max_depth": 10,  # maximum depth of the decision trees being trained
        "alpha": 0.0,
        "lambda": 1.0,
        "gamma": 0.0,
        "subsample": 1.0,
        "min_child_weight": 1.0,
        "colsample_bytree": 1.0,
        "colsample_bylevel": 1.0,
        "eta": 0.05,
        "steps": 1000,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
    }
