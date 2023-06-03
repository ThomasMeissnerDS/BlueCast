from pydantic.dataclasses import dataclass


@dataclass
class TrainingConfig:
    global_random_state: int = 100
    shuffle_during_training: bool = True
    hyperparameter_tuning_rounds: int = 100
    hyperparameter_tuning_max_runtime_secs: int = 3600
    hypertuning_cv_folds: int = 1
    early_stopping_rounds: int = 10


@dataclass
class XgboostParamsConfig:
    max_depth_min: int = 2
    max_depth_max: int = 3
    alpha_min: float = 0.0
    alpha_max: float = 1e6
    lambda_min: float = 0.0
    lambda_max: float = 1e6
    num_leaves_min: int = 2
    num_leaves_max: int = 16
    sub_sample_min: float = 0.3
    sub_sample_max: float = 1.0
    col_sample_by_tree_min: float = 0.3
    col_sample_by_tree_max: float = 1.0
    col_sample_by_level_min: float = 0.3
    col_sample_by_level_max: float = 1.0
    col_sample_by_node_min: float = 0.3
    col_sample_by_node_max: float = 1.0
    min_child_samples_min: int = 5
    min_child_samples_max: int = 1000
    eta: float = 0.1
    steps_min: int = 2
    steps_max: int = 50000
    num_parallel_tree_min: int = 1
    num_parallel_tree_max: int = 3
