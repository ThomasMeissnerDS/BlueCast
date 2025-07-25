"""Define training and common configuration parameters.

Pydantic dataclasses are used to define the configuration parameters. This allows for type checking and validation of
the configuration parameters. The configuration parameters are used in the training pipeline and in the evaluation
pipeline. Pydantic dataclasses are used to allow users a pythonic way to define the configuration parameters.
Default configurations can be loaded, adjusted and passed into the blueprints.
"""

from typing import Dict, List, Optional, Tuple

from bluecast.config.config_validations import check_types_init


class TrainingConfig:
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
    :param hypertuning_cv_repeats: Number of repetitions for each cross-validation fold during hyperparameter
        tuning. Not used when custom ML model is passed.
    :param sample_data_during_tuning: Whether to sample the data during tuning. Not used when custom ML model is passed.
    :param sample_data_during_tuning_alpha: Alpha value for sampling the data during tuning. The higher alpha the
        fewer samples will be left. Not used when custom ML model is passed.
    :param class_weight_during_dmatrix_creation: Whether to use class weights during DMatrix creation. Not used when
        custom ML model is passed.
    :param early_stopping_rounds: Number of early stopping rounds during final training or when hyperparameter tuning
        follows a single train-test split. Not used when custom ML model is passed.
    :param autotune_model: Whether to autotune the model. Not used when custom ML model is passed.
    :param autotune_on_device: Whether to autotune on CPU or GPU. Chose any of [F"gpu", "cpu"].
        Not used when custom ML model is passed.
    :param autotune_n_random_seeds: Number of random seeds to use for autotuning. This changes Optuna's random seed only.
        Will be updated back after every nth trial back again. Not used when custom ML model is passed.
    :param plot_hyperparameter_tuning_overview: Whether to plot the hyperparameter tuning overview. Not used when custom
        ML model is passed.
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
    :param cardinality_threshold_for_onehot_encoding: Categorical features with a cardinality of less or equal
        this threshold will be onehot encoded. The rest will be target encoded. Will be ignored if
        cat_encoding_via_ml_algorithm is set to true.
    :param infrequent_categories_threshold: Categories with a frequency of less this threshold will be
        grouped into a common group. This is done to reduce the risk of overfitting. Will be ignored if
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
    :param out_of_fold_dataset_store_path: Path to store the out of fold dataset. If None, the out of fold dataset will
        not be stored. Shall end with a slash. Only used when BlueCast instances are called with fit_eval method.
    :param optuna_db_backend_path: Path to the Optuna database backend file. If provided as a string, Optuna will use
        a persistent SQLite database to store hyperparameter tuning progress, allowing resumption if tuning fails.
        If None (default), Optuna will use in-memory storage. Example: "/path/to/optuna_study.db"
    """

    def __init__(
        self,
        global_random_state: int = 33,
        increase_random_state_in_bluecast_cv_by: int = 200,
        shuffle_during_training: bool = True,
        hyperparameter_tuning_rounds: int = 200,
        hyperparameter_tuning_max_runtime_secs: int = 3600,
        hypertuning_cv_folds: int = 5,
        hypertuning_cv_repeats: int = 1,
        sample_data_during_tuning: bool = False,
        sample_data_during_tuning_alpha: float = 2.0,
        precise_cv_tuning: bool = False,
        early_stopping_rounds: Optional[int] = 20,
        autotune_model: bool = True,
        autotune_on_device: str = "cpu",
        autotune_n_random_seeds: int = 1,
        plot_hyperparameter_tuning_overview: bool = True,
        enable_feature_selection: bool = False,
        calculate_shap_values: bool = True,
        shap_waterfall_indices: Optional[List[Optional[int]]] = None,
        show_dependence_plots_of_top_n_features: int = 0,
        store_shap_values_in_instance: bool = False,
        train_size: float = 0.8,
        train_split_stratify: bool = True,
        use_full_data_for_final_model: bool = True,
        cardinality_threshold_for_onehot_encoding: int = 5,
        infrequent_categories_threshold: int = 5,
        cat_encoding_via_ml_algorithm: bool = False,
        show_detailed_tuning_logs: bool = False,
        optuna_sampler_n_startup_trials: int = 10,
        enable_grid_search_fine_tuning: bool = False,
        gridsearch_tuning_max_runtime_secs: int = 3600,
        gridsearch_nb_parameters_per_grid: int = 5,
        bluecast_cv_train_n_model: Tuple[int, int] = (5, 1),
        logging_file_path: Optional[str] = None,
        experiment_name: str = "new experiment",
        out_of_fold_dataset_store_path: Optional[str] = None,
        optuna_db_backend_path: Optional[str] = None,
    ):
        self.global_random_state = global_random_state
        self.increase_random_state_in_bluecast_cv_by = (
            increase_random_state_in_bluecast_cv_by
        )
        self.shuffle_during_training = shuffle_during_training
        self.hyperparameter_tuning_rounds = hyperparameter_tuning_rounds
        self.hyperparameter_tuning_max_runtime_secs = (
            hyperparameter_tuning_max_runtime_secs
        )
        self.hypertuning_cv_folds = hypertuning_cv_folds
        self.hypertuning_cv_repeats = hypertuning_cv_repeats
        self.sample_data_during_tuning = sample_data_during_tuning
        self.sample_data_during_tuning_alpha = sample_data_during_tuning_alpha
        self.precise_cv_tuning = precise_cv_tuning
        self.early_stopping_rounds = early_stopping_rounds
        self.autotune_model = autotune_model
        self.autotune_on_device = autotune_on_device
        self.autotune_n_random_seeds = autotune_n_random_seeds
        self.plot_hyperparameter_tuning_overview = plot_hyperparameter_tuning_overview
        self.enable_feature_selection = enable_feature_selection
        self.calculate_shap_values = calculate_shap_values

        if shap_waterfall_indices is None:
            self.shap_waterfall_indices: List[Optional[int]] = []

        self.show_dependence_plots_of_top_n_features = (
            show_dependence_plots_of_top_n_features
        )
        self.store_shap_values_in_instance = store_shap_values_in_instance
        self.train_size = train_size
        self.train_split_stratify = train_split_stratify
        self.use_full_data_for_final_model = use_full_data_for_final_model
        self.cardinality_threshold_for_onehot_encoding = (
            cardinality_threshold_for_onehot_encoding
        )
        self.infrequent_categories_threshold = infrequent_categories_threshold
        self.cat_encoding_via_ml_algorithm = cat_encoding_via_ml_algorithm
        self.show_detailed_tuning_logs = show_detailed_tuning_logs
        self.optuna_sampler_n_startup_trials = optuna_sampler_n_startup_trials
        self.enable_grid_search_fine_tuning = enable_grid_search_fine_tuning
        self.gridsearch_tuning_max_runtime_secs = gridsearch_tuning_max_runtime_secs
        self.gridsearch_nb_parameters_per_grid = gridsearch_nb_parameters_per_grid
        self.bluecast_cv_train_n_model = bluecast_cv_train_n_model
        self.logging_file_path = logging_file_path
        self.experiment_name = experiment_name
        self.out_of_fold_dataset_store_path = out_of_fold_dataset_store_path
        self.optuna_db_backend_path = optuna_db_backend_path

    def dict(self):
        """
        Return dictionary with all class attributes.

        The implementation keeps backwards compatibility as this class has been a Pydantic Basemodel before.
        """
        return vars(self)


# Xgboost
class XgboostTuneParamsConfig:
    """Define hyperparameter tuning search space.

    :param max_depth_min: Minimum value for the maximum depth of the trees. Defaults to 1.
    :param max_depth_max: Maximum value for the maximum depth of the trees. Defaults to 10.
    :param alpha_min: Minimum value for L1 regularization term (alpha). Defaults to 1e-8.
    :param alpha_max: Maximum value for L1 regularization term (alpha). Defaults to 100.
    :param lambda_min: Minimum value for L2 regularization term (lambda). Defaults to 1.
    :param lambda_max: Maximum value for L2 regularization term (lambda). Defaults to 100.
    :param gamma_min: Minimum value for minimum loss reduction required to make a further partition on a leaf node of the tree (gamma). Defaults to 1e-8.
    :param gamma_max: Maximum value for minimum loss reduction required to make a further partition on a leaf node of the tree (gamma). Defaults to 10.
    :param min_child_weight_min: Minimum value for minimum sum of instance weight (hessian) needed in a child. Defaults to 1.
    :param min_child_weight_max: Maximum value for minimum sum of instance weight (hessian) needed in a child. Defaults to 100.
    :param sub_sample_min: Minimum value of subsample ratio of the training instances. Defaults to 0.1.
    :param sub_sample_max: Maximum value of subsample ratio of the training instances. Defaults to 1.0.
    :param col_sample_by_tree_min: Minimum value of subsample ratio of columns when constructing each tree. Defaults to 0.1.
    :param col_sample_by_tree_max: Maximum value of subsample ratio of columns when constructing each tree. Defaults to 1.0.
    :param col_sample_by_level_min: Minimum value of subsample columns for each split in each level. Defaults to 1.0.
    :param col_sample_by_level_max: Maximum value of subsample columns for each split in each level. Defaults to 1.0.
    :param max_bin_min: Minimum value for maximum number of bins. Defaults to 128.
    :param max_bin_max: Maximum value for maximum number of bins. Defaults to 1024.
    :param eta_min: Minimum value for learning rate (eta). Defaults to 1e-3.
    :param eta_max: Maximum value for learning rate (eta). Defaults to 0.3.
    :param steps_min: Minimum number of boosting rounds. Defaults to 1000.
    :param steps_max: Maximum number of boosting rounds. Defaults to 1000.
    :param verbosity_during_hyperparameter_tuning: Verbosity level during hyperparameter tuning. Defaults to 0.
    :param verbosity_during_final_model_training: Verbosity level during final model training. Defaults to 0.
    :param booster: List of booster types. Defaults to ["gbtree"].
    :param grow_policy: List of grow policies. Defaults to ["depthwise", "lossguide"].
    :param tree_method: List of tree building methods. Defaults to ["exact", "approx", "hist"].
    :param xgboost_objective: XGBoost objective. Defaults to "multi:softprob".
    :param xgboost_eval_metric: XGBoost evaluation metric. Defaults to "mlogloss".
    :param xgboost_eval_metric_tune_direction: Direction to tune the evaluation metric. Defaults to "minimize". Must be any of ['minimize', 'maximize']
    """

    @check_types_init
    def __init__(
        self,
        max_depth_min: int = 1,
        max_depth_max: int = 10,
        alpha_min: float = 1e-8,
        alpha_max: float = 100.0,
        lambda_min: float = 1.0,
        lambda_max: float = 100.0,
        gamma_min: float = 1e-8,
        gamma_max: float = 10.0,
        min_child_weight_min: float = 1.0,
        min_child_weight_max: float = 100.0,
        sub_sample_min: float = 0.1,
        sub_sample_max: float = 1.0,
        col_sample_by_tree_min: float = 0.1,
        col_sample_by_tree_max: float = 1.0,
        col_sample_by_level_min: float = 1.0,
        col_sample_by_level_max: float = 1.0,
        max_bin_min: int = 128,
        max_bin_max: int = 1024,
        eta_min: float = 1e-3,
        eta_max: float = 0.3,
        steps_min: int = 1000,
        steps_max: int = 1000,
        verbosity_during_hyperparameter_tuning: int = 0,
        verbosity_during_final_model_training: int = 0,
        booster: Optional[List[str]] = None,
        grow_policy: Optional[List[str]] = None,
        tree_method: Optional[List[str]] = None,
        xgboost_objective: str = "multi:softprob",
        xgboost_eval_metric: str = "mlogloss",
        xgboost_eval_metric_tune_direction: str = "minimize",
    ):
        if booster is None:
            booster = ["gbtree"]
        if grow_policy is None:
            grow_policy = ["depthwise", "lossguide"]
        if tree_method is None:
            tree_method = ["exact", "approx", "hist"]

        self.max_depth_min = max_depth_min
        self.max_depth_max = max_depth_max
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.min_child_weight_min = min_child_weight_min
        self.min_child_weight_max = min_child_weight_max
        self.sub_sample_min = sub_sample_min
        self.sub_sample_max = sub_sample_max
        self.col_sample_by_tree_min = col_sample_by_tree_min
        self.col_sample_by_tree_max = col_sample_by_tree_max
        self.col_sample_by_level_min = col_sample_by_level_min
        self.col_sample_by_level_max = col_sample_by_level_max
        self.max_bin_min = max_bin_min
        self.max_bin_max = max_bin_max
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.steps_min = steps_min
        self.steps_max = steps_max
        self.verbosity_during_hyperparameter_tuning = (
            verbosity_during_hyperparameter_tuning
        )
        self.verbosity_during_final_model_training = (
            verbosity_during_final_model_training
        )
        self.booster = booster
        self.grow_policy = grow_policy
        self.tree_method = tree_method
        self.xgboost_objective = xgboost_objective
        self.xgboost_eval_metric = xgboost_eval_metric
        self.xgboost_eval_metric_tune_direction = xgboost_eval_metric_tune_direction

    def dict(self):
        """
        Return dictionary with all class attributes.

        The implementation keeps backwards compatibility as this class has been a Pydantic Basemodel before.
        """
        return vars(self)


class XgboostTuneParamsRegressionConfig:
    """Define hyperparameter tuning search space.

    :param max_depth_min: Minimum value for the maximum depth of the trees. Defaults to 1.
    :param max_depth_max: Maximum value for the maximum depth of the trees. Defaults to 10.
    :param alpha_min: Minimum value for L1 regularization term (alpha). Defaults to 1e-8.
    :param alpha_max: Maximum value for L1 regularization term (alpha). Defaults to 100.
    :param lambda_min: Minimum value for L2 regularization term (lambda). Defaults to 1.
    :param lambda_max: Maximum value for L2 regularization term (lambda). Defaults to 100.
    :param gamma_min: Minimum value for minimum loss reduction required to make a further partition on a leaf node of the tree (gamma). Defaults to 1e-8.
    :param gamma_max: Maximum value for minimum loss reduction required to make a further partition on a leaf node of the tree (gamma). Defaults to 10.
    :param min_child_weight_min: Minimum value for minimum sum of instance weight (hessian) needed in a child. Defaults to 1.
    :param min_child_weight_max: Maximum value for minimum sum of instance weight (hessian) needed in a child. Defaults to 100.
    :param sub_sample_min: Minimum value of subsample ratio of the training instances. Defaults to 0.1.
    :param sub_sample_max: Maximum value of subsample ratio of the training instances. Defaults to 1.0.
    :param col_sample_by_tree_min: Minimum value of subsample ratio of columns when constructing each tree. Defaults to 0.1.
    :param col_sample_by_tree_max: Maximum value of subsample ratio of columns when constructing each tree. Defaults to 1.0.
    :param col_sample_by_level_min: Minimum value of subsample columns for each split in each level. Defaults to 1.0.
    :param col_sample_by_level_max: Maximum value of subsample columns for each split in each level. Defaults to 1.0.
    :param max_bin_min: Minimum value for maximum number of bins. Defaults to 128.
    :param max_bin_max: Maximum value for maximum number of bins. Defaults to 1024.
    :param eta_min: Minimum value for learning rate (eta). Defaults to 1e-3.
    :param eta_max: Maximum value for learning rate (eta). Defaults to 0.3.
    :param steps_min: Minimum number of boosting rounds. Defaults to 1000.
    :param steps_max: Maximum number of boosting rounds. Defaults to 1000.
    :param verbosity_during_hyperparameter_tuning: Verbosity level during hyperparameter tuning. Defaults to 0.
    :param verbosity_during_final_model_training: Verbosity level during final model training. Defaults to 0.
    :param booster: List of booster types. Defaults to ["gbtree"].
    :param grow_policy: List of grow policies. Defaults to ["depthwise", "lossguide"].
    :param tree_method: List of tree building methods. Defaults to ["exact", "approx", "hist"].
    :param xgboost_objective: XGBoost objective. Defaults to "reg:squarederror".
    :param xgboost_eval_metric: XGBoost evaluation metric. Defaults to "rmse".
    :param xgboost_eval_metric_tune_direction: Direction to tune the evaluation metric. Defaults to "minimize". Must be any of ['minimize', 'maximize']
    """

    @check_types_init
    def __init__(
        self,
        max_depth_min: int = 1,
        max_depth_max: int = 10,
        alpha_min: float = 1e-8,
        alpha_max: float = 100.0,
        lambda_min: float = 1.0,
        lambda_max: float = 100.0,
        gamma_min: float = 1e-8,
        gamma_max: float = 10.0,
        min_child_weight_min: float = 1.0,
        min_child_weight_max: float = 100.0,
        sub_sample_min: float = 0.1,
        sub_sample_max: float = 1.0,
        col_sample_by_tree_min: float = 0.1,
        col_sample_by_tree_max: float = 1.0,
        col_sample_by_level_min: float = 1.0,
        col_sample_by_level_max: float = 1.0,
        max_bin_min: int = 128,
        max_bin_max: int = 1024,
        eta_min: float = 1e-3,
        eta_max: float = 0.3,
        steps_min: int = 1000,
        steps_max: int = 1000,
        verbosity_during_hyperparameter_tuning: int = 0,
        verbosity_during_final_model_training: int = 0,
        booster: Optional[List[str]] = None,
        grow_policy: Optional[List[str]] = None,
        tree_method: Optional[List[str]] = None,
        xgboost_objective: str = "reg:squarederror",
        xgboost_eval_metric: str = "rmse",
        xgboost_eval_metric_tune_direction: str = "minimize",
    ):
        if booster is None:
            booster = ["gbtree"]
        if grow_policy is None:
            grow_policy = ["depthwise", "lossguide"]
        if tree_method is None:
            tree_method = ["exact", "approx", "hist"]

        self.max_depth_min = max_depth_min
        self.max_depth_max = max_depth_max
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.min_child_weight_min = min_child_weight_min
        self.min_child_weight_max = min_child_weight_max
        self.sub_sample_min = sub_sample_min
        self.sub_sample_max = sub_sample_max
        self.col_sample_by_tree_min = col_sample_by_tree_min
        self.col_sample_by_tree_max = col_sample_by_tree_max
        self.col_sample_by_level_min = col_sample_by_level_min
        self.col_sample_by_level_max = col_sample_by_level_max
        self.max_bin_min = max_bin_min
        self.max_bin_max = max_bin_max
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.steps_min = steps_min
        self.steps_max = steps_max
        self.verbosity_during_hyperparameter_tuning = (
            verbosity_during_hyperparameter_tuning
        )
        self.verbosity_during_final_model_training = (
            verbosity_during_final_model_training
        )
        self.booster = booster
        self.grow_policy = grow_policy
        self.tree_method = tree_method
        self.xgboost_objective = xgboost_objective
        self.xgboost_eval_metric = xgboost_eval_metric
        self.xgboost_eval_metric_tune_direction = xgboost_eval_metric_tune_direction

    def dict(self):
        """
        Return dictionary with all class attributes.

        The implementation keeps backwards compatibility as this class has been a Pydantic Basemodel before.
        """
        return vars(self)


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
        "device": "cpu",
    }
    sample_weight: Optional[Dict[str, float]] = None
    classification_threshold: float = 0.5


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
        "device": "cpu",
    }
    sample_weight: Optional[Dict[str, float]] = None
    classification_threshold: float = 999

    # Catboost


class CatboostTuneParamsConfig:
    """Define hyperparameter tuning search space for CatBoost (classification or multiclass).

    :param depth_min: Minimum value for the depth of the trees. Defaults to 1.
    :param depth_max: Maximum value for the depth of the trees. Defaults to 10.
    :param l2_leaf_reg_min: Minimum value for L2 regularization term (l2_leaf_reg). Defaults to 1e-8.
    :param l2_leaf_reg_max: Maximum value for L2 regularization term (l2_leaf_reg). Defaults to 100.
    :param bagging_temperature_min: Minimum value for bagging temperature when bootstrap_type='Bayesian'. Defaults to 0.0.
    :param bagging_temperature_max: Maximum value for bagging temperature when bootstrap_type='Bayesian'. Defaults to 10.0.
    :param random_strength_min: Minimum value for the random strength. Defaults to 0.0.
    :param random_strength_max: Maximum value for the random strength. Defaults to 10.0.
    :param subsample_min: Minimum value of subsample ratio of the training instances. Defaults to 0.1.
    :param subsample_max: Maximum value of subsample ratio of the training instances. Defaults to 1.0.
    :param border_count_min: Minimum value for the number of splits for numerical features. Defaults to 32.
    :param border_count_max: Maximum value for the number of splits for numerical features. Defaults to 255.
    :param learning_rate_min: Minimum value for learning rate. Defaults to 1e-3.
    :param learning_rate_max: Maximum value for learning rate. Defaults to 0.3.
    :param iterations_min: Minimum number of boosting rounds (iterations). Defaults to 1000.
    :param iterations_max: Maximum number of boosting rounds (iterations). Defaults to 1000.
    :param verbosity_during_hyperparameter_tuning: Verbosity level during hyperparameter tuning. Defaults to 0.
    :param verbosity_during_final_model_training: Verbosity level during final model training. Defaults to 0.
    :param bootstrap_type: List of bootstrap types to consider. Defaults to ["Bayesian", "Poisson", "MVS", "No"].
    :param grow_policy: List of grow policies. Defaults to ["SymmetricTree"].
    :param catboost_objective: CatBoost objective. Defaults to "MultiClass".
    :param catboost_eval_metric: CatBoost evaluation metric. Defaults to "MultiClass".
    :param catboost_eval_metric_tune_direction: Direction to tune the evaluation metric. Defaults to "minimize".
                                                Must be any of ['minimize', 'maximize']
    """

    @check_types_init
    def __init__(
        self,
        depth_min: int = 1,
        depth_max: int = 10,
        l2_leaf_reg_min: float = 1e-8,
        l2_leaf_reg_max: float = 100.0,
        bagging_temperature_min: float = 0.0,
        bagging_temperature_max: float = 10.0,
        random_strength_min: float = 0.0,
        random_strength_max: float = 10.0,
        subsample_min: float = 0.1,
        subsample_max: float = 1.0,
        border_count_min: int = 32,
        border_count_max: int = 255,
        learning_rate_min: float = 1e-3,
        learning_rate_max: float = 0.3,
        iterations_min: int = 1000,
        iterations_max: int = 1000,
        verbosity_during_hyperparameter_tuning: int = 0,
        verbosity_during_final_model_training: int = 0,
        bootstrap_type: Optional[List[str]] = None,
        grow_policy: Optional[List[str]] = None,
        catboost_objective: str = "MultiClass",
        catboost_eval_metric: str = "MultiClass",
        catboost_eval_metric_tune_direction: str = "minimize",
    ):
        if bootstrap_type is None:
            bootstrap_type = [
                "Bayesian",
                "No",
            ]  # Poisson not possible on CPU, "MVS" requires min samples
        if grow_policy is None:
            grow_policy = ["SymmetricTree"]

        self.depth_min = depth_min
        self.depth_max = depth_max
        self.l2_leaf_reg_min = l2_leaf_reg_min
        self.l2_leaf_reg_max = l2_leaf_reg_max
        self.bagging_temperature_min = bagging_temperature_min
        self.bagging_temperature_max = bagging_temperature_max
        self.random_strength_min = random_strength_min
        self.random_strength_max = random_strength_max
        self.subsample_min = subsample_min
        self.subsample_max = subsample_max
        self.border_count_min = border_count_min
        self.border_count_max = border_count_max
        self.learning_rate_min = learning_rate_min
        self.learning_rate_max = learning_rate_max
        self.iterations_min = iterations_min
        self.iterations_max = iterations_max
        self.verbosity_during_hyperparameter_tuning = (
            verbosity_during_hyperparameter_tuning
        )
        self.verbosity_during_final_model_training = (
            verbosity_during_final_model_training
        )
        self.bootstrap_type = bootstrap_type
        self.grow_policy = grow_policy
        self.catboost_objective = catboost_objective
        self.catboost_eval_metric = catboost_eval_metric
        self.catboost_eval_metric_tune_direction = catboost_eval_metric_tune_direction

    def dict(self):
        """
        Return dictionary with all class attributes.

        The implementation keeps backwards compatibility as this class mimics a Pydantic BaseModel.
        """
        return vars(self)


class CatboostTuneParamsRegressionConfig:
    """Define hyperparameter tuning search space for CatBoost (regression).

    :param depth_min: Minimum value for the depth of the trees. Defaults to 1.
    :param depth_max: Maximum value for the depth of the trees. Defaults to 10.
    :param l2_leaf_reg_min: Minimum value for L2 regularization term (l2_leaf_reg). Defaults to 1e-8.
    :param l2_leaf_reg_max: Maximum value for L2 regularization term (l2_leaf_reg). Defaults to 100.
    :param bagging_temperature_min: Minimum value for bagging temperature when bootstrap_type='Bayesian'. Defaults to 0.0.
    :param bagging_temperature_max: Maximum value for bagging temperature when bootstrap_type='Bayesian'. Defaults to 10.0.
    :param random_strength_min: Minimum value for the random strength. Defaults to 0.0.
    :param random_strength_max: Maximum value for the random strength. Defaults to 10.0.
    :param subsample_min: Minimum value of subsample ratio of the training instances. Defaults to 0.1.
    :param subsample_max: Maximum value of subsample ratio of the training instances. Defaults to 1.0.
    :param border_count_min: Minimum value for the number of splits for numerical features. Defaults to 32.
    :param border_count_max: Maximum value for the number of splits for numerical features. Defaults to 255.
    :param learning_rate_min: Minimum value for learning rate. Defaults to 1e-3.
    :param learning_rate_max: Maximum value for learning rate. Defaults to 0.3.
    :param iterations_min: Minimum number of boosting rounds (iterations). Defaults to 1000.
    :param iterations_max: Maximum number of boosting rounds (iterations). Defaults to 1000.
    :param verbosity_during_hyperparameter_tuning: Verbosity level during hyperparameter tuning. Defaults to 0.
    :param verbosity_during_final_model_training: Verbosity level during final model training. Defaults to 0.
    :param bootstrap_type: List of bootstrap types to consider. Defaults to ["Bayesian", "Poisson", "MVS", "No"].
    :param grow_policy: List of grow policies. Defaults to ["SymmetricTree"].
    :param catboost_objective: CatBoost objective. Defaults to "RMSE".
    :param catboost_eval_metric: CatBoost evaluation metric. Defaults to "RMSE".
    :param catboost_eval_metric_tune_direction: Direction to tune the evaluation metric. Defaults to "minimize".
                                                Must be any of ['minimize', 'maximize']
    """

    @check_types_init
    def __init__(
        self,
        depth_min: int = 1,
        depth_max: int = 10,
        l2_leaf_reg_min: float = 1e-8,
        l2_leaf_reg_max: float = 100.0,
        bagging_temperature_min: float = 0.0,
        bagging_temperature_max: float = 10.0,
        random_strength_min: float = 0.0,
        random_strength_max: float = 10.0,
        subsample_min: float = 0.1,
        subsample_max: float = 1.0,
        border_count_min: int = 32,
        border_count_max: int = 255,
        learning_rate_min: float = 1e-3,
        learning_rate_max: float = 0.3,
        iterations_min: int = 1000,
        iterations_max: int = 1000,
        verbosity_during_hyperparameter_tuning: int = 0,
        verbosity_during_final_model_training: int = 0,
        bootstrap_type: Optional[List[str]] = None,
        grow_policy: Optional[List[str]] = None,
        catboost_objective: str = "RMSE",
        catboost_eval_metric: str = "RMSE",
        catboost_eval_metric_tune_direction: str = "minimize",
    ):
        if bootstrap_type is None:
            bootstrap_type = ["Bayesian", "No"]  # "Poisson", "MVS"
        if grow_policy is None:
            grow_policy = ["SymmetricTree"]

        self.depth_min = depth_min
        self.depth_max = depth_max
        self.l2_leaf_reg_min = l2_leaf_reg_min
        self.l2_leaf_reg_max = l2_leaf_reg_max
        self.bagging_temperature_min = bagging_temperature_min
        self.bagging_temperature_max = bagging_temperature_max
        self.random_strength_min = random_strength_min
        self.random_strength_max = random_strength_max
        self.subsample_min = subsample_min
        self.subsample_max = subsample_max
        self.border_count_min = border_count_min
        self.border_count_max = border_count_max
        self.learning_rate_min = learning_rate_min
        self.learning_rate_max = learning_rate_max
        self.iterations_min = iterations_min
        self.iterations_max = iterations_max
        self.verbosity_during_hyperparameter_tuning = (
            verbosity_during_hyperparameter_tuning
        )
        self.verbosity_during_final_model_training = (
            verbosity_during_final_model_training
        )
        self.bootstrap_type = bootstrap_type
        self.grow_policy = grow_policy
        self.catboost_objective = catboost_objective
        self.catboost_eval_metric = catboost_eval_metric
        self.catboost_eval_metric_tune_direction = catboost_eval_metric_tune_direction

    def dict(self):
        """
        Return dictionary with all class attributes.

        The implementation keeps backwards compatibility as this class mimics a Pydantic BaseModel.
        """
        return vars(self)


class CatboostFinalParamConfig:
    """Define final hyperparameters for CatBoost (classification or multiclass) using CatBoost defaults."""

    params = {
        "iterations": 1000,
        "depth": 6,
        "learning_rate": 0.03,
        "l2_leaf_reg": 3.0,
        "eval_metric": "MultiClass",
        "loss_function": "MultiClass",
        "random_seed": 0,
        "logging_level": "Silent",
    }
    sample_weight: Optional[Dict[str, float]] = None
    classification_threshold: float = 0.5


class CatboostRegressionFinalParamConfig:
    """Define final hyperparameters for CatBoost (regression) using CatBoost defaults."""

    params = {
        "iterations": 1000,
        "depth": 6,
        "learning_rate": 0.03,
        "l2_leaf_reg": 3.0,
        "eval_metric": "RMSE",
        "loss_function": "RMSE",
        "random_seed": 0,
        "logging_level": "Silent",
    }
    sample_weight: Optional[Dict[str, float]] = None
    classification_threshold: float = (
        999  # Not typically used in regression but kept for compatibility
    )
