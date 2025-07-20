"""Base classes for all ML models."""

import logging
import pickle
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TypeVar, Union

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.utils import class_weight

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
from bluecast.evaluation.eval_metrics import (
    ClassificationEvalWrapper,
    RegressionEvalWrapper,
)
from bluecast.experimentation.tracking import ExperimentTracker
from bluecast.preprocessing.custom import CustomPreprocessing

warnings.filterwarnings("ignore", "is_sparse is deprecated")

PredictedProbas = TypeVar("PredictedProbas", np.ndarray, pd.Series)
PredictedClasses = TypeVar("PredictedClasses", np.ndarray, pd.Series)


class BaseClassMlModel(ABC):
    """Base class for all ML models.

    Enforces the implementation of the fit and predict methods.
    If hyperparameter tuning is required, then the fit method should implement the tuning.
    """

    @abstractmethod
    def fit(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> Optional[Any]:
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> Tuple[PredictedProbas, PredictedClasses]:
        """
        Predict on unseen data.

        :return tuple of predicted probabilities and predicted classes
        """
        pass


class BaseClassMlRegressionModel(ABC):
    """Base class for all ML models.

    Enforces the implementation of the fit and predict methods.
    If hyperparameter tuning is required, then the fit method should implement the tuning.
    """

    @abstractmethod
    def fit(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> Optional[Any]:
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict on unseen data.

        :return numpy array of predictions
        """
        pass


# Xgboost specific base class
class XgboostBaseModel:
    def __init__(
        self,
        class_problem: Union[Literal["binary", "multiclass"], Literal["regression"]],
        conf_training: Optional[TrainingConfig] = None,
        conf_xgboost: Optional[
            Union[XgboostTuneParamsConfig, XgboostTuneParamsRegressionConfig]
        ] = None,
        conf_params_xgboost: Optional[
            Union[XgboostFinalParamConfig, XgboostRegressionFinalParamConfig]
        ] = None,
        experiment_tracker: Optional[ExperimentTracker] = None,
        custom_in_fold_preprocessor: Optional[CustomPreprocessing] = None,
        cat_columns: Optional[List[Union[str, float, int]]] = None,
        single_fold_eval_metric_func: Optional[
            Union[ClassificationEvalWrapper, RegressionEvalWrapper]
        ] = None,
    ):
        self.model: Optional[Union[xgb.XGBClassifier, xgb.XGBRegressor]] = None
        self.class_problem = class_problem

        self._load_xgboost_training_config(conf_xgboost)
        self._load_xgboost_final_params(conf_params_xgboost)
        self._load_training_settings_config(conf_training)
        self._load_experiment_tracker(experiment_tracker)

        self.custom_in_fold_preprocessor = custom_in_fold_preprocessor
        self.single_fold_eval_metric_func = single_fold_eval_metric_func
        self.random_generator = np.random.default_rng(
            self.conf_training.global_random_state
        )
        self.cat_columns = cat_columns
        self.best_score: float = np.inf

    def _load_xgboost_training_config(self, conf_xgboost) -> None:
        if conf_xgboost is None and self.class_problem in ["binary", "multiclass"]:
            logging.info("Load default XgboostTuneParamsConfig.")
            self.conf_xgboost: Union[
                XgboostTuneParamsConfig, XgboostTuneParamsRegressionConfig
            ] = XgboostTuneParamsConfig()
        elif conf_xgboost is None and self.class_problem in ["regression"]:
            logging.info("Load default XgboostTuneParamsRegressionConfig.")
            self.conf_xgboost = XgboostTuneParamsRegressionConfig()
        elif conf_xgboost is None:
            raise ValueError
        else:
            logging.info("Found provided XgboostTuneParamsConfig.")
            self.conf_xgboost = conf_xgboost

    def _load_xgboost_final_params(self, conf_params_xgboost) -> None:
        if conf_params_xgboost is None and self.class_problem in [
            "binary",
            "multiclass",
        ]:
            logging.info("Load default XgboostFinalParamConfig.")
            self.conf_params_xgboost: Union[
                XgboostFinalParamConfig, XgboostRegressionFinalParamConfig
            ] = XgboostFinalParamConfig()
        elif conf_params_xgboost is None and self.class_problem in ["regression"]:
            self.conf_params_xgboost = XgboostRegressionFinalParamConfig()
        elif conf_params_xgboost is None:
            raise ValueError
        else:
            logging.info("Found provided XgboostFinalParamConfig.")
            self.conf_params_xgboost = conf_params_xgboost

    def _load_training_settings_config(self, conf_training) -> None:
        if conf_training is None:
            logging.info("Load default TrainingConfig.")
            self.conf_training = TrainingConfig()
        else:
            logging.info("Load default TrainingConfig.")
            self.conf_training = conf_training

    def _load_experiment_tracker(self, experiment_tracker) -> None:
        if experiment_tracker is None:
            self.experiment_tracker = ExperimentTracker()
        else:
            self.experiment_tracker = experiment_tracker

    def _create_d_matrices(self, x_train, y_train, x_test, y_test):
        if self.conf_params_xgboost.sample_weight and self.class_problem in [
            "binary",
            "multiclass",
        ]:
            classes_weights = class_weight.compute_sample_weight(
                class_weight="balanced", y=y_train
            )
            d_train = xgb.DMatrix(
                x_train,
                label=y_train,
                weight=classes_weights,
                enable_categorical=self.conf_training.cat_encoding_via_ml_algorithm,
            )
        else:
            d_train = xgb.DMatrix(
                x_train,
                label=y_train,
                enable_categorical=self.conf_training.cat_encoding_via_ml_algorithm,
            )
        d_test = xgb.DMatrix(
            x_test,
            label=y_test,
            enable_categorical=self.conf_training.cat_encoding_via_ml_algorithm,
        )
        return d_train, d_test

    def concat_prepare_full_train_datasets(
        self,
        *,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ):
        """
        Prepare training dataset and concat with test data.

        This is only recommended if early stopping is not used or not used on the same eval set.

        :param x_train: Pandas DataFrame with data without labels.
        :param y_train: Pandas Series with labels.
        :param x_test: Pandas DataFrame with data without labels.
        :param y_test: Pandas Series with labels.
        :return: Prepared training dataset as Pandas DataFrame, Pandas Series (labels)
        """
        logging.info(
            f"""{datetime.utcnow()}: Union train and test data for final model training based on TrainingConfig
             param 'use_full_data_for_final_model'"""
        )
        x_train = pd.concat([x_train, x_test]).reset_index(drop=True)
        y_train = pd.concat([y_train, y_test]).reset_index(drop=True)

        if self.cat_columns and self.conf_training.cat_encoding_via_ml_algorithm:
            x_train[self.cat_columns] = x_train[self.cat_columns].astype("category")

        return x_train, y_train

    def get_early_stopping_callback(self) -> Optional[List[xgb.callback.EarlyStopping]]:
        """Create early stopping callback if configured."""
        if self.conf_training.early_stopping_rounds:
            early_stop = xgb.callback.EarlyStopping(
                rounds=self.conf_training.early_stopping_rounds,
                metric_name=self.conf_xgboost.xgboost_eval_metric,
                data_name="test",
                save_best=self.conf_params_xgboost.params["booster"] != "gblinear",
            )
            callbacks = [early_stop]
        else:
            callbacks = None
        return callbacks

    def autotune(
        self,
        *,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ):
        raise NotImplementedError("Method autotune has not been defined.")

    def fine_tune(
        self,
        *,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ):
        raise NotImplementedError("Method fine_tune has not been defined.")

    def create_fine_tune_search_space(self) -> Dict[str, np.array]:
        if (
            isinstance(self.conf_params_xgboost.params["min_child_weight"], float)
            and isinstance(self.conf_params_xgboost.params["lambda"], float)
            and isinstance(self.conf_params_xgboost.params["gamma"], float)
            and isinstance(self.conf_params_xgboost.params["eta"], float)
        ):
            search_space = {
                "min_child_weight": np.linspace(
                    self.conf_params_xgboost.params["min_child_weight"] * 0.9,
                    self.conf_params_xgboost.params["min_child_weight"] * 1.1,
                    self.conf_training.gridsearch_nb_parameters_per_grid,
                    dtype=float,
                ),
                "lambda": np.linspace(
                    self.conf_params_xgboost.params["lambda"] * 0.9,
                    self.conf_params_xgboost.params["lambda"] * 1.1,
                    self.conf_training.gridsearch_nb_parameters_per_grid,
                    dtype=float,
                ),
                "gamma": np.linspace(
                    self.conf_params_xgboost.params["gamma"] * 0.9,
                    self.conf_params_xgboost.params["gamma"] * 1.1,
                    self.conf_training.gridsearch_nb_parameters_per_grid,
                    dtype=float,
                ),
                "eta": np.linspace(
                    self.conf_params_xgboost.params["eta"] * 0.9,
                    self.conf_params_xgboost.params["eta"] * 1.1,
                    self.conf_training.gridsearch_nb_parameters_per_grid,
                    dtype=float,
                ),
            }
            return search_space
        else:
            raise ValueError("Some parameters are not floats or integers")

    def _get_param_space_fpr_grid_search(
        self, trial: optuna.trial
    ) -> Dict[str, np.array]:
        if (
            isinstance(self.conf_params_xgboost.params["min_child_weight"], float)
            and isinstance(self.conf_params_xgboost.params["lambda"], float)
            and isinstance(self.conf_params_xgboost.params["gamma"], float)
            and isinstance(self.conf_params_xgboost.params["eta"], float)
        ):
            # copy best params to not overwrite them
            tuned_params = deepcopy(self.conf_params_xgboost.params)
            min_child_weight_space = trial.suggest_float(
                "min_child_weight",
                self.conf_params_xgboost.params["min_child_weight"] * 0.9,
                self.conf_params_xgboost.params["min_child_weight"] * 1.1,
                log=False,
            )
            lambda_space = trial.suggest_float(
                "lambda",
                self.conf_params_xgboost.params["lambda"] * 0.9,
                self.conf_params_xgboost.params["lambda"] * 1.1,
                log=False,
            )
            gamma_space = trial.suggest_float(
                "gamma",
                self.conf_params_xgboost.params["gamma"] * 0.9,
                self.conf_params_xgboost.params["gamma"] * 1.1,
                log=False,
            )
            eta_space = trial.suggest_float(
                "eta",
                self.conf_params_xgboost.params["eta"] * 0.9,
                self.conf_params_xgboost.params["eta"] * 1.1,
                log=False,
            )

            tuned_params["lambda"] = lambda_space
            tuned_params["min_child_weight"] = min_child_weight_space
            tuned_params["gamma"] = gamma_space
            tuned_params["eta"] = eta_space
            return tuned_params
        else:
            raise ValueError("Some parameters are not floats or integers")

    def _optimize_and_plot_grid_search_study(
        self, objective: Callable, search_space: Dict[str, np.array]
    ) -> None:
        study = self._create_optuna_study(
            direction=self.conf_xgboost.xgboost_eval_metric_tune_direction,
            sampler=optuna.samplers.GridSampler(search_space),
            study_name="xgboost_grid_search",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=50),
        )
        study.optimize(
            objective,
            n_trials=self.conf_training.gridsearch_nb_parameters_per_grid
            ** len(search_space.keys()),
            timeout=self.conf_training.gridsearch_tuning_max_runtime_secs,
            gc_after_trial=True,
            show_progress_bar=True,
        )

        if self.conf_training.plot_hyperparameter_tuning_overview:
            try:
                fig = optuna.visualization.plot_optimization_history(study)
                fig.show()
                fig = optuna.visualization.plot_param_importances(
                    study  # , evaluator=FanovaImportanceEvaluator()
                )
                fig.show()
            except (ZeroDivisionError, RuntimeError, ValueError):
                pass

        best_score_cv = self.best_score

        if study.best_value < self.best_score or not self.conf_training.autotune_model:
            self.best_score = study.best_value
            xgboost_grid_best_param = study.best_trial.params
            self.conf_params_xgboost.params["min_child_weight"] = (
                xgboost_grid_best_param["min_child_weight"]
            )
            self.conf_params_xgboost.params["lambda"] = xgboost_grid_best_param[
                "lambda"
            ]
            self.conf_params_xgboost.params["gamma"] = xgboost_grid_best_param["gamma"]
            self.conf_params_xgboost.params["eta"] = xgboost_grid_best_param["eta"]
            logging.info(
                f"Grid search improved eval metric from {best_score_cv} to {self.best_score}."
            )
            logging.info(f"Best params: {self.conf_params_xgboost.params}")
            print(f"Best params: {self.conf_params_xgboost.params}")
        else:
            logging.info(
                f"Grid search could not improve eval metric of {best_score_cv}. Best score reached was {study.best_value}"
            )

    def orchestrate_hyperparameter_tuning(
        self,
        *,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ):
        if not self.conf_training.show_detailed_tuning_logs:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        if self.conf_training.autotune_model:
            self.autotune(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
            )
            print("Finished hyperparameter tuning")

        if self.conf_training.enable_grid_search_fine_tuning:
            self.fine_tune(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
            )
            print("Finished Grid search fine tuning")

    def _create_optuna_study(
        self,
        direction: str,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        study_name: str = "hyperparameter_tuning",
        pruner: Optional[optuna.pruners.BasePruner] = None,
    ) -> optuna.Study:
        """
        Create an Optuna study with optional database backend support.

        :param direction: Direction to optimize ('minimize' or 'maximize')
        :param sampler: Optuna sampler to use
        :param study_name: Name of the study
        :param pruner: Optuna pruner to use
        :return: Configured Optuna study
        """
        study_kwargs: Dict[str, Any] = {
            "direction": direction,
            "study_name": study_name,
        }

        if sampler is not None:
            study_kwargs["sampler"] = sampler
        if pruner is not None:
            study_kwargs["pruner"] = pruner

        # Add database backend if configured
        if self.conf_training.optuna_db_backend_path is not None and isinstance(
            self.conf_training.optuna_db_backend_path, str
        ):
            storage_name = f"sqlite:///{self.conf_training.optuna_db_backend_path}"
            study_kwargs["storage"] = storage_name
            study_kwargs["load_if_exists"] = True

            # Save the sampler state for resumption if database backend is used
            if sampler is not None and hasattr(sampler, "seed"):
                sampler_path = self.conf_training.optuna_db_backend_path.replace(
                    ".db", "_sampler.pkl"
                )
                try:
                    with open(sampler_path, "wb") as fout:
                        pickle.dump(sampler, fout)
                    logging.info(f"Saved sampler state to {sampler_path}")
                except Exception as e:
                    logging.warning(f"Could not save sampler state: {e}")

        return optuna.create_study(**study_kwargs)


# Catboost specific base class
class CatboostBaseModel:
    """
    Example base model class for CatBoost, replicating the structure and logic
    of your XgboostBaseModel.
    """

    def __init__(
        self,
        class_problem: Union[Literal["binary", "multiclass"], Literal["regression"]],
        conf_training: Optional["TrainingConfig"] = None,
        conf_catboost: Optional[
            Union["CatboostTuneParamsConfig", "CatboostTuneParamsRegressionConfig"]
        ] = None,
        conf_params_catboost: Optional[
            Union["CatboostFinalParamConfig", "CatboostRegressionFinalParamConfig"]
        ] = None,
        experiment_tracker: Optional["ExperimentTracker"] = None,
        custom_in_fold_preprocessor: Optional["CustomPreprocessing"] = None,
        cat_columns: Optional[List[Union[str, float, int]]] = None,
        single_fold_eval_metric_func: Optional[
            Union["ClassificationEvalWrapper", "RegressionEvalWrapper"]
        ] = None,
    ):
        # CatBoost model references
        self.model: Optional[Union[CatBoostClassifier, CatBoostRegressor]] = None

        self.class_problem = class_problem

        # Load configurations
        self._load_catboost_training_config(conf_catboost)
        self._load_catboost_final_params(conf_params_catboost)
        self._load_training_settings_config(conf_training)
        self._load_experiment_tracker(experiment_tracker)

        # Additional references (preprocessor, columns, random generator, etc.)
        self.custom_in_fold_preprocessor = custom_in_fold_preprocessor
        self.single_fold_eval_metric_func = single_fold_eval_metric_func
        self.random_generator = np.random.default_rng(
            self.conf_training.global_random_state
        )
        self.cat_columns = cat_columns

        # Track the best score from hyperparameter searches
        self.best_score: float = np.inf

    def _load_catboost_training_config(self, conf_catboost) -> None:
        """
        Loads CatBoost tuning configuration. If none is provided, uses
        either the classification or the regression default class.
        """
        if conf_catboost is None and self.class_problem in ["binary", "multiclass"]:
            logging.info("Load default CatboostTuneParamsConfig.")
            self.conf_catboost: Union[
                CatboostTuneParamsConfig, CatboostTuneParamsRegressionConfig
            ] = CatboostTuneParamsConfig()
        elif conf_catboost is None and self.class_problem in ["regression"]:
            logging.info("Load default CatboostTuneParamsRegressionConfig.")
            self.conf_catboost = CatboostTuneParamsRegressionConfig()
        elif conf_catboost is None:
            raise ValueError(
                "No CatBoost config provided and class_problem not recognized."
            )
        else:
            logging.info("Found provided CatboostTuneParamsConfig/RegressionConfig.")
            self.conf_catboost = conf_catboost

    def _load_catboost_final_params(self, conf_params_catboost) -> None:
        """
        Loads CatBoost final parameters. If none is provided, uses
        either the classification or the regression default class.
        """
        if conf_params_catboost is None and self.class_problem in [
            "binary",
            "multiclass",
        ]:
            logging.info("Load default CatboostFinalParamConfig.")
            self.conf_params_catboost: Union[
                CatboostFinalParamConfig, CatboostRegressionFinalParamConfig
            ] = CatboostFinalParamConfig()
        elif conf_params_catboost is None and self.class_problem in ["regression"]:
            logging.info("Load default CatboostRegressionFinalParamConfig.")
            self.conf_params_catboost = CatboostRegressionFinalParamConfig()
        elif conf_params_catboost is None:
            raise ValueError(
                "No CatBoost final config provided and class_problem not recognized."
            )
        else:
            logging.info(
                "Found provided CatboostFinalParamConfig/RegressionFinalParamConfig."
            )
            self.conf_params_catboost = conf_params_catboost

    def _load_training_settings_config(self, conf_training) -> None:
        """
        Loads or creates a default TrainingConfig.
        """
        if conf_training is None:
            logging.info("Load default TrainingConfig.")
            self.conf_training = TrainingConfig()
        else:
            logging.info("Load custom TrainingConfig.")
            self.conf_training = conf_training

    def _load_experiment_tracker(self, experiment_tracker) -> None:
        """
        Loads or creates a default ExperimentTracker.
        """
        if experiment_tracker is None:
            self.experiment_tracker = ExperimentTracker()
        else:
            self.experiment_tracker = experiment_tracker

    def _create_pools(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ):
        """
        Creates CatBoost Pools for training and testing.
        Potentially uses sample weights if available.
        Also sets cat_features if self.cat_columns is provided.
        """
        if self.conf_params_catboost.sample_weight:
            # Just as an example, for classification we might compute balanced sample weights
            if self.class_problem in ["binary", "multiclass"]:
                classes_weights = class_weight.compute_sample_weight(
                    class_weight="balanced", y=y_train
                )
                train_pool = Pool(
                    data=x_train,
                    label=y_train,
                    weight=classes_weights,
                    cat_features=self.cat_columns,
                )
            else:
                # For regression or if no special weighting is needed:
                train_pool = Pool(
                    data=x_train, label=y_train, cat_features=self.cat_columns
                )
        else:
            # No sample weighting
            train_pool = Pool(
                data=x_train, label=y_train, cat_features=self.cat_columns
            )

        if x_test.empty:
            test_pool = None
        else:
            test_pool = Pool(data=x_test, label=y_test, cat_features=self.cat_columns)
        return train_pool, test_pool

    def concat_prepare_full_train_datasets(
        self,
        *,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ):
        """
        Prepare training dataset and optionally concatenate with test data for final model training,
        if your approach is to train on all data at once (like in the XGBoost base class).
        """
        logging.info(
            "Union train and test data for final model training based on TrainingConfig param "
            "'use_full_data_for_final_model'"
        )
        x_train = pd.concat([x_train, x_test]).reset_index(drop=True)
        y_train = pd.concat([y_train, y_test]).reset_index(drop=True)

        if self.cat_columns and self.conf_training.cat_encoding_via_ml_algorithm:
            # Convert columns to 'category' dtype if desired
            x_train[self.cat_columns] = x_train[self.cat_columns].astype("category")

        return x_train, y_train

    def get_early_stopping_callback(self):
        """
        In CatBoost, early stopping is handled by setting `od_type` (overfitting detector)
        and `od_wait` (similar to early_stopping_rounds). Example:

        If you want to replicate the 'EarlyStopping' from XGBoost, you can just set:

        model = CatBoostClassifier(
            iterations=...,
            od_type='Iter',
            od_wait=self.conf_training.early_stopping_rounds,
            ...
        )

        For consistency with your structure, we return None or a dictionary of parameters.
        """
        if self.conf_training.early_stopping_rounds:
            # We'll return something that can be used in `fit()` as 'params' or handle differently.
            return {
                "od_type": "Iter",
                "od_wait": self.conf_training.early_stopping_rounds,
            }
        else:
            return None

    def autotune(
        self,
        *,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ):
        raise NotImplementedError("Method autotune has not been defined for CatBoost.")

    def fine_tune(
        self,
        *,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ):
        raise NotImplementedError("Method fine_tune has not been defined for CatBoost.")

    def create_fine_tune_search_space(self) -> Dict[str, np.array]:
        l2_leaf_reg = self.conf_params_catboost.params.get("l2_leaf_reg")
        learning_rate = self.conf_params_catboost.params.get("learning_rate")

        if isinstance(l2_leaf_reg, (float, int)) and isinstance(
            learning_rate, (float, int)
        ):
            l2_leaf_reg = float(l2_leaf_reg)
            learning_rate = float(learning_rate)

            return {
                "l2_leaf_reg": np.linspace(
                    l2_leaf_reg * 0.9,
                    l2_leaf_reg * 1.1,
                    self.conf_training.gridsearch_nb_parameters_per_grid,
                    dtype=float,
                ),
                "learning_rate": np.linspace(
                    learning_rate * 0.9,
                    learning_rate * 1.1,
                    self.conf_training.gridsearch_nb_parameters_per_grid,
                    dtype=float,
                ),
            }
        else:
            raise ValueError("Some parameters are not floats or not found in params.")

    def _get_param_space_fpr_grid_search(self, trial: optuna.trial) -> Dict[str, Any]:
        """
        Similar to XGBoost method for an Optuna-based grid or random search.
        For CatBoost, adjust to whichever parameters you want to tweak.
        """
        l2_leaf_reg = self.conf_params_catboost.params.get("l2_leaf_reg")
        learning_rate = self.conf_params_catboost.params.get("learning_rate")
        tuned_params = deepcopy(self.conf_params_catboost.params)

        if isinstance(l2_leaf_reg, (float, int)) and isinstance(
            learning_rate, (float, int)
        ):
            l2_leaf_reg = float(l2_leaf_reg)
            learning_rate = float(learning_rate)

            l2_leaf_reg_space = trial.suggest_float(
                "l2_leaf_reg",
                l2_leaf_reg * 0.9,
                l2_leaf_reg * 1.1,
                log=False,
            )
            learning_rate_space = trial.suggest_float(
                "learning_rate",
                learning_rate * 0.9,
                learning_rate * 1.1,
                log=False,
            )

            tuned_params["l2_leaf_reg"] = l2_leaf_reg_space
            tuned_params["learning_rate"] = learning_rate_space

            return tuned_params
        else:
            raise ValueError("Some parameters are not floats or not found in params.")

    def _optimize_and_plot_grid_search_study(
        self, objective: Callable, search_space: Dict[str, np.array]
    ) -> None:
        """
        Similar to the XGBoost method.
        We create an Optuna study with a GridSampler (or any other sampler),
        run it, and track the best score. We then update self.conf_params_catboost
        accordingly if improvements are found.
        """
        study = self._create_optuna_study(
            direction=self.conf_catboost.catboost_eval_metric_tune_direction,
            sampler=optuna.samplers.GridSampler(search_space),
            study_name="catboost_grid_search",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=50),
        )
        study.optimize(
            objective,
            n_trials=self.conf_training.gridsearch_nb_parameters_per_grid
            ** len(search_space.keys()),
            timeout=self.conf_training.gridsearch_tuning_max_runtime_secs,
            gc_after_trial=True,
            show_progress_bar=True,
        )

        # Optionally visualize
        if self.conf_training.plot_hyperparameter_tuning_overview:
            try:
                fig = optuna.visualization.plot_optimization_history(study)
                fig.show()
                fig = optuna.visualization.plot_param_importances(study)
                fig.show()
            except (ZeroDivisionError, RuntimeError, ValueError):
                pass

        best_score_cv = self.best_score

        # Since direction could be 'minimize' or 'maximize', compare accordingly
        if (
            self.conf_catboost.catboost_eval_metric_tune_direction == "minimize"
            and study.best_value < self.best_score
        ) or (
            self.conf_catboost.catboost_eval_metric_tune_direction == "maximize"
            and study.best_value > self.best_score
        ):
            self.best_score = study.best_value
            catboost_grid_best_param = study.best_trial.params
            self.conf_params_catboost.params["l2_leaf_reg"] = catboost_grid_best_param[
                "l2_leaf_reg"
            ]
            self.conf_params_catboost.params["learning_rate"] = (
                catboost_grid_best_param["learning_rate"]
            )
            logging.info(
                f"Grid search improved eval metric from {best_score_cv} to {self.best_score}."
            )
            logging.info(f"Best params: {self.conf_params_catboost.params}")
            print(f"Best params: {self.conf_params_catboost.params}")
        else:
            logging.info(
                f"Grid search could not improve eval metric of {best_score_cv}. "
                f"Best score reached was {study.best_value}"
            )

    def orchestrate_hyperparameter_tuning(
        self,
        *,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ):
        """
        Mirrors the XGBoost orchestrate_hyperparameter_tuning approach.
        """
        if not self.conf_training.show_detailed_tuning_logs:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        if self.conf_training.autotune_model:
            self.autotune(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
            )
            print("Finished hyperparameter tuning")

        if self.conf_training.enable_grid_search_fine_tuning:
            self.fine_tune(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
            )
            print("Finished Grid search fine tuning")

    def _create_optuna_study(
        self,
        direction: str,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        study_name: str = "catboost_hyperparameter_tuning",
        pruner: Optional[optuna.pruners.BasePruner] = None,
    ) -> optuna.Study:
        """
        Create an Optuna study with optional database backend support for CatBoost.

        :param direction: Direction to optimize ('minimize' or 'maximize')
        :param sampler: Optuna sampler to use
        :param study_name: Name of the study
        :param pruner: Optuna pruner to use
        :return: Configured Optuna study
        """
        study_kwargs: Dict[str, Any] = {
            "direction": direction,
            "study_name": study_name,
        }

        if sampler is not None:
            study_kwargs["sampler"] = sampler
        if pruner is not None:
            study_kwargs["pruner"] = pruner

        # Add database backend if configured
        if self.conf_training.optuna_db_backend_path is not None and isinstance(
            self.conf_training.optuna_db_backend_path, str
        ):
            storage_name = f"sqlite:///{self.conf_training.optuna_db_backend_path}"
            study_kwargs["storage"] = storage_name
            study_kwargs["load_if_exists"] = True

            # Save the sampler state for resumption if database backend is used
            if sampler is not None and hasattr(sampler, "seed"):
                sampler_path = self.conf_training.optuna_db_backend_path.replace(
                    ".db", "_sampler.pkl"
                )
                try:
                    with open(sampler_path, "wb") as fout:
                        pickle.dump(sampler, fout)
                    logging.info(f"Saved sampler state to {sampler_path}")
                except Exception as e:
                    logging.warning(f"Could not save sampler state: {e}")

        return optuna.create_study(**study_kwargs)
