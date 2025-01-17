"""Base classes for all ML models."""

import logging
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, List, Literal, Optional, Tuple, TypeVar, Union

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.utils import class_weight

from bluecast.config.training_config import (
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
