"""Base classes for all ML models."""

import logging
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, List, Literal, Optional, Tuple, TypeVar, Union

import numpy as np
import pandas as pd
import xgboost as xgb

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

        if conf_training is None:
            logging.info("Load default TrainingConfig.")
            self.conf_training = TrainingConfig()
        else:
            logging.info("Load default TrainingConfig.")
            self.conf_training = conf_training

        if experiment_tracker is None:
            raise ValueError("Experiment tracker not found.")
        else:
            self.experiment_tracker = experiment_tracker

        self.custom_in_fold_preprocessor = custom_in_fold_preprocessor
        self.single_fold_eval_metric_func = single_fold_eval_metric_func
        self.random_generator = np.random.default_rng(
            self.conf_training.global_random_state
        )
        self.cat_columns = cat_columns
        self.best_score: float = np.inf

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
