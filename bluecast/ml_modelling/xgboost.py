"""Xgboost classification model.

This module contains a wrapper for the Xgboost classification model. It can be used to train and/or tune the model.
It also calculates class weights for imbalanced datasets. The weights may or may not be used deepending on the
hyperparameter tuning.
"""
from datetime import datetime
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import matthews_corrcoef
from sklearn.utils import class_weight

from bluecast.config.training_config import (
    TrainingConfig,
    XgboostFinalParamConfig,
    XgboostTuneParamsConfig,
)
from bluecast.general_utils.general_utils import check_gpu_support, logger
from bluecast.ml_modelling.base_classes import BaseClassMlModel


class XgboostModel(BaseClassMlModel):
    """Train and/or tune Xgboost classification model."""

    def __init__(
        self,
        class_problem: Literal["binary", "multiclass"],
        conf_training: Optional[TrainingConfig] = None,
        conf_xgboost: Optional[XgboostTuneParamsConfig] = None,
        conf_params_xgboost: Optional[XgboostFinalParamConfig] = None,
    ):
        self.model: Optional[xgb.XGBClassifier] = None
        self.class_problem = class_problem
        self.conf_training = conf_training
        self.conf_xgboost = conf_xgboost
        self.conf_params_xgboost = conf_params_xgboost

    def calculate_class_weights(self, y: pd.Series) -> Dict[str, float]:
        """Calculate class weights of target column."""
        logger(f"{datetime.utcnow()}: Start calculating target class weights.")
        classes_weights = class_weight.compute_sample_weight(
            class_weight="balanced", y=y
        )
        return classes_weights

    def check_load_confs(self):
        """Load multiple configs or load default configs instead."""
        logger(f"{datetime.utcnow()}: Start loading existing or default config files..")
        if not self.conf_training:
            self.conf_training = TrainingConfig()
            logger(f"{datetime.utcnow()}: Load default TrainingConfig.")

        if not self.conf_xgboost:
            self.conf_xgboost = XgboostTuneParamsConfig()
            logger(f"{datetime.utcnow()}: Load default XgboostTuneParamsConfig.")

        if not self.conf_params_xgboost:
            self.conf_params_xgboost = XgboostFinalParamConfig()
            logger(f"{datetime.utcnow()}: Load default XgboostFinalParamConfig.")

    def fit(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> xgb.Booster:
        """Train Xgboost model. Includes hyperparameter tuning on default."""
        logger(f"{datetime.utcnow()}: Start fitting Xgboost model.")
        self.check_load_confs()

        if not self.conf_params_xgboost or not self.conf_training:
            raise ValueError("conf_params_xgboost or conf_training is None")

        if self.conf_training.autotune_model:
            self.autotune(x_train, x_test, y_train, y_test)

        print("Finished hyperparameter tuning")

        if self.conf_params_xgboost.sample_weight:
            classes_weights = self.calculate_class_weights(y_train)
            d_train = xgb.DMatrix(x_train, label=y_train, weight=classes_weights)
        else:
            d_train = xgb.DMatrix(x_train, label=y_train)
        d_test = xgb.DMatrix(x_test, label=y_test)
        eval_set = [(d_train, "train"), (d_test, "test")]

        self.model = xgb.train(
            self.conf_params_xgboost.params,
            d_train,
            num_boost_round=self.conf_params_xgboost.params["steps"],
            early_stopping_rounds=self.conf_training.early_stopping_rounds,
            evals=eval_set,
        )
        print("Finished training")
        return self.model

    def autotune(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        """Tune hyperparameters.

        An alternative config can be provided to overwrite the hyperparameter search space.
        """
        logger(f"{datetime.utcnow()}: Start hyperparameter tuning of Xgboost model.")
        d_test = xgb.DMatrix(x_test, label=y_test)
        train_on = check_gpu_support()

        self.check_load_confs()

        if (
            not self.conf_params_xgboost
            or not self.conf_training
            or not self.conf_xgboost
        ):
            raise ValueError(
                "At least one of the configs is None, which is not allowed"
            )

        def objective(trial):
            param = {
                "objective": self.conf_xgboost.model_objective,
                "eval_metric": self.conf_xgboost.model_eval_metric,
                "verbose": self.conf_xgboost.model_verbosity,
                "tree_method": train_on,
                "num_class": y_train.nunique(),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    self.conf_xgboost.max_depth_min,
                    self.conf_xgboost.max_depth_max,
                ),
                "alpha": trial.suggest_float(
                    "alpha", self.conf_xgboost.alpha_min, self.conf_xgboost.alpha_max
                ),
                "lambda": trial.suggest_float(
                    "lambda", self.conf_xgboost.lambda_min, self.conf_xgboost.lambda_max
                ),
                "num_leaves": trial.suggest_int(
                    "num_leaves",
                    self.conf_xgboost.num_leaves_min,
                    self.conf_xgboost.num_leaves_max,
                ),
                "subsample": trial.suggest_float(
                    "subsample",
                    self.conf_xgboost.sub_sample_min,
                    self.conf_xgboost.sub_sample_max,
                ),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree",
                    self.conf_xgboost.col_sample_by_tree_min,
                    self.conf_xgboost.col_sample_by_tree_max,
                ),
                "colsample_bylevel": trial.suggest_float(
                    "colsample_bylevel",
                    self.conf_xgboost.col_sample_by_level_min,
                    self.conf_xgboost.col_sample_by_level_max,
                ),
                "colsample_bynode": trial.suggest_float(
                    "colsample_bynode",
                    self.conf_xgboost.col_sample_by_node_min,
                    self.conf_xgboost.col_sample_by_node_max,
                ),
                "min_child_samples": trial.suggest_int(
                    "min_child_samples",
                    self.conf_xgboost.min_child_samples_min,
                    self.conf_xgboost.min_child_samples_max,
                ),
                "eta": self.conf_xgboost.eta,
                "steps": trial.suggest_int(
                    "steps", self.conf_xgboost.steps_min, self.conf_xgboost.steps_max
                ),
                "num_parallel_tree": trial.suggest_int(
                    "num_parallel_tree",
                    self.conf_xgboost.num_parallel_tree_min,
                    self.conf_xgboost.num_parallel_tree_max,
                ),
            }
            sample_weight = trial.suggest_categorical("sample_weight", [True, False])
            if sample_weight:
                classes_weights = self.calculate_class_weights(y_train)
                d_train = xgb.DMatrix(x_train, label=y_train, weight=classes_weights)
            else:
                d_train = xgb.DMatrix(x_train, label=y_train)

            pruning_callback = optuna.integration.XGBoostPruningCallback(
                trial, "test-mlogloss"
            )

            if self.conf_training.hypertuning_cv_folds == 1:
                eval_set = [(d_train, "train"), (d_test, "test")]
                model = xgb.train(
                    param,
                    d_train,
                    num_boost_round=param["steps"],
                    early_stopping_rounds=self.conf_training.early_stopping_rounds,
                    evals=eval_set,
                    callbacks=[pruning_callback],
                    verbose_eval=self.conf_xgboost.model_verbosity,
                )
                preds = model.predict(d_test)
                pred_labels = np.asarray([np.argmax(line) for line in preds])
                matthew = matthews_corrcoef(y_test, pred_labels) * -1
                return matthew
            else:
                result = xgb.cv(
                    params=param,
                    dtrain=d_train,
                    num_boost_round=param["steps"],
                    early_stopping_rounds=self.conf_training.early_stopping_rounds,
                    nfold=self.conf_training.hypertuning_cv_folds,
                    as_pandas=True,
                    seed=self.conf_training.global_random_state,
                    callbacks=[pruning_callback],
                    shuffle=self.conf_training.shuffle_during_training,
                )

                return result["test-mlogloss-mean"].mean()

        algorithm = "xgboost"
        sampler = optuna.samplers.TPESampler(
            multivariate=True, seed=self.conf_training.global_random_state
        )
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            study_name=f"{algorithm} tuning",
        )

        study.optimize(
            objective,
            n_trials=self.conf_training.hyperparameter_tuning_rounds,
            timeout=self.conf_training.hyperparameter_tuning_max_runtime_secs,
            gc_after_trial=True,
            show_progress_bar=True,
        )
        try:
            fig = optuna.visualization.plot_optimization_history(study)
            fig.show()
            fig = optuna.visualization.plot_param_importances(study)
            fig.show()
        except (ZeroDivisionError, RuntimeError, ValueError):
            pass

        xgboost_best_param = study.best_trial.params
        self.conf_params_xgboost.params = {
            "objective": self.conf_xgboost.model_objective,  # OR  'binary:logistic' #the loss function being used
            "eval_metric": self.conf_xgboost.model_eval_metric,
            "verbose": self.conf_xgboost.model_verbosity,
            "tree_method": train_on,  # use GPU for training
            "num_class": y_train.nunique(),
            "max_depth": xgboost_best_param[
                "max_depth"
            ],  # maximum depth of the decision trees being trained
            "alpha": xgboost_best_param["alpha"],
            "lambda": xgboost_best_param["lambda"],
            "num_leaves": xgboost_best_param["num_leaves"],
            "subsample": xgboost_best_param["subsample"],
            "colsample_bytree": xgboost_best_param["colsample_bytree"],
            "colsample_bylevel": xgboost_best_param["colsample_bylevel"],
            "colsample_bynode": xgboost_best_param["colsample_bynode"],
            "min_child_samples": xgboost_best_param["min_child_samples"],
            "eta": self.conf_xgboost.eta,
            "steps": xgboost_best_param["steps"],
            "num_parallel_tree": xgboost_best_param["num_parallel_tree"],
        }
        print("Best params: ", self.conf_params_xgboost.params)
        self.conf_params_xgboost.sample_weight = xgboost_best_param["sample_weight"]

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict on unseen data."""
        logger(
            f"{datetime.utcnow()}: Start predicting on new data using Xgboost model."
        )
        print("++++++++++++++++++++++++++++")
        d_test = xgb.DMatrix(df)
        if not self.model:
            raise Exception("No trained model has been found.")

        if not self.conf_params_xgboost:
            raise Exception("No model configuration file has been found.")

        partial_probs = self.model.predict(d_test)
        if self.class_problem == "binary":
            predicted_probs = np.asarray([line[1] for line in partial_probs])
            predicted_classes = (
                predicted_probs > self.conf_params_xgboost.classification_threshold
            )
        else:
            predicted_probs = partial_probs
            predicted_classes = np.asarray([np.argmax(line) for line in partial_probs])
        print("Finished predicting")
        return predicted_probs, predicted_classes
