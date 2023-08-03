"""Xgboost regression model.

This module contains a wrapper for the Xgboost regression model. It can be used to train and/or tune the model.
"""
from datetime import datetime
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.utils import class_weight

from bluecast.config.training_config import (
    TrainingConfig,
    XgboostFinalParamConfig,
    XgboostTuneParamsConfig,
)
from bluecast.general_utils.general_utils import check_gpu_support, logger
from bluecast.ml_modelling.base_classes import BaseClassMlModel


class XgboostRegressorModel(BaseClassMlModel):
    """Train and/or tune Xgboost classification model."""

    def __init__(
        self,
        class_problem: Literal["regression"],
        conf_training: Optional[TrainingConfig] = None,
        conf_xgboost: Optional[XgboostTuneParamsConfig] = None,
        conf_params_xgboost: Optional[XgboostFinalParamConfig] = None,
    ):
        self.model: Optional[xgb.XGBClassifier] = None
        self.class_problem = class_problem
        self.conf_training = conf_training
        self.conf_xgboost = conf_xgboost
        self.conf_params_xgboost = conf_params_xgboost

    def check_load_confs(self):
        """Load multiple configs or load default configs instead."""
        logger(f"{datetime.utcnow()}: Start loading existing or default config files..")
        if not self.conf_training:
            self.conf_training = TrainingConfig()
            logger(f"{datetime.utcnow()}: Load default TrainingConfig.")
        else:
            logger(f"{datetime.utcnow()}: Found provided TrainingConfig.")

        if not self.conf_xgboost:
            self.conf_xgboost = XgboostTuneParamsConfig()
            logger(f"{datetime.utcnow()}: Load default XgboostTuneParamsConfig.")
        else:
            logger(f"{datetime.utcnow()}: Found provided XgboostTuneParamsConfig.")

        if not self.conf_params_xgboost:
            self.conf_params_xgboost = XgboostFinalParamConfig()
            logger(f"{datetime.utcnow()}: Load default XgboostFinalParamConfig.")
        else:
            logger(f"{datetime.utcnow()}: Found provided XgboostFinalParamConfig.")

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

        if not self.conf_training.show_detailed_tuning_logs:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        if self.conf_training.autotune_model:
            self.autotune(x_train, x_test, y_train, y_test)

        print("Finished hyperparameter tuning")

        print("Start training")
        if self.conf_training.use_full_data_for_final_model:
            logger(
                f"""{datetime.utcnow()}: Union train and test data for final model training based on TrainingConfig
             param 'use_full_data_for_final_model'"""
            )
            x_train = pd.concat([x_train, x_test])
            y_train = pd.concat([y_train, y_test])

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
        eval_set = [(d_train, "train"), (d_test, "test")]

        steps = self.conf_params_xgboost.params["steps"]
        del self.conf_params_xgboost.params["steps"]

        if self.conf_training.hypertuning_cv_folds == 1:
            self.model = xgb.train(
                self.conf_params_xgboost.params,
                d_train,
                num_boost_round=steps,
                early_stopping_rounds=self.conf_training.early_stopping_rounds,
                evals=eval_set,
            )
        else:
            self.model = xgb.train(
                self.conf_params_xgboost.params,
                d_train,
                num_boost_round=steps,
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
        if not self.conf_params_xgboost or not self.conf_training:
            raise ValueError("conf_params_xgboost or conf_training is None")

        d_test = xgb.DMatrix(
            x_test,
            label=y_test,
            enable_categorical=self.conf_training.cat_encoding_via_ml_algorithm,
        )
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
                "booster": self.conf_xgboost.booster,
                "eval_metric": self.conf_xgboost.model_eval_metric,
                "tree_method": train_on,
                "eta": trial.suggest_float(
                    "eta", self.conf_xgboost.eta_min, self.conf_xgboost.eta_max
                ),
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
                "min_child_weight": trial.suggest_float(
                    "min_child_weight",
                    self.conf_xgboost.min_child_weight_min,
                    self.conf_xgboost.min_child_weight_max,
                ),
                "max_leaves": trial.suggest_int(
                    "max_leaves",
                    self.conf_xgboost.max_leaves_min,
                    self.conf_xgboost.max_leaves_max,
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
                "steps": trial.suggest_int(
                    "steps", self.conf_xgboost.steps_min, self.conf_xgboost.steps_max
                ),
            }

            d_train = xgb.DMatrix(
                x_train,
                label=y_train,
                enable_categorical=self.conf_training.cat_encoding_via_ml_algorithm,
            )

            pruning_callback = optuna.integration.XGBoostPruningCallback(
                trial, "test-mae"
            )

            steps = param["steps"]
            del param["steps"]

            if self.conf_training.hypertuning_cv_folds == 1:
                eval_set = [(d_train, "train"), (d_test, "test")]
                model = xgb.train(
                    param,
                    d_train,
                    num_boost_round=steps,
                    early_stopping_rounds=self.conf_training.early_stopping_rounds,
                    evals=eval_set,
                    callbacks=[pruning_callback],
                    verbose_eval=self.conf_xgboost.model_verbosity,
                )
                preds = model.predict(D_test)
                mae = mean_absolute_error(Y_test, preds)
                return mae
            else:
                result = xgb.cv(
                    params=param,
                    dtrain=d_train,
                    num_boost_round=steps,
                    # early_stopping_rounds=self.conf_training.early_stopping_rounds,
                    nfold=self.conf_training.hypertuning_cv_folds,
                    as_pandas=True,
                    seed=self.conf_training.global_random_state,
                    callbacks=[pruning_callback],
                    shuffle=self.conf_training.shuffle_during_training,
                )

                return result["test-mae-mean"].mean()

        algorithm = "xgboost"
        sampler = optuna.samplers.TPESampler(
            multivariate=True,
            seed=self.conf_training.global_random_state,
            n_startup_trials=self.conf_training.optuna_sampler_n_startup_trials,
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
            "booster": self.conf_xgboost.booster,
            "eval_metric": self.conf_xgboost.model_eval_metric,
            "verbose": self.conf_xgboost.model_verbosity,
            "tree_method": train_on,  # use GPU for training
            "max_depth": xgboost_best_param[
                "max_depth"
            ],  # maximum depth of the decision trees being trained
            "alpha": xgboost_best_param["alpha"],
            "lambda": xgboost_best_param["lambda"],
            "max_leaves": xgboost_best_param["max_leaves"],
            "subsample": xgboost_best_param["subsample"],
            "colsample_bytree": xgboost_best_param["colsample_bytree"],
            "colsample_bylevel": xgboost_best_param["colsample_bylevel"],
            "min_child_weight": xgboost_best_param["min_child_weight"],
            "eta": xgboost_best_param["eta"],
            "steps": xgboost_best_param["steps"],
        }
        print("Best params: ", self.conf_params_xgboost.params)
        self.conf_params_xgboost.sample_weight = xgboost_best_param["sample_weight"]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict on unseen data."""
        logger(
            f"{datetime.utcnow()}: Start predicting on new data using Xgboost model."
        )
        if not self.conf_xgboost or not self.conf_training:
            raise ValueError("conf_params_xgboost or conf_training is None")

        d_test = xgb.DMatrix(
            df,
            enable_categorical=self.conf_training.cat_encoding_via_ml_algorithm,
        )

        if not self.model:
            raise Exception("No trained model has been found.")

        if not self.conf_params_xgboost:
            raise Exception("No model configuration file has been found.")

        preds = self.model.predict(d_test)
        print("Finished predicting")
        return preds
