"""Xgboost classification model.

This module contains a wrapper for the Xgboost classification model. It can be used to train and/or tune the model.
It also calculates class weights for imbalanced datasets. The weights may or may not be used deepending on the
hyperparameter tuning.
"""
from copy import deepcopy
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
from bluecast.experimentation.tracking import ExperimentTracker
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
        experiment_tracker: Optional[ExperimentTracker] = None,
    ):
        self.model: Optional[xgb.XGBClassifier] = None
        self.class_problem = class_problem
        self.conf_training = conf_training
        self.conf_xgboost = conf_xgboost
        self.conf_params_xgboost = conf_params_xgboost
        self.experiment_tracker = experiment_tracker

    def calculate_class_weights(self, y: pd.Series) -> Dict[str, float]:
        """Calculate class weights of target column."""
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

        if (
            not self.conf_params_xgboost
            or not self.conf_training
            or not self.experiment_tracker
        ):
            raise ValueError(
                "conf_params_xgboost, conf_training or experiment_tracker is None"
            )

        if not self.conf_training.show_detailed_tuning_logs:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        if self.conf_training.autotune_model:
            self.autotune(x_train, x_test, y_train, y_test)
            print("Finished hyperparameter tuning")

        if self.conf_training.enable_grid_search_fine_tuning:
            self.fine_tune(x_train, x_test, y_train, y_test)
            print("Finished Grid search fine tuning")

        logger("Start final model training")
        if self.conf_training.use_full_data_for_final_model:
            logger(
                f"""{datetime.utcnow()}: Union train and test data for final model training based on TrainingConfig
             param 'use_full_data_for_final_model'"""
            )
            x_train = pd.concat([x_train, x_test])
            y_train = pd.concat([y_train, y_test])

        if self.conf_params_xgboost.sample_weight:
            classes_weights = self.calculate_class_weights(y_train)
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
        eval_set = [(d_train, "train"), (d_test, "test")]

        steps = self.conf_params_xgboost.params["steps"]
        del self.conf_params_xgboost.params["steps"]

        if self.conf_training.hypertuning_cv_folds == 1 and self.conf_xgboost:
            self.model = xgb.train(
                self.conf_params_xgboost.params,
                d_train,
                num_boost_round=steps,
                early_stopping_rounds=self.conf_training.early_stopping_rounds,
                evals=eval_set,
                verbose_eval=self.conf_xgboost.model_verbosity_during_final_training,
            )
        elif self.conf_xgboost:
            self.model = xgb.train(
                self.conf_params_xgboost.params,
                d_train,
                num_boost_round=steps,
                early_stopping_rounds=self.conf_training.early_stopping_rounds,
                evals=eval_set,
                verbose_eval=self.conf_xgboost.model_verbosity_during_final_training,
            )
        logger("Finished training")
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
        if (
            not self.conf_params_xgboost
            or not self.conf_training
            or not self.experiment_tracker
        ):
            raise ValueError(
                "conf_params_xgboost, conf_training or experiment_tracker is None"
            )

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
            or not self.experiment_tracker
        ):
            raise ValueError(
                "At least one of the configs or experiment_tracker is None, which is not allowed"
            )

        def objective(trial):
            param = {
                "objective": self.conf_xgboost.model_objective,
                "booster": self.conf_xgboost.booster,
                "eval_metric": self.conf_xgboost.model_eval_metric,
                "tree_method": train_on,
                "num_class": y_train.nunique(),
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
                "gamma": trial.suggest_float(
                    "gamma", self.conf_xgboost.lambda_min, self.conf_xgboost.lambda_max
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
            sample_weight = trial.suggest_categorical("sample_weight", [True, False])
            if sample_weight:
                classes_weights = self.calculate_class_weights(y_train)
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

            pruning_callback = optuna.integration.XGBoostPruningCallback(
                trial, "test-mlogloss"
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
                preds = model.predict(d_test)
                pred_labels = np.asarray([np.argmax(line) for line in preds])
                matthew = matthews_corrcoef(y_test, pred_labels) * -1

                # track results
                if len(self.experiment_tracker.experiment_id) == 0:
                    new_id = 0
                else:
                    new_id = self.experiment_tracker.experiment_id[-1] + 1
                self.experiment_tracker.add_results(
                    experiment_id=new_id,
                    score_category="simple_train_test_score",
                    training_config=self.conf_training,
                    model_parameters=param,
                    eval_scores=matthew,
                    metric_used="matthew_inverse",
                    metric_higher_is_better=False,
                )
                return matthew
            else:
                random_seed = trial.suggest_categorical(
                    "random_seed",
                    [self.conf_training.global_random_state + i for i in range(100)],
                )
                result = xgb.cv(
                    params=param,
                    dtrain=d_train,
                    num_boost_round=steps,
                    # early_stopping_rounds=self.conf_training.early_stopping_rounds,
                    nfold=self.conf_training.hypertuning_cv_folds,
                    as_pandas=True,
                    seed=random_seed,
                    callbacks=[pruning_callback],
                    shuffle=self.conf_training.shuffle_during_training,
                )

                adjusted_score = result["test-mlogloss-mean"].mean() + (
                    result["test-mlogloss-mean"].std() ** 0.7
                )

                # track results
                if len(self.experiment_tracker.experiment_id) == 0:
                    new_id = 0
                else:
                    new_id = self.experiment_tracker.experiment_id[-1] + 1
                self.experiment_tracker.add_results(
                    experiment_id=new_id,
                    score_category="cv_score",
                    training_config=self.conf_training,
                    model_parameters=param,
                    eval_scores=adjusted_score,
                    metric_used="adjusted ml logloss",
                    metric_higher_is_better=False,
                )

                return adjusted_score

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
        if self.conf_training.hypertuning_cv_folds > 1:
            self.conf_training.global_random_state = xgboost_best_param["random_seed"]

        self.conf_params_xgboost.params = {
            "objective": self.conf_xgboost.model_objective,  # OR  'binary:logistic' #the loss function being used
            "booster": self.conf_xgboost.booster,
            "eval_metric": self.conf_xgboost.model_eval_metric,
            "tree_method": train_on,  # use GPU for training
            "num_class": y_train.nunique(),
            "max_depth": xgboost_best_param[
                "max_depth"
            ],  # maximum depth of the decision trees being trained
            "alpha": xgboost_best_param["alpha"],
            "lambda": xgboost_best_param["lambda"],
            "gamma": xgboost_best_param["gamma"],
            "max_leaves": xgboost_best_param["max_leaves"],
            "subsample": xgboost_best_param["subsample"],
            "colsample_bytree": xgboost_best_param["colsample_bytree"],
            "colsample_bylevel": xgboost_best_param["colsample_bylevel"],
            "eta": xgboost_best_param["eta"],
            "steps": xgboost_best_param["steps"],
        }
        logger(f"Best params: {self.conf_params_xgboost.params}")
        self.conf_params_xgboost.sample_weight = xgboost_best_param["sample_weight"]

    def fine_tune(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        logger(f"{datetime.utcnow()}: Start grid search fine tuning of Xgboost model.")
        if (
            not self.conf_params_xgboost
            or not self.conf_training
            or not self.conf_xgboost
            or not self.experiment_tracker
        ):
            raise ValueError(
                "At least one of the configs or experiment_tracker is None, which is not allowed"
            )

        d_test = xgb.DMatrix(
            x_test,
            label=y_test,
            enable_categorical=self.conf_training.cat_encoding_via_ml_algorithm,
        )
        self.conf_training.global_random_state += (
            1000  # to have correct tracking information and different splits
        )

        def objective(trial):
            if self.conf_params_xgboost.sample_weight:
                classes_weights = self.calculate_class_weights(y_train)
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

            pruning_callback = optuna.integration.XGBoostPruningCallback(
                trial, "test-mlogloss"
            )
            # copy best params to not overwrite them
            tuned_params = deepcopy(self.conf_params_xgboost.params)
            alpha_space = trial.suggest_float(
                "alpha",
                self.conf_params_xgboost.params["alpha"] * 0.9,
                self.conf_params_xgboost.params["alpha"] * 1.1,
            )
            lambda_space = trial.suggest_float(
                "lambda",
                self.conf_params_xgboost.params["lambda"] * 0.9,
                self.conf_params_xgboost.params["lambda"] * 1.1,
            )
            gamma_space = trial.suggest_float(
                "gamma",
                self.conf_params_xgboost.params["gamma"] * 0.9,
                self.conf_params_xgboost.params["gamma"] * 1.1,
            )

            tuned_params["alpha"] = alpha_space
            tuned_params["lambda"] = lambda_space
            tuned_params["gamma"] = gamma_space

            steps = tuned_params["steps"]
            del tuned_params["steps"]

            if self.conf_training.hypertuning_cv_folds == 1:
                eval_set = [(d_train, "train"), (d_test, "test")]
                model = xgb.train(
                    tuned_params,
                    d_train,
                    num_boost_round=steps,
                    early_stopping_rounds=self.conf_training.early_stopping_rounds,
                    evals=eval_set,
                    callbacks=[pruning_callback],
                    verbose_eval=self.conf_xgboost.model_verbosity,
                )
                preds = model.predict(d_test)
                pred_labels = np.asarray([np.argmax(line) for line in preds])
                matthew = matthews_corrcoef(y_test, pred_labels) * -1

                # track results
                if len(self.experiment_tracker.experiment_id) == 0:
                    new_id = 0
                else:
                    new_id = self.experiment_tracker.experiment_id[-1] + 1
                self.experiment_tracker.add_results(
                    experiment_id=new_id,
                    score_category="simple_train_test_score",
                    training_config=self.conf_training,
                    model_parameters=tuned_params,
                    eval_scores=matthew,
                    metric_used="matthew_inverse",
                    metric_higher_is_better=False,
                )
                return matthew
            else:
                result = xgb.cv(
                    params=tuned_params,
                    dtrain=d_train,
                    num_boost_round=steps,
                    # early_stopping_rounds=self.conf_training.early_stopping_rounds,
                    nfold=self.conf_training.hypertuning_cv_folds,
                    as_pandas=True,
                    seed=self.conf_training.global_random_state,
                    callbacks=[pruning_callback],
                    shuffle=self.conf_training.shuffle_during_training,
                )

                adjusted_score = result["test-mlogloss-mean"].mean() + (
                    result["test-mlogloss-mean"].std() ** 0.7
                )

                # track results
                if len(self.experiment_tracker.experiment_id) == 0:
                    new_id = 0
                else:
                    new_id = self.experiment_tracker.experiment_id[-1] + 1
                self.experiment_tracker.add_results(
                    experiment_id=new_id,
                    score_category="cv_score",
                    training_config=self.conf_training,
                    model_parameters=tuned_params,
                    eval_scores=adjusted_score,
                    metric_used="adjusted ml logloss",
                    metric_higher_is_better=False,
                )

                return adjusted_score

        self.check_load_confs()
        if (
            isinstance(self.conf_params_xgboost.params["alpha"], float)
            and isinstance(self.conf_params_xgboost.params["lambda"], float)
            and isinstance(self.conf_params_xgboost.params["gamma"], float)
        ):
            search_space = {
                "alpha": np.linspace(
                    self.conf_params_xgboost.params["alpha"]
                    * 0.9,  # TODO: fix design flaw in config and get rid of nested dict
                    self.conf_params_xgboost.params["alpha"] * 1.1,
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
            }
        else:
            ValueError("Some parameters are not floats or strings")

        if (
            self.conf_training.autotune_model
            and self.conf_training.hypertuning_cv_folds == 1
        ):
            best_score_cv = self.experiment_tracker.get_best_score(
                target_metric="matthew_inverse"
            )
        elif (
            self.conf_training.autotune_model
            and self.conf_training.hypertuning_cv_folds > 1
        ):
            best_score_cv = self.experiment_tracker.get_best_score(
                target_metric="adjusted ml logloss"
            )
        else:
            best_score_cv = np.inf

        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.GridSampler(search_space)
        )
        study.optimize(
            objective,
            n_trials=self.conf_training.gridsearch_nb_parameters_per_grid
            ** len(search_space.keys()),
            timeout=self.conf_training.gridsearch_tuning_max_runtime_secs,
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

        if (
            self.conf_training.autotune_model
            and self.conf_training.hypertuning_cv_folds == 1
        ):
            best_score_cv_grid = self.experiment_tracker.get_best_score(
                target_metric="matthew_inverse"
            )
        elif (
            self.conf_training.autotune_model
            and self.conf_training.hypertuning_cv_folds > 1
        ):
            best_score_cv_grid = self.experiment_tracker.get_best_score(
                target_metric="adjusted ml logloss"
            )
        else:
            best_score_cv_grid = np.inf

        if best_score_cv_grid < best_score_cv or not self.conf_training.autotune_model:
            xgboost_grid_best_param = study.best_trial.params
            self.conf_params_xgboost.params["alpha"] = xgboost_grid_best_param["alpha"]
            self.conf_params_xgboost.params["lambda"] = xgboost_grid_best_param[
                "lambda"
            ]
            self.conf_params_xgboost.params["gamma"] = xgboost_grid_best_param["gamma"]
            logger(
                f"Grid search improved eval metric from {best_score_cv} to {best_score_cv_grid}."
            )
            logger(f"Best params: {self.conf_params_xgboost.params}")
        else:
            logger(
                f"Grid search could not improve eval metric of {best_score_cv}. Best score reached was {best_score_cv_grid}"
            )

        self.conf_training.global_random_state -= 1000  # back to original setting

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
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

        partial_probs = self.model.predict(d_test)
        if self.class_problem == "binary":
            predicted_probs = np.asarray([line[1] for line in partial_probs])
            predicted_classes = (
                predicted_probs > self.conf_params_xgboost.classification_threshold
            )
        else:
            predicted_probs = partial_probs
            predicted_classes = np.asarray([np.argmax(line) for line in partial_probs])
        logger("Finished predicting")
        return predicted_probs, predicted_classes
