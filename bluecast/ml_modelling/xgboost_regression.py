"""Xgboost classification model.

This module contains a wrapper for the Xgboost classification model. It can be used to train and/or tune the model.
It also calculates class weights for imbalanced datasets. The weights may or may not be used deepending on the
hyperparameter tuning.
"""

import logging
import warnings
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb

try:
    from sklearn.metrics import root_mean_squared_error as mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error

from sklearn.model_selection import KFold, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder

from bluecast.config.training_config import (
    TrainingConfig,
    XgboostRegressionFinalParamConfig,
    XgboostTuneParamsRegressionConfig,
)
from bluecast.evaluation.eval_metrics import RegressionEvalWrapper
from bluecast.experimentation.tracking import ExperimentTracker
from bluecast.ml_modelling.base_classes import BaseClassMlRegressionModel
from bluecast.ml_modelling.parameter_tuning_utils import (
    get_params_based_on_device,
    sample_data,
    update_hyperparam_space_after_nth_trial,
    update_params_based_on_tree_method,
    update_params_with_best_params,
)
from bluecast.preprocessing.custom import CustomPreprocessing

warnings.filterwarnings("ignore", "is_sparse is deprecated")


class XgboostModelRegression(BaseClassMlRegressionModel):
    """Train and/or tune Xgboost regression model."""

    def __init__(
        self,
        class_problem: Literal["regression"],
        conf_training: Optional[TrainingConfig] = None,
        conf_xgboost: Optional[XgboostTuneParamsRegressionConfig] = None,
        conf_params_xgboost: Optional[XgboostRegressionFinalParamConfig] = None,
        experiment_tracker: Optional[ExperimentTracker] = None,
        custom_in_fold_preprocessor: Optional[CustomPreprocessing] = None,
        cat_columns: Optional[List[Union[str, float, int]]] = None,
        single_fold_eval_metric_func: Optional[RegressionEvalWrapper] = None,
    ):
        self.model: Optional[xgb.XGBRegressor] = None
        self.class_problem = class_problem
        self.conf_training = conf_training
        self.conf_params_xgboost = conf_params_xgboost
        self.conf_xgboost = conf_xgboost

        self.experiment_tracker = experiment_tracker
        self.custom_in_fold_preprocessor = custom_in_fold_preprocessor
        if self.conf_training:
            self.random_generator = np.random.default_rng(
                self.conf_training.global_random_state
            )
        else:
            self.random_generator = np.random.default_rng(0)

        self.cat_columns = cat_columns
        self.single_fold_eval_metric_func = single_fold_eval_metric_func
        self.best_score: float = np.inf

        if not self.single_fold_eval_metric_func:
            self.single_fold_eval_metric_func = RegressionEvalWrapper(
                higher_is_better=False,
                metric_func=mean_squared_error,
                metric_name="Mean squared error",
                **{"squared": False},
            )

    def check_load_confs(self):
        """Load multiple configs or load default configs instead."""
        logging.info("Start loading existing or default config files..")
        if not self.conf_training:
            self.conf_training = TrainingConfig()
            logging.info("Load default TrainingConfig.")
        else:
            logging.info("Found provided TrainingConfig.")

        if not self.conf_xgboost:
            self.conf_xgboost = XgboostTuneParamsRegressionConfig()
            logging.info("Load default XgboostTuneParamsRegressionConfig.")
        else:
            logging.info("Found provided XgboostTuneParamsRegressionConfig.")

        if not self.conf_params_xgboost:
            self.conf_params_xgboost = XgboostRegressionFinalParamConfig()
            logging.info("Load default XgboostRegressionFinalParamConfig.")
        else:
            logging.info("Found provided XgboostRegressionFinalParamConfig.")

    def fit(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> xgb.Booster:
        """Train Xgboost model. Includes hyperparameter tuning on default."""
        logging.info("Start fitting Xgboost model.")
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

        logging.info("Start final model training")
        if self.custom_in_fold_preprocessor:
            x_train, y_train = self.custom_in_fold_preprocessor.fit_transform(
                x_train, y_train
            )
            x_test, y_test = self.custom_in_fold_preprocessor.transform(
                x_test, y_test, predicton_mode=False
            )

        if self.conf_training.use_full_data_for_final_model:
            logging.info(
                f"""{datetime.utcnow()}: Union train and test data for final model training based on TrainingConfig
             param 'use_full_data_for_final_model'"""
            )
            x_train = pd.concat([x_train, x_test]).reset_index(drop=True)
            y_train = pd.concat([y_train, y_test]).reset_index(drop=True)

            if self.cat_columns and self.conf_training.cat_encoding_via_ml_algorithm:
                x_train[self.cat_columns] = x_train[self.cat_columns].astype("category")

        d_train, d_test = self.create_d_matrices(x_train, y_train, x_test, y_test)
        eval_set = [(d_test, "test")]

        steps = self.conf_params_xgboost.params.pop("steps", 300)

        if self.conf_training.early_stopping_rounds and self.conf_xgboost:
            early_stop = xgb.callback.EarlyStopping(
                rounds=self.conf_training.early_stopping_rounds,
                metric_name=self.conf_xgboost.xgboost_eval_metric,
                data_name="test",
                save_best=self.conf_params_xgboost.params["booster"] != "gblinear",
            )
            callbacks = [early_stop]
        else:
            callbacks = None

        if self.conf_training.hypertuning_cv_folds == 1 and self.conf_xgboost:
            self.model = xgb.train(
                self.conf_params_xgboost.params,
                d_train,
                num_boost_round=steps,
                early_stopping_rounds=self.conf_training.early_stopping_rounds,
                evals=eval_set,
                verbose_eval=self.conf_xgboost.verbosity_during_final_model_training,
                callbacks=callbacks,
            )
        elif self.conf_xgboost:
            self.model = xgb.train(
                self.conf_params_xgboost.params,
                d_train,
                num_boost_round=steps,
                early_stopping_rounds=self.conf_training.early_stopping_rounds,
                evals=eval_set,
                verbose_eval=self.conf_xgboost.verbosity_during_final_model_training,
                callbacks=callbacks,
            )
        logging.info("Finished training")
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
        logging.info("Start hyperparameter tuning of Xgboost model.")
        if (
            not self.conf_params_xgboost
            or not self.conf_training
            or not self.experiment_tracker
        ):
            raise ValueError(
                "conf_params_xgboost, conf_training or experiment_tracker is None"
            )

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

        train_on = get_params_based_on_device(
            self.conf_training, self.conf_params_xgboost, self.conf_xgboost
        )

        x_train, x_test, y_train, y_test = sample_data(
            x_train, x_test, y_train, y_test, self.conf_training
        )

        def objective(trial):
            self.conf_xgboost = update_hyperparam_space_after_nth_trial(
                trial,
                self.conf_xgboost,
                self.conf_training.update_hyperparameter_search_space_after_nth_trial,
            )

            param = {
                "validate_parameters": False,
                "objective": self.conf_xgboost.xgboost_objective,
                "booster": self.conf_xgboost.booster,
                "eval_metric": self.conf_xgboost.xgboost_eval_metric,
                "eta": trial.suggest_float(
                    "eta",
                    self.conf_xgboost.eta_min,
                    self.conf_xgboost.eta_max,
                    log=True,
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    self.conf_xgboost.max_depth_min,
                    self.conf_xgboost.max_depth_max,
                    log=False,
                ),
                "alpha": trial.suggest_float(
                    "alpha",
                    self.conf_xgboost.alpha_min,
                    self.conf_xgboost.alpha_max,
                    log=True,
                ),
                "lambda": trial.suggest_float(
                    "lambda",
                    self.conf_xgboost.lambda_min,
                    self.conf_xgboost.lambda_max,
                    log=True,
                ),
                "gamma": trial.suggest_float(
                    "gamma",
                    self.conf_xgboost.gamma_min,
                    self.conf_xgboost.gamma_max,
                    log=True,
                ),
                "min_child_weight": trial.suggest_float(
                    "min_child_weight",
                    self.conf_xgboost.min_child_weight_min,
                    self.conf_xgboost.min_child_weight_max,
                    log=True,
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
                    "steps",
                    self.conf_xgboost.steps_min,
                    self.conf_xgboost.steps_max,
                    log=True,
                ),
            }
            params = {**param, **train_on}

            params = update_params_based_on_tree_method(
                params, trial, self.conf_xgboost
            )

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

            pruning_callback = optuna.integration.XGBoostPruningCallback(
                trial, f"test-{self.conf_xgboost.xgboost_eval_metric}"
            )

            steps = params.pop("steps", 300)

            if self.conf_training.hypertuning_cv_folds == 1:
                if self.conf_training.hypertuning_cv_folds == 1:
                    try:
                        return self.train_single_fold_model(
                            d_train, d_test, y_test, params, steps, pruning_callback
                        )
                    except Exception as e:
                        logging.error(f"Error during training: {e}. Pruning trial")
                        trial.should_prune()
            elif (
                self.conf_training.hypertuning_cv_folds > 1
                and self.conf_training.precise_cv_tuning
            ):

                return self._fine_tune_precise(params, x_train, y_train, x_test, y_test)
            else:
                # make regression cv startegy stratified
                le = LabelEncoder()
                y_binned = le.fit_transform(pd.qcut(y_train, 10, duplicates="drop"))
                skf = RepeatedStratifiedKFold(
                    n_splits=self.conf_training.hypertuning_cv_folds,
                    n_repeats=self.conf_training.hypertuning_cv_repeats,
                    random_state=self.conf_training.global_random_state,
                )
                folds = []
                for train_index, test_index in skf.split(x_train, y_binned):
                    folds.append((train_index.tolist(), test_index.tolist()))

                result = xgb.cv(
                    params=params,
                    dtrain=d_train,
                    num_boost_round=steps,
                    # early_stopping_rounds=self.conf_training.early_stopping_rounds, # not recommended as per docs: https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html
                    nfold=self.conf_training.hypertuning_cv_folds,
                    as_pandas=True,
                    seed=self.conf_training.global_random_state,
                    callbacks=[pruning_callback],
                    shuffle=self.conf_training.shuffle_during_training,
                    folds=folds,
                )

                adjusted_score = result[
                    f"test-{self.conf_xgboost.xgboost_eval_metric}-mean"
                ].values[-1]

                # track results
                if len(self.experiment_tracker.experiment_id) == 0:
                    new_id = 0
                else:
                    new_id = self.experiment_tracker.experiment_id[-1] + 1
                self.experiment_tracker.add_results(
                    experiment_id=new_id,
                    score_category="cv_score",
                    training_config=self.conf_training,
                    model_parameters=params,
                    eval_scores=adjusted_score,
                    metric_used="adjusted rmse",
                    metric_higher_is_better=False,
                )

                return adjusted_score

        for rst in range(self.conf_training.autotune_n_random_seeds):
            logging.info(
                f"Hyperparameter tuning using random seed {self.conf_training.global_random_state + rst}"
            )

            sampler = optuna.samplers.TPESampler(
                multivariate=True,
                seed=self.conf_training.global_random_state,
                n_startup_trials=self.conf_training.optuna_sampler_n_startup_trials,
                warn_independent_sampling=False,
            )
            study = optuna.create_study(
                direction=self.conf_xgboost.xgboost_eval_metric_tune_direction,
                sampler=sampler,
                study_name="xgboost regression tuning",
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=20,
                    n_warmup_steps=20,
                ),
            )

            study.optimize(
                objective,
                n_trials=self.conf_training.hyperparameter_tuning_rounds,
                timeout=self.conf_training.hyperparameter_tuning_max_runtime_secs,
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

            if study.best_value < self.best_score:
                self.best_score = study.best_value
                logging.info(
                    f"New best score: {study.best_value} from random seed  {self.conf_training.global_random_state + rst}"
                )
                xgboost_best_param = study.best_trial.params

                self.conf_params_xgboost.params = {
                    "validate_parameters": False,
                    "objective": self.conf_xgboost.xgboost_objective,  # OR  'binary:logistic' #the loss function being used
                    "booster": self.conf_xgboost.booster,
                    "eval_metric": self.conf_xgboost.xgboost_eval_metric,
                    "max_depth": xgboost_best_param[
                        "max_depth"
                    ],  # maximum depth of the decision trees being trained
                    "alpha": xgboost_best_param["alpha"],
                    "lambda": xgboost_best_param["lambda"],
                    "gamma": xgboost_best_param["gamma"],
                    "min_child_weight": xgboost_best_param["min_child_weight"],
                    "subsample": xgboost_best_param["subsample"],
                    "colsample_bytree": xgboost_best_param["colsample_bytree"],
                    "colsample_bylevel": xgboost_best_param["colsample_bylevel"],
                    "eta": xgboost_best_param["eta"],
                    "steps": xgboost_best_param["steps"],
                }
                self.conf_params_xgboost.params = {
                    **self.conf_params_xgboost.params,
                    **train_on,
                }
                self.conf_params_xgboost.params = update_params_with_best_params(
                    self.conf_params_xgboost.params, xgboost_best_param
                )
                logging.info(f"Best params: {self.conf_params_xgboost.params}")
                print(f"Best params: {self.conf_params_xgboost.params}")

    def create_d_matrices(self, x_train, y_train, x_test, y_test):
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

    def train_single_fold_model(
        self, d_train, d_test, y_test, param, steps, pruning_callback
    ):
        eval_set = [(d_test, "test")]
        if self.conf_training.early_stopping_rounds and self.conf_xgboost:
            early_stop = xgb.callback.EarlyStopping(
                rounds=self.conf_training.early_stopping_rounds,
                metric_name=self.conf_xgboost.xgboost_eval_metric,
                data_name="test",
                save_best=param["booster"] != "gblinear",
            )
            callbacks = [early_stop]
        else:
            callbacks = None

        model = xgb.train(
            param,
            d_train,
            num_boost_round=steps,
            evals=eval_set,
            callbacks=callbacks,
            verbose_eval=self.conf_xgboost.verbosity_during_hyperparameter_tuning,
        )
        preds = model.predict(d_test)
        mse = self.single_fold_eval_metric_func.regression_eval_func_wrapper(
            y_test, preds
        )

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
            eval_scores=mse,
            metric_used="root_mean_squared_error",
            metric_higher_is_better=False,
        )
        return mse

    def _fine_tune_precise(
        self,
        tuned_params: Dict[str, Any],
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ):
        steps = tuned_params.pop("steps", 300)

        if not self.conf_training:
            self.conf_training = TrainingConfig()
            logging.info(
                "Could not find Training config. Falling back to default values"
            )

        stratifier = KFold(
            n_splits=self.conf_training.hypertuning_cv_folds,
            shuffle=True,
            random_state=self.conf_training.global_random_state,
        )

        fold_losses = []
        for _fn, (trn_idx, val_idx) in enumerate(
            stratifier.split(x_train, y_train.astype(int))
        ):
            X_train_fold, X_val_fold = (
                x_train.iloc[trn_idx],
                x_train.iloc[val_idx],
            )
            y_train_fold, y_val_fold = (
                y_train.iloc[trn_idx],
                y_train.iloc[val_idx],
            )
            if self.custom_in_fold_preprocessor:
                (
                    X_train_fold,
                    y_train_fold,
                ) = self.custom_in_fold_preprocessor.fit_transform(
                    X_train_fold, y_train_fold
                )
                (
                    X_test_fold,
                    y_test_fold,
                ) = self.custom_in_fold_preprocessor.transform(x_test, y_test)
                (
                    X_val_fold,
                    y_val_fold,
                ) = self.custom_in_fold_preprocessor.transform(
                    X_val_fold, y_val_fold, predicton_mode=False
                )
            else:
                X_test_fold, y_test_fold = x_test, y_test

            d_test = xgb.DMatrix(
                X_val_fold,
                label=y_val_fold,
                enable_categorical=self.conf_training.cat_encoding_via_ml_algorithm,
            )

            if not self.conf_params_xgboost:
                self.conf_params_xgboost = XgboostRegressionFinalParamConfig()
                logging.info(
                    "Could not find XgboostRegressionFinalParamConfig. Falling back to default settings."
                )

            d_train = xgb.DMatrix(
                X_train_fold,
                label=y_train_fold,
                enable_categorical=self.conf_training.cat_encoding_via_ml_algorithm,
            )
            eval_set = [(d_test, "test")]

            if not self.conf_xgboost:
                self.conf_xgboost = XgboostTuneParamsRegressionConfig()
                logging.info(
                    "Could not find XgboostTuneParamsRegressionConfig. Falling back to defaults."
                )

            model = xgb.train(
                tuned_params,
                d_train,
                num_boost_round=steps,
                evals=eval_set,
                verbose_eval=self.conf_xgboost.verbosity_during_hyperparameter_tuning,
            )
            d_eval = xgb.DMatrix(
                X_test_fold,
                label=y_test_fold,
                enable_categorical=self.conf_training.cat_encoding_via_ml_algorithm,
            )
            preds = model.predict(d_eval)

            if self.single_fold_eval_metric_func:
                loss = self.single_fold_eval_metric_func.regression_eval_func_wrapper(
                    y_test_fold, preds
                )
            else:
                raise ValueError("No single_fold_eval_metric_func could be found")

            fold_losses.append(loss)

        mse_mean = np.mean(np.asarray(fold_losses))

        if self.experiment_tracker and self.conf_training:
            # track results
            if len(self.experiment_tracker.experiment_id) == 0:
                new_id = 0
            else:
                new_id = self.experiment_tracker.experiment_id[-1] + 1
            self.experiment_tracker.add_results(
                experiment_id=new_id,
                score_category="oof_score",
                training_config=self.conf_training,
                model_parameters=tuned_params,
                eval_scores=mse_mean,
                metric_used="root_mean_squared_error",
                metric_higher_is_better=False,
            )
        return mse_mean

    def fine_tune(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        logging.info("Start grid search fine tuning of Xgboost model.")
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
            d_train, d_test = self.create_d_matrices(x_train, y_train, x_test, y_test)

            pruning_callback = optuna.integration.XGBoostPruningCallback(
                trial, f"test-{self.conf_xgboost.xgboost_eval_metric}"
            )
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

            steps = tuned_params.pop("steps", 300)

            if self.conf_training.hypertuning_cv_folds == 1:
                try:
                    return self.train_single_fold_model(
                        d_train, d_test, y_test, tuned_params, steps, pruning_callback
                    )
                except Exception as e:
                    logging.error(f"Error during training: {e}. Pruning trial")
                    trial.should_prune()
            elif (
                self.conf_training.hypertuning_cv_folds > 1
                and self.conf_training.precise_cv_tuning
            ):
                return self._fine_tune_precise(
                    tuned_params,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                )
            else:
                # make regression cv startegy stratified
                le = LabelEncoder()
                y_binned = le.fit_transform(pd.qcut(y_train, 10, duplicates="drop"))
                skf = RepeatedStratifiedKFold(
                    n_splits=self.conf_training.hypertuning_cv_folds,
                    n_repeats=self.conf_training.hypertuning_cv_repeats,
                    random_state=self.conf_training.global_random_state,
                )
                folds = []
                for train_index, test_index in skf.split(x_train, y_binned):
                    folds.append((train_index.tolist(), test_index.tolist()))

                result = xgb.cv(
                    params=tuned_params,
                    dtrain=d_train,
                    num_boost_round=steps,
                    # early_stopping_rounds=self.conf_training.early_stopping_rounds,  # not recommended as per docs: https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html
                    nfold=self.conf_training.hypertuning_cv_folds,
                    as_pandas=True,
                    seed=self.conf_training.global_random_state,
                    callbacks=[pruning_callback],
                    shuffle=self.conf_training.shuffle_during_training,
                    folds=folds,
                )

                adjusted_score = result[
                    f"test-{self.conf_xgboost.xgboost_eval_metric}-mean"
                ].values[-1]

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
                    metric_used="root_mean_squared_error",
                    metric_higher_is_better=False,
                )

                return adjusted_score

        self.check_load_confs()
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
        else:
            ValueError("Some parameters are not floats or strings")

        study = optuna.create_study(
            direction=self.conf_xgboost.xgboost_eval_metric_tune_direction,
            sampler=optuna.samplers.GridSampler(search_space),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=50),
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

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict on unseen data."""
        logging.info("Start predicting on new data using Xgboost model.")
        if not self.conf_xgboost or not self.conf_training:
            raise ValueError("conf_params_xgboost or conf_training is None")

        if self.custom_in_fold_preprocessor:
            df, _ = self.custom_in_fold_preprocessor.transform(
                df, None, predicton_mode=True
            )

        d_test = xgb.DMatrix(
            df,
            enable_categorical=self.conf_training.cat_encoding_via_ml_algorithm,
        )

        if not self.model:
            raise Exception("No trained model has been found.")

        if not self.conf_params_xgboost:
            raise Exception("No model configuration file has been found.")

        preds = self.model.predict(d_test)
        logging.info("Finished predicting")
        return preds
