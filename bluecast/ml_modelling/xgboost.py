"""Xgboost classification model.

This module contains a wrapper for the Xgboost classification model. It can be used to train and/or tune the model.
It also calculates class weights for imbalanced datasets. The weights may or may not be used deepending on the
hyperparameter tuning.
"""

from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight

from bluecast.config.training_config import (
    TrainingConfig,
    XgboostFinalParamConfig,
    XgboostTuneParamsConfig,
)
from bluecast.experimentation.tracking import ExperimentTracker
from bluecast.general_utils.general_utils import check_gpu_support, log_sampling, logger
from bluecast.ml_modelling.base_classes import BaseClassMlModel
from bluecast.preprocessing.custom import CustomPreprocessing


class XgboostModel(BaseClassMlModel):
    """Train and/or tune Xgboost classification model."""

    def __init__(
        self,
        class_problem: Literal["binary", "multiclass"],
        conf_training: Optional[TrainingConfig] = None,
        conf_xgboost: Optional[XgboostTuneParamsConfig] = None,
        conf_params_xgboost: Optional[XgboostFinalParamConfig] = None,
        experiment_tracker: Optional[ExperimentTracker] = None,
        custom_in_fold_preprocessor: Optional[CustomPreprocessing] = None,
    ):
        self.model: Optional[xgb.XGBClassifier] = None
        self.class_problem = class_problem
        self.conf_training = conf_training
        self.conf_xgboost = conf_xgboost
        self.conf_params_xgboost = conf_params_xgboost
        self.experiment_tracker = experiment_tracker
        self.custom_in_fold_preprocessor = custom_in_fold_preprocessor
        if self.conf_training:
            self.random_generator = np.random.default_rng(
                self.conf_training.global_random_state
            )
        else:
            self.random_generator = np.random.default_rng(0)

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
        if self.custom_in_fold_preprocessor:
            x_train, y_train = self.custom_in_fold_preprocessor.fit_transform(
                x_train, y_train
            )
            x_test, y_test = self.custom_in_fold_preprocessor.transform(
                x_test, y_test, predicton_mode=False
            )

        if self.conf_training.use_full_data_for_final_model:
            logger(
                f"""{datetime.utcnow()}: Union train and test data for final model training based on TrainingConfig
             param 'use_full_data_for_final_model'"""
            )
            x_train = pd.concat([x_train, x_test])
            y_train = pd.concat([y_train, y_test])

        d_train, d_test = self.create_d_matrices(x_train, y_train, x_test, y_test)
        eval_set = [(d_train, "train"), (d_test, "test")]

        steps = self.conf_params_xgboost.params.pop("steps", 300)

        if self.conf_training.hypertuning_cv_folds == 1 and self.conf_xgboost:
            self.model = xgb.train(
                self.conf_params_xgboost.params,
                d_train,
                num_boost_round=steps,
                early_stopping_rounds=self.conf_training.early_stopping_rounds,
                evals=eval_set,
                verbose_eval=self.conf_xgboost.verbosity_during_final_model_training,
            )
        elif self.conf_xgboost:
            self.model = xgb.train(
                self.conf_params_xgboost.params,
                d_train,
                num_boost_round=steps,
                early_stopping_rounds=self.conf_training.early_stopping_rounds,
                evals=eval_set,
                verbose_eval=self.conf_xgboost.verbosity_during_final_model_training,
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

        if self.conf_training.sample_data_during_tuning:
            nb_samples_train = log_sampling(
                len(x_train.index),
                alpha=self.conf_training.sample_data_during_tuning_alpha,
            )
            nb_samples_test = log_sampling(
                len(x_test.index),
                alpha=self.conf_training.sample_data_during_tuning_alpha,
            )

            x_train = x_train.sample(
                nb_samples_train, random_state=self.conf_training.global_random_state
            )
            y_train = y_train.loc[x_train.index]
            x_test = x_test.sample(
                nb_samples_test, random_state=self.conf_training.global_random_state
            )
            y_test = y_test.loc[x_test.index]

        def objective(trial):
            param = {
                "objective": self.conf_xgboost.xgboost_objective,
                "booster": self.conf_xgboost.booster,
                "eval_metric": self.conf_xgboost.xgboost_eval_metric,
                "num_class": y_train.nunique(),
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
                    log=True,
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
                "min_child_weight": trial.suggest_int(
                    "min_child_weight",
                    self.conf_xgboost.min_child_weight_min,
                    self.conf_xgboost.min_child_weight_max,
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
            param = {**param, **train_on}
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

            d_test = xgb.DMatrix(
                x_test,
                label=y_test,
                enable_categorical=self.conf_training.cat_encoding_via_ml_algorithm,
            )

            pruning_callback = optuna.integration.XGBoostPruningCallback(
                trial, "test-mlogloss"
            )

            steps = param.pop("steps", 300)

            if self.conf_training.hypertuning_cv_folds == 1:
                return self.train_single_fold_model(
                    d_train, d_test, y_test, param, steps, pruning_callback
                )
            elif (
                self.conf_training.hypertuning_cv_folds > 1
                and self.conf_training.precise_cv_tuning
            ):

                return self._fine_tune_precise(param, x_train, y_train, x_test, y_test)
            else:
                result = xgb.cv(
                    params=param,
                    dtrain=d_train,
                    num_boost_round=steps,
                    early_stopping_rounds=self.conf_training.early_stopping_rounds,
                    nfold=self.conf_training.hypertuning_cv_folds,
                    as_pandas=True,
                    seed=self.conf_training.global_random_state,
                    callbacks=[pruning_callback],
                    shuffle=self.conf_training.shuffle_during_training,
                )

                adjusted_score = result["test-mlogloss-mean"].mean()

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
            fig = optuna.visualization.plot_param_importances(
                study  # , evaluator=FanovaImportanceEvaluator()
            )
            fig.show()
        except (ZeroDivisionError, RuntimeError, ValueError):
            pass

        xgboost_best_param = study.best_trial.params

        self.conf_params_xgboost.params = {
            "objective": self.conf_xgboost.xgboost_objective,  # OR  'binary:logistic' #the loss function being used
            "booster": self.conf_xgboost.booster,
            "eval_metric": self.conf_xgboost.xgboost_eval_metric,
            "num_class": y_train.nunique(),
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
        logger(f"Best params: {self.conf_params_xgboost.params}")
        self.conf_params_xgboost.sample_weight = xgboost_best_param["sample_weight"]

    def get_best_score(self):
        if self.conf_training.autotune_model and (
            self.conf_training.hypertuning_cv_folds == 1
            or self.conf_training.precise_cv_tuning
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
        return best_score_cv_grid

    def create_d_matrices(self, x_train, y_train, x_test, y_test):
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
        return d_train, d_test

    def train_single_fold_model(
        self, d_train, d_test, y_test, param, steps, pruning_callback
    ):
        eval_set = [(d_train, "train"), (d_test, "test")]
        model = xgb.train(
            param,
            d_train,
            num_boost_round=steps,
            early_stopping_rounds=self.conf_training.early_stopping_rounds,
            evals=eval_set,
            callbacks=[pruning_callback],
            verbose_eval=self.conf_xgboost.verbosity_during_hyperparameter_tuning,
        )
        preds = model.predict(d_test)
        pred_labels = np.asarray([np.argmax(line) for line in preds])
        matthew = matthews_corrcoef(y_test.tolist(), pred_labels.tolist()) * -1

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

    def increasing_noise_evaluator(
        self, ml_model, eval_df: pd.DataFrame, y_true: pd.Series, iterations: int = 100
    ):
        """Function to add increasing noise and evaluate it.

        The function expects a trained model and a dataframe with the same columns as the training data.
        The training data should be normally distributed (consider using a power transformer with yeo-johnson).

        The function will apply increasingly noise to the eval dataframe and evaluate the model on it.

        Returns a list of losses.
        """
        # from sklearn.metrics import roc_auc_score
        losses = []
        for i in range(iterations):
            mu, sigma = 0, 0.2 * i
            N, D = eval_df.shape
            noise = self.random_generator.normal(mu, sigma, [N, D])
            eval_df_mod = eval_df + noise
            if self.conf_training:
                d_eval = xgb.DMatrix(
                    eval_df_mod,
                    label=y_true,
                    enable_categorical=self.conf_training.cat_encoding_via_ml_algorithm,
                )
            else:
                raise ValueError("No training_config could be found")
            y_hat = ml_model.predict(d_eval)
            y_classes = np.asarray([np.argmax(line) for line in y_hat])

            loss = matthews_corrcoef(y_true.values.tolist(), y_classes.tolist()) * -1
            losses.append(loss)

        return losses

    def constant_loss_degregation_factor(self, losses: List[float]) -> float:
        """Calculate a weighted loss based on the number of times the loss decreased.

        Expects a list of losses coming from increasing_noise_evaluator. Checks how many times the loss decreased and
        calculates a weighted loss based on the number of times the loss decreased.

        Returns the weighted loss.
        """
        nb_loss_decreased = 0
        for idx in range(len(losses)):
            if idx + 1 > len(losses) - 1:
                break
            if losses[idx] > losses[idx + 1]:
                nb_loss_decreased += 1

        # apply penalty
        if nb_loss_decreased == 0:
            weighted_loss = 999
        else:
            nb_losses = len(losses)
            weighted_loss = losses[0] - np.std(losses[:nb_loss_decreased]) ** (
                nb_losses / (nb_losses - nb_loss_decreased)
            )

        # print(
        #     f"Score decreased {nb_loss_decreased} times with weighted loss of {weighted_loss}"
        # )
        return weighted_loss

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
            logger("Could not find Training config. Falling back to default values")

        stratifier = StratifiedKFold(
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
                self.conf_params_xgboost = XgboostFinalParamConfig()
                logger(
                    "Could not find XgboostFinalParamConfig. Falling back to default settings."
                )

            if self.conf_params_xgboost.sample_weight:
                classes_weights = self.calculate_class_weights(y_train_fold)
                d_train = xgb.DMatrix(
                    X_train_fold,
                    label=y_train_fold,
                    weight=classes_weights,
                    enable_categorical=self.conf_training.cat_encoding_via_ml_algorithm,
                )
            else:
                d_train = xgb.DMatrix(
                    X_train_fold,
                    label=y_train_fold,
                    enable_categorical=self.conf_training.cat_encoding_via_ml_algorithm,
                )
            eval_set = [(d_train, "train"), (d_test, "test")]

            if not self.conf_xgboost:
                self.conf_xgboost = XgboostTuneParamsConfig()
                logger(
                    "Could not find XgboostTuneParamsConfig. Falling back to defaults."
                )
            model = xgb.train(
                tuned_params,
                d_train,
                num_boost_round=steps,
                early_stopping_rounds=self.conf_training.early_stopping_rounds,
                evals=eval_set,
                verbose_eval=self.conf_xgboost.verbosity_during_hyperparameter_tuning,
            )
            # d_eval = xgb.DMatrix(
            #    X_test_fold,
            #    label=y_test_fold,
            #    enable_categorical=self.conf_training.cat_encoding_via_ml_algorithm,
            # )
            # preds = model.predict(d_eval)
            losses = self.increasing_noise_evaluator(
                model, X_test_fold, y_test_fold, 100
            )
            constant_loss_degregation = self.constant_loss_degregation_factor(losses)
            fold_losses.append(constant_loss_degregation)
            # pred_labels = np.asarray([np.argmax(line) for line in preds])
            # fold_losses.append(matthews_corrcoef(y_test_fold, pred_labels) * -1)

        matthews_mean = np.mean(np.asarray(fold_losses))

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
                eval_scores=matthews_mean,
                metric_used="matthew_inverse",
                metric_higher_is_better=False,
            )
        return matthews_mean

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

        def objective(trial):
            d_train, d_test = self.create_d_matrices(x_train, y_train, x_test, y_test)

            pruning_callback = optuna.integration.XGBoostPruningCallback(
                trial, "test-mlogloss"
            )
            # copy best params to not overwrite them
            tuned_params = deepcopy(self.conf_params_xgboost.params)
            alpha_space = trial.suggest_float(
                "alpha",
                self.conf_params_xgboost.params["alpha"] * 0.9,
                self.conf_params_xgboost.params["alpha"] * 1.1,
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

            tuned_params["alpha"] = alpha_space
            tuned_params["lambda"] = lambda_space
            tuned_params["gamma"] = gamma_space
            tuned_params["eta"] = eta_space

            steps = tuned_params.pop("steps", 300)

            if self.conf_training.hypertuning_cv_folds == 1:
                return self.train_single_fold_model(
                    d_train, d_test, y_test, tuned_params, steps, pruning_callback
                )
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
                result = xgb.cv(
                    params=tuned_params,
                    dtrain=d_train,
                    num_boost_round=steps,
                    early_stopping_rounds=self.conf_training.early_stopping_rounds,
                    nfold=self.conf_training.hypertuning_cv_folds,
                    as_pandas=True,
                    seed=self.conf_training.global_random_state,
                    callbacks=[pruning_callback],
                    shuffle=self.conf_training.shuffle_during_training,
                )

                adjusted_score = result["test-mlogloss-mean"].mean()

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
            and isinstance(self.conf_params_xgboost.params["eta"], float)
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
                "eta": np.linspace(
                    self.conf_params_xgboost.params["eta"] * 0.9,
                    self.conf_params_xgboost.params["eta"] * 1.1,
                    self.conf_training.gridsearch_nb_parameters_per_grid,
                    dtype=float,
                ),
            }
        else:
            ValueError("Some parameters are not floats or strings")

        best_score_cv = self.get_best_score()

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

        best_score_cv_grid = self.get_best_score()

        if best_score_cv_grid < best_score_cv or not self.conf_training.autotune_model:
            xgboost_grid_best_param = study.best_trial.params
            self.conf_params_xgboost.params["alpha"] = xgboost_grid_best_param["alpha"]
            self.conf_params_xgboost.params["lambda"] = xgboost_grid_best_param[
                "lambda"
            ]
            self.conf_params_xgboost.params["gamma"] = xgboost_grid_best_param["gamma"]
            self.conf_params_xgboost.params["eta"] = xgboost_grid_best_param["eta"]
            logger(
                f"Grid search improved eval metric from {best_score_cv} to {best_score_cv_grid}."
            )
            logger(f"Best params: {self.conf_params_xgboost.params}")
        else:
            logger(
                f"Grid search could not improve eval metric of {best_score_cv}. Best score reached was {best_score_cv_grid}"
            )

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict on unseen data."""
        logger(
            f"{datetime.utcnow()}: Start predicting on new data using Xgboost model."
        )
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

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict class scores on unseen data."""
        logger(
            f"{datetime.utcnow()}: Start predicting on new data using Xgboost model."
        )
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

        partial_probs = self.model.predict(d_test)
        if self.class_problem == "binary":
            predicted_probs = np.asarray([line[1] for line in partial_probs])
        else:
            predicted_probs = partial_probs
        logger("Finished predicting")
        return predicted_probs
