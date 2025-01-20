"""Xgboost classification model.

This module contains a wrapper for the Xgboost classification model. It can be used to train and/or tune the model.
It also calculates class weights for imbalanced datasets. The weights may or may not be used deepending on the
hyperparameter tuning.
"""

import logging
import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.utils import class_weight

from bluecast.config.training_config import (
    TrainingConfig,
    XgboostFinalParamConfig,
    XgboostTuneParamsConfig,
)
from bluecast.evaluation.eval_metrics import ClassificationEvalWrapper
from bluecast.experimentation.tracking import ExperimentTracker
from bluecast.ml_modelling.base_classes import XgboostBaseModel
from bluecast.ml_modelling.parameter_tuning_utils import (
    get_params_based_on_device_xgboost,
    sample_data,
    update_params_based_on_tree_method,
    update_params_with_best_params,
)
from bluecast.preprocessing.custom import CustomPreprocessing

warnings.filterwarnings("ignore", "is_sparse is deprecated")


class XgboostModel(XgboostBaseModel):
    """Train and/or tune Xgboost classification model."""

    def __init__(
        self,
        class_problem: Literal["binary", "multiclass"],
        conf_training: Optional[TrainingConfig] = None,
        conf_xgboost: Optional[XgboostTuneParamsConfig] = None,
        conf_params_xgboost: Optional[XgboostFinalParamConfig] = None,
        experiment_tracker: Optional[ExperimentTracker] = None,
        custom_in_fold_preprocessor: Optional[CustomPreprocessing] = None,
        cat_columns: Optional[List[Union[str, float, int]]] = None,
        single_fold_eval_metric_func: Optional[ClassificationEvalWrapper] = None,
    ):
        super().__init__(
            class_problem,
            conf_training,
            conf_xgboost,
            conf_params_xgboost,
            experiment_tracker,
            custom_in_fold_preprocessor,
            cat_columns,
            single_fold_eval_metric_func,
        )

        if not self.single_fold_eval_metric_func:
            self.single_fold_eval_metric_func: ClassificationEvalWrapper = (
                ClassificationEvalWrapper()
            )

    def fit(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> xgb.Booster:
        """Train Xgboost model. Includes hyperparameter tuning on default."""
        logging.info("Start fitting Xgboost model.")

        self.orchestrate_hyperparameter_tuning(
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test
        )

        logging.info("Start final model training")
        if self.custom_in_fold_preprocessor:
            x_train, y_train = self.custom_in_fold_preprocessor.fit_transform(
                x_train, y_train
            )
            x_test, y_test = self.custom_in_fold_preprocessor.transform(
                x_test, y_test, predicton_mode=False
            )

        if self.conf_training.use_full_data_for_final_model:
            x_train, y_train = self.concat_prepare_full_train_datasets(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
            )

        d_train, d_test = self._create_d_matrices(x_train, y_train, x_test, y_test)
        eval_set = [(d_test, "test")]

        steps = self.conf_params_xgboost.params.pop("steps", 300)

        if self.conf_training.hypertuning_cv_folds == 1:
            self.model = xgb.train(
                self.conf_params_xgboost.params,
                d_train,
                num_boost_round=steps,
                evals=eval_set,
                verbose_eval=self.conf_xgboost.verbosity_during_final_model_training,
                callbacks=self.get_early_stopping_callback(),
            )
        elif self.conf_xgboost:
            self.model = xgb.train(
                self.conf_params_xgboost.params,
                d_train,
                num_boost_round=steps,
                early_stopping_rounds=self.conf_training.early_stopping_rounds,
                evals=eval_set,
                verbose_eval=self.conf_xgboost.verbosity_during_final_model_training,
                callbacks=self.get_early_stopping_callback(),
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

        train_on = get_params_based_on_device_xgboost(
            self.conf_training, self.conf_params_xgboost, self.conf_xgboost
        )

        x_train, x_test, y_train, y_test = sample_data(
            x_train, x_test, y_train, y_test, self.conf_training
        )

        def objective(trial):
            param = {
                "validate_parameters": False,
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

            sample_weight = trial.suggest_categorical("sample_weight", [True, False])
            params = update_params_based_on_tree_method(
                params, trial, self.conf_xgboost
            )

            if sample_weight:
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

            pruning_callback = optuna.integration.XGBoostPruningCallback(
                trial, f"test-{self.conf_xgboost.xgboost_eval_metric}"
            )

            steps = params.pop("steps", 300)

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
                skf = RepeatedStratifiedKFold(
                    n_splits=self.conf_training.hypertuning_cv_folds,
                    n_repeats=self.conf_training.hypertuning_cv_repeats,
                    random_state=self.conf_training.global_random_state,
                )
                folds = []
                for train_index, test_index in skf.split(x_train, y_train.tolist()):
                    folds.append((train_index.tolist(), test_index.tolist()))

                result = xgb.cv(
                    params=params,
                    dtrain=d_train,
                    num_boost_round=steps,
                    # early_stopping_rounds=self.conf_training.early_stopping_rounds,  # not recommended as per docs: https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html
                    nfold=self.conf_training.hypertuning_cv_folds,
                    as_pandas=True,
                    seed=self.conf_training.global_random_state,
                    callbacks=[pruning_callback],
                    shuffle=self.conf_training.shuffle_during_training,
                    # stratified=True,
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
                    metric_used="adjusted ml logloss",
                    metric_higher_is_better=False,
                )

                return adjusted_score

        for rst in range(self.conf_training.autotune_n_random_seeds):
            logging.info(
                f"Hyperparameter tuning using random seed {self.conf_training.global_random_state + rst}"
            )
            sampler = optuna.samplers.TPESampler(
                multivariate=True,
                seed=self.conf_training.global_random_state + rst,
                n_startup_trials=self.conf_training.optuna_sampler_n_startup_trials,
                warn_independent_sampling=False,
            )
            study = optuna.create_study(
                direction=self.conf_xgboost.xgboost_eval_metric_tune_direction,
                sampler=sampler,
                study_name="xgboost tuning",
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=10,
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
                self.conf_params_xgboost.params = update_params_with_best_params(
                    self.conf_params_xgboost.params, xgboost_best_param
                )
                logging.info(f"Best params: {self.conf_params_xgboost.params}")
                print(f"Best params: {self.conf_params_xgboost.params}")
                self.conf_params_xgboost.sample_weight = xgboost_best_param[
                    "sample_weight"
                ]

    def train_single_fold_model(
        self, d_train, d_test, y_test, param, steps, pruning_callback
    ):
        eval_set = [(d_test, "test")]

        model = xgb.train(
            param,
            d_train,
            num_boost_round=steps,
            evals=eval_set,
            callbacks=self.get_early_stopping_callback(),
            verbose_eval=self.conf_xgboost.verbosity_during_hyperparameter_tuning,
        )
        preds = model.predict(d_test)
        matthew = self.single_fold_eval_metric_func.classification_eval_func_wrapper(
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
            eval_scores=matthew,
            metric_used="matthew_inverse",
            metric_higher_is_better=False,
        )
        return matthew

    def _fine_tune_precise(
        self,
        tuned_params: Dict[str, Any],
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ):
        steps = tuned_params.pop("steps", 300)

        stratifier = RepeatedStratifiedKFold(
            n_splits=self.conf_training.hypertuning_cv_folds,
            n_repeats=self.conf_training.hypertuning_cv_repeats,
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

            if self.conf_params_xgboost.sample_weight:
                classes_weights = class_weight.compute_sample_weight(
                    class_weight="balanced", y=y_train_fold
                )
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
            eval_set = [(d_test, "test")]

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
                loss = (
                    self.single_fold_eval_metric_func.classification_eval_func_wrapper(
                        y_test_fold, preds
                    )
                )
            else:
                raise ValueError("No single_fold_eval_metric_func could be found")
            fold_losses.append(loss)

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
        logging.info("Start grid search fine tuning of Xgboost model.")

        def objective(trial):  # TODO: Move to baseclass as grid_search_objective
            d_train, d_test = self._create_d_matrices(x_train, y_train, x_test, y_test)

            pruning_callback = optuna.integration.XGBoostPruningCallback(
                trial, f"test-{self.conf_xgboost.xgboost_eval_metric}"
            )
            # copy best params to not overwrite them
            tuned_params = self._get_param_space_fpr_grid_search(trial)

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
                skf = RepeatedStratifiedKFold(
                    n_splits=self.conf_training.hypertuning_cv_folds,
                    n_repeats=self.conf_training.hypertuning_cv_repeats,
                    random_state=self.conf_training.global_random_state,
                )
                folds = []
                for train_index, test_index in skf.split(x_train, y_train.astype(int)):
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
                    # stratified=True,
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
                    metric_used="adjusted ml logloss",
                    metric_higher_is_better=False,
                )

                return adjusted_score

        search_space = self.create_fine_tune_search_space()
        self._optimize_and_plot_grid_search_study(objective, search_space)

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
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

        partial_probs = self.model.predict(d_test)
        if self.class_problem == "binary":
            predicted_probs = np.asarray([line[1] for line in partial_probs])
            predicted_classes = np.asarray(
                [
                    int(line[1] > self.conf_params_xgboost.classification_threshold)
                    for line in partial_probs
                ]
            )
        else:
            predicted_probs = partial_probs
            predicted_classes = np.asarray([np.argmax(line) for line in partial_probs])
        logging.info("Finished predicting")
        return predicted_probs, predicted_classes

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict class scores on unseen data."""
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

        partial_probs = self.model.predict(d_test)
        if self.class_problem == "binary":
            predicted_probs = np.asarray([line[1] for line in partial_probs])
        else:
            predicted_probs = partial_probs
        logging.info("Finished predicting")
        return predicted_probs
