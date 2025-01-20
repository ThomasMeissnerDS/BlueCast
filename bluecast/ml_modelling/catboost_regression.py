import logging
import warnings
from typing import Any, Dict, List, Literal, Optional, Union

import catboost as cb
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor, Pool

try:
    from sklearn.metrics import root_mean_squared_error as mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error

from sklearn.model_selection import KFold, RepeatedStratifiedKFold

from bluecast.config.training_config import (
    CatboostRegressionFinalParamConfig,
    CatboostTuneParamsRegressionConfig,
    TrainingConfig,
)
from bluecast.evaluation.eval_metrics import RegressionEvalWrapper
from bluecast.experimentation.tracking import ExperimentTracker
from bluecast.ml_modelling.base_classes import CatboostBaseModel
from bluecast.ml_modelling.parameter_tuning_utils import (
    get_params_based_on_device_catboost,
    sample_data,
    update_params_with_best_params,
)
from bluecast.preprocessing.custom import CustomPreprocessing

warnings.filterwarnings("ignore", "is_sparse is deprecated")


class CatboostModelRegression(CatboostBaseModel):
    """Train and/or tune a CatBoost regression model."""

    def __init__(
        self,
        class_problem: Literal["regression"],
        conf_training: Optional["TrainingConfig"] = None,
        conf_catboost: Optional["CatboostTuneParamsRegressionConfig"] = None,
        conf_params_catboost: Optional["CatboostRegressionFinalParamConfig"] = None,
        experiment_tracker: Optional["ExperimentTracker"] = None,
        custom_in_fold_preprocessor: Optional["CustomPreprocessing"] = None,
        cat_columns: Optional[List[Union[str, float, int]]] = None,
        single_fold_eval_metric_func: Optional["RegressionEvalWrapper"] = None,
    ):
        super().__init__(
            class_problem=class_problem,
            conf_training=conf_training,
            conf_catboost=conf_catboost,
            conf_params_catboost=conf_params_catboost,
            experiment_tracker=experiment_tracker,
            custom_in_fold_preprocessor=custom_in_fold_preprocessor,
            cat_columns=cat_columns,
            single_fold_eval_metric_func=single_fold_eval_metric_func,
        )

        if not self.single_fold_eval_metric_func:
            self.single_fold_eval_metric_func: RegressionEvalWrapper = (
                RegressionEvalWrapper(
                    higher_is_better=False,
                    metric_func=mean_squared_error,  # Will measure RMSE if squared=False
                    metric_name="Root Mean Squared Error",
                    squared=False,
                )
            )

    def fit(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> cb.CatBoost:
        """
        Train the CatBoost regressor. Includes hyperparameter tuning by default,
        then trains on full or partial data as specified.
        """
        logging.info("Start fitting CatBoost regression model.")

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
            x_test = pd.DataFrame()
            y_test = pd.Series(dtype=y_train.dtype)

        if not x_test.empty and not y_test.empty:
            train_pool, test_pool = self._create_pools(x_train, y_train, x_test, y_test)
        else:
            train_pool = self._create_pools(
                x_train,
                y_train,
                pd.DataFrame(),
                pd.Series(dtype=y_train.dtype),
            )[0]
            test_pool = None

        final_params = dict(self.conf_params_catboost.params)
        if "logging_level" not in final_params:
            final_params["logging_level"] = "Silent"

        early_stopping_dict = self.get_early_stopping_callback()
        if early_stopping_dict:
            final_params.update(early_stopping_dict)

        self.model = CatBoostRegressor(**final_params)

        if test_pool is not None and not test_pool.is_empty():
            self.model.fit(
                train_pool,
                eval_set=test_pool,
                use_best_model=bool(early_stopping_dict),
                verbose=self.conf_training.show_detailed_tuning_logs,
            )
        else:
            self.model.fit(
                train_pool,
                use_best_model=False,
                verbose=self.conf_training.show_detailed_tuning_logs,
            )

        logging.info("Finished training CatBoost regression.")
        return self.model

    def autotune(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        logging.info("Start hyperparameter tuning of CatBoost regression model.")

        # Merge device or other global settings
        train_on = get_params_based_on_device_catboost(
            self.conf_training,
            self.conf_params_catboost,
            self.conf_catboost,
        )

        # Possibly sample data for faster hyperparameter search
        x_train, x_test, y_train, y_test = sample_data(
            x_train, x_test, y_train, y_test, self.conf_training
        )

        def objective(trial):
            # Typical CatBoost regression params
            params = {
                "objective": self.conf_catboost.catboost_objective,
                "eval_metric": self.conf_catboost.catboost_eval_metric,
                "random_seed": self.conf_training.global_random_state,
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    self.conf_catboost.learning_rate_min,
                    self.conf_catboost.learning_rate_max,
                    log=True,
                ),
                "depth": trial.suggest_int(
                    "depth",
                    self.conf_catboost.depth_min,
                    self.conf_catboost.depth_max,
                ),
                "l2_leaf_reg": trial.suggest_float(
                    "l2_leaf_reg",
                    self.conf_catboost.l2_leaf_reg_min,
                    self.conf_catboost.l2_leaf_reg_max,
                    log=True,
                ),
                "bagging_temperature": trial.suggest_float(
                    "bagging_temperature",
                    self.conf_catboost.bagging_temperature_min,
                    self.conf_catboost.bagging_temperature_max,
                ),
                "random_strength": trial.suggest_float(
                    "random_strength",
                    self.conf_catboost.random_strength_min,
                    self.conf_catboost.random_strength_max,
                ),
                "subsample": trial.suggest_float(
                    "subsample",
                    self.conf_catboost.subsample_min,
                    self.conf_catboost.subsample_max,
                ),
                "border_count": trial.suggest_int(
                    "border_count",
                    self.conf_catboost.border_count_min,
                    self.conf_catboost.border_count_max,
                ),
                "bootstrap_type": trial.suggest_categorical(
                    "bootstrap_type",
                    self.conf_catboost.bootstrap_type,
                ),
                "grow_policy": trial.suggest_categorical(
                    "grow_policy",
                    self.conf_catboost.grow_policy,
                ),
                "iterations": trial.suggest_int(
                    "iterations",
                    self.conf_catboost.iterations_min,
                    self.conf_catboost.iterations_max,
                    log=True,
                ),
            }
            params = {**params, **train_on}

            train_pool = Pool(x_train, label=y_train, cat_features=self.cat_columns)
            test_pool = Pool(x_test, label=y_test, cat_features=self.cat_columns)

            if self.conf_training.hypertuning_cv_folds == 1:
                return self.train_single_fold_model(
                    train_pool, test_pool, y_test, params
                )
            elif (
                self.conf_training.hypertuning_cv_folds > 1
                and self.conf_training.precise_cv_tuning
            ):
                return self._fine_tune_precise(params, x_train, y_train, x_test, y_test)
            else:
                y_binned = pd.qcut(y_train, q=10, duplicates="drop", labels=False)
                skf = RepeatedStratifiedKFold(
                    n_splits=self.conf_training.hypertuning_cv_folds,
                    n_repeats=self.conf_training.hypertuning_cv_repeats,
                    random_state=self.conf_training.global_random_state,
                )

                fold_scores = []
                for train_idx, valid_idx in skf.split(x_train, y_binned):
                    X_tr, X_val = x_train.iloc[train_idx], x_train.iloc[valid_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]

                    fold_train_pool = Pool(
                        X_tr, label=y_tr, cat_features=self.cat_columns
                    )
                    fold_val_pool = Pool(
                        X_val, label=y_val, cat_features=self.cat_columns
                    )

                    model = CatBoostRegressor(**params)
                    model.fit(fold_train_pool, eval_set=fold_val_pool, verbose=False)

                    preds = model.predict(fold_val_pool)
                    score = (
                        self.single_fold_eval_metric_func.regression_eval_func_wrapper(
                            y_val, preds
                        )
                    )
                    fold_scores.append(score)

                avg_score = np.mean(fold_scores)

                # Track in experiment tracker
                if len(self.experiment_tracker.experiment_id) == 0:
                    new_id = 0
                else:
                    new_id = self.experiment_tracker.experiment_id[-1] + 1

                self.experiment_tracker.add_results(
                    experiment_id=new_id,
                    score_category="cv_score",
                    training_config=self.conf_training,
                    model_parameters=params,
                    eval_scores=avg_score,
                    metric_used="catboost regression cv average",
                    metric_higher_is_better=False,
                )
                return avg_score

        for rst in range(self.conf_training.autotune_n_random_seeds):
            logging.info(
                f"Hyperparameter tuning using random seed "
                f"{self.conf_training.global_random_state + rst}"
            )
            sampler = optuna.samplers.TPESampler(
                multivariate=True,
                seed=self.conf_training.global_random_state + rst,
                n_startup_trials=self.conf_training.optuna_sampler_n_startup_trials,
                warn_independent_sampling=False,
            )
            study = optuna.create_study(
                direction=self.conf_catboost.catboost_eval_metric_tune_direction,
                sampler=sampler,
                study_name="catboost regression tuning",
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=10, n_warmup_steps=20
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
                    fig = optuna.visualization.plot_param_importances(study)
                    fig.show()
                except (ZeroDivisionError, RuntimeError, ValueError):
                    pass

            if (
                self.conf_catboost.catboost_eval_metric_tune_direction == "minimize"
                and study.best_value < self.best_score
            ) or (
                self.conf_catboost.catboost_eval_metric_tune_direction == "maximize"
                and study.best_value > self.best_score
            ):
                self.best_score = study.best_value
                logging.info(
                    f"New best score: {study.best_value} from random seed "
                    f"{self.conf_training.global_random_state + rst}"
                )
                catboost_best_param = study.best_trial.params

                final_best_params = {
                    "objective": self.conf_catboost.catboost_objective,
                    "eval_metric": self.conf_catboost.catboost_eval_metric,
                    "random_seed": self.conf_training.global_random_state,
                    "depth": catboost_best_param["depth"],
                    "learning_rate": catboost_best_param["learning_rate"],
                    "l2_leaf_reg": catboost_best_param["l2_leaf_reg"],
                    "bagging_temperature": catboost_best_param["bagging_temperature"],
                    "random_strength": catboost_best_param["random_strength"],
                    "subsample": catboost_best_param["subsample"],
                    "border_count": catboost_best_param["border_count"],
                    "bootstrap_type": catboost_best_param["bootstrap_type"],
                    "grow_policy": catboost_best_param["grow_policy"],
                    "iterations": catboost_best_param["iterations"],
                }
                # Merge device or other settings
                final_best_params = {**final_best_params, **train_on}

                # Optionally apply a custom function to finalize best params
                final_best_params = update_params_with_best_params(
                    final_best_params, catboost_best_param
                )

                self.conf_params_catboost.params = final_best_params
                logging.info(f"Best params: {self.conf_params_catboost.params}")
                print(f"Best params: {self.conf_params_catboost.params}")

    def train_single_fold_model(
        self,
        train_pool: Pool,
        test_pool: Pool,
        y_test: pd.Series,
        params: Dict[str, Any],
    ) -> float:
        """
        Single-fold training approach. Trains a quick model and returns the
        metric from the single_fold_eval_metric_func.
        """
        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=test_pool, verbose=False)

        preds = model.predict(test_pool)
        score = self.single_fold_eval_metric_func.regression_eval_func_wrapper(
            y_test, preds
        )

        # Track the result
        if len(self.experiment_tracker.experiment_id) == 0:
            new_id = 0
        else:
            new_id = self.experiment_tracker.experiment_id[-1] + 1

        self.experiment_tracker.add_results(
            experiment_id=new_id,
            score_category="simple_train_test_score",
            training_config=self.conf_training,
            model_parameters=params,
            eval_scores=score,
            metric_used="catboost_single_fold_regression",
            metric_higher_is_better=False,
        )
        return score

    def _fine_tune_precise(
        self,
        tuned_params: Dict[str, Any],
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> float:
        kf = KFold(
            n_splits=self.conf_training.hypertuning_cv_folds,
            shuffle=True,
            random_state=self.conf_training.global_random_state,
        )

        fold_losses = []
        for _fn, (trn_idx, val_idx) in enumerate(kf.split(x_train, y_train)):
            X_train_fold, X_val_fold = x_train.iloc[trn_idx], x_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[trn_idx], y_train.iloc[val_idx]

            # Apply custom preprocessing if available
            if self.custom_in_fold_preprocessor:
                X_train_fold, y_train_fold = (
                    self.custom_in_fold_preprocessor.fit_transform(
                        X_train_fold, y_train_fold
                    )
                )
                X_val_fold, y_val_fold = self.custom_in_fold_preprocessor.transform(
                    X_val_fold, y_val_fold, predicton_mode=False
                )
                X_test_fold, y_test_fold = self.custom_in_fold_preprocessor.transform(
                    x_test, y_test
                )
            else:
                X_test_fold, y_test_fold = x_test, y_test

            train_pool = Pool(
                X_train_fold, label=y_train_fold, cat_features=self.cat_columns
            )
            val_pool = Pool(X_val_fold, label=y_val_fold, cat_features=self.cat_columns)

            model = CatBoostRegressor(**tuned_params)
            model.fit(train_pool, eval_set=val_pool, verbose=False)

            test_pool = Pool(
                X_test_fold, label=y_test_fold, cat_features=self.cat_columns
            )
            preds = model.predict(test_pool)

            if self.single_fold_eval_metric_func:
                loss = self.single_fold_eval_metric_func.regression_eval_func_wrapper(
                    y_test_fold, preds
                )
            else:
                raise ValueError("No single_fold_eval_metric_func could be found")

            fold_losses.append(loss)

        mean_score = np.mean(fold_losses)

        # Track the result
        if len(self.experiment_tracker.experiment_id) == 0:
            new_id = 0
        else:
            new_id = self.experiment_tracker.experiment_id[-1] + 1

        self.experiment_tracker.add_results(
            experiment_id=new_id,
            score_category="oof_score",
            training_config=self.conf_training,
            model_parameters=tuned_params,
            eval_scores=mean_score,
            metric_used="catboost_oof_regression",
            metric_higher_is_better=False,
        )
        return mean_score

    def fine_tune(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        logging.info("Start grid search fine tuning of CatBoost regression model.")

        def objective(trial):
            tuned_params = self._get_param_space_fpr_grid_search(trial)

            train_pool = Pool(x_train, label=y_train, cat_features=self.cat_columns)
            test_pool = Pool(x_test, label=y_test, cat_features=self.cat_columns)

            if self.conf_training.hypertuning_cv_folds == 1:
                return self.train_single_fold_model(
                    train_pool, test_pool, y_test, tuned_params
                )
            elif (
                self.conf_training.hypertuning_cv_folds > 1
                and self.conf_training.precise_cv_tuning
            ):
                return self._fine_tune_precise(
                    tuned_params, x_train, y_train, x_test, y_test
                )
            else:
                y_binned = pd.qcut(y_train, 10, duplicates="drop", labels=False)
                skf = RepeatedStratifiedKFold(
                    n_splits=self.conf_training.hypertuning_cv_folds,
                    n_repeats=self.conf_training.hypertuning_cv_repeats,
                    random_state=self.conf_training.global_random_state,
                )
                fold_scores = []
                for train_idx, valid_idx in skf.split(x_train, y_binned):
                    X_tr, X_val = x_train.iloc[train_idx], x_train.iloc[valid_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]

                    fold_train_pool = Pool(
                        X_tr, label=y_tr, cat_features=self.cat_columns
                    )
                    fold_val_pool = Pool(
                        X_val, label=y_val, cat_features=self.cat_columns
                    )

                    model = CatBoostRegressor(**tuned_params)
                    model.fit(fold_train_pool, eval_set=fold_val_pool, verbose=False)

                    preds = model.predict(fold_val_pool)
                    score = (
                        self.single_fold_eval_metric_func.regression_eval_func_wrapper(
                            y_val, preds
                        )
                    )
                    fold_scores.append(score)

                avg_score = np.mean(fold_scores)

                if len(self.experiment_tracker.experiment_id) == 0:
                    new_id = 0
                else:
                    new_id = self.experiment_tracker.experiment_id[-1] + 1

                self.experiment_tracker.add_results(
                    experiment_id=new_id,
                    score_category="cv_score",
                    training_config=self.conf_training,
                    model_parameters=tuned_params,
                    eval_scores=avg_score,
                    metric_used="catboost fine_tune gridsearch regression",
                    metric_higher_is_better=False,
                )
                return avg_score

        search_space = self.create_fine_tune_search_space()
        self._optimize_and_plot_grid_search_study(objective, search_space)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict on unseen data using the trained CatBoost regressor.
        Returns numeric predictions.
        """
        logging.info("Start predicting on new data using CatBoost regression model.")

        if self.custom_in_fold_preprocessor:
            df, _ = self.custom_in_fold_preprocessor.transform(
                df, None, predicton_mode=True
            )

        test_pool = Pool(df, cat_features=self.cat_columns)

        if not self.model:
            raise Exception("No trained CatBoost model found.")

        preds = self.model.predict(test_pool)
        logging.info("Finished predicting.")
        return preds
