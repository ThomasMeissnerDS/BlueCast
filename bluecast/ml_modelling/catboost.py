import logging
import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import catboost as cb
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.utils import class_weight

from bluecast.config.training_config import (
    CatboostFinalParamConfig,
    CatboostTuneParamsConfig,
    TrainingConfig,
)
from bluecast.evaluation.eval_metrics import ClassificationEvalWrapper
from bluecast.experimentation.tracking import ExperimentTracker
from bluecast.ml_modelling.base_classes import CatboostBaseModel
from bluecast.ml_modelling.parameter_tuning_utils import (
    get_params_based_on_device_catboost,
    sample_data,
    update_params_with_best_params,
)
from bluecast.preprocessing.custom import CustomPreprocessing

warnings.filterwarnings("ignore", "is_sparse is deprecated")


class CatboostModel(CatboostBaseModel):
    """
    CatBoost classification model, mirroring the XgboostModel structure.

    This class can train and/or tune a CatBoost model, handle sample weighting,
    and do repeated stratified cross-validation or single-fold training.
    """

    def __init__(
        self,
        class_problem: Literal["binary", "multiclass"],
        conf_training: Optional["TrainingConfig"] = None,
        conf_catboost: Optional["CatboostTuneParamsConfig"] = None,
        conf_params_catboost: Optional["CatboostFinalParamConfig"] = None,
        experiment_tracker: Optional["ExperimentTracker"] = None,
        custom_in_fold_preprocessor: Optional["CustomPreprocessing"] = None,
        cat_columns: Optional[List[Union[str, float, int]]] = None,
        single_fold_eval_metric_func: Optional["ClassificationEvalWrapper"] = None,
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

        # Default to ClassificationEvalWrapper if not provided
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
    ) -> cb.CatBoost:
        """
        Train a CatBoost classification model. This includes optional
        hyperparameter tuning via 'orchestrate_hyperparameter_tuning',
        then final training on the entire data (or a subset if needed).
        """
        logging.info("Start fitting CatBoost classification model.")

        self.orchestrate_hyperparameter_tuning(
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test
        )

        logging.info("Start final model training")

        # Apply any custom preprocessing to the data
        if self.custom_in_fold_preprocessor:
            x_train, y_train = self.custom_in_fold_preprocessor.fit_transform(
                x_train, y_train
            )
            x_test, y_test = self.custom_in_fold_preprocessor.transform(
                x_test, y_test, predicton_mode=False
            )

        # Optionally concatenate train + test if config says so
        if self.conf_training.use_full_data_for_final_model:
            x_train, y_train = self.concat_prepare_full_train_datasets(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
            )
            # After combining, x_test/y_test is no longer used for final training
            x_test = pd.DataFrame()
            y_test = pd.Series(dtype=y_train.dtype)

        # Create training / test Pools
        if not x_test.empty and not y_test.empty:
            train_pool, test_pool = self._create_pools(x_train, y_train, x_test, y_test)
        else:
            # Train on entire data if no separate test
            train_pool = self._create_pools(
                x_train, y_train, pd.DataFrame(), pd.Series(dtype=y_train.dtype)
            )[0]
            test_pool = None

        # Pull out final params (dictionary) from config
        final_params = dict(self.conf_params_catboost.params)
        # Add or override logging level
        if "logging_level" not in final_params:
            final_params["logging_level"] = "Silent"

        # Overfitting detector / early stopping
        early_stopping_dict = self.get_early_stopping_callback()
        if early_stopping_dict:
            final_params.update(early_stopping_dict)

        # Create the final CatBoost model
        self.model = CatBoostClassifier(**final_params)

        # Train model with or without an eval set
        if test_pool is not None and not test_pool.is_empty():
            self.model.fit(
                train_pool,
                eval_set=test_pool,
                use_best_model=bool(early_stopping_dict),  # only if early_stopping
                verbose=self.conf_training.show_detailed_tuning_logs,
            )
        else:
            self.model.fit(
                train_pool,
                use_best_model=False,
                verbose=self.conf_training.show_detailed_tuning_logs,
            )

        logging.info("Finished final CatBoost training")
        return self.model

    def autotune(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        logging.info("Start hyperparameter tuning of CatBoost model.")

        # If you have some function that merges device-based or other settings:
        train_on = get_params_based_on_device_catboost(
            self.conf_training, self.conf_params_catboost, self.conf_catboost
        )

        x_train, x_test, y_train, y_test = sample_data(
            x_train, x_test, y_train, y_test, self.conf_training
        )

        def objective(trial):
            # Example: Build CatBoost param dictionary from conf_catboost
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
                    log=False,
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
                    log=False,
                ),
                "random_strength": trial.suggest_float(
                    "random_strength",
                    self.conf_catboost.random_strength_min,
                    self.conf_catboost.random_strength_max,
                    log=False,
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
                    log=False,
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
            if params["bootstrap_type"] in ["Bayesian", "No"]:
                params["bagging_temperature"] = None
                params["subsample"] = None

            params = {**params, **train_on}

            sample_weight_choice = trial.suggest_categorical(
                "sample_weight", [True, False]
            )

            if sample_weight_choice:
                weights = class_weight.compute_sample_weight("balanced", y_train)
                train_pool = Pool(
                    x_train,
                    label=y_train,
                    weight=weights,
                    cat_features=self.cat_columns,
                )
            else:
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
                skf = RepeatedStratifiedKFold(
                    n_splits=self.conf_training.hypertuning_cv_folds,
                    n_repeats=self.conf_training.hypertuning_cv_repeats,
                    random_state=self.conf_training.global_random_state,
                )
                fold_scores = []
                for train_index, valid_index in skf.split(x_train, y_train):
                    X_trn, X_val = x_train.iloc[train_index], x_train.iloc[valid_index]
                    y_trn, y_val = y_train.iloc[train_index], y_train.iloc[valid_index]
                    if sample_weight_choice:
                        fold_weights = class_weight.compute_sample_weight(
                            "balanced", y_trn
                        )
                        fold_pool = Pool(
                            X_trn,
                            label=y_trn,
                            weight=fold_weights,
                            cat_features=self.cat_columns,
                        )
                    else:
                        fold_pool = Pool(
                            X_trn, label=y_trn, cat_features=self.cat_columns
                        )

                    val_pool = Pool(X_val, label=y_val, cat_features=self.cat_columns)

                    model = CatBoostClassifier(**params)
                    model.fit(
                        fold_pool,
                        eval_set=val_pool,
                        use_best_model=False,
                        verbose=False,
                    )
                    preds = model.predict_proba(val_pool)
                    score = self.single_fold_eval_metric_func.classification_eval_func_wrapper(
                        y_val, preds
                    )
                    fold_scores.append(score)

                final_score = np.mean(fold_scores)

                if len(self.experiment_tracker.experiment_id) == 0:
                    new_id = 0
                else:
                    new_id = self.experiment_tracker.experiment_id[-1] + 1
                self.experiment_tracker.add_results(
                    experiment_id=new_id,
                    score_category="cv_score",
                    training_config=self.conf_training,
                    model_parameters=params,
                    eval_scores=final_score,
                    metric_used="catboost cv average",
                    metric_higher_is_better=False,
                )
                return final_score

        for rst in range(self.conf_training.autotune_n_random_seeds):
            logging.info(
                f"Hyperparameter tuning with random seed {self.conf_training.global_random_state + rst}"
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
                study_name="catboost tuning",
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

            # Optionally plot
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
                    f"New best score: {study.best_value} from random seed {self.conf_training.global_random_state + rst}"
                )
                best_param = study.best_trial.params

                final_best_params = {
                    "objective": self.conf_catboost.catboost_objective,
                    "eval_metric": self.conf_catboost.catboost_eval_metric,
                    "random_seed": self.conf_training.global_random_state,
                    "depth": best_param["depth"],
                    "learning_rate": best_param["learning_rate"],
                    "l2_leaf_reg": best_param["l2_leaf_reg"],
                    "bagging_temperature": best_param["bagging_temperature"],
                    "random_strength": best_param["random_strength"],
                    "subsample": best_param["subsample"],
                    "border_count": best_param["border_count"],
                    "bootstrap_type": best_param["bootstrap_type"],
                    "grow_policy": best_param["grow_policy"],
                    "iterations": best_param["iterations"],
                }
                final_best_params = {**final_best_params, **train_on}

                if final_best_params["bootstrap_type"] in ["Bayesian", "No"]:
                    final_best_params.pop("subsample", None)
                    final_best_params.pop("bagging_temperature", None)

                final_best_params = update_params_with_best_params(
                    final_best_params, best_param
                )

                self.conf_params_catboost.params = final_best_params
                self.conf_params_catboost.sample_weight = best_param["sample_weight"]

                logging.info(f"Best params: {self.conf_params_catboost.params}")
                print(f"Best params: {self.conf_params_catboost.params}")

    def train_single_fold_model(
        self,
        train_pool: Pool,
        test_pool: Pool,
        y_test: pd.Series,
        params: Dict[str, Any],
    ) -> float:
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=test_pool, verbose=False)

        preds = model.predict_proba(test_pool)
        score = self.single_fold_eval_metric_func.classification_eval_func_wrapper(
            y_test, preds
        )

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
            metric_used="catboost_single_fold",
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
        """
        Manual repeated stratified K-fold approach,
        similar to _fine_tune_precise in your XgboostModel.
        """
        stratifier = RepeatedStratifiedKFold(
            n_splits=self.conf_training.hypertuning_cv_folds,
            n_repeats=self.conf_training.hypertuning_cv_repeats,
            random_state=self.conf_training.global_random_state,
        )

        fold_scores = []
        for _fn, (train_idx, val_idx) in enumerate(stratifier.split(x_train, y_train)):
            X_train_fold, X_val_fold = x_train.iloc[train_idx], x_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # Optionally apply custom preprocessing
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

            if self.conf_params_catboost.sample_weight:
                weights_fold = class_weight.compute_sample_weight(
                    "balanced", y_train_fold
                )
                train_pool = Pool(
                    X_train_fold,
                    label=y_train_fold,
                    weight=weights_fold,
                    cat_features=self.cat_columns,
                )
            else:
                train_pool = Pool(
                    X_train_fold, label=y_train_fold, cat_features=self.cat_columns
                )
            val_pool = Pool(X_val_fold, label=y_val_fold, cat_features=self.cat_columns)

            model = CatBoostClassifier(**tuned_params)
            model.fit(train_pool, eval_set=val_pool, verbose=False)
            test_pool = Pool(
                X_test_fold, label=y_test_fold, cat_features=self.cat_columns
            )
            preds = model.predict_proba(test_pool)
            score = self.single_fold_eval_metric_func.classification_eval_func_wrapper(
                y_test_fold, preds
            )
            fold_scores.append(score)

        score_mean = np.mean(fold_scores)

        if len(self.experiment_tracker.experiment_id) == 0:
            new_id = 0
        else:
            new_id = self.experiment_tracker.experiment_id[-1] + 1
        self.experiment_tracker.add_results(
            experiment_id=new_id,
            score_category="oof_score",
            training_config=self.conf_training,
            model_parameters=tuned_params,
            eval_scores=score_mean,
            metric_used="catboost_multi_fold",
            metric_higher_is_better=False,
        )

        return score_mean

    def fine_tune(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        """
        Grid-search style fine tuning, analogous to the XGBoost model's method.
        We create an objective function that:
        1) Uses a small param space from create_fine_tune_search_space()
        2) Calls _optimize_and_plot_grid_search_study() to run an Optuna study with GridSampler
        """
        logging.info("Start grid search fine tuning of CatBoost model.")

        def objective(trial):
            tuned_params = self._get_param_space_fpr_grid_search(trial)

            if self.conf_params_catboost.sample_weight:
                weights = class_weight.compute_sample_weight("balanced", y_train)
                train_pool = Pool(
                    x_train,
                    label=y_train,
                    weight=weights,
                    cat_features=self.cat_columns,
                )
            else:
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
                skf = RepeatedStratifiedKFold(
                    n_splits=self.conf_training.hypertuning_cv_folds,
                    n_repeats=self.conf_training.hypertuning_cv_repeats,
                    random_state=self.conf_training.global_random_state,
                )
                fold_scores = []
                for train_index, valid_index in skf.split(x_train, y_train):
                    X_trn, X_val = x_train.iloc[train_index], x_train.iloc[valid_index]
                    y_trn, y_val = y_train.iloc[train_index], y_train.iloc[valid_index]

                    if self.conf_params_catboost.sample_weight:
                        fold_weights = class_weight.compute_sample_weight(
                            "balanced", y_trn
                        )
                        fold_pool = Pool(
                            X_trn,
                            label=y_trn,
                            weight=fold_weights,
                            cat_features=self.cat_columns,
                        )
                    else:
                        fold_pool = Pool(
                            X_trn, label=y_trn, cat_features=self.cat_columns
                        )
                    val_pool = Pool(X_val, label=y_val, cat_features=self.cat_columns)

                    model = CatBoostClassifier(**tuned_params)
                    model.fit(fold_pool, eval_set=val_pool, verbose=False)

                    preds = model.predict_proba(val_pool)
                    fold_score = self.single_fold_eval_metric_func.classification_eval_func_wrapper(
                        y_val, preds
                    )
                    fold_scores.append(fold_score)

                final_score = np.mean(fold_scores)

                if len(self.experiment_tracker.experiment_id) == 0:
                    new_id = 0
                else:
                    new_id = self.experiment_tracker.experiment_id[-1] + 1
                self.experiment_tracker.add_results(
                    experiment_id=new_id,
                    score_category="cv_score",
                    training_config=self.conf_training,
                    model_parameters=tuned_params,
                    eval_scores=final_score,
                    metric_used="catboost fine_tune gridsearch cv",
                    metric_higher_is_better=False,
                )
                return final_score

        search_space = self.create_fine_tune_search_space()
        self._optimize_and_plot_grid_search_study(objective, search_space)

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict probabilities and classes on new data.
        Returns (predicted_probs, predicted_classes).
        """
        logging.info("Start predicting on new data using CatBoost model.")
        if not self.conf_catboost or not self.conf_training:
            raise ValueError("conf_catboost or conf_training is None")

        if self.custom_in_fold_preprocessor:
            df, _ = self.custom_in_fold_preprocessor.transform(
                df, None, predicton_mode=True
            )

        if not self.model:
            raise Exception("No trained CatBoost model found.")

        if not self.conf_params_catboost:
            raise Exception("No CatBoost model configuration found.")

        pool_test = Pool(df, cat_features=self.cat_columns)
        partial_probs = self.model.predict_proba(pool_test)

        if self.class_problem == "binary":
            # For binary classification, partial_probs is shape (n, 2)
            predicted_probs = np.asarray([line[1] for line in partial_probs])
            threshold = self.conf_params_catboost.classification_threshold
            predicted_classes = np.asarray(
                [int(prob > threshold) for prob in predicted_probs]
            )
        else:
            # For multiclass, partial_probs is shape (n, num_classes)
            predicted_probs = partial_probs
            predicted_classes = np.asarray([np.argmax(line) for line in partial_probs])

        logging.info("Finished predicting with CatBoost model.")
        return predicted_probs, predicted_classes

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for new data (only relevant for classification).
        """
        logging.info("Start predict_proba on new data using CatBoost model.")
        if not self.conf_catboost or not self.conf_training:
            raise ValueError("conf_catboost or conf_training is None")

        if self.custom_in_fold_preprocessor:
            df, _ = self.custom_in_fold_preprocessor.transform(
                df, None, predicton_mode=True
            )

        if not self.model:
            raise Exception("No trained CatBoost model found.")
        if not self.conf_params_catboost:
            raise Exception("No CatBoost model configuration found.")

        pool_test = Pool(df, cat_features=self.cat_columns)
        partial_probs = self.model.predict_proba(pool_test)

        if self.class_problem == "binary":
            predicted_probs = np.asarray([line[1] for line in partial_probs])
        else:
            predicted_probs = partial_probs

        logging.info("Finished predict_proba with CatBoost model.")
        return predicted_probs
