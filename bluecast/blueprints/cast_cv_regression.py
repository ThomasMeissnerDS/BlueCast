import logging
from copy import deepcopy
from typing import Any, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder

from bluecast.blueprints.cast_regression import BlueCastRegression
from bluecast.config.training_config import (
    CatboostRegressionFinalParamConfig,
    CatboostTuneParamsRegressionConfig,
    TrainingConfig,
    XgboostRegressionFinalParamConfig,
    XgboostTuneParamsRegressionConfig,
)
from bluecast.conformal_prediction.conformal_prediction_regression import (
    ConformalPredictionRegressionWrapper,
)
from bluecast.ensemble.ensemble_config import EnsembleConfig
from bluecast.ensemble.hill_climbing import (
    HillClimbingEnsemble,
    _default_regression_metric,
)
from bluecast.ensemble.mean_blending import blend_predictions_mean
from bluecast.ensemble.stacking import StackingEnsemble
from bluecast.evaluation.eval_metrics import RegressionEvalWrapper
from bluecast.experimentation.tracking import ExperimentTracker
from bluecast.preprocessing.custom import CustomPreprocessing
from bluecast.preprocessing.feature_selection import BoostaRootaWrapper


class BlueCastCVRegression:
    """Wrapper to train and predict multiple blueCast instances.

    Check the BlueCast class documentation for additional parameter details.
    A custom splitter can be provided.

    :param :class_problem: Takes a string containing the class problem type. At the moment "regression" only.
    :param :target_column: Takes a string containing the name of the target column.
    :param :cat_columns: Takes a list of strings containing the names of the categorical columns. If not provided,
        BlueCast will infer these automatically.
    :param :date_columns: Takes a list of strings containing the names of the date columns. If not provided,
        BlueCast will infer these automatically.
    :param :time_split_column: Takes a string containing the name of the time split column. If not provided,
        BlueCast will not split the data by time or order, but do a random split instead.
    :param :ml_model: Takes an instance of a CatboostModelRegression class. If not provided, BlueCast will instantiate one.
        This is an API to pass any model class. Inherit the baseclass from ml_modelling.base_model.BaseModel.
    :param custom_in_fold_preprocessor: Takes an instance of a CustomPreprocessing class. Allows users to execute
        preprocessing after the train test split within cv folds. This will be executed only if precise_cv_tuning in
        the conf_Training is True. Custom ML models need to implement this themselves. This step is only useful when
        the preprocessing step has a high chance of overfitting otherwise (i.e: oversampling techniques).
    :param custom_preprocessor: Takes an instance of a CustomPreprocessing class. Allows users to inject custom
        preprocessing steps which take place right after the train test split.
    :param custom_last_mile_computation: Takes an instance of a CustomPreprocessing class. Allows users to inject custom
        preprocessing steps which take place right before the model training.
    :param experiment_tracker: Takes an instance of an ExperimentTracker class. If not provided this will be initialized
        automatically.
    :param single_fold_eval_metric_func: Takes a function which calculates the evaluation metric for a single fold.
       Default is mean_squared_error. This function is used to calculate the evaluation metric for each fold during
       hyperparameter tuning when hyperparameter_tuning_rounds = 1 (default). Lower must be better.
    """

    def __init__(
        self,
        class_problem: Literal["regression"] = "regression",
        cat_columns: Optional[List[Union[str, float, int]]] = None,
        stratifier: Optional[Any] = None,
        conf_training: Optional[TrainingConfig] = None,
        conf_tuning: Optional[
            Union[XgboostTuneParamsRegressionConfig, CatboostTuneParamsRegressionConfig]
        ] = None,
        conf_params: Optional[
            Union[XgboostRegressionFinalParamConfig, CatboostRegressionFinalParamConfig]
        ] = None,
        experiment_tracker: Optional[ExperimentTracker] = None,
        custom_in_fold_preprocessor: Optional[CustomPreprocessing] = None,
        custom_last_mile_computation: Optional[CustomPreprocessing] = None,
        custom_preprocessor: Optional[CustomPreprocessing] = None,
        custom_feature_selector: Optional[
            Union[BoostaRootaWrapper, CustomPreprocessing]
        ] = None,
        ml_model: Optional[Any] = None,
        single_fold_eval_metric_func: Optional[RegressionEvalWrapper] = None,
        ensemble_config: Optional[EnsembleConfig] = None,
    ):
        self.class_problem = class_problem
        self.conf_tuning = conf_tuning
        self.conf_params = conf_params
        self.custom_in_fold_preprocessor = custom_in_fold_preprocessor
        self.custom_preprocessor = custom_preprocessor
        self.custom_feature_selector = custom_feature_selector
        self.custom_last_mile_computation = custom_last_mile_computation
        self.bluecast_models: List[BlueCastRegression] = []
        self.stratifier = stratifier
        self.ml_model = ml_model
        self.single_fold_eval_metric_func = single_fold_eval_metric_func
        self.conformal_prediction_wrapper: Optional[
            ConformalPredictionRegressionWrapper
        ] = None
        self.ensemble_config = ensemble_config or EnsembleConfig(
            hc_blending_method="probability",
            stacking_use_ranks=False,
        )
        self.stacking_ensemble: Optional[StackingEnsemble] = None
        self.hill_climbing_ensemble: Optional[HillClimbingEnsemble] = None

        if not cat_columns:
            self.cat_columns = []
        else:
            self.cat_columns = cat_columns

        if experiment_tracker:
            self.experiment_tracker = experiment_tracker
        else:
            self.experiment_tracker = ExperimentTracker()

        if not self.conf_params:
            self.conf_params = CatboostRegressionFinalParamConfig()

        self.conf_training: TrainingConfig = conf_training or TrainingConfig()

        if not self.conf_tuning:
            self.conf_tuning = CatboostTuneParamsRegressionConfig()

        if not self.single_fold_eval_metric_func:
            self.single_fold_eval_metric_func = RegressionEvalWrapper(
                higher_is_better=False,
                metric_func=mean_squared_error,
                metric_name="Mean squared error",
            )

    def prepare_data(
        self, df: pd.DataFrame, target: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        df = df.reset_index(drop=True)
        y = df[target]
        X = df.drop(target, axis=1)
        return X, y

    def show_oof_scores(self, metric: str = "RMSE") -> Tuple[float, float]:
        """
        Show out of fold scores.

        When calling BlueCastCVRegression's fit_eval function multiple BlueCastRegression
        instances are called and each of them predicts on unseen/oof data.

        This function collects these scores and returns the mean and standard deviation of them.

        :param metric: String indicating which metric shall be returned.
        :return: Tuple with (mean, std) of oof scores
        """
        all_metrics = []
        for bluecast_instance in self.bluecast_models:
            if bluecast_instance.eval_metrics:
                score = bluecast_instance.eval_metrics.get(metric)
                all_metrics.append(score)

        score_mean = np.asarray(all_metrics).mean()
        score_std = np.asarray(all_metrics).std()
        logging.info(
            f"The mean out of fold {metric} score is {score_mean} with an std of {score_std}"
        )
        return score_mean, score_std

    def fit(self, df: pd.DataFrame, target_col: str) -> None:
        """Fit multiple BlueCastRegression instances on different data splits.

        Input df is expected the target column."""
        X, y = self.prepare_data(df, target_col)

        if not self.conf_training:
            self.conf_training = TrainingConfig()

        le = LabelEncoder()
        y_binned = le.fit_transform(pd.qcut(y, 10, duplicates="drop"))

        if not self.stratifier:
            self.stratifier = RepeatedStratifiedKFold(
                n_splits=self.conf_training.bluecast_cv_train_n_model[0],
                n_repeats=self.conf_training.bluecast_cv_train_n_model[1],
                random_state=self.conf_training.global_random_state,
            )

        for fn, (trn_idx, val_idx) in enumerate(self.stratifier.split(X, y_binned)):
            X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]
            x_train = pd.concat([X_train, X_val], ignore_index=True)
            y_train = pd.concat([y_train, y_val], ignore_index=True)

            X_train = x_train.reset_index(drop=True)
            y_train = y_train.reset_index(drop=True)
            X_train[target_col] = y_train.values

            self.conf_training.global_random_state += (
                self.conf_training.increase_random_state_in_bluecast_cv_by
            )
            logging.info(
                f"Start fitting model number {fn} with random seed {self.conf_training.global_random_state}"
            )

            automl = BlueCastRegression(
                class_problem=self.class_problem,
                cat_columns=self.cat_columns,
                conf_training=self.conf_training,
                conf_tuning=self.conf_tuning,
                conf_params=deepcopy(self.conf_params),
                experiment_tracker=self.experiment_tracker,
                custom_in_fold_preprocessor=self.custom_in_fold_preprocessor,
                custom_preprocessor=self.custom_preprocessor,
                custom_feature_selector=self.custom_feature_selector,
                custom_last_mile_computation=self.custom_last_mile_computation,
                ml_model=deepcopy(self.ml_model) if self.ml_model else None,
                single_fold_eval_metric_func=self.single_fold_eval_metric_func,
            )
            automl.fit(X_train, target_col=target_col)
            self.bluecast_models.append(automl)

            # overwrite experiment tracker to pass it into next iteration
            self.experiment_tracker = automl.experiment_tracker

    def fit_eval(self, df: pd.DataFrame, target_col: str) -> Tuple[float, float]:
        """Fit multiple BlueCastRegression instances on different data splits.

        Input df is expected the target column. Evaluation is executed on out-of-fold dataset
        in each split. When using stacking or hill_climbing ensemble strategies, OOF predictions
        are collected and used to fit the ensemble meta-learner.
        :param df: Pandas DataFrame that includes the target column
        :param target_col: String indicating the name of the target column
        :returns Tuple of (oof_mean, oof_std) with scores on unseen data during eval
        """
        X, y = self.prepare_data(df, target_col)

        if not self.conf_training:
            self.conf_training = TrainingConfig()

        le = LabelEncoder()
        y_binned = le.fit_transform(pd.qcut(y, 10, duplicates="drop"))

        if not self.stratifier:
            self.stratifier = RepeatedStratifiedKFold(
                n_splits=self.conf_training.bluecast_cv_train_n_model[0],
                n_repeats=self.conf_training.bluecast_cv_train_n_model[1],
                random_state=self.conf_training.global_random_state,
            )

        needs_oof = self.ensemble_config.ensemble_strategy in (
            "stacking",
            "hill_climbing",
        )
        oof_preds_per_model: List[np.ndarray] = []
        oof_indices_per_fold: List[np.ndarray] = []
        all_splits = list(self.stratifier.split(X, y_binned))

        for fn, (trn_idx, val_idx) in enumerate(all_splits):
            X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]

            X_train.loc[:, target_col] = y_train

            self.conf_training.global_random_state += (
                self.conf_training.increase_random_state_in_bluecast_cv_by
            )
            logging.info(
                f"Start fitting model number {fn} with random seed {self.conf_training.global_random_state}"
            )

            automl = BlueCastRegression(
                class_problem=self.class_problem,
                cat_columns=self.cat_columns,
                conf_training=self.conf_training,
                conf_tuning=self.conf_tuning,
                conf_params=deepcopy(self.conf_params),
                experiment_tracker=self.experiment_tracker,
                custom_in_fold_preprocessor=self.custom_in_fold_preprocessor,
                custom_preprocessor=self.custom_preprocessor,
                custom_feature_selector=self.custom_feature_selector,
                custom_last_mile_computation=self.custom_last_mile_computation,
                ml_model=deepcopy(self.ml_model) if self.ml_model else None,
                single_fold_eval_metric_func=self.single_fold_eval_metric_func,
            )
            automl.fit_eval(X_train, X_val, y_val, target_col=target_col)
            self.bluecast_models.append(automl)

            if needs_oof:
                oof_pred = automl.predict(X_val)
                oof_preds_per_model.append(oof_pred)
                oof_indices_per_fold.append(val_idx)

            self.experiment_tracker = automl.experiment_tracker

        if needs_oof:
            self._fit_ensemble_from_oof(
                oof_preds_per_model, oof_indices_per_fold, y, all_splits
            )

        oof_mean, oof_std = self.show_oof_scores()
        return oof_mean, oof_std

    def _fit_ensemble_from_oof(
        self,
        oof_preds_per_model: List[np.ndarray],
        oof_indices_per_fold: List[np.ndarray],
        y_full: pd.Series,
        all_splits: list,
    ) -> None:
        """Fit stacking or hill climbing ensemble from OOF predictions."""
        n_samples = len(y_full)
        n_models = len(self.bluecast_models)
        oof_matrix = np.full((n_samples, n_models), np.nan)

        for fn in range(n_models):
            val_idx = oof_indices_per_fold[fn]
            oof_matrix[val_idx, fn] = oof_preds_per_model[fn]

        # Impute NaN values with column-wise mean. With K-fold CV each row
        # only has OOF predictions from its validation fold, so most columns
        # are NaN. Mean imputation is the standard stacking approach.
        col_means = np.nanmean(oof_matrix, axis=0)
        for col in range(n_models):
            mask = np.isnan(oof_matrix[:, col])
            oof_matrix[mask, col] = col_means[col]

        # Drop any rows that are still all-NaN (shouldn't happen with valid folds)
        valid_mask = ~np.any(np.isnan(oof_matrix), axis=1)
        oof_valid = oof_matrix[valid_mask]
        y_valid = y_full.values[valid_mask]

        if len(oof_valid) == 0:
            logging.warning(
                "No valid OOF predictions for ensemble fitting. "
                "Falling back to mean blending."
            )
            return

        if self.ensemble_config.ensemble_strategy == "stacking":
            self.stacking_ensemble = StackingEnsemble(
                meta_learner=self.ensemble_config.stacking_meta_learner,
                use_ranks=self.ensemble_config.stacking_use_ranks,
                clip_predictions=False,  # regression targets can exceed [0, 1]
            )
            self.stacking_ensemble.fit(oof_valid, y_valid)
            logging.info("Stacking ensemble fitted on OOF predictions.")

        elif self.ensemble_config.ensemble_strategy == "hill_climbing":
            eval_metric = (
                self.ensemble_config.hc_eval_metric or _default_regression_metric
            )
            self.hill_climbing_ensemble = HillClimbingEnsemble(
                weight_min=self.ensemble_config.hc_weight_min,
                weight_max=self.ensemble_config.hc_weight_max,
                weight_step=self.ensemble_config.hc_weight_step,
                tolerance=self.ensemble_config.hc_tolerance,
                blending_method=self.ensemble_config.hc_blending_method,
                eval_metric=eval_metric,
                is_classification=False,
            )
            oof_list = [oof_valid[:, i] for i in range(n_models)]
            model_names = [f"model_{i}" for i in range(n_models)]
            self.hill_climbing_ensemble.fit(oof_list, y_valid, model_names)
            logging.info("Hill climbing ensemble fitted on OOF predictions.")

    def predict(
        self,
        df: pd.DataFrame,
        return_sub_models_preds: bool = False,
        save_shap_values: bool = False,
        mean_type: Optional[
            Literal["arithmetic", "median", "geometric", "harmonic"]
        ] = None,
    ) -> Union[pd.DataFrame, pd.Series]:
        """Predict on unseen data using multiple trained BlueCastRegression instances.

        :param df: Pandas DataFrame with unseen data
        :param return_sub_models_preds: If true will return a DataFrame with the predictions of each model
            stored in separate columns.
        :param save_shap_values: If True, calculates and saves shap values, so they can be used to plot
            waterfall plots for selected rows on demand.
        :param mean_type: String indicating the type of mean to be used to blend the predictions of the sub models.
            Only used when ensemble_strategy='mean'. If None, uses ensemble_config.mean_type.
        """
        or_cols = df.columns
        pred_cols: list[str] = []
        result_df = pd.DataFrame()

        for fn, pipeline in enumerate(self.bluecast_models):
            y_preds = pipeline.predict(
                df.loc[:, or_cols], save_shap_values=save_shap_values
            )
            result_df[f"preds_{fn}"] = y_preds
            pred_cols.append(f"preds_{fn}")

        if return_sub_models_preds:
            return result_df

        strategy = self.ensemble_config.ensemble_strategy

        if strategy == "stacking" and self.stacking_ensemble is not None:
            predictions_matrix = result_df.loc[:, pred_cols].values
            return pd.Series(
                self.stacking_ensemble.predict(predictions_matrix),
                index=result_df.index,
            )

        elif strategy == "hill_climbing" and self.hill_climbing_ensemble is not None:
            preds_list = [result_df[col].values for col in pred_cols]
            return pd.Series(
                self.hill_climbing_ensemble.predict(preds_list),
                index=result_df.index,
            )

        else:
            effective_mean_type = mean_type or self.ensemble_config.mean_type
            return blend_predictions_mean(result_df, pred_cols, effective_mean_type)

    def calibrate(
        self, x_calibration: pd.DataFrame, y_calibration: pd.Series, **kwargs
    ) -> None:
        """Calibrate the model.

        Via this function the nonconformity measures are taken and used to predict prediction intervals vis the
        predict_interval function. Used is the mean prediction of all sub models.
        :param: x_calibration: Pandas DataFrame without target column, that has not been seen by the model during
            training.
        :param y_calibration: Pandas Series holding the target value, hat has not been seen by the model during
            training.
        """
        if isinstance(y_calibration, np.ndarray):
            y_calibration = pd.Series(y_calibration)

        self.conformal_prediction_wrapper = ConformalPredictionRegressionWrapper(
            self, **kwargs
        )
        self.conformal_prediction_wrapper.calibrate(x_calibration, y_calibration)

    def predict_interval(self, df: pd.DataFrame, alphas: List[float]) -> pd.DataFrame:
        """Create prediction intervals based on a certain confidence levels.

        Conformal prediction guarantees, that the correct value is present in the prediction band with a probability of
        1 - alpha.
        :param df: Pandas DataFrame holding unseen data
        :param alphas: List of floats indicating the desired confidence levels.
        :returns A Pandas DataFrame with  sorted columns 'alpha_XX_low' (alpha) and 'alpha_XX_high' (1 - alpha) for each
            alpha in the provided list of alphas. To obtain the mean prediction call the 'predict' method.
        """
        if self.conformal_prediction_wrapper:
            pred_interval = self.conformal_prediction_wrapper.predict_interval(
                df, alphas=alphas
            )
            return pred_interval
        else:
            raise ValueError(
                """This instance has not been calibrated yet. Make use of calibrate to fit the
            ConformalPredictionWrapper."""
            )
