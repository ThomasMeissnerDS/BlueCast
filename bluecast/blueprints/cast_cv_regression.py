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
from bluecast.evaluation.eval_metrics import RegressionEvalWrapper
from bluecast.experimentation.tracking import ExperimentTracker
from bluecast.ml_modelling.xgboost import XgboostModel
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
    :param :ml_model: Takes an instance of a XgboostModelRegression class. If not provided, BlueCast will instantiate one.
        This is an API to pass any model class. Inherit the baseclass from ml_modelling.base_model.BaseModel.
    :param custom_in_fold_preprocessor: Takes an instance of a CustomPreprocessing class. Allows users to eeecute
        preprocessing after the train test split within cv folds. This will be executed only if precise_cv_tuning in
        the conf_Training is True. Custom ML models need to implement this themselves. This step is only useful when
        the proprocessing step has a high chance of overfitting otherwise (i.e: oversampling techniques).
    :param custom_preprocessor: Takes an instance of a CustomPreprocessing class. Allows users to inject custom
        preprocessing steps which take place right after the train test spit.
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
        conf_xgboost: Optional[
            Union[XgboostTuneParamsRegressionConfig, CatboostTuneParamsRegressionConfig]
        ] = None,
        conf_params_xgboost: Optional[
            Union[XgboostRegressionFinalParamConfig, CatboostRegressionFinalParamConfig]
        ] = None,
        experiment_tracker: Optional[ExperimentTracker] = None,
        custom_in_fold_preprocessor: Optional[CustomPreprocessing] = None,
        custom_last_mile_computation: Optional[CustomPreprocessing] = None,
        custom_preprocessor: Optional[CustomPreprocessing] = None,
        custom_feature_selector: Optional[
            Union[BoostaRootaWrapper, CustomPreprocessing]
        ] = None,
        ml_model: Optional[Union[XgboostModel, Any]] = None,
        single_fold_eval_metric_func: Optional[RegressionEvalWrapper] = None,
    ):
        self.class_problem = class_problem
        self.conf_xgboost = conf_xgboost
        self.conf_params_xgboost = conf_params_xgboost
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

        if not cat_columns:
            self.cat_columns = []
        else:
            self.cat_columns = cat_columns

        if experiment_tracker:
            self.experiment_tracker = experiment_tracker
        else:
            self.experiment_tracker = ExperimentTracker()

        if not self.conf_params_xgboost:
            self.conf_params_xgboost = XgboostRegressionFinalParamConfig()

        self.conf_training: TrainingConfig = conf_training or TrainingConfig()

        if not self.conf_xgboost:
            self.conf_xgboost = XgboostTuneParamsRegressionConfig()

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

        This function collects these scores and return mean and average of them.

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
                conf_xgboost=self.conf_xgboost,
                conf_params_xgboost=deepcopy(self.conf_params_xgboost),
                experiment_tracker=self.experiment_tracker,
                custom_in_fold_preprocessor=self.custom_in_fold_preprocessor,
                custom_preprocessor=self.custom_preprocessor,
                custom_feature_selector=self.custom_feature_selector,
                custom_last_mile_computation=self.custom_last_mile_computation,
                ml_model=self.ml_model,
                single_fold_eval_metric_func=self.single_fold_eval_metric_func,
            )
            automl.fit(X_train, target_col=target_col)
            self.bluecast_models.append(automl)

            # overwrite experiment tracker to pass it into next iteration
            self.experiment_tracker = automl.experiment_tracker

    def fit_eval(self, df: pd.DataFrame, target_col: str) -> Tuple[float, float]:
        """Fit multiple BlueCastRegression instances on different data splits.

        Input df is expected the target column. Evaluation is executed on out-of-fold dataset
        in each split.
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

        for fn, (trn_idx, val_idx) in enumerate(self.stratifier.split(X, y_binned)):
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
                conf_xgboost=self.conf_xgboost,
                conf_params_xgboost=deepcopy(self.conf_params_xgboost),
                experiment_tracker=self.experiment_tracker,
                custom_in_fold_preprocessor=self.custom_in_fold_preprocessor,
                custom_preprocessor=self.custom_preprocessor,
                custom_feature_selector=self.custom_feature_selector,
                custom_last_mile_computation=self.custom_last_mile_computation,
                ml_model=self.ml_model,
                single_fold_eval_metric_func=self.single_fold_eval_metric_func,
            )
            automl.fit_eval(X_train, X_val, y_val, target_col=target_col)
            self.bluecast_models.append(automl)

            # overwrite experiment tracker to pass it into next iteration
            self.experiment_tracker = automl.experiment_tracker

        oof_mean, oof_std = self.show_oof_scores()
        return oof_mean, oof_std

    def predict(
        self,
        df: pd.DataFrame,
        return_sub_models_preds: bool = False,
        save_shap_values: bool = False,
        mean_type: Literal[
            "arithmetic", "median", "geometric", "harmonic"
        ] = "arithmetic",
    ) -> Union[pd.DataFrame, pd.Series]:
        """Predict on unseen data using multiple trained BlueCastRegression instances.

        :param df: Pandas DataFrame with unseen data
        :param return_sub_models_preds: If true will return a DataFrame with the predictions of each model
            stored in separate columns.
        :param save_shap_values: If True, calculates and saves shap values, so they can be used to plot
            waterfall plots for selected rows o demand.
        :param mean_type: String indicating the type of mean to be used to blend the predictions of the sub models.
            Possible values are 'arithmetic', 'geometric' and 'harmonic' (default='arithmetic').
        """
        or_cols = df.columns
        pred_cols: list[str] = []
        result_df = pd.DataFrame()  # Create an empty DataFrame to store results

        for fn, pipeline in enumerate(self.bluecast_models):
            y_preds = pipeline.predict(
                df.loc[:, or_cols], save_shap_values=save_shap_values
            )
            result_df[f"preds_{fn}"] = y_preds
            pred_cols.append(f"preds_{fn}")

        if return_sub_models_preds:
            return result_df
        else:
            if mean_type == "arithmetic":
                return result_df.mean(axis=1)
            elif mean_type == "geometric":
                return np.exp(np.log(result_df.prod(axis=1)) / result_df.notna().sum(1))
            elif mean_type == "harmonic":
                return len(pred_cols) / np.sum(1 / result_df, axis=1)
            elif mean_type == "median":
                return result_df.median(axis=1)
            else:
                return result_df.mean(axis=1)

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
