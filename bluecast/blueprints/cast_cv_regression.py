from typing import Any, List, Literal, Optional, Tuple, Union

import pandas as pd
from sklearn.model_selection import KFold

from bluecast.blueprints.cast_regression import BlueCastRegression
from bluecast.config.training_config import TrainingConfig, XgboostFinalParamConfig
from bluecast.config.training_config import (
    XgboostTuneParamsRegressionConfig as XgboostTuneParamsConfig,
)
from bluecast.experimentation.tracking import ExperimentTracker
from bluecast.general_utils.general_utils import logger
from bluecast.ml_modelling.xgboost import XgboostModel
from bluecast.preprocessing.custom import CustomPreprocessing
from bluecast.preprocessing.feature_selection import RFECVSelector


class BlueCastCVRegression:
    """Wrapper to train and predict multiple blueCast intstances.

    Check the BlueCast class documentation for additional parameter details.
    A custom splitter can be provided.
    """

    def __init__(
        self,
        class_problem: Literal["regression"] = "regression",
        stratifier: Optional[Any] = None,
        conf_training: Optional[TrainingConfig] = None,
        conf_xgboost: Optional[XgboostTuneParamsConfig] = None,
        conf_params_xgboost: Optional[XgboostFinalParamConfig] = None,
        experiment_tracker: Optional[ExperimentTracker] = None,
        custom_in_fold_preprocessor: Optional[CustomPreprocessing] = None,
        custom_last_mile_computation: Optional[CustomPreprocessing] = None,
        custom_preprocessor: Optional[CustomPreprocessing] = None,
        custom_feature_selector: Optional[
            Union[RFECVSelector, CustomPreprocessing]
        ] = None,
        ml_model: Optional[Union[XgboostModel, Any]] = None,
    ):
        self.class_problem = class_problem
        self.conf_xgboost = conf_xgboost
        self.conf_training = conf_training
        self.conf_params_xgboost = conf_params_xgboost
        self.custom_in_fold_preprocessor = custom_in_fold_preprocessor
        self.custom_preprocessor = custom_preprocessor
        self.custom_feature_selector = custom_feature_selector
        self.custom_last_mile_computation = custom_last_mile_computation
        self.bluecast_models: List[BlueCastRegression] = []
        self.stratifier = stratifier
        self.ml_model = ml_model

        if experiment_tracker:
            self.experiment_tracker = experiment_tracker
        else:
            self.experiment_tracker = ExperimentTracker()

    def prepare_data(
        self, df: pd.DataFrame, target: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        df = df.reset_index(drop=True)
        y = df[target]
        X = df.drop(target, axis=1)
        return X, y

    def fit(self, df: pd.DataFrame, target_col: str) -> None:
        """Fit multiple BlueCastRegression instances on different data splits.

        Input df is expected the target column."""
        X, y = self.prepare_data(df, target_col)

        if not self.conf_training:
            self.conf_training = TrainingConfig()

        if not self.stratifier:
            self.stratifier = KFold(
                n_splits=5,
                shuffle=True,
                random_state=self.conf_training.global_random_state,
            )

        for fn, (trn_idx, val_idx) in enumerate(self.stratifier.split(X, y)):
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
            logger(
                f"Start fitting model number {fn} with random seed {self.conf_training.global_random_state}"
            )

            automl = BlueCastRegression(
                class_problem=self.class_problem,
                target_column=target_col,
                conf_training=self.conf_training,
                conf_xgboost=self.conf_xgboost,
                conf_params_xgboost=self.conf_params_xgboost,
                experiment_tracker=self.experiment_tracker,
                custom_in_fold_preprocessor=self.custom_in_fold_preprocessor,
                custom_preprocessor=self.custom_preprocessor,
                custom_feature_selector=self.custom_feature_selector,
                custom_last_mile_computation=self.custom_last_mile_computation,
                ml_model=self.ml_model,
            )
            automl.fit(X_train, target_col=target_col)
            self.bluecast_models.append(automl)

            # overwrite experiment tracker to pass it into next iteration
            self.experiment_tracker = automl.experiment_tracker

    def fit_eval(self, df: pd.DataFrame, target_col: str) -> None:
        """Fit multiple BlueCastRegression instances on different data splits.

        Input df is expected the target column. Evaluation is executed on out-of-fold dataset
        in each split."""
        X, y = self.prepare_data(df, target_col)

        if not self.conf_training:
            self.conf_training = TrainingConfig()

        if not self.stratifier:
            self.stratifier = KFold(
                n_splits=5,
                shuffle=True,
                random_state=self.conf_training.global_random_state,
            )

        for fn, (trn_idx, val_idx) in enumerate(self.stratifier.split(X, y)):
            X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]

            X_train.loc[:, target_col] = y_train

            self.conf_training.global_random_state += (
                self.conf_training.increase_random_state_in_bluecast_cv_by
            )
            logger(
                f"Start fitting model number {fn} with random seed {self.conf_training.global_random_state}"
            )

            automl = BlueCastRegression(
                class_problem=self.class_problem,
                target_column=target_col,
                conf_training=self.conf_training,
                conf_xgboost=self.conf_xgboost,
                conf_params_xgboost=self.conf_params_xgboost,
                experiment_tracker=self.experiment_tracker,
                custom_in_fold_preprocessor=self.custom_in_fold_preprocessor,
                custom_preprocessor=self.custom_preprocessor,
                custom_feature_selector=self.custom_feature_selector,
                custom_last_mile_computation=self.custom_last_mile_computation,
                ml_model=self.ml_model,
            )
            automl.fit_eval(X_train, X_val, y_val, target_col=target_col)
            self.bluecast_models.append(automl)

            # overwrite experiment tracker to pass it into next iteration
            self.experiment_tracker = automl.experiment_tracker

    def predict(
        self, df: pd.DataFrame, return_sub_models_preds: bool = False
    ) -> Union[pd.DataFrame, pd.Series]:
        """Predict on unseen data using multiple trained BlueCastRegression instances"""
        or_cols = df.columns
        pred_cols: list[str] = []
        for fn, pipeline in enumerate(self.bluecast_models):
            y_preds = pipeline.predict(df.loc[:, or_cols])
            df[f"preds_{fn}"] = y_preds
            pred_cols.append(f"proba_{fn}")

        if return_sub_models_preds:
            return df.loc[:, pred_cols]
        else:
            return df.loc[:, pred_cols].mean(axis=1)