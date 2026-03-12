"""Unified BlueCast interface.

Provides a single entry point that internally dispatches to the correct blueprint
class based on the problem type and whether cross-validation is used.
"""

import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from bluecast.config.training_config import TrainingConfig
from bluecast.ensemble.ensemble_config import EnsembleConfig
from bluecast.experimentation.tracking import ExperimentTracker
from bluecast.preprocessing.custom import CustomPreprocessing
from bluecast.preprocessing.feature_selection import BoostaRootaWrapper

logger = logging.getLogger(__name__)


class BlueCastAuto:
    """Unified AutoML interface for classification and regression.

    A single entry point that automatically dispatches to the correct pipeline based on
    `class_problem` and `use_cross_validation`. This replaces the need to choose between
    BlueCast, BlueCastCV, BlueCastRegression, and BlueCastCVRegression.

    :param class_problem: Problem type: 'binary', 'multiclass', or 'regression'.
    :param use_cross_validation: If True, trains multiple models across CV folds and
        ensembles their predictions. If False, trains a single model with train/test split.
    :param cat_columns: List of categorical column names. Auto-detected if None.
    :param date_columns: List of date column names. Auto-detected if None.
    :param time_split_column: Column name for time-based train/test split.
    :param stratifier: Custom CV splitter (e.g. RepeatedStratifiedKFold).
        Only used when use_cross_validation=True.
    :param ml_model: Custom model instance. Must inherit from BaseClassMlModel.
    :param custom_preprocessor: Custom preprocessing after train/test split.
    :param custom_in_fold_preprocessor: Custom preprocessing within CV folds.
    :param custom_last_mile_computation: Custom preprocessing before model training.
    :param custom_feature_selector: Custom feature selection (or BoostaRootaWrapper).
    :param conf_training: Training configuration. Auto-configured if None.
    :param conf_tuning: Hyperparameter tuning configuration. Auto-configured based on
        class_problem if None.
    :param conf_params: Final model parameters. Auto-configured based on class_problem
        if None.
    :param experiment_tracker: Experiment tracker instance. Created if None.
    :param single_fold_eval_metric_func: Custom evaluation metric for hyperparameter tuning.
    :param ensemble_config: Ensemble configuration (mean, stacking, hill_climbing).
        Only used when use_cross_validation=True.

    Example usage::

        from bluecast.blueprints.unified import BlueCastAuto

        # Binary classification with CV and stacking
        automl = BlueCastAuto(
            class_problem="binary",
            use_cross_validation=True,
            ensemble_config=EnsembleConfig(ensemble_strategy="stacking"),
        )
        automl.fit(df_train, target_col="target")
        y_probs, y_classes = automl.predict(df_test)

        # Regression without CV
        automl = BlueCastAuto(class_problem="regression")
        automl.fit(df_train, target_col="price")
        y_preds = automl.predict(df_test)
    """

    def __init__(
        self,
        class_problem: Literal["binary", "multiclass", "regression"] = "binary",
        use_cross_validation: bool = False,
        cat_columns: Optional[List[Union[str, float, int]]] = None,
        date_columns: Optional[List[Union[str, float, int]]] = None,
        time_split_column: Optional[str] = None,
        stratifier: Optional[Any] = None,
        ml_model: Optional[Any] = None,
        custom_preprocessor: Optional[CustomPreprocessing] = None,
        custom_in_fold_preprocessor: Optional[CustomPreprocessing] = None,
        custom_last_mile_computation: Optional[CustomPreprocessing] = None,
        custom_feature_selector: Optional[
            Union[BoostaRootaWrapper, CustomPreprocessing]
        ] = None,
        conf_training: Optional[TrainingConfig] = None,
        conf_tuning: Optional[Any] = None,
        conf_params: Optional[Any] = None,
        experiment_tracker: Optional[ExperimentTracker] = None,
        single_fold_eval_metric_func: Optional[Any] = None,
        ensemble_config: Optional[EnsembleConfig] = None,
    ):
        self.class_problem = class_problem
        self.use_cross_validation = use_cross_validation

        self._inner = self._create_inner_instance(
            class_problem=class_problem,
            use_cross_validation=use_cross_validation,
            cat_columns=cat_columns,
            date_columns=date_columns,
            time_split_column=time_split_column,
            stratifier=stratifier,
            ml_model=ml_model,
            custom_preprocessor=custom_preprocessor,
            custom_in_fold_preprocessor=custom_in_fold_preprocessor,
            custom_last_mile_computation=custom_last_mile_computation,
            custom_feature_selector=custom_feature_selector,
            conf_training=conf_training,
            conf_tuning=conf_tuning,
            conf_params=conf_params,
            experiment_tracker=experiment_tracker,
            single_fold_eval_metric_func=single_fold_eval_metric_func,
            ensemble_config=ensemble_config,
        )

        logger.info(
            f"BlueCastAuto initialized: problem={class_problem}, "
            f"cv={use_cross_validation}, backend={type(self._inner).__name__}"
        )

    @staticmethod
    def _create_inner_instance(
        class_problem: str,
        use_cross_validation: bool,
        **kwargs,
    ) -> Any:
        """Create the appropriate inner blueprint instance."""
        is_regression = class_problem == "regression"
        cat_columns = kwargs.get("cat_columns")
        date_columns = kwargs.get("date_columns")
        time_split_column = kwargs.get("time_split_column")
        stratifier = kwargs.get("stratifier")
        ml_model = kwargs.get("ml_model")
        custom_preprocessor = kwargs.get("custom_preprocessor")
        custom_in_fold_preprocessor = kwargs.get("custom_in_fold_preprocessor")
        custom_last_mile_computation = kwargs.get("custom_last_mile_computation")
        custom_feature_selector = kwargs.get("custom_feature_selector")
        conf_training = kwargs.get("conf_training")
        conf_tuning = kwargs.get("conf_tuning")
        conf_params = kwargs.get("conf_params")
        experiment_tracker = kwargs.get("experiment_tracker")
        single_fold_eval_metric_func = kwargs.get("single_fold_eval_metric_func")
        ensemble_config = kwargs.get("ensemble_config")

        if is_regression and use_cross_validation:
            from bluecast.blueprints.cast_cv_regression import BlueCastCVRegression

            return BlueCastCVRegression(
                class_problem="regression",
                cat_columns=cat_columns,
                stratifier=stratifier,
                conf_training=conf_training,
                conf_tuning=conf_tuning,
                conf_params=conf_params,
                experiment_tracker=experiment_tracker,
                custom_in_fold_preprocessor=custom_in_fold_preprocessor,
                custom_last_mile_computation=custom_last_mile_computation,
                custom_preprocessor=custom_preprocessor,
                custom_feature_selector=custom_feature_selector,
                ml_model=ml_model,
                single_fold_eval_metric_func=single_fold_eval_metric_func,
                ensemble_config=ensemble_config,
            )

        elif is_regression and not use_cross_validation:
            from bluecast.blueprints.cast_regression import BlueCastRegression

            return BlueCastRegression(
                class_problem="regression",
                cat_columns=cat_columns,
                date_columns=date_columns,
                time_split_column=time_split_column,
                ml_model=ml_model,
                custom_in_fold_preprocessor=custom_in_fold_preprocessor,
                custom_last_mile_computation=custom_last_mile_computation,
                custom_preprocessor=custom_preprocessor,
                custom_feature_selector=custom_feature_selector,
                conf_training=conf_training,
                conf_tuning=conf_tuning,
                conf_params=conf_params,
                experiment_tracker=experiment_tracker,
                single_fold_eval_metric_func=single_fold_eval_metric_func,
            )

        elif not is_regression and use_cross_validation:
            from bluecast.blueprints.cast_cv import BlueCastCV

            cls_problem: Any = class_problem
            return BlueCastCV(
                class_problem=cls_problem,
                cat_columns=cat_columns,
                stratifier=stratifier,
                conf_training=conf_training,
                conf_tuning=conf_tuning,
                conf_params=conf_params,
                experiment_tracker=experiment_tracker,
                custom_in_fold_preprocessor=custom_in_fold_preprocessor,
                custom_last_mile_computation=custom_last_mile_computation,
                custom_preprocessor=custom_preprocessor,
                custom_feature_selector=custom_feature_selector,
                ml_model=ml_model,
                single_fold_eval_metric_func=single_fold_eval_metric_func,
                ensemble_config=ensemble_config,
            )

        else:
            from bluecast.blueprints.cast import BlueCast

            cls_problem_single: Any = class_problem
            return BlueCast(
                class_problem=cls_problem_single,
                cat_columns=cat_columns,
                date_columns=date_columns,
                time_split_column=time_split_column,
                ml_model=ml_model,
                custom_in_fold_preprocessor=custom_in_fold_preprocessor,
                custom_last_mile_computation=custom_last_mile_computation,
                custom_preprocessor=custom_preprocessor,
                custom_feature_selector=custom_feature_selector,
                conf_training=conf_training,
                conf_tuning=conf_tuning,
                conf_params=conf_params,
                experiment_tracker=experiment_tracker,
                single_fold_eval_metric_func=single_fold_eval_metric_func,
            )

    def fit(self, df: pd.DataFrame, target_col: str) -> None:
        """Fit the model on training data.

        :param df: DataFrame containing features and target column.
        :param target_col: Name of the target column.
        """
        self._inner.fit(df, target_col=target_col)

    def fit_eval(
        self,
        df: pd.DataFrame,
        target_col: str,
        df_eval: Optional[pd.DataFrame] = None,
        y_eval: Optional[pd.Series] = None,
    ) -> Union[Dict[str, Any], Tuple[float, float]]:
        """Fit and evaluate the model.

        For CV models: uses out-of-fold evaluation (df_eval/y_eval are ignored).
        For single models: requires df_eval and y_eval for holdout evaluation.

        :param df: Training DataFrame containing features and target column.
        :param target_col: Name of the target column.
        :param df_eval: Evaluation DataFrame (only for single-model mode).
        :param y_eval: Evaluation target series (only for single-model mode).
        :returns: Evaluation metrics dict (single model) or (mean, std) tuple (CV).
        """
        if self.use_cross_validation:
            return self._inner.fit_eval(df, target_col=target_col)
        else:
            if df_eval is None or y_eval is None:
                raise ValueError(
                    "df_eval and y_eval are required for single-model fit_eval. "
                    "Use use_cross_validation=True for automatic OOF evaluation."
                )
            return self._inner.fit_eval(df, df_eval, y_eval, target_col=target_col)

    def predict(
        self, df: pd.DataFrame, **kwargs
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray, pd.Series]:
        """Predict on unseen data.

        :param df: DataFrame with features for prediction.
        :param kwargs: Additional keyword arguments passed to the inner model's predict.
        :returns: For classification: (probabilities, classes). For regression: predictions.
        """
        return self._inner.predict(df, **kwargs)

    def predict_proba(self, df: pd.DataFrame, **kwargs) -> Union[np.ndarray, pd.Series]:
        """Predict class probabilities (classification only).

        :param df: DataFrame with features for prediction.
        :returns: Predicted probabilities.
        :raises AttributeError: If the inner model doesn't support predict_proba (regression).
        """
        if self.class_problem == "regression":
            raise AttributeError("predict_proba is not available for regression.")
        return self._inner.predict_proba(df, **kwargs)

    def calibrate(
        self,
        x_calibration: pd.DataFrame,
        y_calibration: pd.Series,
        group_columns: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """Calibrate for conformal prediction.

        :param x_calibration: Calibration features (unseen data).
        :param y_calibration: Calibration targets.
        :param group_columns: Optional column names for group-conditional calibration.
        """
        self._inner.calibrate(x_calibration, y_calibration, **kwargs)

    def predict_sets(
        self,
        df: pd.DataFrame,
        alpha: float = 0.05,
        group_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Predict conformal prediction sets (classification only).

        :param df: Features for prediction.
        :param alpha: Significance level.
        :param group_columns: Optional column names for group-conditional sets.
        """
        if self.class_problem == "regression":
            raise AttributeError(
                "predict_sets is for classification. Use predict_interval for regression."
            )
        return self._inner.predict_sets(df, alpha=alpha)

    def predict_interval(
        self,
        df: pd.DataFrame,
        alphas: Optional[List[float]] = None,
        group_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Predict conformal prediction intervals (regression only).

        :param df: Features for prediction.
        :param alphas: List of significance levels.
        :param group_columns: Optional column names for group-conditional intervals.
        """
        if self.class_problem != "regression":
            raise AttributeError(
                "predict_interval is for regression. Use predict_sets for classification."
            )
        if alphas is None:
            alphas = [0.05]
        return self._inner.predict_interval(df, alphas=alphas)

    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using the fitted preprocessing pipeline (single model only).

        :param df: DataFrame to transform.
        :returns: Transformed DataFrame.
        """
        if self.use_cross_validation:
            raise AttributeError(
                "transform_new_data is not available for CV models. "
                "Use predict() directly instead."
            )
        return self._inner.transform_new_data(df)

    @property
    def experiment_tracker(self) -> ExperimentTracker:
        """Access the experiment tracker."""
        return self._inner.experiment_tracker

    @property
    def eval_metrics(self) -> Optional[Dict[str, Any]]:
        """Access evaluation metrics (single model) or None."""
        return getattr(self._inner, "eval_metrics", None)

    @property
    def bluecast_models(self) -> Optional[list]:
        """Access sub-models (CV only)."""
        return getattr(self._inner, "bluecast_models", None)

    @property
    def inner_model(self) -> Any:
        """Access the inner blueprint instance for advanced use cases."""
        return self._inner

    def show_oof_scores(self, metric: Optional[str] = None) -> Tuple[float, float]:
        """Show out-of-fold scores (CV only).

        :param metric: Metric name. Defaults to 'matthews' for classification, 'RMSE' for regression.
        :returns: (mean, std) of OOF scores.
        """
        if not self.use_cross_validation:
            raise AttributeError("show_oof_scores is only available for CV models.")
        if metric is None:
            metric = "RMSE" if self.class_problem == "regression" else "matthews"
        return self._inner.show_oof_scores(metric=metric)
