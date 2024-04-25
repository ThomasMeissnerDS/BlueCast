"""Run fully configured classification blueprint.

Customization via class attributes is possible. Configs can be instantiated and provided to change Xgboost training.
Default hyperparameter search space is relatively light-weight to speed up the prototyping.
Can deal with binary and multi-class classification problems.
Hyperparameter tuning can be switched off or even strengthened via cross-validation. This behaviour can be controlled
via the config class attributes from config.training_config module.
"""

import warnings
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd

from bluecast.config.training_config import TrainingConfig, XgboostFinalParamConfig
from bluecast.config.training_config import (
    XgboostTuneParamsRegressionConfig as XgboostTuneParamsConfig,
)
from bluecast.conformal_prediction.conformal_prediction_regression import (
    ConformalPredictionRegressionWrapper,
)
from bluecast.evaluation.eval_metrics import eval_regressor
from bluecast.evaluation.shap_values import (
    shap_dependence_plots,
    shap_explanations,
    shap_waterfall_plot,
)
from bluecast.experimentation.tracking import ExperimentTracker
from bluecast.general_utils.general_utils import check_gpu_support, logger
from bluecast.ml_modelling.xgboost_regression import XgboostModelRegression
from bluecast.preprocessing.category_encoder_orchestration import (
    CategoryEncoderOrchestrator,
)
from bluecast.preprocessing.custom import CustomPreprocessing
from bluecast.preprocessing.datetime_features import date_converter
from bluecast.preprocessing.encode_target_labels import TargetLabelEncoder
from bluecast.preprocessing.feature_selection import RFECVSelector
from bluecast.preprocessing.feature_types import FeatureTypeDetector
from bluecast.preprocessing.nulls_and_infs import fill_infinite_values
from bluecast.preprocessing.onehot_encoding import OneHotCategoryEncoder
from bluecast.preprocessing.schema_checks import SchemaDetector
from bluecast.preprocessing.target_encoding import (
    BinaryClassTargetEncoder,
    MultiClassTargetEncoder,
)
from bluecast.preprocessing.train_test_split import train_test_split


class BlueCastRegression:
    """Run fully configured classification blueprint.

    Customization via class attributes is possible. Configs can be instantiated and provided to change Xgboost training.
    Default hyperparameter search space is relatively light-weight to speed up the prototyping.
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
    """

    def __init__(
        self,
        class_problem: Literal["regression"],
        cat_columns: Optional[List[Union[str, float, int]]] = None,
        date_columns: Optional[List[Union[str, float, int]]] = None,
        time_split_column: Optional[str] = None,
        ml_model: Optional[Union[XgboostModelRegression, Any]] = None,
        custom_in_fold_preprocessor: Optional[CustomPreprocessing] = None,
        custom_last_mile_computation: Optional[CustomPreprocessing] = None,
        custom_preprocessor: Optional[CustomPreprocessing] = None,
        custom_feature_selector: Optional[
            Union[RFECVSelector, CustomPreprocessing]
        ] = None,
        conf_training: Optional[TrainingConfig] = None,
        conf_xgboost: Optional[XgboostTuneParamsConfig] = None,
        conf_params_xgboost: Optional[XgboostFinalParamConfig] = None,
        experiment_tracker: Optional[ExperimentTracker] = None,
    ):
        self.class_problem = class_problem
        self.prediction_mode: bool = False
        self.cat_columns = cat_columns
        self.date_columns = date_columns
        self.time_split_column = time_split_column
        self.target_column = "Undefined"
        self.conf_training = conf_training
        self.conf_xgboost = conf_xgboost
        self.conf_params_xgboost = conf_params_xgboost
        self.feat_type_detector: Optional[FeatureTypeDetector] = None
        self.cat_encoder: Optional[
            Union[BinaryClassTargetEncoder, MultiClassTargetEncoder]
        ] = None
        self.onehot_encoder: Optional[OneHotCategoryEncoder] = None
        self.category_encoder_orchestrator: Optional[CategoryEncoderOrchestrator] = None
        self.target_label_encoder: Optional[TargetLabelEncoder] = None
        self.schema_detector: Optional[SchemaDetector] = None
        self.ml_model: Optional[XgboostModelRegression] = ml_model
        self.custom_in_fold_preprocessor = custom_in_fold_preprocessor
        self.custom_last_mile_computation = custom_last_mile_computation
        self.custom_preprocessor = custom_preprocessor
        self.custom_feature_selector = custom_feature_selector
        self.shap_values: Optional[np.ndarray] = None
        self.explainer = None
        self.eval_metrics: Optional[Dict[str, Any]] = None
        self.conformal_prediction_wrapper: Optional[
            ConformalPredictionRegressionWrapper
        ] = None

        if experiment_tracker:
            self.experiment_tracker = experiment_tracker
        else:
            self.experiment_tracker = ExperimentTracker()

        if not self.conf_params_xgboost:
            self.conf_params_xgboost = XgboostFinalParamConfig()

    def initial_checks(self, df: pd.DataFrame) -> None:
        if not self.conf_training:
            self.conf_training = TrainingConfig()
        if not self.conf_training.enable_feature_selection:
            message = """Feature selection is disabled. Update the TrainingConfig param 'enable_feature_selection'
            to enable it or make use of a custom preprocessor to do it manually during the last mile computations step.
            Feature selection is recommended for datasets with many features (>1000). For datasets with a small amount
            of features feature selection is not recommended.
            """
            warnings.warn(message, UserWarning, stacklevel=2)

        if self.conf_training.hypertuning_cv_folds == 1:
            message = """Cross validation is disabled. Update the TrainingConfig param 'hypertuning_cv_folds'
            to enable it. Cross validation is disabled on default to allow fast prototyping. For robust hyperparameter
            tuning using at least 5 folds is recommended."""
            warnings.warn(message, UserWarning, stacklevel=2)

        if (
            self.conf_training.enable_feature_selection
            and not self.custom_feature_selector
        ):
            message = """Feature selection is enabled but no feature selector has been provided. Falling back to
            cross-validated feature elimination. Specifically for small datasets check the logs to verify that not too
            many features have been removed. Otherwise, consider disabling feature selection or providing a custom
            feature selector."""
            warnings.warn(message, UserWarning, stacklevel=2)
        if not self.conf_xgboost:
            message = """No XgboostTuneParamsConfig has been provided. Falling back to default values. Default values
            have been chosen to speed up the prototyping. For robust hyperparameter tuning consider providing a custom
            XgboostTuneParamsConfig with a deeper hyperparameter search space and a custom TrainingConfig to enable
            cross-validation."""
            warnings.warn(message, UserWarning, stacklevel=2)
        if (
            self.conf_training.min_features_to_select >= len(df.columns)
            and self.conf_training.enable_feature_selection
        ):
            message = """The minimum number of features to select is greater or equal to the number of features in
            the dataset while feature selection is enabled. Consider reducing the minimum number of features to
            select or disabling feature selection via TrainingConfig."""
            warnings.warn(message, UserWarning, stacklevel=2)
        if (
            self.conf_training.cat_encoding_via_ml_algorithm
            and self.conf_training.calculate_shap_values
        ):
            self.conf_training.calculate_shap_values = False
            message = """SHAP values cannot be calculated when categorical encoding via ML algorithm is enabled due to
            incompatibility with the shap library. See this GitHub issue for more context:
            https://github.com/slundberg/shap/issues/266

            Calculation of Shap values has been changed to false.
            Consider disabling categorical encoding via ML algorithm in the TrainingConfig if shap values are
            required. Alternatively use Xgboost as a custom model and calculate shap values manually via
            pred_contribs=True."""
            warnings.warn(message, UserWarning, stacklevel=2)
        if self.conf_training.cat_encoding_via_ml_algorithm and self.ml_model:
            message = """Categorical encoding via ML algorithm is enabled. Make sure to handle categorical features
            within the provided ml model or consider disabling categorical encoding via ML algorithm in the
            TrainingConfig alternatively."""
            warnings.warn(message, UserWarning, stacklevel=2)
        if (
            self.conf_training.cat_encoding_via_ml_algorithm
            and self.custom_last_mile_computation
        ):
            message = """Categorical encoding via ML algorithm is enabled. Make sure to handle categorical features
            within the provided last mile computation or consider disabling categorical encoding via ML algorithm in the
            TrainingConfig alternatively."""
            warnings.warn(message, UserWarning, stacklevel=2)
        if self.conf_training.precise_cv_tuning:
            message = """Precise fine tuning has been enabled. Please make sure to transform your data to a normal
            distribution (yeo-johnson). This is an experimental feature as it includes a special
            evaluation (see more in the docs). If you plan to use this feature, please make sure to read the docs."""
            warnings.warn(message, UserWarning, stacklevel=2)
        if (
            self.conf_training.precise_cv_tuning
            and not self.custom_in_fold_preprocessor
        ):
            message = """Precise fine tuning has been enabled, but no custom_in_fold_preprocessor has been provided.
            This will cause long runtimes without benefit. If you plan to execute any overfitting risky preprocessing,
            please consider using custom_in_fold_preprocessor to execute the steps within the cross-validation folds
            using precise_cv_tuning. Otherwise disable precise_cv_tuning to benefit from early pruning of unpromising
            hyperparameter sets."""
            warnings.warn(message, UserWarning, stacklevel=2)
        if (
            self.conf_training.precise_cv_tuning
            and self.conf_training.hypertuning_cv_folds < 2
        ):
            message = """Precise fine tuning has been enabled, but number of hypertuning_cv_folds is less than 2. With
            less than 2 folds precise_cv_tuning will not have any impact. Consider raising the number of folds to two
            or higher or disable precise_cv_tuning."""
            warnings.warn(message, UserWarning, stacklevel=2)

    def fit(self, df: pd.DataFrame, target_col: str) -> None:
        """Train a full ML pipeline."""
        self.target_column = target_col
        check_gpu_support()
        feat_type_detector = FeatureTypeDetector(
            cat_columns=[], num_columns=[], date_columns=[]
        )
        df = feat_type_detector.fit_transform_feature_types(df)
        self.feat_type_detector = feat_type_detector

        self.cat_columns = self.feat_type_detector.cat_columns
        self.date_columns = self.feat_type_detector.date_columns

        if not self.conf_training:
            self.conf_training = TrainingConfig()

        self.initial_checks(df)

        x_train, x_test, y_train, y_test = train_test_split(
            df,
            target_col,
            self.time_split_column,
            self.conf_training.train_size,
            self.conf_training.global_random_state,
            stratify=False,
        )

        if not self.conf_training.autotune_model and self.conf_params_xgboost:
            self.conf_params_xgboost.params["objective"] = (
                self.conf_params_xgboost.params.get("objective", "reg:squarederror")
            )
            self.conf_params_xgboost.params["eval_metric"] = (
                self.conf_params_xgboost.params.get("eval_metric", "rmse")
            )

        if self.custom_preprocessor:
            x_train, y_train = self.custom_preprocessor.fit_transform(
                x_train.copy(), y_train
            )
            x_test, y_test = self.custom_preprocessor.transform(
                x_test.copy(), y_test, predicton_mode=False
            )
            feat_type_detector = FeatureTypeDetector(
                cat_columns=[], num_columns=[], date_columns=[]
            )
            _ = feat_type_detector.fit_transform_feature_types(x_train)
            x_train, y_train = x_train.reset_index(drop=True), y_train.reset_index(
                drop=True
            )
            x_test, y_test = x_test.reset_index(drop=True), y_test.reset_index(
                drop=True
            )
            if target_col in feat_type_detector.cat_columns:
                feat_type_detector.cat_columns.remove(target_col)

        x_train, x_test = fill_infinite_values(x_train), fill_infinite_values(x_test)
        x_train, x_test = date_converter(
            x_train, self.date_columns, date_parts=["month", "day", "dayofweek", "hour"]
        ), date_converter(
            x_test, self.date_columns, date_parts=["month", "day", "dayofweek", "hour"]
        )

        self.schema_detector = SchemaDetector()
        self.schema_detector.fit(x_train)
        x_train = self.schema_detector.transform(x_train)
        x_test = self.schema_detector.transform(x_test)

        if (
            self.cat_columns is not None
            and not self.conf_training.cat_encoding_via_ml_algorithm
        ):
            self.category_encoder_orchestrator = CategoryEncoderOrchestrator(
                self.target_column
            )
            self.category_encoder_orchestrator.fit(
                x_train,
                feat_type_detector.cat_columns,
                self.conf_training.cardinality_threshold_for_onehot_encoding,
            )

            self.onehot_encoder = OneHotCategoryEncoder(
                self.category_encoder_orchestrator.to_onehot_encode, self.target_column
            )
            x_train = self.onehot_encoder.fit_transform(x_train.copy(), y_train)
            x_test = self.onehot_encoder.transform(x_test.copy())

        if (
            self.cat_columns is not None
            and self.class_problem == "regression"
            and self.category_encoder_orchestrator
            and not self.conf_training.cat_encoding_via_ml_algorithm
        ):
            self.cat_encoder = BinaryClassTargetEncoder(
                self.category_encoder_orchestrator.to_target_encode
            )
            x_train = self.cat_encoder.fit_target_encode_binary_class(
                x_train.copy(), y_train
            )
            x_test = self.cat_encoder.transform_target_encode_binary_class(
                x_test.copy()
            )
        elif self.conf_training.cat_encoding_via_ml_algorithm:
            x_train[self.cat_columns] = x_train[self.cat_columns].astype("category")
            x_test[self.cat_columns] = x_test[self.cat_columns].astype("category")

        if self.custom_last_mile_computation:
            x_train, y_train = self.custom_last_mile_computation.fit_transform(
                x_train.copy(), y_train
            )
            x_test, y_test = self.custom_last_mile_computation.transform(
                x_test.copy(), y_test, predicton_mode=False
            )

        if not self.custom_feature_selector:
            self.custom_feature_selector = RFECVSelector(
                random_state=self.conf_training.global_random_state,
                min_features_to_select=self.conf_training.min_features_to_select,
                class_problem=self.class_problem,
            )

        if self.conf_training.enable_feature_selection:
            x_train, y_train = self.custom_feature_selector.fit_transform(
                x_train.copy(), y_train
            )
            x_test, _ = self.custom_feature_selector.transform(
                x_test.copy(), predicton_mode=False
            )

        if not self.ml_model:
            self.ml_model = XgboostModelRegression(
                self.class_problem,
                conf_training=self.conf_training,
                conf_xgboost=self.conf_xgboost,
                conf_params_xgboost=self.conf_params_xgboost,
                experiment_tracker=self.experiment_tracker,
                custom_in_fold_preprocessor=self.custom_in_fold_preprocessor,
            )
        self.ml_model.fit(x_train, x_test, y_train, y_test)

        if self.custom_in_fold_preprocessor:
            x_test, _ = self.custom_in_fold_preprocessor.transform(
                x_test.copy(), None, predicton_mode=True
            )

        if self.conf_training and self.conf_training.calculate_shap_values:
            shap_values, explainer = shap_explanations(self.ml_model.model, x_test)
            if self.conf_training.store_shap_values_in_instance:
                self.shap_values = shap_values
            shap_waterfall_plot(
                explainer, self.conf_training.shap_waterfall_indices, self.class_problem
            )
            shap_dependence_plots(
                shap_values,
                x_test,
                self.conf_training.show_dependence_plots_of_top_n_features,
            )
        self.prediction_mode = True

    def fit_eval(
        self,
        df: pd.DataFrame,
        df_eval: pd.DataFrame,
        target_eval: pd.Series,
        target_col: str,
    ) -> Dict[str, Any]:
        """Train a full ML pipeline and evaluate on a holdout set.

        This is a convenience function to train and evaluate on a holdout set. It is recommended to use this for model
        exploration. On production the simple fit() function should be used.
        :param :df: Takes a pandas DataFrame containing the training data and the targets.
        :param :df_eval: Takes a pandas DataFrame containing the evaluation data, but not the targets.
        :param :target_eval: Takes a pandas Series containing the evaluation targets.
        :param :target_col: Takes a string containing the name of the target column inside the training data df.
        """
        self.fit(df, target_col)
        y_preds = self.predict(df_eval)

        eval_dict = eval_regressor(target_eval.values, y_preds)
        self.eval_metrics = eval_dict

        if not self.conf_training:
            raise ValueError("Could not find any training config")

        if not self.conf_params_xgboost:
            raise ValueError("Could not find Xgboost params")

        if len(self.experiment_tracker.experiment_id) == 0:
            self.experiment_tracker.experiment_id.append(0)

        # enrich experiment tracker
        for metric, higher_is_better in zip(
            [
                "mae",
                "r2_score",
                "MSE",
                "RMSE",
                "median_absolute_error",
            ],
            [False, False, False, False, False],
        ):
            self.experiment_tracker.add_results(
                experiment_id=self.experiment_tracker.experiment_id[-1],
                score_category="oof_score",
                training_config=self.conf_training,
                model_parameters=self.conf_params_xgboost.params,  # noqa
                eval_scores=self.eval_metrics["RMSE"],
                metric_used=metric,
                metric_higher_is_better=higher_is_better,
            )
        return eval_dict

    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data according to preprocessing pipeline."""
        check_gpu_support()
        if not self.feat_type_detector:
            raise Exception("Feature type converter could not be found.")

        if not self.conf_training:
            raise Exception("Training configuration could not be found.")

        df = self.feat_type_detector.transform_feature_types(
            df, ignore_cols=[self.target_column]
        )

        if self.custom_preprocessor:
            df, _ = self.custom_preprocessor.transform(df, predicton_mode=True)
            df = df.reset_index(drop=True)

        df = fill_infinite_values(df)
        df = date_converter(
            df, self.date_columns, date_parts=["month", "day", "dayofweek", "hour"]
        )

        if self.schema_detector:
            df = self.schema_detector.transform(df)

        if (
            self.cat_columns is not None
            and self.onehot_encoder
            and not self.conf_training.cat_encoding_via_ml_algorithm
        ):
            df = self.onehot_encoder.transform(df.copy())

        if (
            self.cat_columns
            and self.cat_encoder
            and self.class_problem == "regression"
            and isinstance(self.cat_encoder, BinaryClassTargetEncoder)
            and self.category_encoder_orchestrator
            and not self.conf_training.cat_encoding_via_ml_algorithm
        ):
            df = self.cat_encoder.transform_target_encode_binary_class(df.copy())
        elif self.conf_training.cat_encoding_via_ml_algorithm:
            df[self.cat_columns] = df[self.cat_columns].astype("category")

        if self.custom_last_mile_computation:
            df, _ = self.custom_last_mile_computation.transform(
                df.copy(), predicton_mode=True
            )

        if self.custom_feature_selector and self.conf_training.enable_feature_selection:
            df, _ = self.custom_feature_selector.transform(
                df.copy(), predicton_mode=True
            )

        return df

    def predict(self, df: pd.DataFrame, save_shap_values: bool = False) -> np.ndarray:
        """Predict on unseen data.

        Return the predicted probabilities and the predicted classes:
        y_probs, y_classes = predict(df)
        :param df: Pandas DataFrame with unseen data
        :param save_shap_values: If True, calculates and saves shap values, so they can be used to plot
            waterfall plots for selected rows o demand.
        """
        if not self.ml_model:
            raise Exception("Ml model could not be found")

        if not self.feat_type_detector:
            raise Exception("Feature type converter could not be found.")

        if not self.conf_training:
            raise ValueError("conf_training is None")

        df = self.transform_new_data(df)

        logger(f"{datetime.utcnow()}: Predicting...")
        y_preds = self.ml_model.predict(df)

        if save_shap_values:
            self.shap_values, self.explainer = shap_explanations(
                self.ml_model.model, df
            )

        return y_preds

    def calibrate(
        self, x_calibration: pd.DataFrame, y_calibration: pd.Series, **kwargs
    ) -> None:
        """Calibrate the model.

        Via this function the nonconformity measures are taken and used to predict prediction intervals vis the
        predict_interval function.
        :param: x_calibration: Pandas DataFrame without target column, that has not been seen by the model during
            training.
        :param y_calibration: Pandas Series holding the target value, hat has not been seen by the model during
            training.
        """
        x_calibration = self.transform_new_data(x_calibration)
        self.conformal_prediction_wrapper = ConformalPredictionRegressionWrapper(
            self.ml_model, **kwargs
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
            df = self.transform_new_data(df)
            pred_interval = self.conformal_prediction_wrapper.predict_interval(
                df, alphas=alphas
            )
            return pred_interval
        else:
            raise ValueError(
                """This instance has not been calibrated yet. Make use of calibrate to fit the
            ConformalPredictionWrapper."""
            )
