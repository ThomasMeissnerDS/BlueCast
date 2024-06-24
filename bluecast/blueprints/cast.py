"""Run fully configured classification blueprint.

Customization via class attributes is possible. Configs can be instantiated and provided to change Xgboost training.
Default hyperparameter search space is relatively light-weight to speed up the prototyping.
Can deal with binary and multi-class classification problems.
Hyperparameter tuning can be switched off or even strengthened via cross-validation. This behaviour can be controlled
via the config class attributes from config.training_config module.
"""

import logging
import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from bluecast.config.training_config import (
    TrainingConfig,
    XgboostFinalParamConfig,
    XgboostTuneParamsConfig,
)
from bluecast.conformal_prediction.conformal_prediction import (
    ConformalPredictionWrapper,
)
from bluecast.evaluation.eval_metrics import ClassificationEvalWrapper, eval_classifier
from bluecast.evaluation.shap_values import (
    shap_dependence_plots,
    shap_explanations,
    shap_waterfall_plot,
)
from bluecast.experimentation.tracking import ExperimentTracker
from bluecast.general_utils.general_utils import check_gpu_support
from bluecast.ml_modelling.xgboost import XgboostModel
from bluecast.preprocessing.category_encoder_orchestration import (
    CategoryEncoderOrchestrator,
)
from bluecast.preprocessing.custom import CustomPreprocessing
from bluecast.preprocessing.datetime_features import date_converter
from bluecast.preprocessing.encode_target_labels import TargetLabelEncoder
from bluecast.preprocessing.feature_selection import BoostaRootaWrapper
from bluecast.preprocessing.feature_types import FeatureTypeDetector
from bluecast.preprocessing.nulls_and_infs import fill_infinite_values
from bluecast.preprocessing.onehot_encoding import OneHotCategoryEncoder
from bluecast.preprocessing.schema_checks import SchemaDetector
from bluecast.preprocessing.target_encoding import (
    BinaryClassTargetEncoder,
    MultiClassTargetEncoder,
)
from bluecast.preprocessing.train_test_split import train_test_split


class BlueCast:
    """Run fully configured classification blueprint.

    Customization via class attributes is possible. Configs can be instantiated and provided to change Xgboost training.
    Default hyperparameter search space is relatively light-weight to speed up the prototyping.
    :param :class_problem: Takes a string containing the class problem type. Either "binary" or "multiclass".
    :param :target_column: Takes a string containing the name of the target column.
    :param :cat_columns: Takes a list of strings containing the names of the categorical columns. If not provided,
        BlueCast will infer these automatically.
    :param :date_columns: Takes a list of strings containing the names of the date columns. If not provided,
        BlueCast will infer these automatically.
    :param :time_split_column: Takes a string containing the name of the time split column. If not provided,
        BlueCast will not split the data by time or order, but do a random split instead.
    :param :ml_model: Takes an instance of a XgboostModel class. If not provided, BlueCast will instantiate one.
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
           Default is matthews_corrcoef. This function is used to calculate the evaluation metric for each fold during
           hyperparameter tuning when hyperparameter_tuning_rounds = 1 (default). Lower must be better.
    """

    def __init__(
        self,
        class_problem: Literal["binary", "multiclass"],
        cat_columns: Optional[List[Union[str, float, int]]] = None,
        date_columns: Optional[List[Union[str, float, int]]] = None,
        time_split_column: Optional[str] = None,
        ml_model: Optional[Union[XgboostModel, Any]] = None,
        custom_in_fold_preprocessor: Optional[CustomPreprocessing] = None,
        custom_last_mile_computation: Optional[CustomPreprocessing] = None,
        custom_preprocessor: Optional[CustomPreprocessing] = None,
        custom_feature_selector: Optional[
            Union[BoostaRootaWrapper, CustomPreprocessing]
        ] = None,
        conf_training: Optional[TrainingConfig] = None,
        conf_xgboost: Optional[XgboostTuneParamsConfig] = None,
        conf_params_xgboost: Optional[XgboostFinalParamConfig] = None,
        experiment_tracker: Optional[ExperimentTracker] = None,
        single_fold_eval_metric_func: Optional[ClassificationEvalWrapper] = None,
    ):
        self.class_problem = class_problem
        self.prediction_mode: bool = False

        if not cat_columns:
            self.cat_columns = []
        else:
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
        self.ml_model: Optional[XgboostModel] = ml_model
        self.custom_in_fold_preprocessor = custom_in_fold_preprocessor
        self.custom_last_mile_computation = custom_last_mile_computation
        self.custom_preprocessor = custom_preprocessor
        self.custom_feature_selector = custom_feature_selector
        self.shap_values: Optional[np.ndarray] = None
        self.explainer = None
        self.eval_metrics: Optional[Dict[str, Any]] = None
        self.conformal_prediction_wrapper: Optional[ConformalPredictionWrapper] = None
        self.single_fold_eval_metric_func = single_fold_eval_metric_func

        if experiment_tracker:
            self.experiment_tracker = experiment_tracker
        else:
            self.experiment_tracker = ExperimentTracker()

        if not self.conf_params_xgboost:
            self.conf_params_xgboost = XgboostFinalParamConfig()

        if not self.conf_training:
            self.conf_training = TrainingConfig()

        if not self.conf_xgboost:
            self.conf_xgboost = XgboostTuneParamsConfig()

        if not self.single_fold_eval_metric_func:
            self.single_fold_eval_metric_func = ClassificationEvalWrapper()

        logging.basicConfig(
            filename=self.conf_training.logging_file_path,
            filemode="w",
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=logging.INFO,
            # stream=sys.stdout,
            force=True,
        )

        logging.info("BlueCast blueprint initialized.")

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
        if self.class_problem == "binary" and df[self.target_column].nunique() > 2:
            message = """During class instantiation class_problem = 'binary' has been passed. However more than 2
            unique target classes have been found. Did you mean 'multiclass' instead?"""
            warnings.warn(message, UserWarning, stacklevel=2)
        if self.class_problem == "multiclass" and df[self.target_column].nunique() < 3:
            message = """During class instantiation class_problem = 'multiclass' has been passed. However less than 3
            unique target classes have been found. Did you mean 'binary' instead?"""
            warnings.warn(message, UserWarning, stacklevel=2)

    def fit(self, df: pd.DataFrame, target_col: str) -> None:
        """Train a full ML pipeline."""

        self.target_column = target_col

        feat_type_detector = FeatureTypeDetector(
            cat_columns=self.cat_columns, num_columns=[], date_columns=[]
        )
        df = feat_type_detector.fit_transform_feature_types(df)
        self.feat_type_detector = feat_type_detector

        if self.feat_type_detector.cat_columns:
            if self.target_column in self.feat_type_detector.cat_columns:
                self.target_label_encoder = TargetLabelEncoder()
                df.loc[:, self.target_column] = (
                    self.target_label_encoder.fit_transform_target_labels(
                        df.loc[:, self.target_column]
                    )
                )

        self.cat_columns = self.feat_type_detector.cat_columns
        self.date_columns = self.feat_type_detector.date_columns

        if not self.conf_training:
            self.conf_training = TrainingConfig()

        check_gpu_support()

        self.initial_checks(df)

        x_train, x_test, y_train, y_test = train_test_split(
            df,
            target_col,
            self.time_split_column,
            self.conf_training.train_size,
            self.conf_training.global_random_state,
            self.conf_training.train_split_stratify,
        )

        if not self.conf_training.autotune_model and self.conf_params_xgboost:
            self.conf_params_xgboost.params["num_class"] = (
                self.conf_params_xgboost.params.get("num_class", y_test.nunique())
            )

        if self.custom_preprocessor:
            x_train, y_train = self.custom_preprocessor.fit_transform(x_train, y_train)
            x_test, y_test = self.custom_preprocessor.transform(
                x_test, y_test, predicton_mode=False
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
            x_train,
            self.date_columns,
            date_parts=["year", "week_of_year", "month", "day", "dayofweek", "hour"],
        ), date_converter(
            x_test,
            self.date_columns,
            date_parts=["year", "week_of_year", "month", "day", "dayofweek", "hour"],
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
            and self.class_problem == "binary"
            and self.category_encoder_orchestrator
            and not self.conf_training.cat_encoding_via_ml_algorithm
        ):
            self.cat_encoder = BinaryClassTargetEncoder(
                self.category_encoder_orchestrator.to_target_encode
            )
            x_train = self.cat_encoder.fit_target_encode_binary_class(x_train, y_train)
            x_test = self.cat_encoder.transform_target_encode_binary_class(x_test)
        elif (
            self.cat_columns is not None
            and self.class_problem == "multiclass"
            and self.category_encoder_orchestrator
            and not self.conf_training.cat_encoding_via_ml_algorithm
        ):
            self.cat_encoder = MultiClassTargetEncoder(
                self.category_encoder_orchestrator.to_target_encode, self.target_column
            )
            x_train = self.cat_encoder.fit_target_encode_multiclass(
                x_train.copy(), y_train
            )
            x_test = self.cat_encoder.transform_target_encode_multiclass(x_test.copy())
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
            self.custom_feature_selector = BoostaRootaWrapper(
                random_state=self.conf_training.global_random_state,
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
            self.ml_model = XgboostModel(
                self.class_problem,
                conf_training=self.conf_training,
                conf_xgboost=self.conf_xgboost,
                conf_params_xgboost=self.conf_params_xgboost,
                experiment_tracker=self.experiment_tracker,
                custom_in_fold_preprocessor=self.custom_in_fold_preprocessor,
                cat_columns=self.cat_columns,
                single_fold_eval_metric_func=self.single_fold_eval_metric_func,
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
        y_probs, y_classes = self.predict(df_eval)

        if self.feat_type_detector:
            if self.target_label_encoder and self.feat_type_detector:
                eval_df = pd.DataFrame(target_eval.values, columns=[target_col])
                y_true = self.target_label_encoder.transform_target_labels(
                    eval_df, target_col
                )
            else:
                y_true = target_eval.values
        else:
            y_true = target_eval.values

        eval_dict = eval_classifier(y_true, y_probs, y_classes)
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
                "accuracy",
                "recall",
                "f1_score_weighted",
                "log_loss",
                "balanced_logloss",
                "roc_auc",
                "matthews",
            ],
            [True, True, True, False, False, True, True],
        ):
            self.experiment_tracker.add_results(
                experiment_id=self.experiment_tracker.experiment_id[-1],
                score_category="oof_score",
                training_config=self.conf_training,
                model_parameters=self.conf_params_xgboost.params,  # noqa
                eval_scores=self.eval_metrics["accuracy"],
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
            df,
            self.date_columns,
            date_parts=["year", "week_of_year", "month", "day", "dayofweek", "hour"],
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
            and self.class_problem == "binary"
            and isinstance(self.cat_encoder, BinaryClassTargetEncoder)
            and not self.conf_training.cat_encoding_via_ml_algorithm
        ):
            df = self.cat_encoder.transform_target_encode_binary_class(df.copy())
        elif (
            self.cat_columns
            and self.cat_encoder
            and self.class_problem == "multiclass"
            and isinstance(self.cat_encoder, MultiClassTargetEncoder)
            and not self.conf_training.cat_encoding_via_ml_algorithm
        ):
            df = self.cat_encoder.transform_target_encode_multiclass(df.copy())
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

    def predict(
        self, df: pd.DataFrame, save_shap_values: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
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

        check_gpu_support()
        df = self.transform_new_data(df)

        logging.info("Predicting...")
        y_probs, y_classes = self.ml_model.predict(df)
        if save_shap_values:
            self.shap_values, self.explainer = shap_explanations(
                self.ml_model.model, df
            )

        if self.feat_type_detector.cat_columns:
            if (
                self.target_column in self.feat_type_detector.cat_columns
                and self.target_label_encoder
                and self.feat_type_detector
            ):
                y_classes = self.target_label_encoder.label_encoder_reverse_transform(
                    pd.Series(y_classes)
                )

        return y_probs, y_classes

    def predict_proba(
        self, df: pd.DataFrame, save_shap_values: bool = False
    ) -> np.ndarray:
        """Predict class scores on unseen data.

        Return the predicted probabilities and the predicted classes:
        y_probs = predict_proba(df)
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

        logging.info("Predicting...")
        y_probs, _y_classes = self.ml_model.predict(df)
        if save_shap_values:
            self.shap_values, self.explainer = shap_explanations(
                self.ml_model.model, df
            )

        return y_probs

    def calibrate(
        self, x_calibration: pd.DataFrame, y_calibration: pd.Series, **kwargs
    ) -> None:
        """Calibrate the model.

        Via this function the nonconformity measures are taken and used to predict calibrated sets via the
        predict_sets function, or to return p-values of a row for being the class via the predict_p_values function.
        :param: x_calibration: Pandas DataFrame without target column, that has not been seen by the model during
            training.
        :param y_calibration: Pandas Series holding the target value, hat has not been seen by the model during
            training.
        """
        x_calibration = self.transform_new_data(x_calibration)

        if self.target_label_encoder:
            x_calibration[self.target_column] = y_calibration
            x_calibration = self.target_label_encoder.transform_target_labels(
                x_calibration, self.target_column
            )
            y_calibration = x_calibration.pop(self.target_column)

        self.conformal_prediction_wrapper = ConformalPredictionWrapper(
            self.ml_model, **kwargs
        )
        self.conformal_prediction_wrapper.calibrate(x_calibration, y_calibration)

    def predict_p_values(self, df: pd.DataFrame) -> np.ndarray:
        """Create p-values for each class.

        The p_values indicate the probability of being the correct class.
        :param df: Pandas DataFrame holding unseen data
        :returns: Numpy array where each column holds p-values of a row being the class. If string labels were passed
            each column maps the index of target_label_encoder.target_label_mapping stored in this class.
        """
        if self.conformal_prediction_wrapper:
            df = self.transform_new_data(df)
            pred_interval = self.conformal_prediction_wrapper.predict_interval(df)
            return pred_interval
        else:
            raise ValueError(
                """This instance has not been calibrated yet. Make use of calibrate to fit the
            ConformalPredictionWrapper."""
            )

    def predict_sets(self, df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
        """Create prediction sets based on a certain confidence level.

        Conformal prediction guarantees, that the correct label is present in the prediction sets with a probability of
        1 - alpha.
        :param df: Pandas DataFrame holding unseen data
        :param alpha: Float indicating the desired confidence level.
        :returns a Pandas DataFrame with a column called 'prediction_set' holding a nested set with predicted classes.
        """
        if self.conformal_prediction_wrapper:
            check_gpu_support()
            df = self.transform_new_data(df)
            pred_sets = self.conformal_prediction_wrapper.predict_sets(df, alpha)
            # transform numerical values back to original strings for the end user
            if self.target_label_encoder:
                reverse_mapping = {
                    value: key
                    for key, value in self.target_label_encoder.target_label_mapping.items()
                }

                string_pred_sets = []
                for numerical_set in pred_sets:
                    # Convert numerical labels to string labels
                    string_set = {reverse_mapping[label] for label in numerical_set}
                    string_pred_sets.append(string_set)
                return pd.DataFrame({"prediction_set": string_pred_sets})
            else:
                return pd.DataFrame({"prediction_set": pred_sets})
        else:
            raise ValueError(
                """This instance has not been calibrated yet. Make use of calibrate to fit the
            ConformalPredictionWrapper."""
            )
