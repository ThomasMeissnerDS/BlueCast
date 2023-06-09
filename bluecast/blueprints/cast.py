"""Run fully configured classification blueprint.

Customization via class attributes is possible. Configs can be instantiated and provided to change Xgboost training.
Default hyperparameter search space is relatively light-weight to speed up the prototyping.
Can deal with binary and multi-class classification problems.
Hyperparameter tuning can be switched off or even strengthened via cross-validation. This behaviour can be controlled
via the config class attributes from config.training_config module.
"""
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from bluecast.config.training_config import (
    TrainingConfig,
    XgboostFinalParamConfig,
    XgboostTuneParamsConfig,
)
from bluecast.evaluation.eval_metrics import eval_classifier
from bluecast.evaluation.shap_values import shap_explanations
from bluecast.general_utils.general_utils import check_gpu_support
from bluecast.ml_modelling.xgboost import XgboostModel
from bluecast.preprocessing.custom import CustomPreprocessing
from bluecast.preprocessing.datetime_features import date_converter
from bluecast.preprocessing.encode_target_labels import TargetLabelEncoder
from bluecast.preprocessing.feature_types import FeatureTypeDetector
from bluecast.preprocessing.nulls_and_infs import fill_infinite_values
from bluecast.preprocessing.schema_checks import SchemaDetector
from bluecast.preprocessing.target_encoding import (
    BinaryClassTargetEncoder,
    MultiClassTargetEncoder,
)
from bluecast.preprocessing.train_test_split import (
    train_test_split_cross,
    train_test_split_time,
)


class BlueCast:
    """Run fully configured classification blueprint.

    Customization via class attributes is possible. Configs can be instantiated and provided to change Xgboost training.
    Default hyperparameter search space is relatively light-weight to speed up the prototyping.
    :param :class_problem: Takes a string containing the class problem type. Either "binary" or "multiclass".
    :param :target_column: Takes a string containing the name of the target column.
    :param :cat_columns: Takes a list of strings containing the names of the categorical columns. If not provided,
    BlueCast will infer these automaically.
    :param :date_columns: Takes a list of strings containing the names of the date columns. If not provided,
    BlueCast will infer these automaically.
    :param :time_split_column: Takes a string containing the name of the time split column. If not provided,
    BlueCast will not split the data by time or order, but do a random split instead.
    :param :ml_model: Takes an instance of a XgboostModel class. If not provided, BlueCast will instantiate one.
    This is an API to pass any model class. Inherit the baseclass from ml_modelling.base_model.BaseModel.
    """

    def __init__(
        self,
        class_problem: Literal["binary", "multiclass"],
        target_column: Union[str, float, int],
        cat_columns: Optional[List[Union[str, float, int]]] = None,
        date_columns: Optional[List[Union[str, float, int]]] = None,
        time_split_column: Optional[str] = None,
        ml_model: Optional[Union[XgboostModel, Any]] = None,
        custom_preprocessor: Optional[CustomPreprocessing] = None,
        conf_training: Optional[TrainingConfig] = None,
        conf_xgboost: Optional[XgboostTuneParamsConfig] = None,
        conf_params_xgboost: Optional[XgboostFinalParamConfig] = None,
    ):
        self.class_problem = class_problem
        self.prediction_mode: bool = False
        self.cat_columns = cat_columns
        self.date_columns = date_columns
        self.time_split_column = time_split_column
        self.target_column = target_column
        self.conf_training = conf_training
        self.conf_xgboost = conf_xgboost
        self.conf_params_xgboost = conf_params_xgboost
        self.feat_type_detector: Optional[FeatureTypeDetector] = None
        self.cat_encoder: Optional[
            Union[BinaryClassTargetEncoder, MultiClassTargetEncoder]
        ] = None
        self.target_label_encoder: Optional[TargetLabelEncoder] = None
        self.schema_detector: Optional[SchemaDetector] = None
        self.ml_model: Optional[XgboostModel] = ml_model
        self.custom_preprocessor = custom_preprocessor
        self.shap_values: Optional[np.ndarray] = None

    def fit(self, df: pd.DataFrame, target_col: str) -> None:
        """Train a full ML pipeline."""
        check_gpu_support()
        self.feat_type_detector = FeatureTypeDetector()
        df = self.feat_type_detector.fit_transform_feature_types(df)

        if self.feat_type_detector.cat_columns:
            if self.target_column in self.feat_type_detector.cat_columns:
                self.target_label_encoder = TargetLabelEncoder()
                df[
                    self.target_column
                ] = self.target_label_encoder.fit_transform_target_labels(
                    df[self.target_column]
                )

        self.cat_columns = self.feat_type_detector.cat_columns
        self.date_columns = self.feat_type_detector.date_columns

        df = fill_infinite_values(df)
        df = date_converter(df, self.date_columns)

        if self.time_split_column is not None:
            x_train, x_test, y_train, y_test = train_test_split_time(
                df, target_col, self.time_split_column
            )
        else:
            x_train, x_test, y_train, y_test = train_test_split_cross(df, target_col)

        self.schema_detector = SchemaDetector()
        self.schema_detector.fit(x_train)
        x_test = self.schema_detector.transform(x_test)

        if self.cat_columns is not None and self.class_problem == "binary":
            self.cat_encoder = BinaryClassTargetEncoder(self.cat_columns)
            x_train = self.cat_encoder.fit_target_encode_binary_class(x_train, y_train)
            x_test = self.cat_encoder.transform_target_encode_binary_class(x_test)
        elif self.cat_columns is not None and self.class_problem == "multiclass":
            self.cat_encoder = MultiClassTargetEncoder(self.cat_columns)
            x_train = self.cat_encoder.fit_target_encode_multiclass(x_train, y_train)
            x_test = self.cat_encoder.transform_target_encode_multiclass(x_test)

        if self.custom_preprocessor:
            x_train, y_train = self.custom_preprocessor.fit_transform(x_train, y_train)
            x_test, y_test = self.custom_preprocessor.transform(
                x_test, y_train, predicton_mode=False
            )

        if not self.ml_model:
            self.ml_model = XgboostModel(
                self.class_problem,
                conf_training=self.conf_training,
                conf_xgboost=self.conf_xgboost,
                conf_params_xgboost=self.conf_params_xgboost,
            )
        self.ml_model.fit(x_train, x_test, y_train, y_test)
        if self.conf_training and self.conf_training.calculate_shap_values:
            self.shap_values = shap_explanations(self.ml_model.model, x_test, "tree")
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
        eval_dict = eval_classifier(target_eval.values, y_classes)
        return eval_dict

    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data according to preprocessing pipeline."""
        check_gpu_support()
        if not self.feat_type_detector:
            raise Exception("Feature type converter could not be found.")

        df = self.feat_type_detector.transform_feature_types(
            df, ignore_cols=[self.target_column]
        )
        df = fill_infinite_values(df)
        df = date_converter(df, self.date_columns)

        if self.schema_detector:
            df = self.schema_detector.transform(df)

        if (
            self.cat_columns
            and self.cat_encoder
            and self.class_problem == "binary"
            and isinstance(self.cat_encoder, BinaryClassTargetEncoder)
        ):
            df = self.cat_encoder.transform_target_encode_binary_class(df)
        elif (
            self.cat_columns
            and self.cat_encoder
            and self.class_problem == "multiclass"
            and isinstance(self.cat_encoder, MultiClassTargetEncoder)
        ):
            df = self.cat_encoder.transform_target_encode_multiclass(df)

        if self.custom_preprocessor:
            df, _ = self.custom_preprocessor.transform(df, predicton_mode=True)
        return df

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict on unseen data.

        Return the predicted probabilities and the predicted classes:
        y_probs, y_classes = predict(df)
        """
        if not self.ml_model:
            raise Exception("Ml model could not be found")

        if not self.feat_type_detector:
            raise Exception("Feature type converter could not be found.")

        check_gpu_support()
        df = self.transform_new_data(df)

        print("Predicting...")
        y_probs, y_classes = self.ml_model.predict(df)

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
