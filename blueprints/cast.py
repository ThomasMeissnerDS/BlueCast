import numpy as np
import pandas as pd

from config.training_config import TrainingConfig, XgboostTuneParamsConfig, XgboostFinalParamConfig
from ml_modelling.xgboost import XgboostModel
from preprocessing.datetime_features import date_converter
from preprocessing.general_utils import check_gpu_support, FeatureTypeDetector
from preprocessing.nulls_and_infs import fill_infinite_values
from preprocessing.target_encoding import BinaryClassTargetEncoder, MultiClassTargetEncoder
from preprocessing.train_test_split import train_test_split_cross, train_test_split_time
from typing import List, Literal, Optional, Tuple, Union


class BlueCast:
    def __init__(self,
                 class_problem: Literal["binary", "multiclass"],
                 target_column: str,
                 cat_columns: Optional[List[str]],
                 date_columns: Optional[List[str]],
                 time_split_column: Optional[str] = None,
                 conf_training: Optional[TrainingConfig] = None,
                 conf_xgboost: Optional[XgboostTuneParamsConfig] = None,
                 conf_params_xgboost: Optional[XgboostFinalParamConfig] = None
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
        self.cat_encoder: Optional[Union[BinaryClassTargetEncoder, MultiClassTargetEncoder]] = None
        self.ml_model: Optional[XgboostModel] = None

    def fit(self, df: pd.DataFrame, target_col: str):
        check_gpu_support()
        feat_detector = FeatureTypeDetector()
        df = feat_detector.fit_transform_feature_types(df)
        self.cat_columns = feat_detector.cat_columns
        self.date_columns = feat_detector.date_columns

        df = fill_infinite_values(df)
        df = date_converter(df, self.date_columns)
        if self.cat_columns is not None and self.class_problem == "binary":
            self.cat_encoder = BinaryClassTargetEncoder(self.cat_columns)
            df = self.cat_encoder.fit_target_encode_binary_class(df, df[target_col])
        elif self.cat_columns is not None and self.class_problem == "multiclass":
            self.cat_encoder = MultiClassTargetEncoder(self.cat_columns)
            df = self.cat_encoder.fit_target_encode_multiclass(df, df[target_col])

        if self.time_split_column is not None:
            x_train, y_train, x_test, y_test = train_test_split_time(df, target_col, self.time_split_column)
        else:
            x_train, y_train, x_test, y_test = train_test_split_cross(df, target_col)

        self.ml_model = XgboostModel(self.class_problem,
                                     conf_training=self.conf_training,
                                     conf_xgboost=self.conf_xgboost,
                                     conf_params_xgboost=self.conf_params_xgboost)
        self.ml_model = self.ml_model.fit(x_train, y_train, x_test, y_test)
        self.prediction_mode = True

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        check_gpu_support()
        df = fill_infinite_values(df)
        df = date_converter(df, self.date_columns)

        if self.cat_columns is not None and self.class_problem == "binary":
            df = self.cat_encoder.transform_target_encode_binary_class(df)
        elif self.cat_columns is not None and self.class_problem == "multiclass":
            df = self.cat_encoder.transform_target_encode_multiclass(df)

        y_probs, y_classes = self.ml_model.predict(df)

        return y_probs, y_classes

