import logging
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from category_encoders import OneHotEncoder, TargetEncoder
from category_encoders.wrapper import NestedCVWrapper
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", "is_categorical_dtype")


class BinaryClassTargetEncoder:
    """Target encode categorical features in the context of binary classification using NestedCVWrapper."""

    def __init__(
        self, cat_columns: List[Union[str, float, int]], random_state: int = 200
    ):
        self.encoders: Dict[str, NestedCVWrapper] = {}
        self.prediction_mode: bool = False
        self.cat_columns = cat_columns
        self.random_state = random_state

    def fit_target_encode_binary_class(
        self, x: pd.DataFrame, y: pd.Series
    ) -> pd.DataFrame:
        """Fit target encoder using NestedCVWrapper and transform columns."""
        logging.info("Start fitting binary target encoder with NestedCVWrapper.")

        # Check and encode target if it is not numeric (i.e. string labels)
        if y.dtype == "object" or y.dtype.name == "category":
            logging.info(
                "Encoding string targets temporarily for binary classification."
            )
            label_encoder = LabelEncoder()
            y = pd.Series(label_encoder.fit_transform(y))
            y.index = x.index

        smoothing = np.max([np.log10(len(x.index)) * 5, 10])

        # Wrap TargetEncoder with NestedCVWrapper
        enc = NestedCVWrapper(
            TargetEncoder(
                cols=self.cat_columns,
                smoothing=smoothing,
            ),
            cv=5,  # Specify number of folds for cross-validation
            random_state=self.random_state,
        )

        x.loc[:, self.cat_columns] = enc.fit_transform(x[self.cat_columns], y)
        x[self.cat_columns] = x[self.cat_columns].astype(float)
        self.encoders["target_encoder_all_cols"] = enc
        return x

    def transform_target_encode_binary_class(self, x: pd.DataFrame) -> pd.DataFrame:
        """Transform categories based on already trained encoder."""
        logging.info("Start transforming categories with binary target encoder.")
        enc = self.encoders["target_encoder_all_cols"]
        x.loc[:, self.cat_columns] = enc.transform(x[self.cat_columns])
        x[self.cat_columns] = x[self.cat_columns].astype(float)
        return x


class MultiClassTargetEncoder:
    """Target encode categorical features in the context of multiclass classification using NestedCVWrapper."""

    def __init__(
        self,
        cat_columns: List[Union[str, float, int]],
        target_col: Union[str, float, int],
        random_state: int = 200,
    ):
        self.encoders: Dict[str, Union[NestedCVWrapper, OneHotEncoder]] = {}
        self.prediction_mode: bool = False
        self.cat_columns = cat_columns
        self.class_names: List[Optional[Union[str, float, int]]] = []
        self.target_col = target_col
        self.random_state = random_state

    def fit_target_encode_multiclass(
        self, x: pd.DataFrame, y: pd.Series
    ) -> pd.DataFrame:
        """Fit target encoder using NestedCVWrapper and transform columns."""
        logging.info("Start fitting multiclass target encoder with NestedCVWrapper.")

        # Check and encode target if it is not numeric (i.e. string labels)
        if y.dtype == "object" or y.dtype.name == "category":
            logging.info(
                "Encoding string targets temporarily for multiclass classification."
            )
            label_encoder = LabelEncoder()
            y = pd.Series(label_encoder.fit_transform(y))
            y.index = x.index

        algorithm = "multiclass_target_encoding_onehotter"
        enc = OneHotEncoder(
            # drop_invariant=True, handle_unknown="ignore"
        )
        enc.fit(y)
        y_onehot = enc.transform(y)
        self.class_names = y_onehot.columns.to_list()
        if self.target_col in self.cat_columns:
            self.cat_columns.remove(self.target_col)

        x_obj = x.loc[:, self.cat_columns].copy()
        x = x.loc[:, ~x.columns.isin(self.cat_columns)]
        for class_ in self.class_names:
            smoothing = np.max([np.log10(len(x.index)) * 5, 10])

            # Wrap TargetEncoder with NestedCVWrapper for each class
            target_enc = NestedCVWrapper(
                TargetEncoder(
                    cols=self.cat_columns,
                    smoothing=smoothing,
                ),
                cv=5,  # Specify number of folds for cross-validation
                random_state=self.random_state,
            )
            target_enc.fit(x_obj, y_onehot[class_])
            self.encoders[f"multiclass_target_encoder_all_cols_{class_}"] = target_enc
            temp = target_enc.transform(x_obj)
            temp.columns = [str(col) + "_" + str(class_) for col in temp.columns]
            x = pd.concat([x, temp], axis=1)
        self.encoders[f"{algorithm}_all_cols"] = enc
        return x

    def transform_target_encode_multiclass(self, x: pd.DataFrame) -> pd.DataFrame:
        """Transform categories based on already trained encoder."""
        logging.info("Start transforming categories with multiclass target encoder.")
        algorithm = "multiclass_target_encoding_onehotter"
        enc = self.encoders[f"{algorithm}_all_cols"]

        x_obj = x.loc[:, self.cat_columns].copy()
        x = x.loc[:, ~x.columns.isin(self.cat_columns)]
        for class_ in self.class_names:
            target_enc = self.encoders[f"multiclass_target_encoder_all_cols_{class_}"]
            temp = target_enc.transform(x_obj)
            temp.columns = [str(col) + "_" + str(class_) for col in temp.columns]
            x = pd.concat([x, temp], axis=1)
        self.encoders[f"{algorithm}_all_cols"] = enc
        return x


class RegressionTargetEncoder:
    """Target encode categorical features in the context of regression using NestedCVWrapper."""

    def __init__(
        self, cat_columns: List[Union[str, float, int]], random_state: int = 200
    ):
        self.encoders: Dict[str, NestedCVWrapper] = {}
        self.prediction_mode: bool = False
        self.cat_columns = cat_columns
        self.random_state = random_state

    def fit_target_encode_regression(
        self, x: pd.DataFrame, y: pd.Series
    ) -> pd.DataFrame:
        """Fit target encoder using NestedCVWrapper and transform columns."""
        logging.info("Start fitting regression target encoder with NestedCVWrapper.")

        y = pd.qcut(y, q=10, labels=False, duplicates="drop")

        smoothing = np.max([np.log10(len(x.index)) * 5, 10])

        # Use KFold for regression
        enc = NestedCVWrapper(
            TargetEncoder(
                cols=self.cat_columns,
                smoothing=smoothing,
            ),
            cv=StratifiedKFold(
                n_splits=5, shuffle=True, random_state=self.random_state
            ),
            random_state=self.random_state,
        )

        x.loc[:, self.cat_columns] = enc.fit_transform(x[self.cat_columns], y)
        x[self.cat_columns] = x[self.cat_columns].astype(float)
        self.encoders["target_encoder_all_cols"] = enc
        return x

    def transform_target_encode_regression(self, x: pd.DataFrame) -> pd.DataFrame:
        """Transform categories based on already trained encoder."""
        logging.info("Start transforming categories with regression target encoder.")
        enc = self.encoders["target_encoder_all_cols"]
        x.loc[:, self.cat_columns] = enc.transform(x[self.cat_columns])
        x[self.cat_columns] = x[self.cat_columns].astype(float)
        return x
