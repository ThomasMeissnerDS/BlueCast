"""
Target encoding is a method to encode categorical features. It is a supervised encoding technique, meaning it uses the
target variable to encode the features. The target variable is used to calculate the mean of the target for each
category and then replace the category variable with the mean value. This is a powerful technique that can be used to
create new features from categorical variables. It is also a powerful technique to deal with high cardinality features
as it reduces the dimensionality of the categorical features.

The target encoding technique is implemented in the category_encoders library. The library offers a variety of
different encoding techniques. The target encoding technique is implemented in the TargetEncoder class. For multiclass
uses cases a special implementation is available as the category-encoders implementation is not suitable for
multiclass.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union

import pandas as pd
from category_encoders import OneHotEncoder, TargetEncoder

from bluecast.general_utils.general_utils import logger


class BinaryClassTargetEncoder:
    """Target encode categorical features in the context of binary classification."""

    def __init__(self, cat_columns: List[Union[str, float, int]]):
        self.encoders: Dict[str, TargetEncoder] = {}
        self.prediction_mode: bool = False
        self.cat_columns = cat_columns

    def fit_target_encode_binary_class(
        self, x: pd.DataFrame, y: pd.Series
    ) -> pd.DataFrame:
        """Fit target encoder and transform column."""
        logger(f"{datetime.utcnow()}: Start fitting binary target encoder.")
        enc = TargetEncoder(cols=self.cat_columns)
        x.loc[:, self.cat_columns] = enc.fit_transform(x[self.cat_columns], y)
        x[self.cat_columns] = x[self.cat_columns].astype(float)
        self.encoders["target_encoder_all_cols"] = enc
        return x

    def transform_target_encode_binary_class(self, x: pd.DataFrame) -> pd.DataFrame:
        """Transform categories based on already trained encoder."""
        logger(
            f"{datetime.utcnow()}: Start transforming categories with binary target encoder."
        )
        enc = self.encoders["target_encoder_all_cols"]
        x.loc[:, self.cat_columns] = enc.transform(x[self.cat_columns])
        x[self.cat_columns] = x[self.cat_columns].astype(float)
        return x


class MultiClassTargetEncoder:
    """Target encode categorical features in the context of multiclass classification."""

    def __init__(
        self,
        cat_columns: List[Union[str, float, int]],
        target_col: Union[str, float, int],
    ):
        self.encoders: Dict[str, Union[TargetEncoder, OneHotEncoder]] = {}
        self.prediction_mode: bool = False
        self.cat_columns = cat_columns
        self.class_names: List[Optional[Union[str, float, int]]] = []
        self.target_col = target_col

    def fit_target_encode_multiclass(
        self, x: pd.DataFrame, y: pd.Series
    ) -> pd.DataFrame:
        """Fit target encoder and transform column."""
        logger(f"{datetime.utcnow()}: Start fitting multiclass target encoder.")
        algorithm = "multiclass_target_encoding_onehotter"
        enc = OneHotEncoder()
        enc.fit(y)
        y_onehot = enc.transform(y)
        self.class_names = y_onehot.columns.to_list()
        if self.target_col in self.cat_columns:
            self.cat_columns.remove(self.target_col)

        x_obj = x.loc[:, self.cat_columns].copy()
        x = x.loc[:, ~x.columns.isin(self.cat_columns)]
        for class_ in self.class_names:
            target_enc = TargetEncoder()
            target_enc.fit(x_obj, y_onehot[class_])
            self.encoders[f"multiclass_target_encoder_all_cols_{class_}"] = target_enc
            temp = target_enc.transform(x_obj)
            temp.columns = [str(x) + "_" + str(class_) for x in temp.columns]
            x = pd.concat([x, temp], axis=1)
        self.encoders[f"{algorithm}_all_cols"] = enc
        return x

    def transform_target_encode_multiclass(self, x: pd.DataFrame) -> pd.DataFrame:
        """Transform categories based on already trained encoder."""
        logger(
            f"{datetime.utcnow()}: Start transforming categories with multiclass target encoder."
        )
        algorithm = "multiclass_target_encoding_onehotter"
        enc = self.encoders[f"{algorithm}_all_cols"]

        x_obj = x.loc[:, self.cat_columns].copy()
        x = x.loc[:, ~x.columns.isin(self.cat_columns)]
        for class_ in self.class_names:
            target_enc = self.encoders[f"multiclass_target_encoder_all_cols_{class_}"]
            temp = target_enc.transform(x_obj)
            temp.columns = [str(x) + "_" + str(class_) for x in temp.columns]
            x = pd.concat([x, temp], axis=1)
        self.encoders[f"{algorithm}_all_cols"] = enc
        return x
