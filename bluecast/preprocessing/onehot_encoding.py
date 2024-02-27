"""
Onehot encoding is a method to encode categorical features. It is an unsupervised encoding technique.
It is not recommended for features with high cardinality.

The onehot encoding technique is implemented in the category_encoders library. The library offers a variety of
different encoding techniques. The onehot encoding technique is implemented in the OneHotEncoder class.
"""

from datetime import datetime
from typing import Dict, List, Union

import pandas as pd
from category_encoders import OneHotEncoder

from bluecast.general_utils.general_utils import logger


class OneHotCategoryEncoder:
    """Onehot encode categorical features."""

    def __init__(
        self,
        cat_columns: List[Union[str, float, int]],
        target_col: Union[str, float, int],
    ):
        self.encoders: Dict[str, OneHotEncoder] = {}
        self.prediction_mode: bool = False
        self.cat_columns = cat_columns
        self.target_col = target_col

    def fit_transform(self, x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit onehot encoder and transform column."""
        logger(f"{datetime.utcnow()}: Start fitting binary target encoder.")
        if self.target_col in self.cat_columns:
            self.cat_columns.remove(self.target_col)

        enc = OneHotEncoder(
            cols=self.cat_columns, drop_invariant=True, use_cat_names=True
        )
        encoded_cats = enc.fit_transform(x[self.cat_columns], y)
        x = x.drop(self.cat_columns, axis=1)
        x[encoded_cats.columns.to_list()] = encoded_cats
        x[encoded_cats.columns.to_list()] = x[encoded_cats.columns.to_list()].astype(
            int
        )
        self.encoders["onehot_encoder_all_cols"] = enc
        return x

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Transform categories based on already trained encoder."""
        logger(
            f"{datetime.utcnow()}: Start transforming categories with binary target encoder."
        )
        enc = self.encoders["onehot_encoder_all_cols"]
        encoded_cats = enc.transform(x[self.cat_columns])
        x = x.drop(self.cat_columns, axis=1)
        x[encoded_cats.columns.to_list()] = encoded_cats
        x[encoded_cats.columns.to_list()] = x[encoded_cats.columns.to_list()].astype(
            int
        )
        return x
