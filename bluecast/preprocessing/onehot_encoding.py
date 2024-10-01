import logging
from typing import Dict, List, Union

import pandas as pd
from category_encoders import OneHotEncoder


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
        logging.info("Start fitting binary target encoder.")
        if self.target_col in self.cat_columns:
            self.cat_columns.remove(self.target_col)

        enc = OneHotEncoder(
            use_cat_names=True,  # drop_invariant=True, use_cat_names=True
        )
        encoded_cats = enc.fit_transform(x[self.cat_columns], y)
        x_new = x.drop(
            self.cat_columns, axis=1
        ).copy()  # copy against high fragmentation
        x_new = self.append_encoded_columns(x_new, encoded_cats)

        self.encoders["onehot_encoder_all_cols"] = enc
        return x_new.copy()  # copy against high fragmentation

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Transform categories based on already trained encoder."""
        logging.info("Start transforming categories with binary target encoder.")
        enc = self.encoders["onehot_encoder_all_cols"]
        encoded_cats = enc.transform(x[self.cat_columns])
        x = x.drop(self.cat_columns, axis=1)
        x = self.append_encoded_columns(x, encoded_cats)
        return x

    def append_encoded_columns(
        self, x: pd.DataFrame, encoded_cats: pd.DataFrame
    ) -> pd.DataFrame:
        """Append encoded columns to the DataFrame."""
        # Instead of assigning the columns one by one, we concatenate them all at once.
        encoded_cats = encoded_cats.astype(int)  # Ensure all columns are integers
        x_new = pd.concat([x, encoded_cats], axis=1)

        # Set all new columns to -1 where all values are 0
        mask_all_zero = (x_new[encoded_cats.columns.to_list()] == 0).all(axis=1)
        x_new.loc[mask_all_zero, encoded_cats.columns.to_list()] = -1

        return x_new
