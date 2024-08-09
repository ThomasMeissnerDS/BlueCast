"""
Infrequent categories may cause overfitting.

This module groups infrequent categories into a common group to reduce the risk of overfitting
"""

import logging
from typing import Dict, List, Union

import pandas as pd


class InFrequentCategoryEncoder:
    """Group infrequent categories into common group."""

    def __init__(
        self,
        cat_columns: List[Union[str, float, int]],
        target_col: Union[str, float, int],
        infrequent_threshold: int = 5,
    ):
        self.frequencies: Dict[Union[str, float, int], pd.Series] = {}
        self.prediction_mode: bool = False
        self.cat_columns = cat_columns
        self.target_col = target_col
        self.infrequent_threshold = infrequent_threshold

    def fit_transform(self, x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Find infrequent categories and transform column."""
        logging.info("Start fitting binary target encoder.")
        if self.target_col in self.cat_columns:
            self.cat_columns.remove(self.target_col)

        for col in self.cat_columns:
            self.frequencies[col] = x[col].value_counts()
            x[col] = x[col].mask(
                x[col].map(self.frequencies[col], na_action="ignore")
                < self.infrequent_threshold,
                "rare categories",
            )
        return x.copy()  # copy against high fragmentation

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Transform categories based on already explored frequencies."""
        logging.info("Start transforming categories with binary target encoder.")
        for col in self.cat_columns:
            x[col] = x[col].mask(
                x[col].map(self.frequencies[col], na_action="ignore")
                < self.infrequent_threshold,
                "rare categories",
            )
        return x
