"""
A module for encoding target column labels.

This is a convenience feature. It is only relevant when target column values are categorical.
In such cases they will be converted to numerical values, but reverse-transformed for the end-user at the end of the
pipeline.
"""

from datetime import datetime
from typing import Dict, Optional, Union

import pandas as pd

from bluecast.general_utils.general_utils import logger


class TargetLabelEncoder:
    """
    Encode target column labels.

    This function is only relevant when target column values are categorical. In such cases they will be converted
    into numerical representation. This encoding can also be reversed to translate back.
    """

    def __init__(self):
        self.target_label_mapping: Dict[str, int] = {}

    def fit_label_encoder(self, targets: pd.DataFrame) -> Dict[str, int]:
        """Iterate through target values and map them to numerics."""
        logger(f"{datetime.utcnow()}: Start fitting target label encoder.")
        targets = targets.astype("category")
        col = targets.name

        if isinstance(targets, pd.Series):
            targets = targets.to_frame()

        values = sorted(targets[col].unique().tolist())

        cat_mapping = {}
        for label, cat in enumerate(values):
            cat_mapping[cat] = label
        return cat_mapping

    def label_encoder_transform(
        self,
        targets: pd.DataFrame,
        mapping: Dict[str, int],
        target_col: Optional[Union[str, int, float]] = None,
    ) -> pd.DataFrame:
        """Transform target column from categorical to numerical representation."""
        logger(f"{datetime.utcnow()}: Start encoding target labels.")
        if (
            isinstance(target_col, str)
            or isinstance(target_col, int)
            or isinstance(target_col, float)
        ):
            col = target_col
        else:
            col = targets.name

        if isinstance(targets, pd.Series):
            targets = targets.astype("category")
        elif isinstance(targets, pd.DataFrame):
            targets[col] = targets[col].astype("category")

        if isinstance(targets, pd.Series):
            targets = targets.to_frame()
        mapping = self.target_label_mapping
        targets[col] = targets.loc[:, col].apply(lambda x: mapping.get(x, 999))
        targets[col] = targets[col].astype("int")
        return targets

    def fit_transform_target_labels(self, targets: pd.DataFrame) -> pd.DataFrame:
        """Wrapper function that creates the mapping and transforms the target column."""
        cat_mapping = self.fit_label_encoder(targets)
        self.target_label_mapping = cat_mapping
        targets = self.label_encoder_transform(targets, self.target_label_mapping)
        return targets

    def transform_target_labels(
        self, targets: pd.DataFrame, target_col: Optional[Union[str, int, float]] = None
    ) -> pd.DataFrame:
        """Transform the target column based on already created mappings."""
        targets = self.label_encoder_transform(
            targets, self.target_label_mapping, target_col
        )
        return targets

    def label_encoder_reverse_transform(self, targets: pd.Series) -> pd.DataFrame:
        """Reverse numerical encodings back to original categories."""
        logger(f"{datetime.utcnow()}: Start reverse-encoding target labels.")
        col = targets.name

        if isinstance(targets, pd.Series):
            targets = targets.to_frame()

        reverse_mapping = {
            value: key for key, value in self.target_label_mapping.items()
        }
        targets = targets.replace({col: reverse_mapping})
        return targets
