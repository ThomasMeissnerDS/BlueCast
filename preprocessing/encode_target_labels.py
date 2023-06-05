import pandas as pd
from typing import Dict


class TargetLabelEncoder:
    def __init__(self):
        self.target_label_mapping: Dict[str, int] = {}

    def fit_label_encoder(self, targets: pd.DataFrame) -> Dict[str, int]:
        targets = targets.astype("category")
        col = targets.name

        if isinstance(targets, pd.Series):
            targets = targets.to_frame()

        values = targets[col].unique()
        cat_mapping = {}
        for label, cat in enumerate(values):
            cat_mapping[cat] = label
        return cat_mapping

    def label_encoder_transform(self, targets: pd.DataFrame, mapping: Dict[str, int]) -> pd.DataFrame:
        targets = targets.astype("category")
        col = targets.name
        if isinstance(targets, pd.Series):
            targets = targets.to_frame()
        mapping = self.target_label_mapping
        targets[col] = targets[col].apply(lambda x: mapping.get(x, 999))
        targets[col] = targets[col].astype("int")
        return targets

    def fit_transform_target_labels(self, targets: pd.DataFrame) -> pd.DataFrame:
        cat_mapping = self.fit_label_encoder(targets)
        self.target_label_mapping = cat_mapping
        targets = self.label_encoder_transform(
            targets, self.target_label_mapping
        )
        return targets

    def transform_target_labels(self, targets: pd.DataFrame) -> pd.DataFrame:
        targets = self.label_encoder_transform(
            targets, self.target_label_mapping
        )
        return targets

    def label_encoder_reverse_transform(self, targets: pd.DataFrame) -> pd.Series:
        col = targets.name

        if isinstance(targets, pd.Series):
            targets = targets.to_frame()

        reverse_mapping = {
            value: key
            for key, value in self.target_label_mapping.items()
        }
        targets = targets.replace({col: reverse_mapping})
        return targets
