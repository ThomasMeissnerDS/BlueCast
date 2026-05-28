from typing import Optional, Tuple

import pandas as pd

from bluecast.preprocessing.custom import CustomPreprocessing
from feature_engineering import engineer_features


class PerfectFitPreprocessor(CustomPreprocessing):
    def __init__(self):
        super().__init__()
        self.x5_median_ = 9.8

    def fit_transform(
        self, df: pd.DataFrame, target: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        # Compute median excluding the sentinel value
        valid_x5 = df.loc[df["x5"] < 999.0, "x5"]
        if len(valid_x5) > 0:
            self.x5_median_ = float(valid_x5.median())
        df = engineer_features(df, x5_median=self.x5_median_)
        return df, target

    def transform(
        self,
        df: pd.DataFrame,
        target: Optional[pd.Series] = None,
        prediction_mode: bool = False,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        df = engineer_features(df, x5_median=self.x5_median_)
        return df, target
