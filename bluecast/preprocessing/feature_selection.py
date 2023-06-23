from datetime import datetime

import pandas as pd

from bluecast.config.training_config import FeatureSelectionConfig
from bluecast.general_utils.general_utils import logger


class FeatureSelector:
    """Select top features based on selection_strategy defined in FeatureSelectionConfig.

    On default cross-validated recursive feature elimination is used. In the config file a different
    RFECV instance can be defined (using a  different classifier, scoring function, random seed etc.).
    :param selection_strategy: Instance of FeatureSelectionConfig.selection_strategy
    """

    def __init__(self, selection_strategy: FeatureSelectionConfig.selection_strategy):
        self.selected_features = None
        self.selection_strategy = selection_strategy

    def fit_transform(self, df: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        logger(
            f"{datetime.utcnow()}: Start feature selection as defined in FeatureSelectionConfig."
        )
        self.selection_strategy.fit(df, target)
        self.selected_features = self.selection_strategy.support_
        df = df.loc[:, self.selected_features]
        logger(f"{datetime.utcnow()}: Selected features are {df.columns}.")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.loc[:, self.selected_features]
        return df
