from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer, matthews_corrcoef, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

from bluecast.general_utils.general_utils import logger
from bluecast.preprocessing.custom import CustomPreprocessing


class RFECVSelector(CustomPreprocessing):
    """Select top features based on selection_strategy defined in FeatureSelectionConfig.

    On default cross-validated recursive feature elimination is used.
    """

    def __init__(
        self, random_state: int = 0, min_features_to_select: int = 5, class_problem=None
    ):
        super().__init__()
        self.selected_features = None
        self.random_state = random_state
        if class_problem in ["regression"]:
            stratifier = KFold(5, random_state=random_state, shuffle=True)
            model = xgb.XGBRegressor()
            scorer = make_scorer(mean_squared_error)
        else:
            stratifier = StratifiedKFold(5, random_state=random_state, shuffle=True)
            model = xgb.XGBClassifier()
            scorer = make_scorer(matthews_corrcoef)

        self.selection_strategy: RFECV = RFECV(
            estimator=model,
            step=1,
            cv=stratifier,
            min_features_to_select=min_features_to_select,
            scoring=scorer,
            n_jobs=2,
        )

    def fit_transform(
        self, df: pd.DataFrame, target: pd.Series
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        logger(
            f"{datetime.utcnow()}: Start feature selection as defined in FeatureSelectionConfig."
        )
        self.selection_strategy.fit(df, target)
        self.selected_features = self.selection_strategy.support_
        df = df.loc[:, self.selected_features]
        logger(f"{datetime.utcnow()}: Selected features are {df.columns}.")
        return df, target

    def transform(
        self,
        df: pd.DataFrame,
        target: Optional[pd.Series] = None,
        predicton_mode: bool = False,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        logger(f"{datetime.utcnow()}: Apply feature selection.")
        df = df.loc[:, self.selected_features]
        return df, target
