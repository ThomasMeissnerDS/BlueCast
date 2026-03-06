from typing import Optional, Tuple

import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

from bluecast.preprocessing.custom import CustomPreprocessing


class MyCustomLastMilePreprocessing(CustomPreprocessing):
    def custom_function(self, df: pd.DataFrame) -> pd.DataFrame:
        df["custom_col"] = 5
        return df

    def fit_transform(
        self, df: pd.DataFrame, target: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        df = self.custom_function(df)
        df = df.head(1000)
        target = target.head(1000)
        return df, target

    def transform(
        self,
        df: pd.DataFrame,
        target: Optional[pd.Series] = None,
        prediction_mode: bool = False,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        df = self.custom_function(df)
        if not prediction_mode and isinstance(target, pd.Series):
            df = df.head(100)
            target = target.head(100)
        return df, target


class RFECVSelector(CustomPreprocessing):
    def __init__(self, estimator=None, cv=None, scoring=None, random_state: int = 0):
        super().__init__()
        self.selected_features = None
        self.random_state = random_state
        if estimator is None:
            estimator = xgb.XGBClassifier()
        if cv is None:
            cv = StratifiedKFold(2, random_state=random_state, shuffle=True)
        if scoring is None:
            scoring = make_scorer(matthews_corrcoef)
        self.selection_strategy: RFECV = RFECV(
            estimator=estimator,
            step=1,
            cv=cv,
            min_features_to_select=1,
            scoring=scoring,
            n_jobs=2,
        )

    def fit_transform(
        self, df: pd.DataFrame, target: pd.Series
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        self.selection_strategy.fit(df, target)
        self.selected_features = self.selection_strategy.support_
        df = df.loc[:, self.selected_features]
        return df, target

    def transform(
        self,
        df: pd.DataFrame,
        target: Optional[pd.Series] = None,
        prediction_mode: bool = False,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        df = df.loc[:, self.selected_features]
        return df, target


class MyCustomPreprocessor(CustomPreprocessing):
    def __init__(self, estimator=None, cv=None, scoring=None, random_state: int = 0):
        super().__init__()
        self.selected_features = None
        self.random_state = random_state
        if estimator is None:
            estimator = xgb.XGBClassifier()
        if cv is None:
            cv = StratifiedKFold(2, random_state=random_state, shuffle=True)
        if scoring is None:
            scoring = make_scorer(matthews_corrcoef)
        self.selection_strategy: RFECV = RFECV(
            estimator=estimator,
            step=1,
            cv=cv,
            min_features_to_select=1,
            scoring=scoring,
            n_jobs=2,
        )

    def fit_transform(
        self, df: pd.DataFrame, target: pd.Series
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        return df, target

    def transform(
        self,
        df: pd.DataFrame,
        target: Optional[pd.Series] = None,
        prediction_mode: bool = False,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        return df, target


class MyCustomInFoldPreprocessor(CustomPreprocessing):
    def __init__(self):
        super().__init__()

    def fit_transform(
        self, df: pd.DataFrame, target: pd.Series
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        df["leakage"] = target
        return df, target

    def transform(
        self,
        df: pd.DataFrame,
        target: Optional[pd.Series] = None,
        prediction_mode: bool = False,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        df["leakage"] = 0
        return df, target
