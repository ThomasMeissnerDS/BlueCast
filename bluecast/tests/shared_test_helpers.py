from typing import Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

from bluecast.ml_modelling.base_classes import (
    BaseClassMlModel,
    BaseClassMlRegressionModel,
    PredictedClasses,
    PredictedProbas,
)
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


class TestCustomPreprocessor(CustomPreprocessing):
    def custom_function(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["custom_feature"] = df["feature1"] * 2
        return df

    def fit_transform(
        self, df: pd.DataFrame, target: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        df = self.custom_function(df)
        return df, target

    def transform(
        self,
        df: pd.DataFrame,
        target: Optional[pd.Series] = None,
        prediction_mode: bool = False,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        df = self.custom_function(df)
        return df, target


class CustomClassificationModel(BaseClassMlModel):
    def __init__(self):
        self.model = None

    def fit(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        self.model = RandomForestClassifier()
        self.model.fit(x_train, y_train)

    def predict(self, df: pd.DataFrame) -> Tuple[PredictedProbas, PredictedClasses]:
        predicted_probas = self.model.predict_proba(df)
        predicted_classes = self.model.predict(df)
        return predicted_probas, predicted_classes


class CustomBinaryClassificationModel(BaseClassMlModel):
    def __init__(self):
        self.model = None

    def fit(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        self.model = RandomForestClassifier()
        self.model.fit(x_train, y_train)

    def predict(self, df: pd.DataFrame) -> Tuple[PredictedProbas, PredictedClasses]:
        predicted_probas = self.model.predict_proba(df)[:, 1]
        predicted_classes = self.model.predict(df)
        return predicted_probas, predicted_classes


class CustomMulticlassClassificationModel(BaseClassMlModel):
    def __init__(self):
        self.model = None

    def fit(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        self.model = RandomForestClassifier()
        self.model.fit(x_train, y_train)

    def predict(self, df: pd.DataFrame) -> Tuple[PredictedProbas, PredictedClasses]:
        predicted_probas = self.model.predict_proba(df)
        predicted_classes = np.asarray([np.argmax(line) for line in predicted_probas])
        return predicted_probas, predicted_classes


class CustomRegressionModel(BaseClassMlRegressionModel):
    def __init__(self):
        self.model = None

    def fit(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        self.model = RandomForestRegressor()
        self.model.fit(x_train, y_train)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(df)
        return preds
