from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from bluecast.ml_modelling.base_classes import (
    PredictedClasses,  # just for linting checks
)
from bluecast.ml_modelling.base_classes import (
    PredictedProbas,  # just for linting checks
)
from bluecast.ml_modelling.base_classes import BaseClassMlModel


class LogisticRegressionModel(BaseClassMlModel):
    def __init__(self, max_iter=100000, random_state=300):
        self.model = LogisticRegression(max_iter=max_iter, random_state=random_state)
        self.random_state = random_state

    def autotune(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ):

        skfold = StratifiedKFold(n_splits=5)

        params = [
            {
                "penalty": ["l2"],
                "C": np.logspace(0.1, 1, 5),
                "class_weight": ["balanced", None],
                "solver": ["newton-cg", "newton-cholesky", "sag", "saga"],
            },
            {
                "penalty": ["elasticnet"],
                "C": np.logspace(0.1, 1, 5),
                "class_weight": ["balanced", None],
                "solver": ["newton-cg", "newton-cholesky", "sag", "saga"],
                "l1_ratio": np.arange(0, 1, 3),
            },
        ]

        gs_lr = GridSearchCV(
            estimator=self.model,
            param_grid=params,
            n_jobs=-1,
            cv=skfold,
            scoring="roc_auc",
            verbose=1,
        )

        gs_lr.fit(x_train, y_train)

        self.model = gs_lr

    def fit(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        self.autotune(x_train, x_test, y_train, y_test)

    def predict(self, df: pd.DataFrame) -> Tuple[PredictedProbas, PredictedClasses]:
        probas = self.model.predict_proba(df)[:, 1]
        classes = self.model.predict(df)

        return probas, classes


class LinearRegressionModel(BaseClassMlModel):
    def __init__(self):
        self.model = LinearRegression()

    def autotune(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ):

        skfold = StratifiedKFold(n_splits=5)

        params = [
            {
                "penalty": ["l2"],
                "C": np.logspace(0.1, 1, 5),
                "class_weight": ["balanced", None],
                "solver": ["newton-cg", "newton-cholesky", "sag", "saga"],
            },
            {
                "penalty": ["elasticnet"],
                "C": np.logspace(0.1, 1, 5),
                "class_weight": ["balanced", None],
                "solver": ["newton-cg", "newton-cholesky", "sag", "saga"],
                "l1_ratio": np.arange(0, 1, 3),
            },
        ]

        gs_lr = GridSearchCV(
            estimator=self.model,
            param_grid=params,
            n_jobs=-1,
            cv=skfold,
            scoring="neg_root_mean_squared_error",
            verbose=1,
        )

        gs_lr.fit(x_train, y_train)

        self.model = gs_lr

    def fit(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        self.autotune(x_train, x_test, y_train, y_test)

    def predict(self, df: pd.DataFrame) -> Tuple[PredictedProbas, PredictedClasses]:
        preds = self.model.predict(df)

        return preds
