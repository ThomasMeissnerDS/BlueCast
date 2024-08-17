from typing import Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegressionCV

from bluecast.ml_modelling.base_classes import (
    PredictedClasses,  # just for linting checks
)
from bluecast.ml_modelling.base_classes import (
    PredictedProbas,  # just for linting checks
)
from bluecast.ml_modelling.base_classes import BaseClassMlModel


class LogisticRegressionModel(BaseClassMlModel):
    def __init__(self, random_state=300):
        self.model = LogisticRegressionCV(random_state=random_state)
        self.random_state = random_state

    def autotune(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ):

        self.model.fit(x_train, y_train)

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
