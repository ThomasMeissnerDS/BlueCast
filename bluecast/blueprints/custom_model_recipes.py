import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold

from bluecast.ml_modelling.base_classes import (
    PredictedClasses,  # just for linting checks
)
from bluecast.ml_modelling.base_classes import (
    PredictedProbas,  # just for linting checks
)
from bluecast.ml_modelling.base_classes import (
    BaseClassMlModel,
    BaseClassMlRegressionModel,
)


class LogisticRegressionModel(BaseClassMlModel):
    """Logistic Regression with GridSearchCV-based hyperparameter tuning.

    Searches over L1, L2, and ElasticNet penalties with a wide regularization range.

    :param max_iter: Maximum iterations for solver convergence.
    :param random_state: Random seed for reproducibility.
    :param scoring: Scoring metric for GridSearchCV (e.g. 'roc_auc', 'f1', 'accuracy').
    :param cv_folds: Number of cross-validation folds.
    """

    def __init__(
        self,
        max_iter: int = 100000,
        random_state: int = 300,
        scoring: str = "roc_auc",
        cv_folds: int = 5,
    ):
        self.logistic_regression_model: LogisticRegression = LogisticRegression(
            max_iter=max_iter, random_state=random_state
        )
        self.model: Optional[GridSearchCV] = None
        self.random_state = random_state
        self.scoring = scoring
        self.cv_folds = cv_folds

    def autotune(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ):
        skfold = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )

        params = [
            {
                "penalty": ["l1"],
                "C": np.logspace(-3, 2, 10),
                "class_weight": ["balanced", None],
                "solver": ["saga"],
            },
            {
                "penalty": ["l2"],
                "C": np.logspace(-3, 2, 10),
                "class_weight": ["balanced", None],
                "solver": ["lbfgs", "newton-cg", "sag", "saga"],
            },
            {
                "penalty": ["elasticnet"],
                "C": np.logspace(-3, 2, 10),
                "class_weight": ["balanced", None],
                "solver": ["saga"],
                "l1_ratio": np.linspace(0.1, 0.9, 5),
            },
        ]

        gs_lr = GridSearchCV(
            estimator=self.logistic_regression_model,
            param_grid=params,
            n_jobs=-1,
            cv=skfold,
            scoring=self.scoring,
            verbose=0,
        )

        gs_lr.fit(x_train, y_train)
        logging.info(
            f"Best LogisticRegression params: {gs_lr.best_params_} "
            f"(score: {gs_lr.best_score_:.4f})"
        )
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
        if isinstance(self.model, GridSearchCV):
            probas = self.model.predict_proba(df)[:, 1]
            classes = self.model.predict(df)
            return probas, classes
        else:
            raise ValueError("No fitted model has been found.")


class RegularizedRegressionModel(BaseClassMlRegressionModel):
    """Regularized regression model with GridSearchCV-based hyperparameter tuning.

    Searches over Ridge, Lasso, and ElasticNet with various alpha values.

    :param scoring: Scoring metric for GridSearchCV (e.g. 'neg_mean_squared_error', 'r2').
    :param cv_folds: Number of cross-validation folds.
    :param random_state: Random seed for reproducibility.
    """

    def __init__(
        self,
        scoring: str = "neg_mean_squared_error",
        cv_folds: int = 5,
        random_state: int = 300,
    ):
        self.model = None
        self.best_model_type: Optional[str] = None
        self.scoring = scoring
        self.cv_folds = cv_folds
        self.random_state = random_state

    def autotune(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ):
        kfold = KFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )
        alphas = np.logspace(-4, 3, 15)

        candidates = {
            "ridge": (
                Ridge(random_state=self.random_state),
                {"alpha": alphas},
            ),
            "lasso": (
                Lasso(max_iter=100000, random_state=self.random_state),
                {"alpha": alphas},
            ),
            "elasticnet": (
                ElasticNet(max_iter=100000, random_state=self.random_state),
                {
                    "alpha": alphas,
                    "l1_ratio": np.linspace(0.1, 0.9, 5),
                },
            ),
        }

        best_score = -np.inf
        best_gs = None

        for name, (estimator, param_grid) in candidates.items():
            gs = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                n_jobs=-1,
                cv=kfold,
                scoring=self.scoring,
                verbose=0,
            )
            gs.fit(x_train, y_train)
            logging.info(
                f"{name} best score: {gs.best_score_:.4f}, "
                f"params: {gs.best_params_}"
            )
            if gs.best_score_ > best_score:
                best_score = gs.best_score_
                best_gs = gs
                self.best_model_type = name

        self.model = best_gs
        logging.info(
            f"Best regression model: {self.best_model_type} "
            f"(score: {best_score:.4f})"
        )

    def fit(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        self.autotune(x_train, x_test, y_train, y_test)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is not None:
            preds = self.model.predict(df)
            return preds
        else:
            raise ValueError("No fitted model has been found.")


class LinearRegressionModel(BaseClassMlRegressionModel):
    """Plain OLS linear regression (no regularization). For regularized models, use
    RegularizedRegressionModel instead."""

    def __init__(self):
        self.linear_regression_model: LinearRegression = LinearRegression()
        self.model: Optional[LinearRegression] = None

    def autotune(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ):
        self.linear_regression_model.fit(x_train, y_train)
        self.model = self.linear_regression_model

    def fit(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        self.autotune(x_train, x_test, y_train, y_test)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if isinstance(self.model, LinearRegression):
            preds = self.model.predict(df)
            return preds
        else:
            raise ValueError("No fitted model has been found.")


# Catboost
