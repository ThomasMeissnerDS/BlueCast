"""Architecture registry for the 'ultimate' multi-model mode.

Maps friendly names to factory functions that return ``BaseClassMlModel``
(or ``None`` for the BlueCast default CatBoost pipeline).
"""

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold

from bluecast.ml_modelling.base_classes import (
    BaseClassMlModel,
    BaseClassMlRegressionModel,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HistGradientBoosting wrapper
# ---------------------------------------------------------------------------


class HistGBClassificationModel(BaseClassMlModel):
    """Sklearn HistGradientBoostingClassifier with GridSearchCV tuning.

    :param scoring: Scoring metric for GridSearchCV.
    :param cv_folds: Number of cross-validation folds.
    :param random_state: Random seed.
    """

    def __init__(
        self,
        scoring: str = "roc_auc",
        cv_folds: int = 5,
        random_state: int = 300,
    ):
        self.model: Optional[GridSearchCV] = None
        self.scoring = scoring
        self.cv_folds = cv_folds
        self.random_state = random_state

    def autotune(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        skfold = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )
        param_grid = {
            "max_iter": [200, 500],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [4, 6, 8],
            "min_samples_leaf": [10, 20, 50],
        }
        gs = GridSearchCV(
            estimator=HistGradientBoostingClassifier(
                random_state=self.random_state,
                early_stopping=True,
                validation_fraction=0.1,
            ),
            param_grid=param_grid,
            n_jobs=-1,
            cv=skfold,
            scoring=self.scoring,
            verbose=0,
        )
        gs.fit(x_train, y_train)
        logger.info(
            f"HistGB classification best params: {gs.best_params_} "
            f"(score: {gs.best_score_:.4f})"
        )
        self.model = gs

    def fit(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        self.autotune(x_train, x_test, y_train, y_test)

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise ValueError("No fitted model has been found.")
        probas = self.model.predict_proba(df)[:, 1]
        classes = self.model.predict(df)
        return probas, classes


class HistGBRegressionModel(BaseClassMlRegressionModel):
    """Sklearn HistGradientBoostingRegressor with GridSearchCV tuning.

    :param scoring: Scoring metric for GridSearchCV.
    :param cv_folds: Number of cross-validation folds.
    :param random_state: Random seed.
    """

    def __init__(
        self,
        scoring: str = "neg_mean_absolute_error",
        cv_folds: int = 5,
        random_state: int = 300,
    ):
        self.model: Optional[GridSearchCV] = None
        self.scoring = scoring
        self.cv_folds = cv_folds
        self.random_state = random_state

    def autotune(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        kfold = KFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )
        param_grid = {
            "max_iter": [200, 500],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [4, 6, 8],
            "min_samples_leaf": [10, 20, 50],
        }
        gs = GridSearchCV(
            estimator=HistGradientBoostingRegressor(
                random_state=self.random_state,
                early_stopping=True,
                validation_fraction=0.1,
            ),
            param_grid=param_grid,
            n_jobs=-1,
            cv=kfold,
            scoring=self.scoring,
            verbose=0,
        )
        gs.fit(x_train, y_train)
        logger.info(
            f"HistGB regression best params: {gs.best_params_} "
            f"(score: {gs.best_score_:.4f})"
        )
        self.model = gs

    def fit(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        self.autotune(x_train, x_test, y_train, y_test)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("No fitted model has been found.")
        preds = self.model.predict(df)
        return preds


# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------

ArchFactory = Callable[[str], Optional[BaseClassMlModel]]


def _make_linear(problem: str) -> BaseClassMlModel:
    """Create a linear model appropriate for the problem type."""
    if problem == "regression":
        from bluecast.blueprints.custom_model_recipes import RegularizedRegressionModel

        return RegularizedRegressionModel()
    else:
        from bluecast.blueprints.custom_model_recipes import LogisticRegressionModel

        return LogisticRegressionModel()


def _make_histgb(problem: str) -> BaseClassMlModel:
    """Create a HistGradientBoosting model appropriate for the problem type."""
    if problem == "regression":
        return HistGBRegressionModel()
    else:
        return HistGBClassificationModel()


ArchInfo = Dict[str, Any]

ARCHITECTURE_REGISTRY: Dict[str, ArchInfo] = {
    "catboost": {
        "name": "CatBoost (default)",
        "factory": lambda problem: None,  # None = use default BlueCast pipeline
        "supports": ["binary", "multiclass", "regression"],
    },
    "xgboost": {
        "name": "XGBoost",
        "factory": lambda problem: None,  # XGBoost uses conf_xgboost in BlueCast
        "supports": ["binary", "multiclass", "regression"],
        "use_xgboost_native": True,
    },
    "linear": {
        "name": "Regularized Linear Model",
        "factory": _make_linear,
        "supports": ["binary", "multiclass", "regression"],
    },
    "histgb": {
        "name": "HistGradientBoosting (sklearn)",
        "factory": _make_histgb,
        "supports": ["binary", "multiclass", "regression"],
    },
}


def get_architectures_for_problem(
    problem: str,
) -> Dict[str, ArchInfo]:
    """Return only architectures that support the given problem type."""
    return {
        name: info
        for name, info in ARCHITECTURE_REGISTRY.items()
        if problem in info["supports"]
    }
