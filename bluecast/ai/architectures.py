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
    RandomForestClassifier,
    RandomForestRegressor,
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
        import optuna
        from optuna.samplers import TPESampler
        from sklearn.model_selection import cross_val_score

        conf_tuning = getattr(self, "conf_tuning", {})
        tuning_rounds = conf_tuning.get("tuning_rounds", 15)

        def objective(trial):
            params = {
                "max_iter": trial.suggest_int("max_iter", conf_tuning.get("histgb_max_iter_min", 100), conf_tuning.get("histgb_max_iter_max", 500)),
                "learning_rate": trial.suggest_float("learning_rate", conf_tuning.get("histgb_lr_min", 0.01), conf_tuning.get("histgb_lr_max", 0.1), log=True),
                "max_depth": trial.suggest_int("max_depth", conf_tuning.get("histgb_depth_min", 3), conf_tuning.get("histgb_depth_max", 9)),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", conf_tuning.get("histgb_min_samples_min", 10), conf_tuning.get("histgb_min_samples_max", 50)),
            }
            model = HistGradientBoostingClassifier(
                random_state=self.random_state, early_stopping=True, validation_fraction=0.1, **params
            )
            skfold = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, x_train, y_train, cv=skfold, scoring=self.scoring, n_jobs=-1)
            return scores.mean()

        tuning_timeout = conf_tuning.get("tuning_max_runtime", 120)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=tuning_rounds, timeout=tuning_timeout)

        logger.info(f"HistGB classification best params: {study.best_params} (score: {study.best_value:.4f})")
        self.model = HistGradientBoostingClassifier(
            random_state=self.random_state, early_stopping=True, validation_fraction=0.1, **study.best_params
        )
        self.model.fit(x_train, y_train)

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
        import optuna
        from optuna.samplers import TPESampler
        from sklearn.model_selection import cross_val_score

        conf_tuning = getattr(self, "conf_tuning", {})
        tuning_rounds = conf_tuning.get("tuning_rounds", 15)

        def objective(trial):
            params = {
                "max_iter": trial.suggest_int("max_iter", conf_tuning.get("histgb_max_iter_min", 100), conf_tuning.get("histgb_max_iter_max", 500)),
                "learning_rate": trial.suggest_float("learning_rate", conf_tuning.get("histgb_lr_min", 0.01), conf_tuning.get("histgb_lr_max", 0.1), log=True),
                "max_depth": trial.suggest_int("max_depth", conf_tuning.get("histgb_depth_min", 3), conf_tuning.get("histgb_depth_max", 9)),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", conf_tuning.get("histgb_min_samples_min", 10), conf_tuning.get("histgb_min_samples_max", 50)),
            }
            loss = "absolute_error" if "absolute_error" in self.scoring else "squared_error"
            model = HistGradientBoostingRegressor(
                random_state=self.random_state, loss=loss, early_stopping=True, validation_fraction=0.1, **params
            )
            kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring=self.scoring, n_jobs=-1)
            return scores.mean()

        tuning_timeout = conf_tuning.get("tuning_max_runtime", 120)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=tuning_rounds, timeout=tuning_timeout)

        logger.info(f"HistGB regression best params: {study.best_params} (score: {study.best_value:.4f})")
        loss = "absolute_error" if "absolute_error" in self.scoring else "squared_error"
        self.model = HistGradientBoostingRegressor(
            random_state=self.random_state, loss=loss, early_stopping=True, validation_fraction=0.1, **study.best_params
        )
        self.model.fit(x_train, y_train)

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


class RandomForestClassificationModel(BaseClassMlModel):
    """Sklearn RandomForestClassifier with GridSearchCV tuning.

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
        import optuna
        from optuna.samplers import TPESampler
        from sklearn.model_selection import cross_val_score
        
        x_train = x_train.fillna(0)
        conf_tuning = getattr(self, "conf_tuning", {})
        tuning_rounds = conf_tuning.get("tuning_rounds", 15)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", conf_tuning.get("rf_estimators_min", 50), conf_tuning.get("rf_estimators_max", 300)),
                "max_depth": trial.suggest_int("max_depth", conf_tuning.get("rf_max_depth_min", 3), conf_tuning.get("rf_max_depth_max", 15)),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", conf_tuning.get("rf_min_samples_min", 1), conf_tuning.get("rf_min_samples_max", 20)),
            }
            model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1, **params)
            skfold = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, x_train, y_train, cv=skfold, scoring=self.scoring, n_jobs=-1)
            return scores.mean()

        tuning_timeout = conf_tuning.get("tuning_max_runtime", 120)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=tuning_rounds, timeout=tuning_timeout)

        logger.info(f"RandomForest classification best params: {study.best_params} (score: {study.best_value:.4f})")
        self.model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1, **study.best_params)
        self.model.fit(x_train, y_train)

    def fit(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        self.autotune(x_train, x_test, y_train, y_test)

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        df = df.fillna(0)
        if self.model is None:
            raise ValueError("No fitted model has been found.")
        probas = self.model.predict_proba(df)[:, 1]
        classes = self.model.predict(df)
        return probas, classes


class RandomForestRegressionModel(BaseClassMlRegressionModel):
    """Sklearn RandomForestRegressor with GridSearchCV tuning.

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
        import optuna
        from optuna.samplers import TPESampler
        from sklearn.model_selection import cross_val_score
        
        x_train = x_train.fillna(0)
        conf_tuning = getattr(self, "conf_tuning", {})
        tuning_rounds = conf_tuning.get("tuning_rounds", 15)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", conf_tuning.get("rf_estimators_min", 50), conf_tuning.get("rf_estimators_max", 300)),
                "max_depth": trial.suggest_int("max_depth", conf_tuning.get("rf_max_depth_min", 3), conf_tuning.get("rf_max_depth_max", 15)),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", conf_tuning.get("rf_min_samples_min", 1), conf_tuning.get("rf_min_samples_max", 20)),
            }
            criterion = "absolute_error" if "absolute_error" in self.scoring else "squared_error"
            model = RandomForestRegressor(random_state=self.random_state, criterion=criterion, n_jobs=-1, **params)
            kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring=self.scoring, n_jobs=-1)
            return scores.mean()

        tuning_timeout = conf_tuning.get("tuning_max_runtime", 120)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=tuning_rounds, timeout=tuning_timeout)

        logger.info(f"RandomForest regression best params: {study.best_params} (score: {study.best_value:.4f})")
        criterion = "absolute_error" if "absolute_error" in self.scoring else "squared_error"
        self.model = RandomForestRegressor(random_state=self.random_state, criterion=criterion, n_jobs=-1, **study.best_params)
        self.model.fit(x_train, y_train)

    def fit(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        self.autotune(x_train, x_test, y_train, y_test)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        df = df.fillna(0)
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


def _make_random_forest(problem: str) -> BaseClassMlModel:
    """Create a RandomForest model appropriate for the problem type."""
    if problem == "regression":
        return RandomForestRegressionModel()
    else:
        return RandomForestClassificationModel()


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
    "randomforest": {
        "name": "RandomForest (sklearn)",
        "factory": _make_random_forest,
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
