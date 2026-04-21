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
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold

from bluecast.ml_modelling.base_classes import (
    BaseClassMlModel,
    BaseClassMlRegressionModel,
)

logger = logging.getLogger(__name__)



class MLPClassificationModel(BaseClassMlModel):
    def __init__(self, scoring: str = "roc_auc", cv_folds: int = 5, random_state: int = 300):
        self.model = None
        self.scoring = scoring
        self.cv_folds = cv_folds
        self.random_state = random_state

    def autotune(self, x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> None:
        import optuna
        from optuna.samplers import TPESampler
        from sklearn.model_selection import cross_val_score
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        import numpy as np

        if y_train.nunique() > 2 and self.scoring == "roc_auc":
            self.scoring = "roc_auc_ovr"

        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        x_train_np = self.scaler.fit_transform(self.imputer.fit_transform(x_train))
        x_train = pd.DataFrame(x_train_np, columns=x_train.columns)

        def objective(trial):
            params = {
                "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes", [(50,), (100,), (50, 50), (100, 50)]),
                "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
                "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True),
            }
            model = MLPClassifier(random_state=self.random_state, max_iter=200, early_stopping=True, **params)
            skfold = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            return cross_val_score(model, x_train, y_train, cv=skfold, scoring=self.scoring).mean()

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=10, timeout=120)

        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        best_params = study.best_params if completed else {"hidden_layer_sizes": (100,)}
        from sklearn.neural_network import MLPClassifier
        self.model = MLPClassifier(random_state=self.random_state, max_iter=200, early_stopping=True, **best_params)
        self.model.fit(x_train, y_train)

    def fit(self, x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> None:
        self.autotune(x_train, x_test, y_train, y_test)

    def predict(self, df: pd.DataFrame):
        df_scaled = pd.DataFrame(self.scaler.transform(self.imputer.transform(df)), columns=df.columns)
        proba_matrix = self.model.predict_proba(df_scaled)
        classes = self.model.predict(df_scaled)
        if proba_matrix.shape[1] == 2:
            return proba_matrix[:, 1], classes
        return proba_matrix, classes

class MLPRegressionModel(BaseClassMlRegressionModel):
    def __init__(self, scoring: str = "neg_mean_absolute_error", cv_folds: int = 5, random_state: int = 300):
        self.model = None
        self.scoring = scoring
        self.cv_folds = cv_folds
        self.random_state = random_state

    def autotune(self, x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> None:
        import optuna
        from optuna.samplers import TPESampler
        from sklearn.model_selection import cross_val_score
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        import numpy as np

        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        x_train_np = self.scaler.fit_transform(self.imputer.fit_transform(x_train))
        x_train = pd.DataFrame(x_train_np, columns=x_train.columns)

        def objective(trial):
            params = {
                "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes", [(50,), (100,), (50, 50), (100, 50)]),
                "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
                "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True),
            }
            model = MLPRegressor(random_state=self.random_state, max_iter=200, early_stopping=True, **params)
            kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            return cross_val_score(model, x_train, y_train, cv=kfold, scoring=self.scoring).mean()

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=10, timeout=120)

        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        best_params = study.best_params if completed else {"hidden_layer_sizes": (100,)}
        from sklearn.neural_network import MLPRegressor
        self.model = MLPRegressor(random_state=self.random_state, max_iter=200, early_stopping=True, **best_params)
        self.model.fit(x_train, y_train)

    def fit(self, x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> None:
        self.autotune(x_train, x_test, y_train, y_test)

    def predict(self, df: pd.DataFrame):
        df_scaled = pd.DataFrame(self.scaler.transform(self.imputer.transform(df)), columns=df.columns)
        return self.model.predict(df_scaled)


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

        # Auto-detect multiclass and fix scoring
        if y_train.nunique() > 2 and self.scoring == "roc_auc":
            self.scoring = "roc_auc_ovr"

        conf_tuning = getattr(self, "conf_tuning", {})
        tuning_rounds = conf_tuning.get("tuning_rounds", 15)

        def objective(trial):
            params = {
                "max_iter": trial.suggest_int("max_iter", conf_tuning.get("histgb_max_iter_min", 100), conf_tuning.get("histgb_max_iter_max", 1000)),
                "learning_rate": trial.suggest_float("learning_rate", conf_tuning.get("histgb_lr_min", 0.01), conf_tuning.get("histgb_lr_max", 0.1), log=True),
                "max_depth": trial.suggest_int("max_depth", conf_tuning.get("histgb_depth_min", 3), conf_tuning.get("histgb_depth_max", 9)),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", conf_tuning.get("histgb_min_samples_min", 10), conf_tuning.get("histgb_min_samples_max", 50)),
                "l2_regularization": trial.suggest_float("l2_regularization", conf_tuning.get("histgb_l2_min", 1e-6), conf_tuning.get("histgb_l2_max", 10.0), log=True),
            }
            model = HistGradientBoostingClassifier(
                random_state=self.random_state, early_stopping=True, validation_fraction=0.1, **params
            )
            skfold = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, x_train, y_train, cv=skfold, scoring=self.scoring)
            return scores.mean()

        tuning_timeout = conf_tuning.get("tuning_max_runtime", 120)

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=tuning_rounds, timeout=tuning_timeout)

        # Guard against no completed trials
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            logger.warning("HistGB: all Optuna trials failed, using defaults.")
            best_params: dict = {}
        else:
            best_params = study.best_params
            logger.info(f"HistGB classification best params: {best_params} (score: {study.best_value:.4f})")

        self.model = HistGradientBoostingClassifier(
            random_state=self.random_state, early_stopping=True, validation_fraction=0.1, **best_params
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
        proba_matrix = self.model.predict_proba(df)
        classes = self.model.predict(df)
        if proba_matrix.shape[1] == 2:
            # Binary: return positive-class probabilities
            return proba_matrix[:, 1], classes
        # Multiclass: return full probability matrix
        return proba_matrix, classes


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
                "max_iter": trial.suggest_int("max_iter", conf_tuning.get("histgb_max_iter_min", 100), conf_tuning.get("histgb_max_iter_max", 1000)),
                "learning_rate": trial.suggest_float("learning_rate", conf_tuning.get("histgb_lr_min", 0.01), conf_tuning.get("histgb_lr_max", 0.1), log=True),
                "max_depth": trial.suggest_int("max_depth", conf_tuning.get("histgb_depth_min", 3), conf_tuning.get("histgb_depth_max", 9)),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", conf_tuning.get("histgb_min_samples_min", 10), conf_tuning.get("histgb_min_samples_max", 50)),
                "l2_regularization": trial.suggest_float("l2_regularization", conf_tuning.get("histgb_l2_min", 1e-6), conf_tuning.get("histgb_l2_max", 10.0), log=True),
            }
            loss = "absolute_error" if "absolute_error" in self.scoring else "squared_error"
            model = HistGradientBoostingRegressor(
                random_state=self.random_state, loss=loss, early_stopping=True, validation_fraction=0.1, **params
            )
            kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring=self.scoring)
            return scores.mean()

        tuning_timeout = conf_tuning.get("tuning_max_runtime", 120)

        optuna.logging.set_verbosity(optuna.logging.ERROR)
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

        # Auto-detect multiclass and fix scoring
        if y_train.nunique() > 2 and self.scoring == "roc_auc":
            self.scoring = "roc_auc_ovr"

        self.imputer = SimpleImputer(strategy="median")
        x_train = pd.DataFrame(self.imputer.fit_transform(x_train), columns=x_train.columns)
        conf_tuning = getattr(self, "conf_tuning", {})
        tuning_rounds = conf_tuning.get("tuning_rounds", 15)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", conf_tuning.get("rf_estimators_min", 50), conf_tuning.get("rf_estimators_max", 300)),
                "max_depth": trial.suggest_int("max_depth", conf_tuning.get("rf_max_depth_min", 3), conf_tuning.get("rf_max_depth_max", 15)),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", conf_tuning.get("rf_min_samples_min", 1), conf_tuning.get("rf_min_samples_max", 20)),
                "max_features": trial.suggest_float("max_features", conf_tuning.get("rf_max_features_min", 0.1), conf_tuning.get("rf_max_features_max", 1.0)),
            }
            model = RandomForestClassifier(random_state=self.random_state, n_jobs=1, **params)
            skfold = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, x_train, y_train, cv=skfold, scoring=self.scoring)
            return scores.mean()

        tuning_timeout = conf_tuning.get("tuning_max_runtime", 120)

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=tuning_rounds, timeout=tuning_timeout)

        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            logger.warning("RandomForest: all Optuna trials failed, using defaults.")
            best_params = {"n_estimators": 100, "max_depth": None}
        else:
            best_params = study.best_params
            logger.info(f"RandomForest classification best params: {best_params} (score: {study.best_value:.4f})")

        self.model = RandomForestClassifier(random_state=self.random_state, n_jobs=1, **best_params)
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
        df = pd.DataFrame(self.imputer.transform(df), columns=df.columns)
        if self.model is None:
            raise ValueError("No fitted model has been found.")
        proba_matrix = self.model.predict_proba(df)
        classes = self.model.predict(df)
        if proba_matrix.shape[1] == 2:
            return proba_matrix[:, 1], classes
        return proba_matrix, classes


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
        
        self.imputer = SimpleImputer(strategy="median")
        x_train = pd.DataFrame(self.imputer.fit_transform(x_train), columns=x_train.columns)
        conf_tuning = getattr(self, "conf_tuning", {})
        tuning_rounds = conf_tuning.get("tuning_rounds", 15)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", conf_tuning.get("rf_estimators_min", 50), conf_tuning.get("rf_estimators_max", 300)),
                "max_depth": trial.suggest_int("max_depth", conf_tuning.get("rf_max_depth_min", 10), conf_tuning.get("rf_max_depth_max", 50)),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", conf_tuning.get("rf_min_samples_min", 1), conf_tuning.get("rf_min_samples_max", 20)),
                "max_features": trial.suggest_float("max_features", conf_tuning.get("rf_max_features_min", 0.1), conf_tuning.get("rf_max_features_max", 1.0)),
            }
            # Always use squared_error internally for RF because absolute_error is computationally prohibitive
            model = RandomForestRegressor(random_state=self.random_state, criterion="squared_error", n_jobs=1, **params)
            kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring=self.scoring)
            return scores.mean()

        tuning_timeout = conf_tuning.get("tuning_max_runtime", 120)

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=tuning_rounds, timeout=tuning_timeout)

        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            logger.warning("RandomForest: all Optuna regression trials failed, using defaults.")
            best_params = {"n_estimators": 100, "max_depth": None}
        else:
            best_params = study.best_params
            logger.info(f"RandomForest regression best params: {best_params} (score: {study.best_value:.4f})")
            
        self.model = RandomForestRegressor(random_state=self.random_state, criterion="squared_error", n_jobs=1, **best_params)
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
        df = pd.DataFrame(self.imputer.transform(df), columns=df.columns)
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
        "mlp": {
        "name": "MLP Neural Network (sklearn)",
        "factory": lambda problem: MLPRegressionModel() if problem == "regression" else MLPClassificationModel(),
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
