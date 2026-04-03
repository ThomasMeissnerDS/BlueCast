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
from sklearn.exceptions import ConvergenceWarning
import warnings
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
        import optuna
        from optuna.samplers import TPESampler
        from sklearn.model_selection import cross_val_score

        x_train = x_train.fillna(0)
        conf_tuning = getattr(self, "conf_tuning", {})
        tuning_rounds = conf_tuning.get("tuning_rounds", 15)

        def objective(trial):
            penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
            C = trial.suggest_float("C", conf_tuning.get("lr_C_min", 1e-3), conf_tuning.get("lr_C_max", 1e2), log=True)
            class_weight = trial.suggest_categorical("class_weight", ["balanced", None])

            if penalty == "l1":
                solver = "saga"
                l1_ratio = None
            elif penalty == "elasticnet":
                solver = "saga"
                l1_ratio = trial.suggest_float("l1_ratio", conf_tuning.get("lr_l1_ratio_min", 0.1), conf_tuning.get("lr_l1_ratio_max", 0.9))
            else:
                solver = trial.suggest_categorical("solver", ["lbfgs", "newton-cg", "sag", "saga"])
                l1_ratio = None

            params = {"penalty": penalty, "C": C, "class_weight": class_weight, "solver": solver}
            if l1_ratio is not None:
                params["l1_ratio"] = l1_ratio

            model = LogisticRegression(random_state=self.random_state, max_iter=1000, **params)
            skfold = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, x_train, y_train, cv=skfold, scoring=self.scoring, n_jobs=-1)
            return scores.mean()

        tuning_timeout = conf_tuning.get("tuning_max_runtime", 120)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=self.random_state))
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            study.optimize(objective, n_trials=tuning_rounds, timeout=tuning_timeout)

        logging.info(f"Best LogisticRegression params: {study.best_params} (score: {study.best_value:.4f})")
        
        best_params = study.best_params.copy()
        penalty = best_params.get("penalty")
        if penalty == "l1":
            best_params["solver"] = "saga"
        elif penalty == "elasticnet":
            best_params["solver"] = "saga"
        
        if "l1_ratio" in best_params and penalty != "elasticnet":
            del best_params["l1_ratio"]

        self.model = LogisticRegression(random_state=self.random_state, max_iter=1000, **best_params)
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
        df = df.fillna(0)
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
        import optuna
        from optuna.samplers import TPESampler
        from sklearn.model_selection import cross_val_score

        x_train = x_train.fillna(0)
        conf_tuning = getattr(self, "conf_tuning", {})
        tuning_rounds = conf_tuning.get("tuning_rounds", 15)

        def objective(trial):
            model_type = trial.suggest_categorical("model_type", ["ridge", "lasso", "elasticnet"])
            alpha = trial.suggest_float("alpha", conf_tuning.get("reg_alpha_min", 1e-4), conf_tuning.get("reg_alpha_max", 1e3), log=True)
            
            if model_type == "ridge":
                estimator = Ridge(alpha=alpha, random_state=self.random_state)
            elif model_type == "lasso":
                estimator = Lasso(alpha=alpha, max_iter=100000, random_state=self.random_state)
            else:
                l1_ratio = trial.suggest_float("l1_ratio", conf_tuning.get("reg_l1_ratio_min", 0.1), conf_tuning.get("reg_l1_ratio_max", 0.9))
                estimator = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=100000, random_state=self.random_state)
                
            kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(estimator, x_train, y_train, cv=kfold, scoring=self.scoring, n_jobs=-1)
            return scores.mean()

        tuning_timeout = conf_tuning.get("tuning_max_runtime", 120)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=self.random_state))
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            study.optimize(objective, n_trials=tuning_rounds, timeout=tuning_timeout)

        best_params = study.best_params.copy()
        model_type = best_params.pop("model_type")
        
        if model_type == "ridge":
            self.model = Ridge(random_state=self.random_state, **best_params)
        elif model_type == "lasso":
            self.model = Lasso(max_iter=100000, random_state=self.random_state, **best_params)
        else:
            self.model = ElasticNet(max_iter=100000, random_state=self.random_state, **best_params)

        self.best_model_type = model_type
        logging.info(f"Best regression model: {self.best_model_type} (score: {study.best_value:.4f})")
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
        x_train = x_train.fillna(0)
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
        df = df.fillna(0)
        if isinstance(self.model, LinearRegression):
            preds = self.model.predict(df)
            return preds
        else:
            raise ValueError("No fitted model has been found.")


# Catboost
