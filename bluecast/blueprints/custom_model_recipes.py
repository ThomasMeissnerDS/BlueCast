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
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        import optuna
        from optuna.samplers import TPESampler
        from sklearn.model_selection import cross_val_score

        # Auto-detect multiclass and fix scoring
        if y_train.nunique() > 2 and self.scoring == "roc_auc":
            self.scoring = "roc_auc_ovr"

        x_train_np = self.scaler.fit_transform(self.imputer.fit_transform(x_train))
        x_train = pd.DataFrame(x_train_np, columns=x_train.columns)
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
            scores = cross_val_score(model, x_train, y_train, cv=skfold, scoring=self.scoring)
            return scores.mean()

        tuning_timeout = conf_tuning.get("tuning_max_runtime", 120)

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=self.random_state))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            study.optimize(objective, n_trials=tuning_rounds, timeout=tuning_timeout)

        # Guard against no completed trials
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            logging.warning("LogisticRegression: all Optuna trials failed, using defaults.")
            best_params: dict = {}
            self.best_tuning_score_ = 0.0
        else:
            logging.info(f"Best LogisticRegression params: {study.best_params} (score: {study.best_value:.4f})")
            best_params = study.best_params.copy()
            self.best_tuning_score_ = study.best_value
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
        df_scaled = pd.DataFrame(self.scaler.transform(self.imputer.transform(df)), columns=df.columns)
        df = df_scaled
        if self.model is None:
            raise ValueError("No fitted model has been found.")
        proba_matrix = self.model.predict_proba(df)
        classes = self.model.predict(df)
        if proba_matrix.shape[1] == 2:
            return proba_matrix[:, 1], classes
        return proba_matrix, classes


class RegularizedRegressionModel(BaseClassMlRegressionModel):
    """Regularized regression model with GridSearchCV-based hyperparameter tuning.

    Searches over Ridge, Lasso, and ElasticNet with various alpha values.

    :param scoring: Scoring metric for GridSearchCV (e.g. 'neg_mean_squared_error', 'r2').
    :param cv_folds: Number of cross-validation folds.
    :param random_state: Random seed for reproducibility.
    """

    def __init__(
        self,
        scoring: str = "neg_mean_absolute_error",
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
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import RobustScaler, QuantileTransformer, FunctionTransformer
        from sklearn.impute import SimpleImputer
        self.scaler = RobustScaler()
        import optuna
        from optuna.samplers import TPESampler
        from sklearn.model_selection import cross_val_score
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.preprocessing import StandardScaler as TargetStdScaler

        conf_tuning = getattr(self, "conf_tuning", {})
        tuning_rounds = conf_tuning.get("tuning_rounds", 15)

        target_transformer_choices = ["standard", "quantile"]

        def objective(trial):
            model_type = trial.suggest_categorical("model_type", ["ridge", "lasso", "elasticnet"])
            alpha = trial.suggest_float("alpha", conf_tuning.get("reg_alpha_min", 1e-4), conf_tuning.get("reg_alpha_max", 1e3), log=True)
            
            target_transformer_type = trial.suggest_categorical(
                "target_transformer_type", target_transformer_choices
            )
            
            imputer_strategy = trial.suggest_categorical(
                "imputer_strategy", ["mean", "median", "most_frequent", "constant"]
            )
            
            if model_type == "ridge":
                estimator = Ridge(alpha=alpha, random_state=self.random_state)
            elif model_type == "lasso":
                estimator = Lasso(alpha=alpha, max_iter=100000, random_state=self.random_state)
            else:
                l1_ratio = trial.suggest_float("l1_ratio", conf_tuning.get("reg_l1_ratio_min", 0.1), conf_tuning.get("reg_l1_ratio_max", 0.9))
                estimator = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=100000, random_state=self.random_state)
            
            if target_transformer_type == "standard":
                target_transformer = TargetStdScaler()
            elif target_transformer_type == "quantile":
                target_transformer = QuantileTransformer(output_distribution="normal", random_state=self.random_state)

            # Wrap in TransformedTargetRegressor so target is scaled during CV
            # and predictions are automatically inverse-transformed for scoring
            wrapped = TransformedTargetRegressor(
                regressor=estimator, transformer=target_transformer
            )
            
            if imputer_strategy == "constant":
                imputer = SimpleImputer(strategy="constant", fill_value=0)
            else:
                imputer = SimpleImputer(strategy=imputer_strategy)
                
            steps = [
                ("imputer", imputer),
                ("scaler", RobustScaler()),
                ("estimator", wrapped)
            ]
                
            pipeline = Pipeline(steps)
            
            from sklearn.preprocessing import LabelEncoder
            from sklearn.model_selection import StratifiedKFold
            le = LabelEncoder()
            y_binned = le.fit_transform(pd.qcut(y_train, 10, duplicates="drop"))
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            splits = list(cv.split(x_train, y_binned))
            scores = cross_val_score(pipeline, x_train, y_train, cv=splits, scoring=self.scoring)
            return scores.mean()

        tuning_timeout = conf_tuning.get("tuning_max_runtime", 120)

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=self.random_state))
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            study.optimize(objective, n_trials=tuning_rounds, timeout=tuning_timeout)

        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            import logging
            logging.error("FALLBACK: RegularizedRegression all Optuna trials failed, using defaults!")
            best_params = {"model_type": "ridge", "alpha": 1.0, "target_transformer_type": "standard", "imputer_strategy": "median"}
            study_best_value = float("-inf")
            self.best_tuning_score_ = 0.0
        else:
            best_params = study.best_params.copy()
            study_best_value = study.best_value
            self.best_tuning_score_ = study.best_value
            
        model_type = best_params.pop("model_type")
        target_transformer_type = best_params.get("target_transformer_type", "standard")
        imputer_strategy = best_params.get("imputer_strategy", "median")
        
        if imputer_strategy == "constant":
            self.imputer = SimpleImputer(strategy="constant", fill_value=0)
        else:
            self.imputer = SimpleImputer(strategy=imputer_strategy)
            
        if model_type == "ridge":
            base_model = Ridge(random_state=self.random_state, **{k: v for k, v in best_params.items() if k in ["alpha"]})
        elif model_type == "lasso":
            base_model = Lasso(max_iter=100000, random_state=self.random_state, **{k: v for k, v in best_params.items() if k in ["alpha"]})
        else:
            base_model = ElasticNet(max_iter=100000, random_state=self.random_state, **{k: v for k, v in best_params.items() if k in ["alpha", "l1_ratio"]})

        if target_transformer_type == "standard":
            self.target_scaler = TargetStdScaler()
        elif target_transformer_type == "quantile":
            self.target_scaler = QuantileTransformer(output_distribution="normal", random_state=self.random_state)

        # Wrap final model with target scaling — predict() auto inverse-transforms
        self.model = TransformedTargetRegressor(
            regressor=base_model, transformer=self.target_scaler
        )

        self.best_model_type = model_type
        import logging
        logging.info(f"Best regression model: {self.best_model_type} (score: {study_best_value:.4f}, target_scaler: {target_transformer_type})")
        
        # Fit final model on fully transformed data
        x_train_np = self.scaler.fit_transform(self.imputer.fit_transform(x_train))
        self.model.fit(x_train_np, y_train)

    def fit(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        self.autotune(x_train, x_test, y_train, y_test)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        x_np = self.scaler.transform(self.imputer.transform(df))
        
        if self.model is not None:
            # TransformedTargetRegressor.predict() auto inverse-transforms
            preds = self.model.predict(x_np)
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
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        x_train_np = self.scaler.fit_transform(self.imputer.fit_transform(x_train))
        x_train = pd.DataFrame(x_train_np, columns=x_train.columns)
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
        df_scaled = pd.DataFrame(self.scaler.transform(self.imputer.transform(df)), columns=df.columns)
        df = df_scaled
        if isinstance(self.model, LinearRegression):
            preds = self.model.predict(df)
            return preds
        else:
            raise ValueError("No fitted model has been found.")


# Catboost
