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
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from bluecast.ml_modelling.base_classes import (
    BaseClassMlModel,
    BaseClassMlRegressionModel,
)

logger = logging.getLogger(__name__)


class MLPClassificationModel(BaseClassMlModel):
    def __init__(
        self, scoring: str = "roc_auc", cv_folds: int = 5, random_state: int = 300
    ):
        self.model = None
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
        from sklearn.preprocessing import StandardScaler

        from bluecast.ml_modelling.pytorch_models import PyTorchMLPClassifier

        if y_train.nunique() > 2 and self.scoring == "roc_auc":
            self.scoring = "roc_auc_ovr"

        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        x_train_np = self.scaler.fit_transform(self.imputer.fit_transform(x_train))
        x_train = pd.DataFrame(x_train_np, columns=x_train.columns)

        def objective(trial):
            params = {
                "hidden_layer_sizes": trial.suggest_categorical(
                    "hidden_layer_sizes", [(50,), (100,), (50, 50), (100, 50)]
                ),
                "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
                "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                "learning_rate_init": trial.suggest_float(
                    "learning_rate_init", 1e-4, 1e-1, log=True
                ),
            }
            model = PyTorchMLPClassifier(
                random_state=self.random_state,
                max_iter=200,
                **params,
            )
            skfold = StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )
            return cross_val_score(
                model, x_train, y_train, cv=skfold, scoring=self.scoring
            ).mean()

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(
            direction="maximize", sampler=TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=10, timeout=120)

        completed = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        best_params = study.best_params if completed else {"hidden_layer_sizes": (100,)}
        self.model = PyTorchMLPClassifier(
            random_state=self.random_state,
            max_iter=200,
            **best_params,
        )  # type: ignore
        self.model.fit(x_train, y_train)  # type: ignore

    def fit(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        self.autotune(x_train, x_test, y_train, y_test)

    def predict(self, df: pd.DataFrame):
        df_scaled = pd.DataFrame(
            self.scaler.transform(self.imputer.transform(df)), columns=df.columns
        )
        proba_matrix = self.model.predict_proba(df_scaled)  # type: ignore
        classes = self.model.predict(df_scaled)  # type: ignore
        if proba_matrix.shape[1] == 2:
            return proba_matrix[:, 1], classes
        return proba_matrix, classes


class MLPRegressionModel(BaseClassMlRegressionModel):
    def __init__(
        self,
        scoring: str = "neg_mean_absolute_error",
        cv_folds: int = 5,
        random_state: int = 300,
    ):
        self.model = None
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
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import (
            QuantileTransformer,
            StandardScaler,
        )

        from bluecast.ml_modelling.pytorch_models import PyTorchMLPRegressor

        self.scaler = StandardScaler()
        target_transformer_choices = ["standard", "quantile"]

        conf_tuning = getattr(self, "conf_tuning", {})
        tuning_rounds = conf_tuning.get("tuning_rounds", 15)
        nn_max_iter = conf_tuning.get("nn_max_iter", 200)

        tuning_timeout = conf_tuning.get("tuning_max_runtime", 120)

        def objective(trial):
            params = {
                "hidden_layer_sizes": trial.suggest_categorical(
                    "hidden_layer_sizes",
                    [(50,), (100,), (200,), (50, 50), (100, 50), (100, 100)],
                ),
                "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
                "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                "learning_rate_init": trial.suggest_float(
                    "learning_rate_init", 1e-4, 1e-1, log=True
                ),
                "batch_size": trial.suggest_categorical(
                    "batch_size", [32, 64, 128, 256, 512]
                ),
                "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5),
            }
            target_transformer_type = trial.suggest_categorical(
                "target_transformer_type", target_transformer_choices
            )

            imputer_strategy = trial.suggest_categorical(
                "imputer_strategy", ["mean", "median", "constant"]
            )

            if target_transformer_type == "standard":
                target_transformer = StandardScaler()
            elif target_transformer_type == "quantile":
                target_transformer = QuantileTransformer(
                    output_distribution="normal", random_state=self.random_state
                )

            base_model = PyTorchMLPRegressor(
                random_state=self.random_state,
                max_iter=nn_max_iter,
                scoring=self.scoring,
                early_stopping_rounds=15,
                training_deadline=getattr(self, "training_deadline_", None),
                **params,
            )
            # Wrap with target scaling so CV scores reflect inverse-transformed predictions
            wrapped = TransformedTargetRegressor(
                regressor=base_model, transformer=target_transformer
            )

            if imputer_strategy == "constant":
                imputer = SimpleImputer(strategy="constant", fill_value=0)
            else:
                imputer = SimpleImputer(strategy=imputer_strategy)

            steps = [
                ("imputer", imputer),
            ]

            steps.extend([("scaler", StandardScaler()), ("estimator", wrapped)])
            pipeline = Pipeline(steps)

            from sklearn.model_selection import StratifiedKFold
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            y_binned = le.fit_transform(pd.qcut(y_train, 5, duplicates="drop"))
            cv = StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )
            splits = list(cv.split(x_train, y_binned))
            return cross_val_score(
                pipeline, x_train, y_train, cv=splits, scoring=self.scoring
            ).mean()

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(
            direction="maximize", sampler=TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=tuning_rounds, timeout=tuning_timeout)

        completed = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]

        self.convergence_info_ = {
            "nn_max_iter_used": nn_max_iter,
            "best_trial_score": study.best_value if completed else None,
            "trials_completed": len(completed),
            "trials_requested": tuning_rounds,
        }

        if not completed:
            best_params = {
                "hidden_layer_sizes": (100,),
                "target_transformer_type": "standard",
                "imputer_strategy": "median",
            }
            self.best_tuning_score_ = 0.0
        else:
            best_params = study.best_params.copy()
            self.best_tuning_score_ = study.best_value

        target_transformer_type = best_params.pop("target_transformer_type", "standard")
        imputer_strategy = best_params.pop("imputer_strategy", "median")

        if imputer_strategy == "constant":
            self.imputer = SimpleImputer(strategy="constant", fill_value=0)
        else:
            self.imputer = SimpleImputer(strategy=imputer_strategy)

        base_mlp = PyTorchMLPRegressor(
            random_state=self.random_state,
            max_iter=min(nn_max_iter * 2, 1000),
            scoring=self.scoring,
            early_stopping_rounds=30,
            training_deadline=getattr(self, "training_deadline_", None),
            **best_params,
        )

        if target_transformer_type == "standard":
            self.target_scaler = StandardScaler()
        elif target_transformer_type == "quantile":
            self.target_scaler = QuantileTransformer(
                output_distribution="normal", random_state=self.random_state
            )

        # Wrap final model — predict() auto inverse-transforms
        self.model = TransformedTargetRegressor(
            regressor=base_mlp, transformer=self.target_scaler
        )  # type: ignore

        x_train_np = self.scaler.fit_transform(self.imputer.fit_transform(x_train))
        self.model.fit(x_train_np, y_train)  # type: ignore

    def fit(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        self.autotune(x_train, x_test, y_train, y_test)

    def predict(self, df: pd.DataFrame):
        x_np = self.scaler.transform(self.imputer.transform(df))

        # TransformedTargetRegressor.predict() auto inverse-transforms
        return self.model.predict(x_np)  # type: ignore


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
                "max_iter": trial.suggest_int(
                    "max_iter",
                    conf_tuning.get("histgb_max_iter_min", 100),
                    conf_tuning.get("histgb_max_iter_max", 1000),
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    conf_tuning.get("histgb_lr_min", 0.01),
                    conf_tuning.get("histgb_lr_max", 0.1),
                    log=True,
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    conf_tuning.get("histgb_depth_min", 3),
                    conf_tuning.get("histgb_depth_max", 9),
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf",
                    conf_tuning.get("histgb_min_samples_min", 10),
                    conf_tuning.get("histgb_min_samples_max", 50),
                ),
                "l2_regularization": trial.suggest_float(
                    "l2_regularization",
                    conf_tuning.get("histgb_l2_min", 1e-6),
                    conf_tuning.get("histgb_l2_max", 10.0),
                    log=True,
                ),
            }
            model = HistGradientBoostingClassifier(
                random_state=self.random_state,
                early_stopping=True,
                validation_fraction=0.1,
                **params,
            )
            skfold = StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )
            scores = cross_val_score(
                model, x_train, y_train, cv=skfold, scoring=self.scoring
            )
            return scores.mean()

        tuning_timeout = conf_tuning.get("tuning_max_runtime", 120)

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(
            direction="maximize", sampler=TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=tuning_rounds, timeout=tuning_timeout)

        # Guard against no completed trials
        completed = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if not completed:
            logger.warning(
                "FALLBACK: HistGB classification all Optuna trials failed, using defaults."
            )
            best_params: dict = {}
            self.best_tuning_score_ = 0.0
        else:
            best_params = study.best_params
            self.best_tuning_score_ = study.best_value
            logger.info(
                f"HistGB classification best params: {best_params} (score: {study.best_value:.4f})"
            )

        self.model = HistGradientBoostingClassifier(
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1,
            **best_params,
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
                "max_iter": trial.suggest_int(
                    "max_iter",
                    conf_tuning.get("histgb_max_iter_min", 100),
                    conf_tuning.get("histgb_max_iter_max", 1000),
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    conf_tuning.get("histgb_lr_min", 0.01),
                    conf_tuning.get("histgb_lr_max", 0.1),
                    log=True,
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    conf_tuning.get("histgb_depth_min", 3),
                    conf_tuning.get("histgb_depth_max", 9),
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf",
                    conf_tuning.get("histgb_min_samples_min", 10),
                    conf_tuning.get("histgb_min_samples_max", 50),
                ),
                "l2_regularization": trial.suggest_float(
                    "l2_regularization",
                    conf_tuning.get("histgb_l2_min", 1e-6),
                    conf_tuning.get("histgb_l2_max", 10.0),
                    log=True,
                ),
            }
            from bluecast.ai.metrics import get_tree_criterion_from_scoring

            loss = get_tree_criterion_from_scoring(self.scoring)
            model = HistGradientBoostingRegressor(
                random_state=self.random_state,
                loss=loss,
                early_stopping=True,
                validation_fraction=0.1,
                **params,
            )
            from sklearn.model_selection import StratifiedKFold
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            y_binned = le.fit_transform(pd.qcut(y_train, 5, duplicates="drop"))
            cv = StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )
            splits = list(cv.split(x_train, y_binned))
            scores = cross_val_score(
                model, x_train, y_train, cv=splits, scoring=self.scoring
            )
            return scores.mean()

        tuning_timeout = conf_tuning.get("tuning_max_runtime", 120)

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(
            direction="maximize", sampler=TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=tuning_rounds, timeout=tuning_timeout)

        self.best_tuning_score_ = study.best_value
        logger.info(
            f"HistGB regression best params: {study.best_params} (score: {study.best_value:.4f})"
        )
        from bluecast.ai.metrics import get_tree_criterion_from_scoring

        loss = get_tree_criterion_from_scoring(self.scoring)
        self.model = HistGradientBoostingRegressor(
            random_state=self.random_state,
            loss=loss,
            early_stopping=True,
            validation_fraction=0.1,
            **study.best_params,
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

        from sklearn.pipeline import Pipeline

        conf_tuning = getattr(self, "conf_tuning", {})
        tuning_rounds = conf_tuning.get("tuning_rounds", 15)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    conf_tuning.get("rf_estimators_min", 50),
                    conf_tuning.get("rf_estimators_max", 150),
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    conf_tuning.get("rf_max_depth_min", 3),
                    conf_tuning.get("rf_max_depth_max", 15),
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf",
                    conf_tuning.get("rf_min_samples_min", 1),
                    conf_tuning.get("rf_min_samples_max", 20),
                ),
                "max_features": trial.suggest_float(
                    "max_features",
                    conf_tuning.get("rf_max_features_min", 0.1),
                    conf_tuning.get("rf_max_features_max", 1.0),
                ),
            }

            imputer_strategy = trial.suggest_categorical(
                "imputer_strategy", ["mean", "median", "constant"]
            )
            if imputer_strategy == "constant":
                imputer = SimpleImputer(strategy="constant", fill_value=0)
            else:
                imputer = SimpleImputer(strategy=imputer_strategy)

            model = RandomForestClassifier(
                random_state=self.random_state, n_jobs=-1, **params
            )
            pipeline = Pipeline([("imputer", imputer), ("model", model)])
            skfold = StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )
            scores = cross_val_score(
                pipeline, x_train, y_train, cv=skfold, scoring=self.scoring
            )
            return scores.mean()

        tuning_timeout = conf_tuning.get("tuning_max_runtime", 900)

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(
            direction="maximize", sampler=TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=tuning_rounds, timeout=tuning_timeout)

        completed = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if not completed:
            logger.warning(
                "FALLBACK: RandomForest classification all Optuna trials failed, using defaults."
            )
            best_params = {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_leaf": 2,
                "max_features": 0.5,
                "imputer_strategy": "median",
            }
            self.best_tuning_score_ = 0.0
        else:
            best_params = study.best_params
            self.best_tuning_score_ = study.best_value
            logger.info(
                f"RandomForest classification best params: {best_params} (score: {study.best_value:.4f})"
            )

        imputer_strategy = best_params.pop("imputer_strategy", "median")
        if imputer_strategy == "constant":
            self.imputer = SimpleImputer(strategy="constant", fill_value=0)
        else:
            self.imputer = SimpleImputer(strategy=imputer_strategy)

        self.model = RandomForestClassifier(
            random_state=self.random_state, n_jobs=-1, **best_params
        )

        x_train_imputed = pd.DataFrame(
            self.imputer.fit_transform(x_train), columns=x_train.columns
        )
        self.model.fit(x_train_imputed, y_train)

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
        df_imputed = pd.DataFrame(self.imputer.transform(df), columns=df.columns)
        proba_matrix = self.model.predict_proba(df_imputed)
        classes = self.model.predict(df_imputed)
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
        from sklearn.pipeline import Pipeline

        conf_tuning = getattr(self, "conf_tuning", {})
        tuning_rounds = conf_tuning.get("tuning_rounds", 15)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    conf_tuning.get("rf_estimators_min", 2),
                    conf_tuning.get("rf_estimators_max", 150),
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    conf_tuning.get("rf_max_depth_min", 2),
                    conf_tuning.get("rf_max_depth_max", 20),
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf",
                    conf_tuning.get("rf_min_samples_min", 1),
                    conf_tuning.get("rf_min_samples_max", 20),
                ),
                "max_features": trial.suggest_float(
                    "max_features",
                    conf_tuning.get("rf_max_features_min", 0.1),
                    conf_tuning.get("rf_max_features_max", 1.0),
                ),
            }

            imputer_strategy = trial.suggest_categorical(
                "imputer_strategy", ["mean", "median", "constant"]
            )
            if imputer_strategy == "constant":
                imputer = SimpleImputer(strategy="constant", fill_value=0)
            else:
                imputer = SimpleImputer(strategy=imputer_strategy)

            from bluecast.ai.metrics import get_tree_criterion_from_scoring

            criterion = get_tree_criterion_from_scoring(self.scoring)
            model = RandomForestRegressor(
                random_state=self.random_state, n_jobs=-1, criterion=criterion, **params
            )
            pipeline = Pipeline([("imputer", imputer), ("model", model)])
            from sklearn.model_selection import StratifiedKFold
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            y_binned = le.fit_transform(pd.qcut(y_train, 5, duplicates="drop"))
            cv = StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )
            splits = list(cv.split(x_train, y_binned))
            scores = cross_val_score(
                pipeline, x_train, y_train, cv=splits, scoring=self.scoring
            )
            return scores.mean()

        tuning_timeout = conf_tuning.get("tuning_max_runtime", 900)

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(
            direction="maximize", sampler=TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=tuning_rounds, timeout=tuning_timeout)

        completed = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if not completed:
            logger.warning(
                "FALLBACK: RandomForest regression all Optuna trials failed, using defaults."
            )
            best_params = {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_leaf": 2,
                "max_features": 0.5,
                "imputer_strategy": "median",
            }
        else:
            best_params = study.best_params
            logger.info(
                f"RandomForest regression best params: {best_params} (score: {study.best_value:.4f})"
            )

        self.best_tuning_score_ = study.best_value
        imputer_strategy = best_params.pop("imputer_strategy", "median")
        if imputer_strategy == "constant":
            self.imputer = SimpleImputer(strategy="constant", fill_value=0)
        else:
            self.imputer = SimpleImputer(strategy=imputer_strategy)

        from bluecast.ai.metrics import get_tree_criterion_from_scoring

        criterion = get_tree_criterion_from_scoring(self.scoring)
        self.model = RandomForestRegressor(
            random_state=self.random_state,
            n_jobs=-1,
            criterion=criterion,
            **best_params,
        )

        x_train_imputed = pd.DataFrame(
            self.imputer.fit_transform(x_train), columns=x_train.columns
        )
        self.model.fit(x_train_imputed, y_train)

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
        df_imputed = pd.DataFrame(self.imputer.transform(df), columns=df.columns)
        preds = self.model.predict(df_imputed)
        return preds


# ---------------------------------------------------------------------------
# SoftOrdering1DCNN wrappers
# ---------------------------------------------------------------------------


class SO1DCNNClassificationModel(BaseClassMlModel):
    """SoftOrdering1DCNN for classification with Optuna-based hyperparameter tuning."""

    def __init__(
        self, scoring: str = "roc_auc", cv_folds: int = 5, random_state: int = 300
    ):
        self.model = None
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
        from sklearn.preprocessing import StandardScaler

        from bluecast.ml_modelling.pytorch_models import PyTorchSO1DCNNClassifier

        if y_train.nunique() > 2 and self.scoring == "roc_auc":
            self.scoring = "roc_auc_ovr"

        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        x_train_np = self.scaler.fit_transform(self.imputer.fit_transform(x_train))
        x_train = pd.DataFrame(x_train_np, columns=x_train.columns)

        conf_tuning = getattr(self, "conf_tuning", {})
        tuning_rounds = conf_tuning.get("tuning_rounds", 10)

        def objective(trial):
            params = {
                "sign_size": trial.suggest_categorical("sign_size", [16, 32, 64]),
                "cha_input": trial.suggest_categorical("cha_input", [8, 16, 32]),
                "cha_hidden": trial.suggest_categorical("cha_hidden", [16, 32, 64]),
                "K": trial.suggest_int("K", 1, 3),
                "dropout_input": trial.suggest_float("dropout_input", 0.0, 0.5),
                "dropout_hidden": trial.suggest_float("dropout_hidden", 0.0, 0.5),
                "dropout_output": trial.suggest_float("dropout_output", 0.0, 0.5),
                "learning_rate_init": trial.suggest_float(
                    "learning_rate_init", 1e-4, 1e-1, log=True
                ),
                "batch_size": trial.suggest_categorical(
                    "batch_size",
                    conf_tuning.get("nn_batch_size_choices", [128, 256, 512, 1024]),
                ),
            }
            model = PyTorchSO1DCNNClassifier(
                random_state=self.random_state,
                max_iter=500,
                early_stopping_rounds=20,
                scoring=self.scoring,
                **params,
            )
            skfold = StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )
            return cross_val_score(
                model, x_train, y_train, cv=skfold, scoring=self.scoring
            ).mean()

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(
            direction="maximize", sampler=TPESampler(seed=self.random_state)
        )
        tuning_timeout = conf_tuning.get("tuning_max_runtime", 120)
        study.optimize(objective, n_trials=tuning_rounds, timeout=tuning_timeout)

        completed = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        best_params = study.best_params if completed else {}
        self.model = PyTorchSO1DCNNClassifier(
            random_state=self.random_state,
            max_iter=500,
            early_stopping_rounds=20,
            **best_params,
        )  # type: ignore
        self.model.fit(x_train, y_train)  # type: ignore

    def fit(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        self.autotune(x_train, x_test, y_train, y_test)

    def predict(self, df: pd.DataFrame):
        df_scaled = pd.DataFrame(
            self.scaler.transform(self.imputer.transform(df)), columns=df.columns
        )
        proba_matrix = self.model.predict_proba(df_scaled)  # type: ignore
        classes = self.model.predict(df_scaled)  # type: ignore
        if proba_matrix.shape[1] == 2:
            return proba_matrix[:, 1], classes
        return proba_matrix, classes


class SO1DCNNRegressionModel(BaseClassMlRegressionModel):
    """SoftOrdering1DCNN for regression with Optuna-based hyperparameter tuning."""

    def __init__(
        self,
        scoring: str = "neg_mean_absolute_error",
        cv_folds: int = 5,
        random_state: int = 300,
    ):
        self.model = None
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
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import (
            QuantileTransformer,
            StandardScaler,
        )

        from bluecast.ml_modelling.pytorch_models import PyTorchSO1DCNNRegressor

        self.scaler = StandardScaler()
        target_transformer_choices = ["standard", "quantile"]

        conf_tuning = getattr(self, "conf_tuning", {})
        tuning_rounds = conf_tuning.get("tuning_rounds", 15)
        nn_max_iter = conf_tuning.get("nn_max_iter", 200)

        tuning_timeout = conf_tuning.get("tuning_max_runtime", 120)

        def objective(trial):
            params = {
                "sign_size": trial.suggest_categorical("sign_size", [16, 32, 64]),
                "cha_input": trial.suggest_categorical("cha_input", [8, 16, 32]),
                "cha_hidden": trial.suggest_categorical("cha_hidden", [16, 32, 64]),
                "K": trial.suggest_int("K", 1, 3),
                "dropout_input": trial.suggest_float("dropout_input", 0.0, 0.5),
                "dropout_hidden": trial.suggest_float("dropout_hidden", 0.0, 0.5),
                "dropout_output": trial.suggest_float("dropout_output", 0.0, 0.5),
                "learning_rate_init": trial.suggest_float(
                    "learning_rate_init", 1e-4, 1e-1, log=True
                ),
                "batch_size": trial.suggest_categorical(
                    "batch_size",
                    conf_tuning.get("nn_batch_size_choices", [128, 256, 512, 1024]),
                ),
            }
            target_transformer_type = trial.suggest_categorical(
                "target_transformer_type", target_transformer_choices
            )
            imputer_strategy = trial.suggest_categorical(
                "imputer_strategy", ["mean", "median", "constant"]
            )

            if target_transformer_type == "standard":
                target_transformer = StandardScaler()
            elif target_transformer_type == "quantile":
                target_transformer = QuantileTransformer(
                    output_distribution="normal", random_state=self.random_state
                )

            base_model = PyTorchSO1DCNNRegressor(
                random_state=self.random_state,
                max_iter=nn_max_iter,
                scoring=self.scoring,
                early_stopping_rounds=15,
                training_deadline=getattr(self, "training_deadline_", None),
                **params,
            )
            wrapped = TransformedTargetRegressor(
                regressor=base_model, transformer=target_transformer
            )

            if imputer_strategy == "constant":
                imputer = SimpleImputer(strategy="constant", fill_value=0)
            else:
                imputer = SimpleImputer(strategy=imputer_strategy)

            steps = [
                ("imputer", imputer),
                ("scaler", StandardScaler()),
                ("estimator", wrapped),
            ]
            pipeline = Pipeline(steps)

            from sklearn.model_selection import StratifiedKFold
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            y_binned = le.fit_transform(pd.qcut(y_train, 5, duplicates="drop"))
            cv = StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )
            splits = list(cv.split(x_train, y_binned))
            return cross_val_score(
                pipeline, x_train, y_train, cv=splits, scoring=self.scoring
            ).mean()

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(
            direction="maximize", sampler=TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=tuning_rounds, timeout=tuning_timeout)

        completed = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]

        self.convergence_info_ = {
            "nn_max_iter_used": nn_max_iter,
            "best_trial_score": study.best_value if completed else None,
            "trials_completed": len(completed),
            "trials_requested": tuning_rounds,
        }

        if not completed:
            best_params = {
                "target_transformer_type": "standard",
                "imputer_strategy": "median",
            }
            self.best_tuning_score_ = 0.0
        else:
            best_params = study.best_params.copy()
            self.best_tuning_score_ = study.best_value

        target_transformer_type = best_params.pop("target_transformer_type", "standard")
        imputer_strategy = best_params.pop("imputer_strategy", "median")

        if imputer_strategy == "constant":
            self.imputer = SimpleImputer(strategy="constant", fill_value=0)
        else:
            self.imputer = SimpleImputer(strategy=imputer_strategy)

        base_cnn = PyTorchSO1DCNNRegressor(
            random_state=self.random_state,
            max_iter=min(nn_max_iter * 2, 1000),
            scoring=self.scoring,
            early_stopping_rounds=30,
            training_deadline=getattr(self, "training_deadline_", None),
            **best_params,
        )

        if target_transformer_type == "standard":
            self.target_scaler = StandardScaler()
        elif target_transformer_type == "quantile":
            self.target_scaler = QuantileTransformer(
                output_distribution="normal", random_state=self.random_state
            )

        # Wrap final model — predict() auto inverse-transforms
        self.model = TransformedTargetRegressor(
            regressor=base_cnn, transformer=self.target_scaler
        )  # type: ignore

        x_train_np = self.scaler.fit_transform(self.imputer.fit_transform(x_train))
        self.model.fit(x_train_np, y_train)  # type: ignore

    def fit(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        self.autotune(x_train, x_test, y_train, y_test)

    def predict(self, df: pd.DataFrame):
        x_np = self.scaler.transform(self.imputer.transform(df))

        # TransformedTargetRegressor.predict() auto inverse-transforms
        return self.model.predict(x_np)  # type: ignore


# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------

ArchFactory = Callable[[str], Optional[BaseClassMlModel]]


def _make_linear(problem: str) -> BaseClassMlModel:
    """Create a linear model appropriate for the problem type."""
    if problem == "regression":
        from bluecast.blueprints.custom_model_recipes import RegularizedRegressionModel

        return RegularizedRegressionModel()  # type: ignore
    else:
        from bluecast.blueprints.custom_model_recipes import LogisticRegressionModel

        return LogisticRegressionModel()


def _make_histgb(problem: str) -> BaseClassMlModel:
    """Create a HistGradientBoosting model appropriate for the problem type."""
    if problem == "regression":
        return HistGBRegressionModel()  # type: ignore
    else:
        return HistGBClassificationModel()


def _make_random_forest(problem: str) -> BaseClassMlModel:
    """Create a RandomForest model appropriate for the problem type."""
    if problem == "regression":
        return RandomForestRegressionModel()  # type: ignore
    else:
        return RandomForestClassificationModel()


def _make_so1dcnn(problem: str) -> BaseClassMlModel:
    """Create a SoftOrdering1DCNN model appropriate for the problem type."""
    if problem == "regression":
        return SO1DCNNRegressionModel()  # type: ignore
    else:
        return SO1DCNNClassificationModel()


ArchInfo = Dict[str, Any]

ARCHITECTURE_REGISTRY: Dict[str, ArchInfo] = {
    # Ordered fast → slow so users get a quick baseline and can cancel early.
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
    "mlp": {
        "name": "MLP Neural Network (PyTorch)",
        "factory": lambda problem: (
            MLPRegressionModel()
            if problem == "regression"
            else MLPClassificationModel()
        ),
        "supports": ["binary", "multiclass", "regression"],
    },
    "so1dcnn": {
        "name": "SoftOrdering1DCNN (PyTorch)",
        "factory": _make_so1dcnn,
        "supports": ["binary", "multiclass", "regression"],
    },
    "xgboost": {
        "name": "XGBoost",
        "factory": lambda problem: None,  # XGBoost uses conf_xgboost in BlueCast
        "supports": ["binary", "multiclass", "regression"],
        "use_xgboost_native": True,
    },
    "catboost": {
        "name": "CatBoost (default)",
        "factory": lambda problem: None,  # None = use default BlueCast pipeline
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
