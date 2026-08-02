import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from bluecast.blueprints.cast_cv import BlueCastCV
from bluecast.blueprints.cast_cv_regression import BlueCastCVRegression
from bluecast.config.training_config import TrainingConfig
from bluecast.ensemble.ensemble_config import EnsembleConfig
from bluecast.tests.shared_test_helpers import (
    CustomBinaryClassificationModel,
    CustomRegressionModel,
)


def test_bluecast_cv_regression_ignores_hill_climbing():
    """
    Validates that BlueCastCVRegression ignores 'hill_climbing' and 'stacking'
    strategies for internal folds, and properly collapses OOF predictions
    into a 1D array without crashing.
    """
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
    df["target"] = y

    # Force hill_climbing on the CV level
    ensemble_config = EnsembleConfig(
        ensemble_strategy="hill_climbing", regression_eval_metric="mae"
    )
    conf_tuning = TrainingConfig()
    conf_tuning.hypertuning_cv_folds = 2
    conf_tuning.hyperparameter_tuning_rounds = 1

    automl = BlueCastCVRegression(
        conf_tuning=conf_tuning,
        ensemble_config=ensemble_config,
    )

    # We use a dummy model to keep it fast
    automl.ml_model = CustomRegressionModel()

    automl.fit_eval(df, target_col="target")

    # 1. Assert OOF predictions were successfully generated and collapsed into a 1D array
    assert hasattr(automl, "oof_predictions_")
    assert isinstance(automl.oof_predictions_, np.ndarray)
    assert automl.oof_predictions_.ndim == 1
    assert len(automl.oof_predictions_) == len(df)

    # 2. Assert no internal hill_climbing meta-learner was instantiated
    assert (
        not hasattr(automl, "hill_climbing_ensemble")
        or getattr(automl, "hill_climbing_ensemble", None) is None
    )

    # 3. Assert predict() returns a valid Series and blends folds via mean
    preds = automl.predict(df.drop("target", axis=1))
    assert isinstance(preds, pd.Series)
    assert len(preds) == len(df)
    assert not preds.isna().any()


def test_bluecast_cv_ignores_hill_climbing():
    """
    Validates that BlueCastCV ignores 'hill_climbing' and 'stacking'
    strategies for internal folds, and properly collapses OOF predictions
    into a 1D/2D array without crashing.
    """
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
    df["target"] = y

    ensemble_config = EnsembleConfig(ensemble_strategy="hill_climbing")
    conf_tuning = TrainingConfig()
    conf_tuning.hypertuning_cv_folds = 2
    conf_tuning.hyperparameter_tuning_rounds = 1

    automl = BlueCastCV(
        conf_tuning=conf_tuning,
        ensemble_config=ensemble_config,
    )
    automl.ml_model = CustomBinaryClassificationModel()

    automl.fit_eval(df, target_col="target")

    # 1. Assert OOF predictions were collapsed properly
    assert hasattr(automl, "oof_predictions_")
    assert isinstance(automl.oof_predictions_, np.ndarray)
    assert len(automl.oof_predictions_) == len(df)

    # 2. Assert predict() returns a valid tuple
    probs, classes = automl.predict(df.drop("target", axis=1))
    assert isinstance(probs, pd.Series)
    assert isinstance(classes, pd.Series)
    assert len(probs) == len(df)
    assert not probs.isna().any()
