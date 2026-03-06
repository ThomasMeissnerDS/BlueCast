"""Shared pytest fixtures for BlueCast tests.

Session-scoped fixtures avoid redundant model training across test files.
"""

import numpy as np
import pandas as pd
import pytest

from bluecast.blueprints.cast import BlueCast
from bluecast.blueprints.cast_regression import BlueCastRegression
from bluecast.config.training_config import (
    CatboostTuneParamsConfig,
    CatboostTuneParamsRegressionConfig,
    TrainingConfig,
)
from bluecast.tests.make_data.create_data import (
    create_synthetic_dataframe,
    create_synthetic_dataframe_regression,
)

_FAST_TRAINING_CONFIG_KWARGS = dict(
    hyperparameter_tuning_rounds=2,
    hypertuning_cv_folds=2,
    autotune_model=False,
    calculate_shap_values=False,
    enable_feature_selection=False,
    cat_encoding_via_ml_algorithm=False,
    use_full_data_for_final_model=False,
    early_stopping_rounds=None,
)


@pytest.fixture(scope="session")
def fast_training_config():
    """TrainingConfig tuned for speed in tests."""
    return TrainingConfig(**_FAST_TRAINING_CONFIG_KWARGS)


@pytest.fixture(scope="session")
def synthetic_binary_data():
    """200-row synthetic binary classification dataset."""
    return create_synthetic_dataframe(200, random_state=42)


@pytest.fixture(scope="session")
def synthetic_regression_data():
    """200-row synthetic regression dataset."""
    return create_synthetic_dataframe_regression(200, random_state=42)


@pytest.fixture(scope="session")
def trained_bluecast_binary(synthetic_binary_data):
    """Pre-trained BlueCast binary classifier (session-scoped)."""
    conf = TrainingConfig(**_FAST_TRAINING_CONFIG_KWARGS)
    automl = BlueCast(class_problem="binary", conf_training=conf)
    df = synthetic_binary_data.copy()
    automl.fit(df, target_col="target")
    return automl


@pytest.fixture(scope="session")
def trained_bluecast_regression(synthetic_regression_data):
    """Pre-trained BlueCast regression model (session-scoped)."""
    conf = TrainingConfig(**_FAST_TRAINING_CONFIG_KWARGS)
    automl = BlueCastRegression(class_problem="regression", conf_training=conf)
    df = synthetic_regression_data.copy()
    automl.fit(df, target_col="target")
    return automl


@pytest.fixture(scope="session")
def small_inline_data():
    """Standard 6-feature, 20-row DataFrame used across many tests."""
    x_train = pd.DataFrame(
        {
            "feature1": [i for i in range(20)],
            "feature2": [i for i in range(20)],
            "feature3": [i for i in range(20)],
            "feature4": [i for i in range(20)],
            "feature5": [i for i in range(20)],
            "feature6": [i for i in range(20)],
        }
    )
    x_test = pd.DataFrame(
        {
            "feature1": [i for i in range(20)],
            "feature2": [i for i in range(20)],
            "feature3": [i for i in range(20)],
            "feature4": [i for i in range(20)],
            "feature5": [i for i in range(20)],
            "feature6": [i for i in range(20)],
        }
    )
    y_train = pd.Series([0] * 10 + [1] * 10)
    y_test = pd.Series([0] * 10 + [1] * 10)
    return x_train, x_test, y_train, y_test


@pytest.fixture(scope="session")
def small_regression_data():
    """Standard 6-feature, 20-row DataFrame for regression tests."""
    x = pd.DataFrame(
        {
            "feature1": list(range(20)),
            "feature2": list(range(20)),
            "feature3": list(range(20)),
            "feature4": list(range(20)),
            "feature5": list(range(20)),
            "feature6": list(range(20)),
        }
    )
    y = pd.Series(np.random.default_rng(42).standard_normal(20))
    return x.copy(), x.copy(), y.copy(), y.copy()
