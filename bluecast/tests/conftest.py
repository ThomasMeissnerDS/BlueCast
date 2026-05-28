"""Shared pytest fixtures for BlueCast tests.

Session-scoped fixtures avoid redundant model training across test files.
"""

import numpy.core._methods as np_methods
import pytest

from bluecast.blueprints.cast import BlueCast
from bluecast.blueprints.cast_regression import BlueCastRegression
from bluecast.config.training_config import TrainingConfig
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


# Monkey-patch numpy to fix pandas coverage crash
original_amax = np_methods._amax
original_amin = getattr(np_methods, "_amin", None)

def patched_amax(*args, **kwargs):
    import numpy.core._methods as np_methods

    a = args[0] if len(args) > 0 else kwargs.get("a")
    axis = args[1] if len(args) > 1 else kwargs.get("axis", None)
    out = args[2] if len(args) > 2 else kwargs.get("out", None)
    keepdims = args[3] if len(args) > 3 else kwargs.get("keepdims", False)
    initial = args[4] if len(args) > 4 else kwargs.get("initial", getattr(np_methods, "_NoValue", None))
    where = args[5] if len(args) > 5 else kwargs.get("where", True)

    if initial is getattr(np_methods, "_NoValue", None) or initial is None:
        initial = None
        
    return np_methods.umr_maximum(a, axis, None, out, keepdims, initial, where)

def patched_amin(*args, **kwargs):
    import numpy.core._methods as np_methods

    a = args[0] if len(args) > 0 else kwargs.get("a")
    axis = args[1] if len(args) > 1 else kwargs.get("axis", None)
    out = args[2] if len(args) > 2 else kwargs.get("out", None)
    keepdims = args[3] if len(args) > 3 else kwargs.get("keepdims", False)
    initial = args[4] if len(args) > 4 else kwargs.get("initial", getattr(np_methods, "_NoValue", None))
    where = args[5] if len(args) > 5 else kwargs.get("where", True)

    if initial is getattr(np_methods, "_NoValue", None) or initial is None:
        initial = None
        
    return np_methods.umr_minimum(a, axis, None, out, keepdims, initial, where)

np_methods._amax = patched_amax
if hasattr(np_methods, "_amin"):
    np_methods._amin = patched_amin
