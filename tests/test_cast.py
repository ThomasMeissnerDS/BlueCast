from blueprints.cast import BlueCast
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_train_df_binary_classification():
    """Create a synthetic train df for binary classification."""
    df = pd.DataFrame({
        "a": [np.nan, 2, 3, "A", 5, 6, 7, 8, 9, np.nan],
        "b": [1, 2, 3, "B", 5, 6, 7, 8, 9, 10],
        "c": [1, 2, 3, "A", 5, 6, 7, 8, np.nan, 10],
        "d": [-50, -237, -8, "C", 0, np.nan, 0.3, 0.98, 0.1, 0.2],
        "target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    })
    return df


@pytest.fixture
def synthetic_test_df_binary_classification(synthetic_train_df_binary_classification):
    """Create a synthetic test df for binary classification."""
    df = synthetic_train_df_binary_classification
    df[["a", "b", "c", "d"]] = df[["a", "b", "c", "d"]] * 2
    df = df.drop(columns=["target"])
    return df


def test_fit(synthetic_df_binary_classification, synthetic_test_df_binary_classification):
    """Test that tests the BlueCast class"""
    df = synthetic_df_binary_classification
    automl = BlueCast(class_problem="binary", target_column="target")
    automl.fit(df, target_col="target")
    y_probs, y_classes = automl.predict(synthetic_test_df_binary_classification)
    assert y_probs.shape[0] == synthetic_test_df_binary_classification.shape[0]
    assert y_classes.shape[0] == synthetic_test_df_binary_classification.shape[0]
    assert y_probs.shape[1] == 2
    assert y_classes.shape[1] == 1
    assert y_probs[0][0] == pytest.approx(0.9999997615814209, 0.0001)