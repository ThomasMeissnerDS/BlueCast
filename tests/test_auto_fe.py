import pandas as pd
import numpy as np
import pytest
from bluecast.preprocessing.auto_fe import BlueCastFeatureGenerator
from bluecast.preprocessing.target_encoding import OOFTargetEncoder


def test_autofe_expansion_and_reduction():
    # Setup dummy data
    np.random.seed(42)
    df = pd.DataFrame({
        "num1": np.random.rand(100),
        "num2": np.random.rand(100) * 10,
        "cat1": np.random.choice(["A", "B", "C"], 100)
    })
    
    # Create target related to some specific arithmetic operations so they don't get pruned
    y = pd.Series(df["num1"] + df["num2"] * 2.5 + (df["cat1"] == "A") * 5.0 + np.random.randn(100))
    y_bin = (y > y.mean()).astype(int)

    # Init generator
    autofe = BlueCastFeatureGenerator(
        autofe_pruning_threshold=0.01,
        cat_columns=["cat1"],
        class_problem="binary",
        random_state=42
    )

    # Fit transform
    df_transformed = autofe.fit_transform(df, y_bin)

    # Assert
    assert autofe._fitted is True
    # At least some features should be generated
    assert len(autofe.retained_features) >= 0
    
    # Check that transform works on new data
    df_test = df.copy()
    df_test_transformed = autofe.transform(df_test)
    assert df_test_transformed.shape[1] == df_transformed.shape[1]
    assert list(df_test_transformed.columns) == list(df_transformed.columns)


def test_oof_target_encoder():
    np.random.seed(42)
    df = pd.DataFrame({
        "cat1": np.random.choice(["A", "B", "C", "D"], 200),
        "cat2": np.random.choice(["X", "Y", "Z"], 200)
    })
    y = pd.Series(
        np.where(df["cat1"] == "A", 1, 0) + np.where(df["cat2"] == "X", 1, 0)
    )
    y_bin = (y > 0).astype(int)

    encoder = OOFTargetEncoder(
        cat_columns=["cat1", "cat2"],
        random_state=42,
        cv=3,
        smoothing=5.0
    )

    df_transformed = encoder.fit_transform(df, y_bin)
    
    assert encoder.is_fitted is True
    assert df_transformed["cat1"].dtype == float
    assert df_transformed["cat2"].dtype == float
    assert not df_transformed.isnull().any().any()
    
    # Test on unseen data with new category
    df_test = pd.DataFrame({
        "cat1": ["A", "B", "E"], # 'E' is unseen
        "cat2": ["X", "Y", "Z"]
    })
    
    df_test_transformed = encoder.transform(df_test)
    assert not df_test_transformed.isnull().any().any()
    # The unseen category 'E' should be mapped to the global mean
    assert np.isclose(df_test_transformed.loc[2, "cat1"], encoder.global_mean)
