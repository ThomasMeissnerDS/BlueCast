import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from bluecast.blueprints.preprocessing_recipes import PreprocessingForLinearModels
from bluecast.preprocessing.custom import CustomPreprocessing


# Mocking remove_correlated_columns for testing purposes
def mock_remove_correlated_columns(df, threshold):
    return df.loc[:, df.columns[:-1]]  # Just drop the last column for simplicity


@pytest.fixture
def sample_data():
    # Create a sample DataFrame with numerical data
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
    target = pd.Series(y, name="target")
    return df, target


@pytest.fixture
def preprocessing_instance():
    # Create an instance of PreprocessingForLinearModels
    return PreprocessingForLinearModels(
        num_columns=["feature_0", "feature_1", "feature_2", "feature_3", "feature_4"]
    )


def test_initialization(preprocessing_instance):
    # Test if the class initializes correctly
    assert isinstance(preprocessing_instance, CustomPreprocessing)
    assert preprocessing_instance.num_columns == [
        "feature_0",
        "feature_1",
        "feature_2",
        "feature_3",
        "feature_4",
    ]
    assert preprocessing_instance.non_correlated_columns == []


def test_fit_transform(sample_data, preprocessing_instance, monkeypatch):
    df, target = sample_data

    # Mock the remove_correlated_columns function
    monkeypatch.setattr(
        "bluecast.preprocessing.remove_collinearity.remove_correlated_columns",
        mock_remove_correlated_columns,
    )

    transformed_df, transformed_target = preprocessing_instance.fit_transform(
        df, target
    )

    # Check if transformed data has the correct shape and type
    assert isinstance(transformed_df, pd.DataFrame)
    assert isinstance(transformed_target, pd.Series)
    assert transformed_df.shape == (100, 4)  # Since one column is removed by mock
    assert transformed_target.shape == (100,)

    # Check if missing values and infinite values are handled correctly
    assert not transformed_df.isnull().any().any()
    assert not np.isinf(transformed_df).any().any()


def test_transform(sample_data, preprocessing_instance, monkeypatch):
    df, target = sample_data

    # Fit-transform first to simulate the normal flow
    monkeypatch.setattr(
        "bluecast.preprocessing.remove_collinearity.remove_correlated_columns",
        mock_remove_correlated_columns,
    )
    preprocessing_instance.fit_transform(df, target)

    # Now transform new data
    new_df = df.copy()
    new_df.loc[0, "feature_0"] = np.nan  # Introduce missing value

    transformed_df, transformed_target = preprocessing_instance.transform(
        new_df, target
    )

    # Check if transformed data has the correct shape and type
    assert isinstance(transformed_df, pd.DataFrame)
    assert isinstance(transformed_target, pd.Series)
    assert transformed_df.shape == (100, 4)
    assert transformed_target.shape == (100,)

    # Check if missing values and infinite values are handled correctly
    assert not transformed_df.isnull().any().any()
    assert not np.isinf(transformed_df).any().any()


def test_no_numerical_columns():
    df = pd.DataFrame({"category": ["A", "B", "C"], "binary": [1, 0, 1]})
    target = pd.Series([1, 0, 1])

    preprocessing = PreprocessingForLinearModels(num_columns=[])
    transformed_df, transformed_target = preprocessing.fit_transform(df, target)

    # Since there are no numerical columns, the DataFrame should remain unchanged
    pd.testing.assert_frame_equal(transformed_df, df)
    pd.testing.assert_series_equal(transformed_target, target)
