import numpy as np
import pandas as pd

from bluecast.preprocessing.feature_types import FeatureTypeDetector


def test_feature_type_detector():
    # Create a sample dataframe for testing
    df = pd.DataFrame(
        {
            "numeric_col1": [1, 2, 3],
            "numeric_col2": [4, 5, 6],
            "boolean_col1": [True, False, True],
            "boolean_col2": [False, True, False],
            "date_col1": ["2021-01-01", "2021-02-01", "2021-03-01"],
            "date_col2": ["2022-01-01", "2022-02-01", "2022-03-01"],
            "categorical_col1": ["cat", "dog", "cat"],
            "categorical_col2": ["dog", "cat", "dog"],
        }
    )

    # Initialize the FeatureTypeDetector object
    detector = FeatureTypeDetector(
        num_columns=["numeric_col1", "numeric_col2"],
        cat_columns=["categorical_col1", "categorical_col2"],
        date_columns=["date_col1", "date_col2"],
    )

    # Test fit_transform_feature_types
    transformed_df = detector.fit_transform_feature_types(df)

    # Numeric columns should remain the same
    assert transformed_df["numeric_col1"].equals(df["numeric_col1"])
    assert transformed_df["numeric_col2"].equals(df["numeric_col2"])

    # Boolean columns should be cast to bool type
    assert transformed_df["boolean_col1"].dtype == np.bool_
    assert transformed_df["boolean_col2"].dtype == np.bool_

    # Date columns should be cast to datetime[ns] type
    assert transformed_df["date_col1"].dtype == np.dtype("datetime64[ns]")
    assert transformed_df["date_col2"].dtype == np.dtype("datetime64[ns]")

    # Categorical columns should remain the same
    assert transformed_df["categorical_col1"].equals(df["categorical_col1"])
    assert transformed_df["categorical_col2"].equals(df["categorical_col2"])

    # Test transform_feature_types
    transformed_df = detector.transform_feature_types(transformed_df, ignore_cols=[])

    # All columns should have the expected data types
    assert transformed_df["numeric_col1"].dtype in ["int64", "float64"]
    assert transformed_df["numeric_col2"].dtype in ["int64", "float64"]
    assert transformed_df["boolean_col1"].dtype == np.bool_
    assert transformed_df["boolean_col2"].dtype == np.bool_
    assert transformed_df["date_col1"].dtype == np.dtype("datetime64[ns]")
    assert transformed_df["date_col2"].dtype == np.dtype("datetime64[ns]")
    assert transformed_df["categorical_col1"].dtype == np.object_
    assert transformed_df["categorical_col2"].dtype == np.object_

    # Ensure the transformed dataframe is the same as the original dataframe
    assert transformed_df.equals(df)
