import numpy as np
import pandas as pd

from bluecast.preprocessing.feature_types import FeatureTypeDetector


def test_feature_type_detector():
    # Create a sample dataframe for testing
    df = pd.DataFrame(
        {
            "numeric_col1": [1, 2, 3, 4],
            "numeric_col2": [4, np.nan, 6, 7],
            "float_col1": [1.1, 2.0, 3.0, 4.0],
            "float_col2": [4.5, np.nan, 6.8, 9.0],
            "float_from_string_col1": ["1.1", "2.", "3.", "4."],
            "float_from_string_col2": ["4.5", "5.6", "6.8", "9."],
            "boolean_col1": [True, False, True, False],
            "boolean_col2": [False, True, False, True],
            "date_col1": ["2021-01-01", "2021-02-01", "2021-03-01", "2021-04-01"],
            "date_col2": ["2022-01-01", "2022-02-01", "2022-03-01", "2022-04-01"],
            "categorical_col1": ["cat", "dog", "cat", "dog"],
            "categorical_col2": ["dog", "cat", "dog", "cat"],
            "categorical_col3": ["e", "f", "h", "o"],
        }
    )

    # Initialize the FeatureTypeDetector object
    detector = FeatureTypeDetector(
        num_columns=["numeric_col1", "numeric_col2", "float_col1", "float_col2"],
        cat_columns=["categorical_col1", "categorical_col2", "categorical_col3"],
        date_columns=["date_col1", "date_col2"],
    )

    # Test fit_transform_feature_types
    transformed_df = detector.fit_transform_feature_types(df)

    # Numeric columns should remain the same
    assert not transformed_df["numeric_col1"].equals(
        df["numeric_col1"]
    )  # changed to pd type Int64
    assert transformed_df["numeric_col2"].equals(df["numeric_col2"])

    assert transformed_df["float_col1"].equals(df["float_col1"])
    assert transformed_df["float_col2"].equals(df["float_col2"])

    assert not transformed_df["float_from_string_col1"].equals(
        df["float_from_string_col1"]
    )
    assert not transformed_df["float_from_string_col2"].equals(
        df["float_from_string_col2"]
    )

    # Boolean columns should be cast to bool type
    assert transformed_df["boolean_col1"].dtype == np.bool_
    assert transformed_df["boolean_col2"].dtype == np.bool_

    # Date columns should be cast to datetime[ns] type
    assert transformed_df["date_col1"].dtype == np.dtype("datetime64[ns]")
    assert transformed_df["date_col2"].dtype == np.dtype("datetime64[ns]")

    # Categorical columns should remain the same
    assert transformed_df["categorical_col1"].equals(df["categorical_col1"])
    assert transformed_df["categorical_col2"].equals(df["categorical_col2"])
    assert transformed_df["categorical_col3"].equals(df["categorical_col3"])

    # Test transform_feature_types
    transformed_df = detector.transform_feature_types(transformed_df, ignore_cols=[])

    print("Transformed after not passing feature types")
    print(transformed_df.info())

    # All columns should have the expected data types
    assert transformed_df["numeric_col1"].dtype in ["int64", "Int64", "float64"]
    assert transformed_df["numeric_col2"].dtype in ["int64", "Int64", "float64"]
    assert transformed_df["float_col1"].dtype in ["float64"]
    assert transformed_df["float_col2"].dtype in ["float64"]
    assert transformed_df["float_from_string_col1"].dtype in ["float64"]
    assert transformed_df["float_from_string_col2"].dtype in ["float64"]
    assert transformed_df["boolean_col1"].dtype == np.bool_
    assert transformed_df["boolean_col2"].dtype == np.bool_
    assert transformed_df["date_col1"].dtype == np.dtype("datetime64[ns]")
    assert transformed_df["date_col2"].dtype == np.dtype("datetime64[ns]")
    assert transformed_df["categorical_col1"].dtype == np.object_
    assert transformed_df["categorical_col2"].dtype == np.object_
    assert transformed_df["categorical_col3"].dtype == np.object_

    # Ensure the transformed dataframe is not the same as the original dataframe (transformed datetime columns)
    assert not transformed_df.equals(df)
