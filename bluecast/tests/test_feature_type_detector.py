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


def test_casting_int_and_float_columns():
    # Create a sample dataframe for testing
    df = pd.DataFrame(
        {
            "int_col": [1, 2, 3, 4],
            "float_col": [1.1, 2.2, 3.3, 4.4],
            "mixed_int_str_col": ["1", "2", "3", "4"],
            "mixed_float_str_col": ["1.1", "2.2", "3.3", "4.4"],
            "non_castable_str_col": ["a", "b", "c", "d"],
        }
    )

    # Initialize the FeatureTypeDetector object
    detector = FeatureTypeDetector()

    # Test fit_transform_feature_types
    transformed_df = detector.fit_transform_feature_types(df)

    # Check if the integer column is correctly cast to Int64
    assert (
        transformed_df["int_col"].dtype == "int64"
    ), f"Expected 'Int64', but got {transformed_df['int_col'].dtype}"

    # Check if the float column is correctly cast to float64
    assert (
        transformed_df["float_col"].dtype == "float64"
    ), f"Expected 'float64', but got {transformed_df['float_col'].dtype}"

    # Check if the mixed integer and string column is correctly cast to Int64
    assert (
        transformed_df["mixed_int_str_col"].dtype == "float64"
    ), f"Expected 'float64', but got {transformed_df['mixed_int_str_col'].dtype}"

    # Check if the mixed float and string column is correctly cast to float64
    assert (
        transformed_df["mixed_float_str_col"].dtype == "float64"
    ), f"Expected 'float64', but got {transformed_df['mixed_float_str_col'].dtype}"

    # Check if the non-castable string column remains as object
    assert (
        transformed_df["non_castable_str_col"].dtype == "object"
    ), f"Expected 'object', but got {transformed_df['non_castable_str_col'].dtype}"


def test_casting_edge_cases():
    # Test case with negative integers, floats and mixed strings
    df = pd.DataFrame(
        {
            "int_col": [-1, -2, -3, -4],
            "float_col": [-1.1, -2.2, -3.3, -4.4],
            "mixed_int_str_col": ["-1", "-2", "-3", "-4"],
            "mixed_float_str_col": ["-1.1", "-2.2", "-3.3", "-4.4"],
        }
    )

    # Initialize the FeatureTypeDetector object
    detector = FeatureTypeDetector()

    # Test fit_transform_feature_types
    transformed_df = detector.fit_transform_feature_types(df)

    # Check if the negative integer column is correctly cast to Int64
    assert (
        transformed_df["int_col"].dtype == "int64"
    ), f"Expected 'int64', but got {transformed_df['int_col'].dtype}"

    # Check if the negative float column is correctly cast to float64
    assert (
        transformed_df["float_col"].dtype == "float64"
    ), f"Expected 'float64', but got {transformed_df['float_col'].dtype}"

    # Check if the mixed negative integer and string column is cast to float64
    assert (
        transformed_df["mixed_int_str_col"].dtype == "float64"
    ), f"Expected 'float64', but got {transformed_df['mixed_int_str_col'].dtype}"

    # Check if the mixed negative float and string column is correctly cast to float64
    assert (
        transformed_df["mixed_float_str_col"].dtype == "float64"
    ), f"Expected 'float64', but got {transformed_df['mixed_float_str_col'].dtype}"


def test_casting_with_nan_values():
    # Create a dataframe with NaN values
    df = pd.DataFrame(
        {
            "int_col": [1, 2, np.nan, 4],
            "float_col": [1.1, 2.2, np.nan, 4.4],
            "mixed_int_str_col": ["1", np.nan, "3", "4"],
            "mixed_float_str_col": ["1.1", "2.2", np.nan, "4.4"],
        }
    )

    # Initialize the FeatureTypeDetector object
    detector = FeatureTypeDetector()

    # Test fit_transform_feature_types
    transformed_df = detector.fit_transform_feature_types(df)

    assert transformed_df["int_col"].isnull().sum().sum() == 1
    assert transformed_df["float_col"].isnull().sum().sum() == 1
    assert transformed_df["mixed_int_str_col"].isnull().sum().sum() == 1
    assert transformed_df["mixed_float_str_col"].isnull().sum().sum() == 1

    # Check if the integer column with NaN is correctly cast to Int64
    assert (
        transformed_df["int_col"].dtype == "float64"
    ), f"Expected 'float64', but got {transformed_df['int_col'].dtype}"

    # Check if the float column with NaN is correctly cast to float64
    assert (
        transformed_df["float_col"].dtype == "float64"
    ), f"Expected 'float64', but got {transformed_df['float_col'].dtype}"

    # Check if the mixed integer and string column with NaN is correctly cast to Int64
    assert (
        transformed_df["mixed_int_str_col"].dtype == "float64"
    ), f"Expected 'float64', but got {transformed_df['mixed_int_str_col'].dtype}"

    # Check if the mixed float and string column with NaN is correctly cast to float64
    assert (
        transformed_df["mixed_float_str_col"].dtype == "float64"
    ), f"Expected 'float64', but got {transformed_df['mixed_float_str_col'].dtype}"


def test_check_if_column_is_int():
    # Initialize the FeatureTypeDetector object
    detector = FeatureTypeDetector()

    # Test cases for integer columns
    int_series = pd.Series([1, 2, 3, 4])
    assert detector.check_if_column_is_int(int_series)

    # Test cases for mixed type (ints and floats)
    mixed_series = pd.Series([1, 2.5, 3, 4.0])
    assert not detector.check_if_column_is_int(mixed_series)  # Has ints

    # Test cases for all floats
    float_series = pd.Series([1.1, 2.2, 3.3, 4.4])
    assert not detector.check_if_column_is_int(float_series)  # No ints

    # Test cases for strings that look like ints
    string_int_series = pd.Series(["1", "2", "3", "4"])
    assert not detector.check_if_column_is_int(string_int_series)  # All strings


def test_check_if_column_is_float():
    # Initialize the FeatureTypeDetector object
    detector = FeatureTypeDetector()

    # Test cases for float columns
    float_series = pd.Series([1.1, 2.2, 3.3, 4.4])
    assert detector.check_if_column_is_float(float_series)

    # Test cases for mixed type (ints and floats)
    mixed_series = pd.Series([1, 2.5, 3, 4.0])
    assert detector.check_if_column_is_float(mixed_series)  # Has floats

    # Test cases for all integers
    int_series = pd.Series([1, 2, 3, 4])
    assert not detector.check_if_column_is_float(int_series)  # No floats

    # Test cases for strings that look like floats
    string_float_series = pd.Series(["1.1", "2.2", "3.3", "4.4"])
    assert not (detector.check_if_column_is_float(string_float_series))  # All strings
