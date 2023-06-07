import numpy as np
import pandas as pd

from bluecast.preprocessing.nulls_and_infs import fill_infinite_values, fill_nulls


def test_fill_nulls():
    # Create a test DataFrame with null values
    df = pd.DataFrame({"A": [1, None, 3, 4], "B": [None, 6, 7, None]})

    # Test case 1: Fill null values with 0
    expected_output = pd.DataFrame({"A": [1, 0, 3, 4], "B": [0, 6, 7, 0]})
    assert (
        fill_nulls(df, fill_with=0).astype(float).equals(expected_output.astype(float))
    )

    # Test case 2: Fill null values with NaN
    df_alt = pd.DataFrame(
        {"A": [1, float("nan"), 3, 4], "B": [float("nan"), 6, 7, float("nan")]}
    )
    assert fill_nulls(df_alt).astype(float).equals(expected_output.astype(float))


def test_fill_infinite_values():
    # Create a sample DataFrame with infinite values
    df = pd.DataFrame({"A": [1, 2, np.inf, 4], "B": [np.inf, 6, -np.inf, 8]})

    # Test with default fill_with value (0)
    expected_result = pd.DataFrame({"A": [1, 2, 0, 4], "B": [0, 6, 0, 8]})
    assert fill_infinite_values(df.astype(float)).equals(expected_result.astype(float))

    # Test with a different fill_with value (10)
    expected_result = pd.DataFrame({"A": [1, 2, 10, 4], "B": [10, 6, 10, 8]})
    assert fill_infinite_values(df, fill_with=10).equals(expected_result.astype(float))

    # Test with a DataFrame that does not contain infinite values
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    expected_result = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    assert fill_infinite_values(df).equals(expected_result)
