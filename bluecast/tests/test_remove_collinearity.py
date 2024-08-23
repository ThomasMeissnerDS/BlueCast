import pandas as pd

from bluecast.preprocessing.remove_collinearity import remove_correlated_columns


def test_remove_correlated_columns_high_correlation():
    # Create a DataFrame with high correlation between columns
    data = {
        "A": [1, 2, 3, 4, 5],
        "B": [2, 4, 6, 8, 10],  # B is perfectly correlated with A
        "C": [5, 4, 3, 2, 1],  # C is not correlated with A or B
    }
    df = pd.DataFrame(data)

    result_df = remove_correlated_columns(df, threshold=0.9)

    # B should be removed because it's highly correlated with A
    expected_df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "C": [5, 4, 3, 2, 1]})

    pd.testing.assert_frame_equal(result_df, expected_df)


def test_remove_correlated_columns_no_removal():
    # Create a DataFrame with no high correlations
    data = {
        "A": [1, 2, 3, 4, 5],
        "B": [2, 3, 4, 5, 6],  # B is not perfectly correlated with A
        "C": [5, 4, 3, 2, 1],  # C is not correlated with A or B
    }
    df = pd.DataFrame(data)

    result_df = remove_correlated_columns(df, threshold=0.9)

    # No columns should be removed
    pd.testing.assert_frame_equal(result_df, df)


def test_remove_correlated_columns_no_correlation():
    # Create a DataFrame where no columns are correlated above the threshold
    data = {
        "A": [1, 2, 3, 4, 5],
        "B": [2, 3, 4, 5, 6],
        "C": [5, 4, 3, 2, 1],
        "D": [1, 2, 1, 2, 1],
    }
    df = pd.DataFrame(data)

    result_df = remove_correlated_columns(df, threshold=0.9)

    # Since no columns are correlated above the threshold, the original DataFrame should be returned
    pd.testing.assert_frame_equal(result_df, df)


def test_remove_correlated_columns_different_threshold():
    # Create a DataFrame with some correlation
    data = {
        "A": [1, 2, 3, 4, 5],
        "B": [2, 4, 6, 8, 10],  # B is perfectly correlated with A
        "C": [5, 5, 5, 5, 5],  # C is constant, should have no correlation
    }
    df = pd.DataFrame(data)

    # Use a higher threshold, so no columns should be removed
    result_df = remove_correlated_columns(df, threshold=0.95)

    pd.testing.assert_frame_equal(result_df, df)

    # Use a lower threshold, so column B should be removed
    result_df = remove_correlated_columns(df, threshold=0.8)

    expected_df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "C": [5, 5, 5, 5, 5]})

    pd.testing.assert_frame_equal(result_df, expected_df)
