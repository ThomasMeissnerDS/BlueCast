import pandas as pd

from bluecast.preprocessing.remove_collinearity import remove_correlated_columns


def test_remove_correlated_columns_positive_correlation():
    data = {
        "A": [1, 2, 3, 4, 5],
        "B": [2, 4, 6, 8, 10],  # B is perfectly positively correlated with A
        "C": [1, 3, 2, 5, 4],  # C has low correlation with A
    }
    df = pd.DataFrame(data)

    result_df = remove_correlated_columns(df, threshold=0.9)

    assert "A" in result_df.columns
    assert "B" not in result_df.columns
    assert "C" in result_df.columns


def test_remove_correlated_columns_negative_correlation():
    data = {
        "A": [1, 2, 3, 4, 5],
        "B": [5, 4, 3, 2, 1],  # B is perfectly negatively correlated with A
        "C": [1, 3, 2, 5, 4],  # C has low correlation
    }
    df = pd.DataFrame(data)

    result_df = remove_correlated_columns(df, threshold=0.9)

    assert "A" in result_df.columns
    assert "B" not in result_df.columns, "Negative correlation should also be caught"
    assert "C" in result_df.columns


def test_remove_correlated_columns_no_removal():
    data = {
        "A": [1, 2, 3, 4, 5],
        "B": [1, 3, 2, 5, 4],  # Low correlation with A
        "C": [3, 1, 4, 2, 5],  # Low correlation with A and B
    }
    df = pd.DataFrame(data)

    result_df = remove_correlated_columns(df, threshold=0.9)

    assert list(result_df.columns) == ["A", "B", "C"]


def test_remove_correlated_columns_does_not_mutate_input():
    data = {
        "A": [1, 2, 3, 4, 5],
        "B": [2, 4, 6, 8, 10],
        "C": [1, 3, 2, 5, 4],
    }
    df = pd.DataFrame(data)
    original_cols = list(df.columns)

    remove_correlated_columns(df, threshold=0.9)

    assert list(df.columns) == original_cols, "Original DataFrame should not be mutated"


def test_remove_correlated_columns_different_threshold():
    data = {
        "A": [1, 2, 3, 4, 5],
        "B": [2, 4, 6, 8, 10],  # Perfectly correlated with A
        "C": [1, 3, 2, 5, 4],  # Low correlation
    }
    df = pd.DataFrame(data)

    result_high = remove_correlated_columns(df, threshold=1.01)
    assert len(result_high.columns) == 3, "No columns removed at threshold > 1.0"

    result_low = remove_correlated_columns(df, threshold=0.8)
    assert "B" not in result_low.columns
