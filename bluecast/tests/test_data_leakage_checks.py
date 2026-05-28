import pandas as pd
import pytest

from bluecast.eda.data_leakage_checks import (
    detect_categorical_leakage,
    detect_leakage_via_correlation,
)


@pytest.fixture
def create_to_target_correlated_features() -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "target": [1, 2, 3, 4, 5],
            "feature1": [2, -1, 3, 10, 0],
            "feature2": [0, 0, 0, 0, 0],
        }
    )
    return data


@pytest.fixture
def create_to_target_correlated_categorical_features() -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "target": ["A", "B", "C", "A", "B"],
            "feature1": ["X", "Y", "Z", "M", "O"],
            "feature2": ["N", "N", "N", "N", "N"],
        }
    )
    return data


def test_detect_leakage_via_correlation(create_to_target_correlated_features):
    # Test when there is no data leakage
    result = detect_leakage_via_correlation(
        create_to_target_correlated_features, "target", threshold=0.9
    )
    assert len(result) == 0
    assert result == []

    # Test when there is potential data leakage (feature1 is highly correlated)
    create_to_target_correlated_features["feature1"] = (
        create_to_target_correlated_features["target"] * 2
    )  # Introduce data leakage
    result = detect_leakage_via_correlation(
        create_to_target_correlated_features, "target", threshold=0.9
    )
    assert len(result) == 1

    # Test when a non-existent target column is provided
    with pytest.raises(
        ValueError,
        match="The target column 'nonexistent_column' is not found in the DataFrame.",
    ):
        detect_leakage_via_correlation(
            create_to_target_correlated_features, "nonexistent_column"
        )


def test_detect_categorical_leakage(create_to_target_correlated_categorical_features):
    # Test when there is no data leakage
    leakage_columns = detect_categorical_leakage(
        create_to_target_correlated_categorical_features, "target", threshold=0.9
    )
    assert leakage_columns == ["feature2"]

    # Test when there is potential data leakage (feature1 is highly related to target)
    create_to_target_correlated_categorical_features["feature1"] = (
        create_to_target_correlated_categorical_features["target"]
    )  # Introduce data leakage
    leakage_columns = detect_categorical_leakage(
        create_to_target_correlated_categorical_features, "target", threshold=0.9
    )
    assert "feature1" in leakage_columns

    # Test when a non-existent target column is provided
    with pytest.raises(
        ValueError,
        match="The target column 'nonexistent_column' is not found in the DataFrame.",
    ):
        detect_categorical_leakage(
            create_to_target_correlated_categorical_features, "nonexistent_column"
        )
