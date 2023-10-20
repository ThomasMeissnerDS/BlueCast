import pandas as pd
import pytest

from bluecast.eda.data_quality_checks import detect_leakage_via_correlation


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


def test_detect_leakage_via_correlation(create_to_target_correlated_features):
    # Test when there is no data leakage
    result = detect_leakage_via_correlation(
        create_to_target_correlated_features, "target", threshold=0.9
    )
    assert result is False

    # Test when there is potential data leakage (feature1 is highly correlated)
    create_to_target_correlated_features["feature1"] = (
        create_to_target_correlated_features["target"] * 2
    )  # Introduce data leakage
    result = detect_leakage_via_correlation(
        create_to_target_correlated_features, "target", threshold=0.9
    )
    assert result is True

    # Test when a non-existent target column is provided
    with pytest.raises(
        ValueError,
        match="The target column 'nonexistent_column' is not found in the DataFrame.",
    ):
        detect_leakage_via_correlation(
            create_to_target_correlated_features, "nonexistent_column"
        )
