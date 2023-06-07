import pandas as pd
import pytest

from bluecast.preprocessing.target_encoding import BinaryClassTargetEncoder


@pytest.fixture
def sample_data():
    # Sample data for testing
    data = pd.DataFrame(
        {
            "col1": ["A", "B", "C", "A", "B", "C"],
            "col2": ["X", "Y", "Z", "X", "Y", "Z"],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )
    return data


def test_transform_target_encode_binary_class(sample_data):
    # Create an instance of the BinaryClassTargetEncoder class
    encoder = BinaryClassTargetEncoder(cat_columns=["col1", "col2"])

    # Fit the encoder on the sample data
    encoder.fit_target_encode_binary_class(
        sample_data[["col1", "col2"]], sample_data["target"]
    )

    # Transform the sample data using the fitted encoder
    transformed_data = encoder.transform_target_encode_binary_class(
        sample_data[["col1", "col2"]]
    )

    # Check if the transformed data has the expected columns
    expected_columns = ["col1", "col2"]
    assert set(transformed_data.columns) == set(expected_columns)

    # Check if the transformed data values are as expected
    expected_values = pd.DataFrame(
        {"col1": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], "col2": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]}
    )
    pd.testing.assert_frame_equal(transformed_data, expected_values)
