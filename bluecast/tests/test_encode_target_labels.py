import pandas as pd
import pytest

from bluecast.preprocessing.encode_target_labels import (
    TargetLabelEncoder,
    cast_bool_to_int,
)


@pytest.fixture
def label_encoder():
    return TargetLabelEncoder()


@pytest.fixture
def sample_data():
    data = pd.DataFrame({"target": ["A", "B", "C", "A", "B", "C"]})
    return data


@pytest.fixture
def trained_label_encoder(label_encoder, sample_data):
    label_encoder.fit_transform_target_labels(sample_data["target"])
    return label_encoder


def test_fit_label_encoder(label_encoder, sample_data):
    mapping = label_encoder.fit_label_encoder(sample_data["target"])
    expected_mapping = {"A": 0, "B": 1, "C": 2}
    assert mapping == expected_mapping


def test_label_encoder_transform(trained_label_encoder, sample_data):
    transformed_data = trained_label_encoder.transform_target_labels(
        sample_data["target"]
    )
    expected_data = pd.DataFrame({"target": [0, 1, 2, 0, 1, 2]})
    pd.testing.assert_frame_equal(transformed_data, expected_data)


def test_fit_transform_target_labels(label_encoder, sample_data):
    transformed_data = label_encoder.fit_transform_target_labels(sample_data["target"])
    expected_data = pd.DataFrame({"target": [0, 1, 2, 0, 1, 2]})
    pd.testing.assert_frame_equal(transformed_data, expected_data)


def test_transform_target_labels(trained_label_encoder, sample_data):
    transformed_data = trained_label_encoder.transform_target_labels(
        sample_data["target"]
    )
    expected_data = pd.DataFrame({"target": [0, 1, 2, 0, 1, 2]})
    pd.testing.assert_frame_equal(transformed_data, expected_data)


def test_label_encoder_reverse_transform(trained_label_encoder, sample_data):
    transformed_data = pd.DataFrame({"target": [0, 1, 2, 0, 1, 2]})
    reversed_data = trained_label_encoder.label_encoder_reverse_transform(
        transformed_data["target"]
    )
    expected_data = pd.DataFrame({"target": ["A", "B", "C", "A", "B", "C"]})
    pd.testing.assert_frame_equal(reversed_data, expected_data)


# Test conversion of target labels to numeric values


def test_cast_bool_to_int_with_bool_column():
    df = pd.DataFrame({"a": [True, False, True]})
    result = cast_bool_to_int(df, "a")
    expected = pd.DataFrame({"a": [1, 0, 1]})
    pd.testing.assert_frame_equal(result, expected)


def test_cast_bool_to_int_with_non_bool_column():
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = cast_bool_to_int(df, "a")
    expected = df.copy()  # No change expected
    pd.testing.assert_frame_equal(result, expected)


def test_cast_bool_to_int_with_nonexistent_column():
    df = pd.DataFrame({"a": [True, False, True]})
    result = cast_bool_to_int(df, "b")
    expected = df.copy()  # No change expected
    pd.testing.assert_frame_equal(result, expected)


def test_cast_bool_to_int_with_mixed_column():
    df = pd.DataFrame({"a": [1, "2", True]})
    result = cast_bool_to_int(df, "a")
    expected = df.copy()  # No change expected
    pd.testing.assert_frame_equal(result, expected)


def test_cast_bool_to_int_with_empty_dataframe():
    df = pd.DataFrame()
    result = cast_bool_to_int(df, "a")
    expected = df.copy()  # No change expected
    pd.testing.assert_frame_equal(result, expected)
