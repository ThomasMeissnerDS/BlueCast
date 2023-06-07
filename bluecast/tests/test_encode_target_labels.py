import pandas as pd
import pytest

from bluecast.preprocessing.encode_target_labels import TargetLabelEncoder


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
