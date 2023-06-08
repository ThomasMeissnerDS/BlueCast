import pandas as pd
import pytest

from bluecast.preprocessing.schema_checks import SchemaDetector


def test_transform_same_schema():
    # Create a SchemaDetector instance
    detector = SchemaDetector()
    # Define the train dataset
    train_data = {"col1": [1, 2, 3], "col2": ["a", "b", "c"]}
    train_df = pd.DataFrame(train_data)
    # Define the test dataset with the same schema as the train dataset
    test_data = {"col1": [4, 5, 6], "col2": ["d", "e", "f"]}
    test_df = pd.DataFrame(test_data)
    # Fit the detector with the train dataset
    detector.fit(train_df)
    # Transform the test dataset
    transformed_df = detector.transform(test_df)
    # Assert that the transformed_df has the same schema as the train dataset
    assert list(transformed_df.columns) == detector.train_schema


def test_transform_missing_columns():
    # Create a SchemaDetector instance
    detector = SchemaDetector()
    # Define the train dataset
    train_data = {"col1": [1, 2, 3], "col2": ["a", "b", "c"]}
    train_df = pd.DataFrame(train_data)
    # Define the test dataset with missing columns compared to the train dataset
    test_data = {"col1": [4, 5, 6]}
    test_df = pd.DataFrame(test_data)
    # Fit the detector with the train dataset
    detector.fit(train_df)
    # Assert that an error is raised when transforming the test dataset
    with pytest.raises(ValueError):
        detector.transform(test_df)


def test_transform_extra_columns():
    # Create a SchemaDetector instance
    detector = SchemaDetector()
    # Define the train dataset
    train_data = {"col1": [1, 2, 3], "col2": ["a", "b", "c"]}
    train_df = pd.DataFrame(train_data)
    # Define the test dataset with extra columns compared to the train dataset
    test_data = {"col1": [4, 5, 6], "col2": ["d", "e", "f"], "col3": [7, 8, 9]}
    test_df = pd.DataFrame(test_data)
    # Fit the detector with the train dataset
    detector.fit(train_df)
    # Assert that an error is raised when transforming the test dataset
    with pytest.raises(ValueError):
        detector.transform(test_df)
