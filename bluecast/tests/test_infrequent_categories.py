import pandas as pd
import pytest

from bluecast.preprocessing.infrequent_categories import InFrequentCategoryEncoder


@pytest.fixture
def sample_data():
    data = {
        "cat1": ["a", "b", "b", "c", "c", "c", "d", "e"],
        "cat2": ["w", "w", "x", "y", "y", "z", "z", "z"],
        "target": [0, 1, 0, 1, 0, 1, 0, 1],
    }
    df = pd.DataFrame(data)
    return df


def test_fit_transform(sample_data):
    encoder = InFrequentCategoryEncoder(
        cat_columns=["cat1", "cat2"], target_col="target", infrequent_threshold=2
    )
    transformed_df = encoder.fit_transform(
        sample_data.drop(columns="target"), sample_data["target"]
    )

    assert transformed_df["cat1"].tolist() == [
        "rare categories",
        "b",
        "b",
        "c",
        "c",
        "c",
        "rare categories",
        "rare categories",
    ]
    assert transformed_df["cat2"].tolist() == [
        "w",
        "w",
        "rare categories",
        "y",
        "y",
        "z",
        "z",
        "z",
    ]


def test_transform(sample_data):
    encoder = InFrequentCategoryEncoder(
        cat_columns=["cat1", "cat2"], target_col="target", infrequent_threshold=2
    )
    encoder.fit_transform(sample_data.drop(columns="target"), sample_data["target"])

    new_data = pd.DataFrame(
        {"cat1": ["a", "b", "c", "d", "f"], "cat2": ["w", "x", "y", "z", "a"]}
    )

    transformed_new_data = encoder.transform(new_data)

    assert transformed_new_data["cat1"].tolist() == [
        "rare categories",
        "b",
        "c",
        "rare categories",
        "f",
    ]
    assert transformed_new_data["cat2"].tolist() == [
        "w",
        "rare categories",
        "y",
        "z",
        "a",
    ]


def test_no_infrequent_categories(sample_data):
    encoder = InFrequentCategoryEncoder(
        cat_columns=["cat1", "cat2"], target_col="target", infrequent_threshold=1
    )
    transformed_df = encoder.fit_transform(
        sample_data.drop(columns="target"), sample_data["target"]
    )

    assert transformed_df["cat1"].tolist() == sample_data["cat1"].tolist()
    assert transformed_df["cat2"].tolist() == sample_data["cat2"].tolist()


def test_all_infrequent_categories(sample_data):
    encoder = InFrequentCategoryEncoder(
        cat_columns=["cat1", "cat2"], target_col="target", infrequent_threshold=10
    )
    transformed_df = encoder.fit_transform(
        sample_data.drop(columns="target"), sample_data["target"]
    )

    assert all(val == "rare categories" for val in transformed_df["cat1"].tolist())
    assert all(val == "rare categories" for val in transformed_df["cat2"].tolist())
