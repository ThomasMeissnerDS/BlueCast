import pandas as pd
import pytest

from bluecast.preprocessing.onehot_encoding import OneHotCategoryEncoder


@pytest.fixture
def example_data():
    data = {
        "category1": ["A", "B", "A", "C"],
        "category2": ["X", "Y", "Z", "X"],
        "target": [1, 0, 1, 0],
    }
    return pd.DataFrame(data)


def test_fit_transform(example_data):
    encoder = OneHotCategoryEncoder(
        cat_columns=["category1", "category2"], target_col="target"
    )
    transformed_data = encoder.fit_transform(
        example_data[["category1", "category2"]], example_data["target"]
    )

    assert transformed_data.equals(
        pd.DataFrame(
            {
                "category1_A": [1, 0, 1, 0],
                "category1_B": [0, 1, 0, 0],
                "category1_C": [0, 0, 0, 1],
                "category2_X": [1, 0, 0, 1],
                "category2_Y": [0, 1, 0, 0],
                "category2_Z": [0, 0, 1, 0],
            }
        )
    )


def test_transform(example_data):
    encoder = OneHotCategoryEncoder(
        cat_columns=["category1", "category2"], target_col="target"
    )
    _ = encoder.fit_transform(
        example_data[["category1", "category2"]], example_data["target"]
    )

    transformed_data = encoder.transform(example_data[["category1", "category2"]])

    assert transformed_data.equals(
        pd.DataFrame(
            {
                "category1_A": [1, 0, 1, 0],
                "category1_B": [0, 1, 0, 0],
                "category1_C": [0, 0, 0, 1],
                "category2_X": [1, 0, 0, 1],
                "category2_Y": [0, 1, 0, 0],
                "category2_Z": [0, 0, 1, 0],
            }
        )
    )
