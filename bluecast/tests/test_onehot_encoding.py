import pandas as pd
import pytest
from category_encoders import OneHotEncoder

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
    encoder = OneHotCategoryEncoder(cat_columns=["category1", "category2"])
    transformed_data = encoder.fit_transform(
        example_data[["category1", "category2"]], example_data["target"]
    )

    assert "target_encoder_all_cols" in encoder.encoders
    assert isinstance(encoder.encoders["target_encoder_all_cols"], OneHotEncoder)
    assert transformed_data.equals(
        pd.DataFrame(
            {
                "category1_1": [1, 0, 1, 0],
                "category1_2": [0, 1, 0, 0],
                "category1_3": [0, 0, 0, 1],
                "category2_1": [1, 0, 0, 1],
                "category2_2": [0, 1, 0, 0],
                "category2_3": [0, 0, 1, 0],
                "target": [1, 0, 1, 0],
            }
        )
    )


def test_transform(example_data):
    encoder = OneHotCategoryEncoder(cat_columns=["category1", "category2"])

    transformed_data = encoder.transform(example_data[["category1", "category2"]])

    assert transformed_data.equals(
        pd.DataFrame(
            {
                "category1_1": [1, 0, 1, 0],
                "category1_2": [0, 1, 0, 0],
                "category1_3": [0, 0, 0, 1],
                "category2_1": [1, 0, 0, 1],
                "category2_2": [0, 1, 0, 0],
                "category2_3": [0, 0, 1, 0],
            }
        )
    )
