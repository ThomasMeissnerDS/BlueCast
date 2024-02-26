import pandas as pd
import pytest

from bluecast.preprocessing.category_encoder_orchestration import (
    CategoryEncoderOrchestrator,
)


@pytest.fixture
def encoder_orchestrator():
    return CategoryEncoderOrchestrator(target_col="num_col")


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "cat_col1": ["a", "b", "c", "d", "e", "f"],
            "cat_col2": ["x", "y", "z", "x", "y", "y"],
            "num_col": [1, 2, 3, 4, 5, 6],
        }
    )


def test_fit_assigns_to_onehot_when_cardinality_below_threshold(
    encoder_orchestrator, sample_data
):
    cat_columns = ["cat_col1", "cat_col2"]
    encoder_orchestrator.fit(sample_data, cat_columns, threshold=5)
    assert encoder_orchestrator.to_onehot_encode == ["cat_col2"]


def test_fit_assigns_to_target_when_cardinality_above_threshold(
    encoder_orchestrator, sample_data
):
    cat_columns = ["cat_col1", "cat_col2"]
    encoder_orchestrator.fit(sample_data, cat_columns, threshold=6)
    assert encoder_orchestrator.to_onehot_encode == ["cat_col1", "cat_col2"]
    assert encoder_orchestrator.to_target_encode == []
