import pandas as pd
import pytest

from bluecast.preprocessing.category_encoder_orchestration import (
    CategoryEncoderOrchestrator,
)


@pytest.fixture
def encoder_orchestrator():
    return CategoryEncoderOrchestrator()


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "cat_col1": ["a", "b", "c", "d", "e"],
            "cat_col2": ["x", "y", "z", "x", "y"],
            "num_col": [1, 2, 3, 4, 5],
        }
    )


def test_fit_assigns_to_onehot_when_cardinality_below_threshold(
    encoder_orchestrator, sample_data
):
    cat_columns = ["cat_col1"]
    encoder_orchestrator.fit(sample_data, cat_columns, threshold=5)
    assert encoder_orchestrator.to_onehot_encode == ["cat_col1"]


def test_fit_assigns_to_target_when_cardinality_above_threshold(
    encoder_orchestrator, sample_data
):
    cat_columns = ["cat_col1"]
    encoder_orchestrator.fit(sample_data, cat_columns, threshold=4)
    assert encoder_orchestrator.to_target_encode == ["cat_col1"]


def test_fit_does_not_assign_numeric_columns(encoder_orchestrator, sample_data):
    cat_columns = ["num_col"]
    encoder_orchestrator.fit(sample_data, cat_columns, threshold=5)
    assert encoder_orchestrator.to_onehot_encode == []
    assert encoder_orchestrator.to_target_encode == []


def test_fit_handles_empty_dataframe(encoder_orchestrator):
    cat_columns = ["cat_col1"]
    sample_data = pd.DataFrame()
    encoder_orchestrator.fit(sample_data, cat_columns)
    assert encoder_orchestrator.to_onehot_encode == []
    assert encoder_orchestrator.to_target_encode == []


def test_fit_handles_empty_categorical_columns(encoder_orchestrator, sample_data):
    cat_columns = []
    encoder_orchestrator.fit(sample_data, cat_columns)
    assert encoder_orchestrator.to_onehot_encode == []
    assert encoder_orchestrator.to_target_encode == []
