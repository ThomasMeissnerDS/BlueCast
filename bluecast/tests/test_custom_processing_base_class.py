import pandas as pd
import pytest

from bluecast.preprocessing.custom import CustomPreprocessing


@pytest.fixture
def custom_preprocessing():
    class CustomPreprocessingTest(CustomPreprocessing):
        def fit_transform(self, df: pd.DataFrame, target: pd.Series = None):
            return df, target

        def transform(
            self,
            df: pd.DataFrame,
            target: pd.Series = None,
            predicton_mode: bool = False,
        ):
            return df, target

    return (
        CustomPreprocessingTest()
    )  # Replace with your implementation of CustomPreprocessing


def test_fit_transform_returns_tuple(custom_preprocessing):
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    target = pd.Series([0, 1, 0])
    result = custom_preprocessing.fit_transform(df, target)
    assert isinstance(result, tuple)


def test_transform_returns_tuple(custom_preprocessing):
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    target = pd.Series([0, 1, 0])
    result = custom_preprocessing.transform(df, target)
    assert isinstance(result, tuple)


def test_transform_returns_optional_target(custom_preprocessing):
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    result = custom_preprocessing.transform(df)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], pd.DataFrame)
    assert result[1] is None


def test_transform_returns_optional_target_prediction_mode(custom_preprocessing):
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    result = custom_preprocessing.transform(df, predicton_mode=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], pd.DataFrame)
    assert result[1] is None
