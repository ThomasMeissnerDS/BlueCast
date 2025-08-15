from unittest.mock import patch

import pandas as pd
import polars as pl
import pytest

from bluecast.eda.analyse import plot_error_distributions
from bluecast.evaluation.base_classes import (
    DataReader,
    ErrorAnalyser,
    ErrorPreprocessor,
)


def test_base_class_datareader_notimplemented_error():
    class FailDataReader(DataReader):
        def __init__(self):
            pass

        def read_data_from_bluecast_instance(self) -> pl.DataFrame:
            raise NotImplementedError

        def read_data_from_bluecast_cv_instance(self) -> pl.DataFrame:
            raise NotImplementedError

    with pytest.raises(NotImplementedError):
        fail_cls = FailDataReader()
        fail_cls.read_data_from_bluecast_instance()

    with pytest.raises(NotImplementedError):
        fail_cls = FailDataReader()
        fail_cls.read_data_from_bluecast_cv_instance()


def test_base_class_erroranalyser_notimplemented_error():
    class FailErrorAnalyser(ErrorAnalyser):
        def __init__(self):
            pass

        def analyse_errors(self, df, descending: bool = True) -> None:
            raise NotImplementedError

    test_df = pl.DataFrame(
        {
            "target": [0, 1, 0, 1],
            "predictions_class_0": [0.8, 0.2, 0.7, 0.3],
            "predictions_class_1": [0.2, 0.8, 0.3, 0.7],
        }
    )

    with pytest.raises(NotImplementedError):
        fail_cls = FailErrorAnalyser()
        fail_cls.analyse_errors(test_df)


def test_base_class_errorpreprocessor_notimplemented_error():
    class FailEErrorPreprocessor(ErrorPreprocessor):
        def __init__(self):
            pass

        def stack_predictions_by_class(self, df: pl.DataFrame) -> pl.DataFrame:
            raise NotImplementedError

        def calculate_errors(self, df) -> pl.DataFrame:
            raise NotImplementedError

    test_df = pl.DataFrame(
        {
            "target": [0, 1, 0, 1],
            "predictions_class_0": [0.8, 0.2, 0.7, 0.3],
            "predictions_class_1": [0.2, 0.8, 0.3, 0.7],
        }
    )

    with pytest.raises(NotImplementedError):
        fail_cls = FailEErrorPreprocessor()
        fail_cls.stack_predictions_by_class(test_df)

    with pytest.raises(NotImplementedError):
        fail_cls = FailEErrorPreprocessor()
        fail_cls.calculate_errors(test_df)


# Test the plot_error_distributions method (converted to plotly)
def test_plot_error_distributions_with_splits():
    # Create a test DataFrame with more than max_x_elements unique values in one column
    test_df = pd.DataFrame(
        {
            "target": ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"],
            "variable_with_many_values": list(range(10)),  # 10 unique values
            "prediction_error": [1, 0.8, 0.9, 1.1, 1.2, 0.7, 1.3, 0.6, 1.4, 0.5],
        }
    )

    # Set the number of max_x_elements to be less than the number of unique values (e.g., 5)
    max_x_elements = 5
    num_cols_grid = 1

    # Mock plotly figure.show() to prevent actual plotting during the test
    with patch("plotly.graph_objects.Figure.show") as mock_show:
        # Call the function with the test data
        plot_error_distributions(
            df=test_df,
            target="target",
            prediction_error="prediction_error",
            num_cols_grid=num_cols_grid,
            max_x_elements=max_x_elements,
        )

        # Check that figure.show() was called multiple times (for each split)
        # Should be split into 2 plots because 10 unique values > 5 max_x_elements
        assert mock_show.call_count == 2


def test_plot_error_distributions_no_split():
    # Create a test DataFrame with fewer unique values than max_x_elements
    test_df = pd.DataFrame(
        {
            "target": ["A", "A", "B", "B"],
            "variable_with_few_values": [1, 2, 1, 2],
            "prediction_error": [1.1, 0.8, 0.9, 1.0],
        }
    )

    # Set the number of max_x_elements to be larger than the number of unique values (e.g., 5)
    max_x_elements = 5
    num_cols_grid = 1

    # Mock plotly figure.show() to prevent actual plotting during the test
    with patch("plotly.graph_objects.Figure.show") as mock_show:
        # Call the function with the test data
        plot_error_distributions(
            df=test_df,
            target="target",
            prediction_error="prediction_error",
            num_cols_grid=num_cols_grid,
            max_x_elements=max_x_elements,
        )

        # Check that figure.show() was called only once (no split)
        # Since 2 unique values <= 5 max_x_elements, no splitting should occur
        assert mock_show.call_count == 1
