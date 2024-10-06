from unittest.mock import patch

import pandas as pd
import polars as pl
import pytest

from bluecast.eda.analyse import plot_error_distributions
from bluecast.evaluation.base_classes import (
    DataReader,
    ErrorAnalyser,
    ErrorDistributionPlotter,
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


class ConcreteDataReader(DataReader):
    def read_data_from_bluecast_instance(self) -> pl.DataFrame:
        super().read_data_from_bluecast_instance()

    def read_data_from_bluecast_cv_instance(self) -> pl.DataFrame:
        super().read_data_from_bluecast_cv_instance()


class ConcreteErrorPreprocessor(ErrorPreprocessor):
    def stack_predictions_by_class(self, df: pl.DataFrame) -> pl.DataFrame:
        super().stack_predictions_by_class(df)

    def calculate_errors(self, df: pl.DataFrame) -> pl.DataFrame:
        super().calculate_errors(df)


class ConcreteErrorAnalyser(ErrorAnalyser):
    def analyse_errors(self, df: pl.DataFrame, descending: bool = True) -> None:
        super().analyse_errors(df, descending)


class ConcreteErrorDistributionPlotter(ErrorDistributionPlotter):
    def plot_error_distributions(
        self, df: pl.DataFrame, hue_column: str = "target_class"
    ) -> None:
        super().plot_error_distributions(df, hue_column)


def test_data_reader_not_implemented():
    reader = ConcreteDataReader()
    with pytest.raises(NotImplementedError):
        reader.read_data_from_bluecast_instance()
    with pytest.raises(NotImplementedError):
        reader.read_data_from_bluecast_cv_instance()


def test_error_preprocessor_not_implemented():
    preprocessor = ConcreteErrorPreprocessor()
    with pytest.raises(NotImplementedError):
        preprocessor.stack_predictions_by_class(pl.DataFrame())
    with pytest.raises(NotImplementedError):
        preprocessor.calculate_errors(pl.DataFrame())


def test_error_analyser_not_implemented():
    analyser = ConcreteErrorAnalyser()
    with pytest.raises(NotImplementedError):
        analyser.analyse_errors(pl.DataFrame())


def test_error_distribution_plotter_not_implemented():
    analyser = ConcreteErrorDistributionPlotter()
    with pytest.raises(NotImplementedError):
        analyser.plot_error_distributions(pl.DataFrame(), "target_column")


# Mock function to test the plot_error_distributions method
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

    # Mock sns.violinplot to prevent actual plotting during the test
    with patch("seaborn.violinplot") as mock_violinplot:
        # Call the function with the test data
        plot_error_distributions(
            df=test_df,
            target="target",
            prediction_error="prediction_error",
            num_cols_grid=num_cols_grid,
            max_x_elements=max_x_elements,
        )

        # Check that sns.violinplot was called multiple times (for each split)
        assert (
            mock_violinplot.call_count == 2
        )  # Should be split into 2 plots because 10 > 5

        # Verify the first call arguments for the first split
        first_call_args = mock_violinplot.call_args_list[0][1]
        assert first_call_args["x"] == "variable_with_many_values"
        assert first_call_args["y"] == "prediction_error"
        assert first_call_args["hue"] == "target"

        # Verify the second call arguments for the second split
        second_call_args = mock_violinplot.call_args_list[1][1]
        assert second_call_args["x"] == "variable_with_many_values"
        assert second_call_args["y"] == "prediction_error"
        assert second_call_args["hue"] == "target"

        # Check that the data in the subset only contains the correct range of unique values
        assert sorted(first_call_args["order"]) == list(range(5))  # First split (0-4)
        assert sorted(second_call_args["order"]) == list(
            range(5, 10)
        )  # Second split (5-9)


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

    # Mock sns.violinplot to prevent actual plotting during the test
    with patch("seaborn.violinplot") as mock_violinplot:
        # Call the function with the test data
        plot_error_distributions(
            df=test_df,
            target="target",
            prediction_error="prediction_error",
            num_cols_grid=num_cols_grid,
            max_x_elements=max_x_elements,
        )

        # Check that sns.violinplot was called only once (no split)
        assert mock_violinplot.call_count == 1

        # Verify the call arguments
        call_args = mock_violinplot.call_args_list[0][1]
        assert call_args["x"] == "variable_with_few_values"
        assert call_args["y"] == "prediction_error"
        assert call_args["hue"] == "target"

        # Ensure the entire dataset was used without splitting
        assert sorted(call_args["order"]) == [1, 2]  # No splitting occurred
