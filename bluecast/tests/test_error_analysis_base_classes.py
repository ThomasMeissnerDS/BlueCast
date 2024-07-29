import polars as pl
import pytest

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
