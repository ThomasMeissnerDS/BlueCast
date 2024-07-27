import polars as pl
import pytest

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
