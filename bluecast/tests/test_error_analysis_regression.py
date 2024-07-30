import tempfile

import pandas as pd
import polars as pl
import pytest

from bluecast.blueprints.cast_cv_regression import BlueCastCVRegression
from bluecast.blueprints.cast_regression import BlueCastRegression
from bluecast.evaluation.error_analysis_regression import (
    ErrorAnalyserRegression,
    ErrorAnalyserRegressionCV,
    ErrorAnalyserRegressionMixin,
    OutOfFoldDataReaderRegression,
    OutOfFoldDataReaderRegressionCV,
)


@pytest.fixture
def create_test_bluecast_instance():
    # Create a mock or a test instance of BlueCast for regression
    return BlueCastRegression(class_problem="regression")


def test_out_of_fold_data_reader(create_test_bluecast_instance):
    bluecast_instance = create_test_bluecast_instance
    bluecast_instance.target_column = "target"

    data_reader = OutOfFoldDataReaderRegression(bluecast_instance)

    # Test read_data_from_bluecast_instance
    with pytest.raises(ValueError):
        bluecast_instance.conf_training.out_of_fold_dataset_store_path = None
        data_reader.read_data_from_bluecast_instance()

    with tempfile.TemporaryDirectory() as tmpdir:
        bluecast_instance.conf_training.out_of_fold_dataset_store_path = tmpdir + "/"
        oof_data = pl.DataFrame(
            {
                "target": [1.0, 2.0, 1.5, 2.5],
                "feature_1": [0.1, 0.2, 0.1, 0.2],
                "feature_2": [1, 2, 1, 2],
                "predictions": [0.9, 2.1, 1.4, 2.6],
            }
        )

        oof_data.write_parquet(f"{tmpdir}/oof_data_33.parquet")

        # test full pipeline
        error_analyser = ErrorAnalyserRegression(bluecast_instance)
        analysis_result = error_analyser.analyse_segment_errors()
        assert isinstance(analysis_result, pl.DataFrame)


def test_error_analyser_regression(create_test_bluecast_instance):
    bluecast_instance = create_test_bluecast_instance
    bluecast_instance.target_column = "target"

    error_analyser = ErrorAnalyserRegression(bluecast_instance)

    oof_data = pl.DataFrame(
        {
            "target": [1.0, 1.5, 2.0, 2.5],
            "feature_1": [0.1, 0.2, 0.1, 0.2],
            "feature_2": [1, 2, 1, 2],
            "predictions": [0.9, 1.4, 2.2, 2.9],
        }
    )

    stacked_data = error_analyser.stack_predictions_by_class(oof_data)

    assert stacked_data.shape[0] == 4
    assert stacked_data.shape[1] == 5


def test_calculate_errors_regression(create_test_bluecast_instance):
    bluecast_instance = create_test_bluecast_instance

    error_analyser = ErrorAnalyserRegression(bluecast_instance)

    stacked_data = pl.DataFrame(
        {
            "target": [1.0, 1.5, 2.0, 2.5],
            "feature_1": [0.1, 0.2, 0.1, 0.2],
            "feature_2": [1, 2, 1, 2],
            "predictions": [0.1, 1.0, 2.1, 2.5],
        }
    )

    expected_errors = pl.DataFrame(
        {
            "target": [1.0, 1.5, 2.0, 2.5],
            "feature_1": [0.1, 0.2, 0.1, 0.2],
            "feature_2": [1, 2, 1, 2],
            "predictions": [0.1, 1.0, 2.1, 2.5],
            "prediction_error": [0.9, 0.5, 0.1, 0.0],
        }
    )

    calculated_errors = error_analyser.calculate_errors(stacked_data)

    assert (
        pd.testing.assert_frame_equal(
            calculated_errors.to_pandas(),
            expected_errors.to_pandas(),
            check_dtype=False,
        )
        is None
    )


@pytest.fixture
def create_test_bluecast_cv_instance():
    # Create a mock or a test instance of BlueCastCV for regression
    bluecast_cv_instance = BlueCastCVRegression(class_problem="regression")
    bluecast_cv_instance.bluecast_models = []

    # Initialize bluecast_models with dummy models to avoid IndexError
    for _i in range(5):
        model = BlueCastRegression(class_problem="regression")
        model.target_column = "target"
        bluecast_cv_instance.bluecast_models.append(model)
    return bluecast_cv_instance


def test_out_of_fold_data_reader_cv(create_test_bluecast_cv_instance):
    bluecast_cv_instance = create_test_bluecast_cv_instance

    data_reader_cv = OutOfFoldDataReaderRegressionCV(bluecast_cv_instance)

    # Test read_data_from_bluecast_cv_instance
    with pytest.raises(ValueError):
        bluecast_cv_instance.conf_training.out_of_fold_dataset_store_path = None
        data_reader_cv.read_data_from_bluecast_cv_instance()

    with tempfile.TemporaryDirectory() as tmpdir:
        bluecast_cv_instance.conf_training.out_of_fold_dataset_store_path = tmpdir + "/"

        for model in bluecast_cv_instance.bluecast_models:
            model.conf_training.out_of_fold_dataset_store_path = tmpdir + "/"

        for i in range(5):
            global_random_state = (
                bluecast_cv_instance.conf_training.global_random_state
                + i
                * bluecast_cv_instance.conf_training.increase_random_state_in_bluecast_cv_by
            )
            oof_data = pl.DataFrame(
                {
                    "target": [1.0, 2.0, 1.5, 2.5],
                    "feature_1": [0.1, 0.2, 0.1, 0.2],
                    "feature_2": [1, 2, 1, 2],
                    "predictions": [0.1, 1.5, 2.1, 2.5],
                }
            )

            oof_data.write_parquet(f"{tmpdir}/oof_data_{global_random_state}.parquet")

        # Read and validate the data
        data_reader_cv = OutOfFoldDataReaderRegressionCV(bluecast_cv_instance)
        read_data = data_reader_cv.read_data_from_bluecast_cv_instance()

        assert isinstance(read_data, pl.DataFrame)
        assert "target" in read_data.columns
        assert "predictions" in read_data.columns

        # test full pipeline
        error_analyser = ErrorAnalyserRegressionCV(bluecast_cv_instance)
        analysis_result = error_analyser.analyse_segment_errors()
        assert isinstance(analysis_result, pl.DataFrame)


def test_error_analyser_regression_cv(create_test_bluecast_cv_instance):
    bluecast_cv_instance = create_test_bluecast_cv_instance

    error_analyser_cv = ErrorAnalyserRegressionCV(bluecast_cv_instance)

    oof_data = pl.DataFrame(
        {
            "target": [1.0, 2.0, 1.5, 2.5],
            "feature_1": [0.1, 0.2, 0.1, 0.2],
            "feature_2": [1, 2, 1, 2],
            "predictions": [0.1, 1.5, 2.1, 2.5],
        }
    )

    stacked_data_cv = error_analyser_cv.stack_predictions_by_class(oof_data)

    assert stacked_data_cv.shape[0] == 4
    assert stacked_data_cv.shape[1] == 5


def test_calculate_errors_regression_cv(create_test_bluecast_cv_instance):
    bluecast_cv_instance = create_test_bluecast_cv_instance

    error_analyser_cv = ErrorAnalyserRegressionCV(bluecast_cv_instance)

    stacked_data = pl.DataFrame(
        {
            "target": [1.0, 1.5, 2.0, 2.5],
            "feature_1": [0.1, 0.2, 0.1, 0.2],
            "feature_2": [1, 2, 1, 2],
            "predictions": [0.1, 1.0, 2.1, 2.5],
        }
    )

    expected_errors = pl.DataFrame(
        {
            "target": [1.0, 1.5, 2.0, 2.5],
            "feature_1": [0.1, 0.2, 0.1, 0.2],
            "feature_2": [1, 2, 1, 2],
            "predictions": [0.1, 1.0, 2.1, 2.5],
            "prediction_error": [0.9, 0.5, 0.1, 0.0],
        }
    )

    calculated_errors = error_analyser_cv.calculate_errors(stacked_data)

    assert (
        pd.testing.assert_frame_equal(
            calculated_errors.to_pandas(),
            expected_errors.to_pandas(),
            check_dtype=False,
        )
        is None
    )


@pytest.fixture
def create_test_error_analyser_mixin_instance():
    return ErrorAnalyserRegressionMixin()


def test_analyse_errors_regression(create_test_error_analyser_mixin_instance):
    analyser_instance = create_test_error_analyser_mixin_instance

    df = pl.DataFrame(
        {
            "target_quantiles": [1.0, 1.5, 2.0, 2.5],
            "feature_1": [0.1, 0.2, 0.1, 0.2],
            "feature_2": [1, 2, 1, 2],
            "feature_3": ["male", "female", "male", "female"],
            "prediction_error": [0.5, 0.6, 0.7, 0.8],
        }
    )

    result_df = analyser_instance.analyse_errors(df)

    assert result_df.shape[0] == 12
    assert result_df.shape[1] == 4
