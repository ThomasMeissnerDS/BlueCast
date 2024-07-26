import tempfile

import pandas as pd
import polars as pl
import pytest

from bluecast.blueprints.cast import BlueCast
from bluecast.blueprints.cast_cv import BlueCastCV
from bluecast.evaluation.error_analysis import (
    ErrorAnalyserClassification,
    ErrorAnalyserClassificationCV,
    OutOfFoldDataReader,
    OutOfFoldDataReaderCV,
)


@pytest.fixture
def create_test_bluecast_instance():
    # Create a mock or a test instance of BlueCast
    return BlueCast(class_problem="binary")


def test_out_of_fold_data_reader(create_test_bluecast_instance):
    bluecast_instance = create_test_bluecast_instance

    data_reader = OutOfFoldDataReader(bluecast_instance)

    # Test read_data_from_bluecast_instance
    with pytest.raises(ValueError):
        bluecast_instance.conf_training.out_of_fold_dataset_store_path = None
        data_reader.read_data_from_bluecast_instance()

    with tempfile.TemporaryDirectory() as tmpdir:
        bluecast_instance.conf_training.out_of_fold_dataset_store_path = tmpdir
        oof_data = pl.DataFrame(
            {
                "target": [0, 1, 0, 1],
                "predictions_class_0": [0.8, 0.2, 0.7, 0.3],
                "predictions_class_1": [0.2, 0.8, 0.3, 0.7],
            }
        )

        oof_data.write_parquet(f"{tmpdir}/oof_data_42.parquet")
        # Add more assertions or tests as needed


def test_error_analyser_classification(create_test_bluecast_instance):
    bluecast_instance = create_test_bluecast_instance

    error_analyser = ErrorAnalyserClassification(bluecast_instance)

    oof_data = pl.DataFrame(
        {
            "target": [0, 1, 0, 1],
            "predictions_class_0": [0.8, 0.2, 0.7, 0.3],
            "predictions_class_1": [0.2, 0.8, 0.3, 0.7],
        }
    )

    error_analyser.target_classes = [0, 1]
    error_analyser.prediction_columns = ["predictions_class_0", "predictions_class_1"]
    error_analyser.target_column = "target"

    stacked_data = error_analyser.stack_predictions_by_class(oof_data)
    expected_stacked_data = pl.DataFrame(
        {
            "target": [0, 0, 1, 1],
            "prediction": [0.8, 0.7, 0.8, 0.7],
            "target_class": [0, 0, 1, 1],
        }
    )

    assert (
        pd.testing.assert_frame_equal(
            stacked_data.to_pandas(),
            expected_stacked_data.to_pandas(),
            check_dtype=False,
        )
        is None
    )


@pytest.fixture
def create_test_bluecast_cv_instance():
    # Create a mock or a test instance of BlueCastCV
    # Create a mock or a test instance of BlueCastCV with initialized bluecast_models
    bluecast_cv_instance = BlueCastCV(class_problem="binary")
    bluecast_cv_instance.bluecast_models = []

    # Initialize bluecast_models with dummy models to avoid IndexError
    for _i in range(5):
        model = BlueCast(class_problem="binary")
        model.target_column = "target"
        bluecast_cv_instance.bluecast_models.append(model)
    return bluecast_cv_instance


def test_out_of_fold_data_reader_cv(create_test_bluecast_cv_instance):
    bluecast_cv_instance = create_test_bluecast_cv_instance

    data_reader_cv = OutOfFoldDataReaderCV(bluecast_cv_instance)

    # Test read_data_from_bluecast_instance
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
                    "target": [0, 1, 0, 1],
                    "predictions_class_0": [0.8, 0.2, 0.7, 0.3],
                    "predictions_class_1": [0.2, 0.8, 0.3, 0.7],
                }
            )

            oof_data.write_parquet(f"{tmpdir}/oof_data_{global_random_state}.parquet")

        # Read and validate the data
        data_reader_cv = OutOfFoldDataReaderCV(bluecast_cv_instance)
        read_data = data_reader_cv.read_data_from_bluecast_cv_instance()

        # You can add more detailed checks to validate read_data against expected values

        assert isinstance(read_data, pl.DataFrame)
        assert "target" in read_data.columns
        assert "predictions_class_0" in read_data.columns
        assert "predictions_class_1" in read_data.columns


def test_error_analyser_classification_cv(create_test_bluecast_cv_instance):
    bluecast_cv_instance = create_test_bluecast_cv_instance

    error_analyser_cv = ErrorAnalyserClassificationCV(bluecast_cv_instance)

    oof_data = pl.DataFrame(
        {
            "target": [0, 1, 0, 1],
            "predictions_class_0": [0.8, 0.2, 0.7, 0.3],
            "predictions_class_1": [0.2, 0.8, 0.3, 0.7],
        }
    )

    error_analyser_cv.target_classes = [0, 1]
    error_analyser_cv.prediction_columns = [
        "predictions_class_0",
        "predictions_class_1",
    ]

    stacked_data_cv = error_analyser_cv.stack_predictions_by_class(oof_data)
    expected_stacked_data_cv = pl.DataFrame(
        {
            "target": [0, 0, 1, 1],
            "prediction": [0.8, 0.7, 0.8, 0.7],
            "target_class": [0, 0, 1, 1],
        }
    )

    assert (
        pd.testing.assert_frame_equal(
            stacked_data_cv.to_pandas(),
            expected_stacked_data_cv.to_pandas(),
            check_dtype=False,
        )
        is None
    )
