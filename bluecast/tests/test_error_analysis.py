import tempfile

import pandas as pd
import polars as pl
import pytest

from bluecast.blueprints.cast import BlueCast
from bluecast.blueprints.cast_cv import BlueCastCV
from bluecast.evaluation.error_analysis import (
    ErrorAnalyserClassification,
    ErrorAnalyserClassificationCV,
    ErrorAnalyserClassificationMixin,
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


def test_calculate_errors_classification(create_test_bluecast_instance):
    bluecast_instance = create_test_bluecast_instance

    error_analyser = ErrorAnalyserClassification(bluecast_instance)

    stacked_data = pl.DataFrame(
        {
            "target": [0, 0, 1, 1],
            "prediction": [0.8, 0.7, 0.8, 0.7],
            "target_class": [0, 0, 1, 1],
            "target_class_predicted_probas": [0.2, 0.3, 0.8, 0.7],
        }
    )

    expected_errors = pl.DataFrame(
        {
            "target": [0, 0, 1, 1],
            "prediction": [0.8, 0.7, 0.8, 0.7],
            "target_class": [0, 0, 1, 1],
            "target_class_predicted_probas": [0.2, 0.3, 0.8, 0.7],
            "prediction_error": [0.8, 0.7, 0.2, 0.3],
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
                    "target_class_predicted_probas": [0.2, 0.8, 0.7, 0.7],
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

        # test full pipeline
        error_analyser = ErrorAnalyserClassificationCV(bluecast_cv_instance)
        analysis_result = error_analyser.analyse_segment_errors()
        assert isinstance(analysis_result, pl.DataFrame)


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


def test_calculate_errors_classification_cv(create_test_bluecast_cv_instance):
    bluecast_cv_instance = create_test_bluecast_cv_instance

    error_analyser_cv = ErrorAnalyserClassificationCV(bluecast_cv_instance)

    stacked_data = pl.DataFrame(
        {
            "target": [0, 0, 1, 1],
            "prediction": [0.8, 0.7, 0.8, 0.7],
            "target_class": [0, 0, 1, 1],
            "target_class_predicted_probas": [0.2, 0.3, 0.8, 0.7],
        }
    )

    expected_errors = pl.DataFrame(
        {
            "target": [0, 0, 1, 1],
            "prediction": [0.8, 0.7, 0.8, 0.7],
            "target_class": [0, 0, 1, 1],
            "target_class_predicted_probas": [0.2, 0.3, 0.8, 0.7],
            "prediction_error": [0.8, 0.7, 0.2, 0.3],
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
def create_test_bluecast_instance_multiclass():
    # Create a mock or a test instance of BlueCast
    return BlueCast(class_problem="multiclass")


def test_out_of_fold_data_reader_multiclass(create_test_bluecast_instance_multiclass):
    bluecast_instance = create_test_bluecast_instance_multiclass
    bluecast_instance.target_column = "target"

    data_reader = OutOfFoldDataReader(bluecast_instance)

    # Test read_data_from_bluecast_instance
    with pytest.raises(ValueError):
        bluecast_instance.conf_training.out_of_fold_dataset_store_path = None
        data_reader.read_data_from_bluecast_instance()

    with tempfile.TemporaryDirectory() as tmpdir:

        # Test error when Bluecast instance type mismatches data reader type (CV vs non-CV)
        with pytest.raises(ValueError):
            bluecast_instance.conf_training.out_of_fold_dataset_store_path = (
                tmpdir + "/"
            )
            data_reader.read_data_from_bluecast_cv_instance()

        bluecast_instance.conf_training.out_of_fold_dataset_store_path = tmpdir + "/"
        oof_data = pl.DataFrame(
            {
                "target": [0, 1, 2, 0, 1, 2],
                "predictions_class_0": [0.7, 0.1, 0.2, 0.6, 0.2, 0.2],
                "predictions_class_1": [0.2, 0.7, 0.1, 0.3, 0.6, 0.1],
                "predictions_class_2": [0.1, 0.2, 0.7, 0.1, 0.2, 0.7],
                "target_class_predicted_probas": [0.7, 0.7, 0.7, 0.6, 0.6, 0.7],
            }
        )

        oof_data.write_parquet(f"{tmpdir}/oof_data_33.parquet")
        # Add more assertions or tests as needed
        data_reader = OutOfFoldDataReader(bluecast_instance)
        read_data = data_reader.read_data_from_bluecast_instance()
        assert isinstance(read_data, pl.DataFrame)
        assert sorted(data_reader.prediction_columns) == sorted(
            [
                f"predictions_class_{target_class}"
                for target_class in oof_data.unique(subset=["target"])
                .select("target")
                .to_series()
                .to_list()
            ]
        )

        # test full pipeline
        error_analyser = ErrorAnalyserClassification(bluecast_instance)
        analysis_result = error_analyser.analyse_segment_errors()
        assert isinstance(analysis_result, pl.DataFrame)


def test_error_analyser_classification_multiclass(
    create_test_bluecast_instance_multiclass,
):
    bluecast_instance = create_test_bluecast_instance_multiclass

    error_analyser = ErrorAnalyserClassification(bluecast_instance)

    oof_data = pl.DataFrame(
        {
            "target": [0, 1, 2, 0, 1, 2],
            "predictions_class_0": [0.7, 0.1, 0.2, 0.6, 0.2, 0.2],
            "predictions_class_1": [0.2, 0.7, 0.1, 0.3, 0.6, 0.1],
            "predictions_class_2": [0.1, 0.2, 0.7, 0.1, 0.2, 0.7],
        }
    )

    error_analyser.target_classes = [0, 1, 2]
    error_analyser.prediction_columns = [
        "predictions_class_0",
        "predictions_class_1",
        "predictions_class_2",
    ]
    error_analyser.target_column = "target"

    stacked_data = error_analyser.stack_predictions_by_class(oof_data)
    expected_stacked_data = pl.DataFrame(
        {
            "target": [0, 0, 1, 1, 2, 2],
            "prediction": [0.7, 0.6, 0.7, 0.6, 0.7, 0.7],
            "target_class": [0, 0, 1, 1, 2, 2],
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
def create_test_bluecast_cv_instance_multiclass():
    # Create a mock or a test instance of BlueCastCV with initialized bluecast_models
    bluecast_cv_instance = BlueCastCV(class_problem="multiclass")
    bluecast_cv_instance.bluecast_models = []

    # Initialize bluecast_models with dummy models to avoid IndexError
    for _i in range(5):
        model = BlueCast(class_problem="multiclass")
        model.target_column = "target"
        bluecast_cv_instance.bluecast_models.append(model)
    return bluecast_cv_instance


def test_out_of_fold_data_reader_cv_multiclass(
    create_test_bluecast_cv_instance_multiclass,
):
    bluecast_cv_instance = create_test_bluecast_cv_instance_multiclass

    data_reader_cv = OutOfFoldDataReaderCV(bluecast_cv_instance)

    # Test read_data_from_bluecast_instance
    with pytest.raises(ValueError):
        bluecast_cv_instance.conf_training.out_of_fold_dataset_store_path = None
        data_reader_cv.read_data_from_bluecast_cv_instance()

    with tempfile.TemporaryDirectory() as tmpdir:

        # Test error when Bluecast instance type mismatches data reader type (CV vs non-CV)
        with pytest.raises(ValueError):
            bluecast_cv_instance.bluecast_models[
                0
            ].conf_training.out_of_fold_dataset_store_path = (tmpdir + "/")
            data_reader_cv.read_data_from_bluecast_instance()

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
                    "target": [0, 1, 2, 0, 1, 2],
                    "predictions_class_0": [0.7, 0.1, 0.2, 0.6, 0.2, 0.2],
                    "predictions_class_1": [0.2, 0.7, 0.1, 0.3, 0.6, 0.1],
                    "predictions_class_2": [0.1, 0.2, 0.7, 0.1, 0.2, 0.7],
                    "target_class_predicted_probas": [0.7, 0.7, 0.7, 0.6, 0.6, 0.7],
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
        assert "predictions_class_2" in read_data.columns

        # test full pipeline
        error_analyser = ErrorAnalyserClassificationCV(bluecast_cv_instance)
        analysis_result = error_analyser.analyse_segment_errors()
        assert isinstance(analysis_result, pl.DataFrame)


def test_error_analyser_classification_cv_multiclass(
    create_test_bluecast_cv_instance_multiclass,
):
    bluecast_cv_instance = create_test_bluecast_cv_instance_multiclass

    error_analyser_cv = ErrorAnalyserClassificationCV(bluecast_cv_instance)

    oof_data = pl.DataFrame(
        {
            "target": [0, 1, 2, 0, 1, 2],
            "predictions_class_0": [0.7, 0.1, 0.2, 0.6, 0.2, 0.2],
            "predictions_class_1": [0.2, 0.7, 0.1, 0.3, 0.6, 0.1],
            "predictions_class_2": [0.1, 0.2, 0.7, 0.1, 0.2, 0.7],
        }
    )

    error_analyser_cv.target_classes = [0, 1, 2]
    error_analyser_cv.prediction_columns = [
        "predictions_class_0",
        "predictions_class_1",
        "predictions_class_2",
    ]

    stacked_data_cv = error_analyser_cv.stack_predictions_by_class(oof_data)
    expected_stacked_data_cv = pl.DataFrame(
        {
            "target": [0, 0, 1, 1, 2, 2],
            "prediction": [0.7, 0.6, 0.7, 0.6, 0.7, 0.7],
            "target_class": [0, 0, 1, 1, 2, 2],
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


@pytest.fixture
def create_test_error_analyser_mixin_instance():
    return ErrorAnalyserClassificationMixin()


def test_analyse_errors_with_numeric_columns(create_test_error_analyser_mixin_instance):
    analyser_instance = create_test_error_analyser_mixin_instance

    df = pl.DataFrame(
        {
            "target_class": [0, 0, 1, 1, 2, 2],
            "feature_1": [0.1, 0.2, 0.1, 0.2, 0.3, 0.4],
            "feature_2": [1, 2, 1, 2, 3, 4],
            "prediction_error": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        }
    )

    result = analyser_instance.analyse_errors(df)

    expected_columns = [
        "target_class",
        "column_subset",
        "prediction_error",
        "column_name",
    ]
    assert all(col in result.columns for col in expected_columns)


def test_analyse_errors_with_categorical_columns(
    create_test_error_analyser_mixin_instance,
):
    analyser_instance = create_test_error_analyser_mixin_instance

    df = pl.DataFrame(
        {
            "target_class": [0, 0, 1, 1, 2, 2],
            "feature_1": ["A", "B", "A", "B", "C", "D"],
            "feature_2": ["X", "Y", "X", "Y", "Z", "W"],
            "prediction_error": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        }
    )

    result = analyser_instance.analyse_errors(df)

    expected_columns = [
        "target_class",
        "column_subset",
        "prediction_error",
        "column_name",
    ]
    assert all(col in result.columns for col in expected_columns)


def test_analyse_errors_with_mixed_columns(create_test_error_analyser_mixin_instance):
    analyser_instance = create_test_error_analyser_mixin_instance

    df = pl.DataFrame(
        {
            "target_class": [0, 0, 1, 1, 2, 2],
            "feature_1": [0.1, 0.2, 0.1, 0.2, 0.3, 0.4],
            "feature_2": ["A", "B", "A", "B", "C", "D"],
            "prediction_error": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        }
    )

    result = analyser_instance.analyse_errors(df)

    expected_columns = [
        "target_class",
        "column_subset",
        "prediction_error",
        "column_name",
    ]
    assert all(col in result.columns for col in expected_columns)


def test_analyse_errors_descending_false(create_test_error_analyser_mixin_instance):
    analyser_instance = create_test_error_analyser_mixin_instance

    df = pl.DataFrame(
        {
            "target_class": [0, 0, 1, 1, 2, 2],
            "feature_1": [0.1, 0.2, 0.1, 0.2, 0.3, 0.4],
            "feature_2": ["A", "B", "A", "B", "C", "D"],
            "prediction_error": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        }
    )

    result = analyser_instance.analyse_errors(df, descending=False)

    expected_columns = [
        "target_class",
        "column_subset",
        "prediction_error",
        "column_name",
    ]
    assert all(col in result.columns for col in expected_columns)


def test_analyse_errors_with_empty_dataframe(create_test_error_analyser_mixin_instance):
    analyser_instance = create_test_error_analyser_mixin_instance

    df = pl.DataFrame(
        {
            "target_class": [],
            "feature_1": [],
            "feature_2": [],
            "prediction_error": [],
        }
    )

    result = analyser_instance.analyse_errors(df)

    expected_columns = [
        "target_class",
        "column_subset",
        "prediction_error",
        "column_name",
    ]
    assert all(col in result.columns for col in expected_columns)
    assert result.shape == (0, len(expected_columns))
