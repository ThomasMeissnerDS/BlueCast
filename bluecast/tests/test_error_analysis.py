import tempfile

import pandas as pd
import polars as pl
import pytest

from bluecast.blueprints.cast import BlueCast
from bluecast.blueprints.cast_cv import BlueCastCV
from bluecast.evaluation.error_analysis import (
    DuckDBErrorAnalysisEngine,
    ErrorAnalyserClassification,
    ErrorAnalyserClassificationCV,
    ErrorAnalyserClassificationMixin,
    ErrorDistributionPlotterMixin,
    OutOfFoldDataReader,
    OutOfFoldDataReaderCV,
)


@pytest.fixture
def create_test_bluecast_instance():
    """Create a mock or a test instance of BlueCast"""
    return BlueCast(class_problem="binary")


@pytest.fixture
def create_test_bluecast_cv_instance():
    """Create a mock BlueCastCV instance for testing"""
    return BlueCastCV(class_problem="binary")


@pytest.fixture
def sample_error_data():
    """Create sample error analysis data for testing"""
    return pd.DataFrame(
        {
            "target_class": ["A", "B", "A", "B", "A"],
            "prediction_error": [0.1, 0.2, 0.05, 0.15, 0.3],
            "feature_1": [1, 2, 3, 4, 5],
            "feature_2": ["x", "y", "x", "y", "x"],
        }
    )


@pytest.fixture
def duckdb_engine():
    """Create a DuckDB error analysis engine for testing"""
    engine = DuckDBErrorAnalysisEngine()
    yield engine
    engine.close()


def test_duckdb_error_analysis_engine_init(duckdb_engine):
    """Test DuckDB error analysis engine initialization"""
    assert duckdb_engine.db_path is not None
    # Test that tables were created
    import duckdb

    with duckdb.connect(duckdb_engine.db_path) as conn:
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [table[0] for table in tables]
        assert "error_analysis" in table_names
        assert "error_statistics" in table_names


def test_duckdb_load_data(duckdb_engine, sample_error_data):
    """Test loading data into DuckDB"""
    experiment_id = "test_exp_001"
    target_column = "target_class"

    duckdb_engine.load_data(sample_error_data, experiment_id, target_column)

    # Verify data was loaded
    import duckdb

    with duckdb.connect(duckdb_engine.db_path) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM error_analysis WHERE experiment_id = ?",
            [experiment_id],
        ).fetchone()[0]
        assert count == len(sample_error_data)


def test_duckdb_compute_statistics(duckdb_engine, sample_error_data):
    """Test computing error statistics"""
    experiment_id = "test_exp_002"
    target_column = "target_class"

    duckdb_engine.load_data(sample_error_data, experiment_id, target_column)
    stats = duckdb_engine.compute_error_statistics(experiment_id)

    assert "overall_statistics" in stats
    assert "class_statistics" in stats
    assert "top_errors" in stats

    # Check overall statistics
    overall = stats["overall_statistics"]
    assert not overall.empty
    assert overall["total_samples"].iloc[0] == len(sample_error_data)

    # Check class statistics
    class_stats = stats["class_statistics"]
    assert not class_stats.empty
    assert len(class_stats) == 2  # Two classes: A and B


def test_duckdb_create_visualizations(duckdb_engine, sample_error_data):
    """Test creating error visualizations"""
    experiment_id = "test_exp_003"
    target_column = "target_class"

    duckdb_engine.load_data(sample_error_data, experiment_id, target_column)
    figures = duckdb_engine.create_error_visualizations(experiment_id, target_column)

    # Check that figures were created
    assert isinstance(figures, dict)
    assert "error_distribution" in figures
    assert "error_by_class" in figures

    # Verify figures are plotly objects
    import plotly.graph_objects as go

    for fig in figures.values():
        assert isinstance(fig, go.Figure)


def test_out_of_fold_data_reader(create_test_bluecast_instance):
    """Test OutOfFoldDataReader with proper error handling"""
    bluecast_instance = create_test_bluecast_instance
    bluecast_instance.target_column = "target"

    data_reader = OutOfFoldDataReader(bluecast_instance)

    # Test with None path
    with pytest.raises(
        ValueError, match="out_of_fold_dataset_store_path has not been configured"
    ):
        bluecast_instance.conf_training.out_of_fold_dataset_store_path = None
        data_reader.read_data_from_bluecast_instance()

    # Test with valid path and data
    with tempfile.TemporaryDirectory() as tmpdir:
        bluecast_instance.conf_training.out_of_fold_dataset_store_path = tmpdir + "/"
        bluecast_instance.conf_training.global_random_state = 42

        oof_data = pl.DataFrame(
            {
                "target": [0, 1, 0, 1],
                "predictions_class_0": [0.8, 0.2, 0.7, 0.3],
                "predictions_class_1": [0.2, 0.8, 0.3, 0.7],
                "feature_1": [1, 2, 3, 4],
            }
        )

        oof_data.write_parquet(f"{tmpdir}/oof_data_42.parquet")

        # Should work now
        loaded_data = data_reader.read_data_from_bluecast_instance()
        assert len(loaded_data) == 4
        assert "target" in loaded_data.columns


def test_out_of_fold_data_reader_cv_error():
    """Test that OutOfFoldDataReader properly fails for CV instances"""
    bluecast_instance = BlueCast(class_problem="binary")
    data_reader = OutOfFoldDataReader(bluecast_instance)

    with pytest.raises(
        ValueError, match="Please use OutOfFoldDataReaderCV class instead"
    ):
        data_reader.read_data_from_bluecast_cv_instance()


def test_out_of_fold_data_reader_cv(create_test_bluecast_cv_instance):
    """Test OutOfFoldDataReaderCV"""
    bluecast_cv_instance = create_test_bluecast_cv_instance

    # Create mock bluecast models
    mock_model1 = BlueCast(class_problem="binary")
    mock_model1.target_column = "target"
    mock_model1.conf_training.global_random_state = 42

    mock_model2 = BlueCast(class_problem="binary")
    mock_model2.target_column = "target"
    mock_model2.conf_training.global_random_state = 43

    bluecast_cv_instance.bluecast_models = [mock_model1, mock_model2]

    data_reader = OutOfFoldDataReaderCV(bluecast_cv_instance)

    # Test error when path not configured
    with pytest.raises(
        ValueError, match="out_of_fold_dataset_store_path has not been configured"
    ):
        data_reader.read_data_from_bluecast_cv_instance()

    # Test with valid configuration
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, model in enumerate([mock_model1, mock_model2]):
            model.conf_training.out_of_fold_dataset_store_path = tmpdir + "/"

            oof_data = pl.DataFrame(
                {
                    "target": [0, 1],
                    "predictions_class_0": [0.8, 0.2],
                    "predictions_class_1": [0.2, 0.8],
                    "feature_1": [i + 1, i + 2],
                }
            )

            oof_data.write_parquet(
                f"{tmpdir}/oof_data_{model.conf_training.global_random_state}.parquet"
            )

        # Should work now
        loaded_data = data_reader.read_data_from_bluecast_cv_instance()
        assert len(loaded_data) == 4  # Combined data from both models


def test_error_analyser_classification_mixin():
    """Test ErrorAnalyserClassificationMixin functionality"""
    mixin = ErrorAnalyserClassificationMixin()

    # Test with sample data
    sample_df = pd.DataFrame(
        {
            "target_class": ["A", "B", "A", "B"],
            "prediction_error": [0.1, 0.2, 0.05, 0.15],
            "feature_1": [1, 2, 3, 4],
        }
    )

    # This should create visualizations and return enhanced analysis
    result = mixin.analyse_errors(sample_df)

    assert isinstance(result, pl.DataFrame)
    assert not result.is_empty()

    # Clean up
    mixin.duckdb_engine.close()


def test_error_analyser_classification_full_pipeline(create_test_bluecast_instance):
    """Test full ErrorAnalyserClassification pipeline"""
    bluecast_instance = create_test_bluecast_instance
    bluecast_instance.target_column = "target"

    # Create temporary out-of-fold data
    with tempfile.TemporaryDirectory() as tmpdir:
        bluecast_instance.conf_training.out_of_fold_dataset_store_path = tmpdir + "/"
        bluecast_instance.conf_training.global_random_state = 42

        # Create realistic out-of-fold data
        oof_data = pl.DataFrame(
            {
                "target": [0, 1, 0, 1, 0, 1],
                "predictions_class_0": [0.8, 0.2, 0.7, 0.3, 0.9, 0.1],
                "predictions_class_1": [0.2, 0.8, 0.3, 0.7, 0.1, 0.9],
                "target_class_predicted_probas": [0.8, 0.8, 0.7, 0.7, 0.9, 0.9],
                "feature_1": [1, 2, 3, 4, 5, 6],
                "feature_2": ["x", "y", "x", "y", "x", "y"],
            }
        )

        oof_data.write_parquet(f"{tmpdir}/oof_data_42.parquet")

        # Test the full pipeline
        analyser = ErrorAnalyserClassification(bluecast_instance)

        # Test individual methods
        loaded_data = analyser.read_data_from_bluecast_instance()
        assert len(loaded_data) > 0

        stacked_data = analyser.stack_predictions_by_class(loaded_data)
        assert len(stacked_data) > 0
        assert "target_class" in stacked_data.columns

        errors_data = analyser.calculate_errors(stacked_data)
        assert "prediction_error" in errors_data.columns

        # Clean up DuckDB engine
        analyser.duckdb_engine.close()


def test_error_analyser_classification_cv_full_pipeline(
    create_test_bluecast_cv_instance,
):
    """Test full ErrorAnalyserClassificationCV pipeline"""
    bluecast_cv_instance = create_test_bluecast_cv_instance

    # Create mock models
    mock_model1 = BlueCast(class_problem="binary")
    mock_model1.target_column = "target"
    mock_model1.conf_training.global_random_state = 42

    mock_model2 = BlueCast(class_problem="binary")
    mock_model2.target_column = "target"
    mock_model2.conf_training.global_random_state = 43

    bluecast_cv_instance.bluecast_models = [mock_model1, mock_model2]

    with tempfile.TemporaryDirectory() as tmpdir:
        for model in [mock_model1, mock_model2]:
            model.conf_training.out_of_fold_dataset_store_path = tmpdir + "/"

            oof_data = pl.DataFrame(
                {
                    "target": [0, 1, 0],
                    "predictions_class_0": [0.8, 0.2, 0.7],
                    "predictions_class_1": [0.2, 0.8, 0.3],
                    "target_class_predicted_probas": [0.8, 0.8, 0.7],
                    "feature_1": [1, 2, 3],
                }
            )

            oof_data.write_parquet(
                f"{tmpdir}/oof_data_{model.conf_training.global_random_state}.parquet"
            )

        # Test CV analyser
        analyser_cv = ErrorAnalyserClassificationCV(bluecast_cv_instance)

        # Test CV-specific method
        loaded_data = analyser_cv.read_data_from_bluecast_cv_instance()
        assert len(loaded_data) == 6  # Combined from both models

        # Clean up
        analyser_cv.duckdb_engine.close()


def test_error_analyser_wrong_method_calls():
    """Test that wrong method calls raise appropriate errors"""
    # Test OutOfFoldDataReader with CV method
    bluecast_instance = BlueCast(class_problem="binary")
    reader = OutOfFoldDataReader(bluecast_instance)

    with pytest.raises(
        ValueError, match="Please use OutOfFoldDataReaderCV class instead"
    ):
        reader.read_data_from_bluecast_cv_instance()

    # Test OutOfFoldDataReaderCV with non-CV method
    bluecast_cv = BlueCastCV(class_problem="binary")
    bluecast_cv.bluecast_models = [BlueCast(class_problem="binary")]
    reader_cv = OutOfFoldDataReaderCV(bluecast_cv)

    with pytest.raises(
        ValueError, match="Please use OutOfFoldDataReader class instead"
    ):
        reader_cv.read_data_from_bluecast_instance()


def test_error_distribution_plotter_mixin():
    """Test ErrorDistributionPlotterMixin"""
    # Test with ignore columns
    ignore_cols = ["ignore_me"]
    plotter = ErrorDistributionPlotterMixin(
        ignore_columns_during_visualization=ignore_cols
    )
    assert plotter.ignore_columns_during_visualization == ignore_cols

    # Test with None (should default to empty list)
    plotter2 = ErrorDistributionPlotterMixin(ignore_columns_during_visualization=None)
    assert plotter2.ignore_columns_during_visualization == []

    # Test error handling for missing columns
    test_df = pl.DataFrame({"feature_1": [1, 2, 3], "feature_2": ["a", "b", "c"]})

    with pytest.raises(ValueError, match="Required columns missing"):
        plotter.plot_error_distributions(test_df)
