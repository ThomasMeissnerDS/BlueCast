import tempfile

import pandas as pd
import polars as pl
import pytest

from bluecast.blueprints.cast_cv_regression import BlueCastCVRegression
from bluecast.blueprints.cast_regression import BlueCastRegression
from bluecast.evaluation.error_analysis_regression import (
    DuckDBRegressionErrorAnalysisEngine,
    ErrorAnalyserRegression,
    ErrorAnalyserRegressionCV,
    ErrorAnalyserRegressionMixin,
    OutOfFoldDataReaderRegression,
    OutOfFoldDataReaderRegressionCV,
)


@pytest.fixture
def create_test_bluecast_instance():
    """Create a mock or a test instance of BlueCast for regression"""
    bluecast_instance = BlueCastRegression(class_problem="regression")
    bluecast_instance.target_column = "target"
    return bluecast_instance


@pytest.fixture
def create_test_bluecast_cv_instance():
    """Create a mock BlueCastCVRegression instance for testing"""
    return BlueCastCVRegression(class_problem="regression")


@pytest.fixture
def sample_regression_error_data():
    """Create sample regression error analysis data for testing"""
    return pd.DataFrame(
        {
            "target": [1.0, 2.0, 1.5, 2.5, 3.0],
            "predictions": [0.9, 2.1, 1.4, 2.6, 2.8],
            "prediction_error": [0.1, 0.1, 0.1, 0.1, 0.2],
            "target_quantiles": ["low", "high", "low", "high", "high"],
            "feature_1": [1, 2, 3, 4, 5],
            "feature_2": ["a", "b", "a", "b", "a"],
        }
    )


@pytest.fixture
def duckdb_regression_engine():
    """Create a DuckDB regression error analysis engine for testing"""
    engine = DuckDBRegressionErrorAnalysisEngine()
    yield engine
    engine.close()


def test_duckdb_regression_engine_init(duckdb_regression_engine):
    """Test DuckDB regression error analysis engine initialization"""
    assert duckdb_regression_engine.db_path is not None
    # Test that tables were created
    import duckdb

    with duckdb.connect(duckdb_regression_engine.db_path) as conn:
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [table[0] for table in tables]
        assert "regression_error_analysis" in table_names
        assert "regression_error_statistics" in table_names


def test_duckdb_load_regression_data(
    duckdb_regression_engine, sample_regression_error_data
):
    """Test loading regression data into DuckDB"""
    experiment_id = "reg_test_exp_001"
    target_column = "target"  # Use numeric target column

    duckdb_regression_engine.load_regression_data(
        sample_regression_error_data, experiment_id, target_column
    )

    # Verify data was loaded
    import duckdb

    with duckdb.connect(duckdb_regression_engine.db_path) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM regression_error_analysis WHERE experiment_id = ?",
            [experiment_id],
        ).fetchone()[0]
        assert count == len(sample_regression_error_data)


def test_duckdb_compute_regression_statistics(
    duckdb_regression_engine, sample_regression_error_data
):
    """Test computing regression error statistics"""
    experiment_id = "reg_test_exp_002"
    target_column = "target"  # Use numeric target column

    duckdb_regression_engine.load_regression_data(
        sample_regression_error_data, experiment_id, target_column
    )
    stats = duckdb_regression_engine.compute_regression_statistics(experiment_id)

    assert "overall_statistics" in stats
    assert "quantile_statistics" in stats
    assert "residual_statistics" in stats
    assert "top_errors" in stats

    # Check overall statistics
    overall = stats["overall_statistics"]
    assert not overall.empty
    assert overall["total_samples"].iloc[0] == len(sample_regression_error_data)

    # Check quantile statistics
    quantile_stats = stats["quantile_statistics"]
    assert not quantile_stats.empty


def test_duckdb_create_regression_visualizations(
    duckdb_regression_engine, sample_regression_error_data
):
    """Test creating regression error visualizations"""
    experiment_id = "reg_test_exp_003"

    duckdb_regression_engine.load_regression_data(
        sample_regression_error_data,
        experiment_id,
        "target",  # Use numeric target column
    )
    figures = duckdb_regression_engine.create_regression_visualizations(experiment_id)

    # Check that figures were created
    assert isinstance(figures, dict)
    assert "residual_plot" in figures
    assert "predicted_vs_actual" in figures
    assert "error_distribution" in figures

    # Verify figures are plotly objects
    import plotly.graph_objects as go

    for fig in figures.values():
        assert isinstance(fig, go.Figure)


def test_out_of_fold_data_reader_regression(create_test_bluecast_instance):
    """Test OutOfFoldDataReaderRegression with proper error handling"""
    bluecast_instance = create_test_bluecast_instance
    data_reader = OutOfFoldDataReaderRegression(bluecast_instance)

    # Test with None path
    with pytest.raises(
        ValueError, match="out_of_fold_dataset_store_path has not been configured"
    ):
        bluecast_instance.conf_training.out_of_fold_dataset_store_path = None
        data_reader.read_data_from_bluecast_instance()

    # Test with valid path and data
    with tempfile.TemporaryDirectory() as tmpdir:
        bluecast_instance.conf_training.out_of_fold_dataset_store_path = tmpdir + "/"
        bluecast_instance.conf_training.global_random_state = 33

        oof_data = pl.DataFrame(
            {
                "target": [1.0, 2.0, 1.5, 2.5],
                "feature_1": [0.1, 0.2, 0.1, 0.2],
                "feature_2": [1, 2, 1, 2],
                "predictions": [0.9, 2.1, 1.4, 2.6],
            }
        )

        oof_data.write_parquet(f"{tmpdir}/oof_data_33.parquet")

        # Should work now
        loaded_data = data_reader.read_data_from_bluecast_instance()
        assert len(loaded_data) == 4
        assert "target" in loaded_data.columns
        assert "predictions" in loaded_data.columns


def test_out_of_fold_data_reader_regression_cv_error():
    """Test that OutOfFoldDataReaderRegression properly fails for CV instances"""
    bluecast_instance = BlueCastRegression(class_problem="regression")
    data_reader = OutOfFoldDataReaderRegression(bluecast_instance)

    with pytest.raises(
        ValueError, match="Please use OutOfFoldDataReaderRegressionCV class instead"
    ):
        data_reader.read_data_from_bluecast_cv_instance()


def test_out_of_fold_data_reader_regression_cv(create_test_bluecast_cv_instance):
    """Test OutOfFoldDataReaderRegressionCV"""
    bluecast_cv_instance = create_test_bluecast_cv_instance

    # Create mock bluecast models
    mock_model1 = BlueCastRegression(class_problem="regression")
    mock_model1.target_column = "target"
    mock_model1.conf_training.global_random_state = 42

    mock_model2 = BlueCastRegression(class_problem="regression")
    mock_model2.target_column = "target"
    mock_model2.conf_training.global_random_state = 43

    bluecast_cv_instance.bluecast_models = [mock_model1, mock_model2]

    data_reader = OutOfFoldDataReaderRegressionCV(bluecast_cv_instance)

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
                    "target": [1.0, 2.0],
                    "predictions": [0.9, 2.1],
                    "feature_1": [i + 1, i + 2],
                }
            )

            oof_data.write_parquet(
                f"{tmpdir}/oof_data_{model.conf_training.global_random_state}.parquet"
            )

        # Should work now
        loaded_data = data_reader.read_data_from_bluecast_cv_instance()
        assert len(loaded_data) == 4  # Combined data from both models


def test_error_analyser_regression_mixin():
    """Test ErrorAnalyserRegressionMixin functionality"""
    mixin = ErrorAnalyserRegressionMixin()

    # Test with sample data - use numeric target
    sample_df = pd.DataFrame(
        {
            "target": [1.0, 2.0, 1.5, 2.5],  # Use numeric target
            "predictions": [0.9, 2.1, 1.4, 2.6],  # Add predictions column
            "prediction_error": [0.1, 0.2, 0.05, 0.15],
            "feature_1": [1, 2, 3, 4],
        }
    )

    # This should create visualizations and return enhanced analysis
    result = mixin.analyse_errors(sample_df, target_column="target")

    assert isinstance(result, pl.DataFrame)
    assert not result.is_empty()

    # Clean up
    mixin.duckdb_engine.close()


def test_error_analyser_regression_full_pipeline(create_test_bluecast_instance):
    """Test full ErrorAnalyserRegression pipeline"""
    bluecast_instance = create_test_bluecast_instance

    # Create temporary out-of-fold data
    with tempfile.TemporaryDirectory() as tmpdir:
        bluecast_instance.conf_training.out_of_fold_dataset_store_path = tmpdir + "/"
        bluecast_instance.conf_training.global_random_state = 33

        # Create realistic out-of-fold data
        oof_data = pl.DataFrame(
            {
                "target": [1.0, 2.0, 1.5, 2.5, 3.0, 2.2],
                "predictions": [0.9, 2.1, 1.4, 2.6, 2.8, 2.3],
                "feature_1": [1, 2, 3, 4, 5, 6],
                "feature_2": ["x", "y", "x", "y", "x", "y"],
            }
        )

        oof_data.write_parquet(f"{tmpdir}/oof_data_33.parquet")

        # Test the full pipeline
        analyser = ErrorAnalyserRegression(bluecast_instance)

        # Test individual methods
        loaded_data = analyser.read_data_from_bluecast_instance()
        assert len(loaded_data) > 0

        stacked_data = analyser.stack_predictions_by_class(loaded_data)
        assert len(stacked_data) > 0
        assert "target_quantiles" in stacked_data.columns

        errors_data = analyser.calculate_errors(stacked_data)
        assert "prediction_error" in errors_data.columns

        # Clean up DuckDB engine
        analyser.duckdb_engine.close()


def test_error_analyser_regression_cv_full_pipeline(create_test_bluecast_cv_instance):
    """Test full ErrorAnalyserRegressionCV pipeline"""
    bluecast_cv_instance = create_test_bluecast_cv_instance

    # Create mock models
    mock_model1 = BlueCastRegression(class_problem="regression")
    mock_model1.target_column = "target"
    mock_model1.conf_training.global_random_state = 42

    mock_model2 = BlueCastRegression(class_problem="regression")
    mock_model2.target_column = "target"
    mock_model2.conf_training.global_random_state = 43

    bluecast_cv_instance.bluecast_models = [mock_model1, mock_model2]

    with tempfile.TemporaryDirectory() as tmpdir:
        for model in [mock_model1, mock_model2]:
            model.conf_training.out_of_fold_dataset_store_path = tmpdir + "/"

            oof_data = pl.DataFrame(
                {
                    "target": [1.0, 2.0, 1.5],
                    "predictions": [0.9, 2.1, 1.4],
                    "feature_1": [1, 2, 3],
                }
            )

            oof_data.write_parquet(
                f"{tmpdir}/oof_data_{model.conf_training.global_random_state}.parquet"
            )

        # Test CV analyser
        analyser_cv = ErrorAnalyserRegressionCV(bluecast_cv_instance)

        # Test CV-specific method
        loaded_data = analyser_cv.read_data_from_bluecast_cv_instance()
        assert len(loaded_data) == 6  # Combined from both models

        # Clean up
        analyser_cv.duckdb_engine.close()


def test_error_analyser_regression_wrong_method_calls():
    """Test that wrong method calls raise appropriate errors"""
    # Test OutOfFoldDataReaderRegression with CV method
    bluecast_instance = BlueCastRegression(class_problem="regression")
    reader = OutOfFoldDataReaderRegression(bluecast_instance)

    with pytest.raises(
        ValueError, match="Please use OutOfFoldDataReaderRegressionCV class instead"
    ):
        reader.read_data_from_bluecast_cv_instance()

    # Test OutOfFoldDataReaderRegressionCV with non-CV method
    bluecast_cv = BlueCastCVRegression(class_problem="regression")
    bluecast_cv.bluecast_models = [BlueCastRegression(class_problem="regression")]
    reader_cv = OutOfFoldDataReaderRegressionCV(bluecast_cv)

    with pytest.raises(
        ValueError, match="Please use OutOfFoldDataReaderRegression class instead"
    ):
        reader_cv.read_data_from_bluecast_instance()


def test_error_distribution_regression_plotter_mixin():
    """Test ErrorDistributionRegressionPlotterMixin"""
    from bluecast.evaluation.error_analysis_regression import (
        ErrorDistributionRegressionPlotterMixin,
    )

    # Test with ignore columns
    ignore_cols = ["ignore_me"]
    plotter = ErrorDistributionRegressionPlotterMixin(
        ignore_columns_during_visualization=ignore_cols
    )
    assert plotter.ignore_columns_during_visualization == ignore_cols

    # Test with None (should default to empty list)
    plotter2 = ErrorDistributionRegressionPlotterMixin(
        ignore_columns_during_visualization=None
    )
    assert plotter2.ignore_columns_during_visualization == []

    # Test error handling for missing columns
    test_df = pl.DataFrame({"feature_1": [1, 2, 3], "feature_2": ["a", "b", "c"]})

    with pytest.raises(ValueError, match="Required columns missing"):
        plotter.plot_error_distributions(test_df)


def test_regression_stack_predictions_by_class():
    """Test that stack_predictions_by_class works correctly for regression"""
    bluecast_instance = BlueCastRegression(class_problem="regression")
    bluecast_instance.target_column = "target"

    analyser = ErrorAnalyserRegression(bluecast_instance)

    # Test data
    test_data = pl.DataFrame(
        {
            "target": [1.0, 2.0, 1.5, 2.5, 3.0],
            "predictions": [0.9, 2.1, 1.4, 2.6, 2.8],
            "feature_1": [1, 2, 3, 4, 5],
        }
    )

    result = analyser.stack_predictions_by_class(test_data)

    # Check that target_quantiles column was added
    assert "target_quantiles" in result.columns
    assert len(result) == len(test_data)

    # Clean up
    analyser.duckdb_engine.close()


def test_regression_calculate_errors():
    """Test error calculation for regression"""
    bluecast_instance = BlueCastRegression(class_problem="regression")
    bluecast_instance.target_column = "target"

    analyser = ErrorAnalyserRegression(bluecast_instance)

    # Test data
    test_data = pl.DataFrame(
        {
            "target": [1.0, 2.0, 1.5],
            "predictions": [0.9, 2.1, 1.4],
            "feature_1": [1, 2, 3],
        }
    )

    result = analyser.calculate_errors(test_data)

    # Check that prediction_error column was added
    assert "prediction_error" in result.columns

    # Verify error calculations
    errors = result.to_pandas()
    expected_errors = [0.1, 0.1, 0.1]  # abs(target - predictions)

    for i, expected in enumerate(expected_errors):
        assert abs(errors["prediction_error"].iloc[i] - expected) < 0.001

    # Clean up
    analyser.duckdb_engine.close()
