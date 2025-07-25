import os
import tempfile

import pytest

from bluecast.config.training_config import TrainingConfig
from bluecast.experimentation.tracking import ExperimentTracker


@pytest.fixture
def experiment_tracker():
    """Create a temporary DuckDB-based experiment tracker for testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_experiment_tracker.duckdb")
    tracker = ExperimentTracker(db_path=db_path)
    yield tracker
    tracker.close()


@pytest.fixture
def sample_training_config():
    """Create a sample training config for testing."""
    return TrainingConfig()


def test_add_hyperparameter_results(experiment_tracker, sample_training_config):
    """Test adding hyperparameter tuning results."""
    experiment_id = 1
    score_category = "cv_score"
    model_parameters = {"param1": 1, "param2": "abc", "learning_rate": 0.1}
    eval_scores = 0.95
    metric_used = "accuracy"
    metric_higher_is_better = True

    experiment_tracker.add_results(
        experiment_id,
        score_category,
        sample_training_config,
        model_parameters,
        eval_scores,
        metric_used,
        metric_higher_is_better,
    )

    # Check if the data was added correctly to hyperparameter table
    hyperparameter_results = experiment_tracker.get_hyperparameter_results()
    assert len(hyperparameter_results) == 1
    assert hyperparameter_results["experiment_id"].iloc[0] == 1
    assert hyperparameter_results["score_category"].iloc[0] == "cv_score"
    assert hyperparameter_results["eval_scores"].iloc[0] == 0.95
    assert hyperparameter_results["metric_used"].iloc[0] == "accuracy"
    assert hyperparameter_results["metric_higher_is_better"].iloc[0] == True


def test_add_evaluation_results(experiment_tracker, sample_training_config):
    """Test adding evaluation results."""
    experiment_id = 1
    score_category = "oof_score"  # This should go to evaluation table
    model_parameters = {"param1": 2, "param2": "def"}
    eval_scores = 0.88
    metric_used = "f1_score"
    metric_higher_is_better = True

    experiment_tracker.add_results(
        experiment_id,
        score_category,
        sample_training_config,
        model_parameters,
        eval_scores,
        metric_used,
        metric_higher_is_better,
    )

    # Check if the data was added correctly to evaluation table
    evaluation_results = experiment_tracker.get_evaluation_results()
    assert len(evaluation_results) == 1
    assert evaluation_results["experiment_id"].iloc[0] == 1
    assert evaluation_results["score_category"].iloc[0] == "oof_score"
    assert evaluation_results["eval_scores"].iloc[0] == 0.88
    assert evaluation_results["metric_used"].iloc[0] == "f1_score"


def test_retrieve_results_as_df(experiment_tracker, sample_training_config):
    """Test retrieving results as DataFrame."""
    # Add data to both tables
    experiment_tracker.add_results(
        1, "cv_score", sample_training_config, {"param1": 1}, 0.95, "accuracy", True
    )
    experiment_tracker.add_results(
        1, "oof_score", sample_training_config, {"param1": 2}, 0.88, "f1_score", True
    )

    # Retrieve all results
    all_results = experiment_tracker.retrieve_results_as_df()
    assert len(all_results) == 2
    assert "table_source" in all_results.columns
    assert "hyperparameter" in all_results["table_source"].values
    assert "evaluation" in all_results["table_source"].values

    # Retrieve only hyperparameter results
    hyper_results = experiment_tracker.retrieve_results_as_df(
        table="hyperparameter_experiments"
    )
    assert len(hyper_results) == 1
    assert hyper_results["score_category"].iloc[0] == "cv_score"

    # Retrieve only evaluation results
    eval_results = experiment_tracker.retrieve_results_as_df(
        table="evaluation_experiments"
    )
    assert len(eval_results) == 1
    assert eval_results["score_category"].iloc[0] == "oof_score"


def test_get_best_score_empty(experiment_tracker):
    """Test getting best score when no results exist."""
    with pytest.raises(
        ValueError, match="No results have been found in experiment tracker"
    ):
        experiment_tracker.get_best_score(target_metric="loss")


def test_get_best_score_higher_is_better(experiment_tracker, sample_training_config):
    """Test getting best score for higher-is-better metric."""
    # Add multiple results with same metric
    experiment_tracker.add_results(
        1, "cv_score", sample_training_config, {"param1": 1}, 0.95, "accuracy", True
    )
    experiment_tracker.add_results(
        2, "cv_score", sample_training_config, {"param1": 2}, 0.92, "accuracy", True
    )
    experiment_tracker.add_results(
        3, "oof_score", sample_training_config, {"param1": 3}, 0.98, "accuracy", True
    )

    # Get best score across all tables
    best_score = experiment_tracker.get_best_score(target_metric="accuracy")
    assert best_score == 0.98

    # Get best score from hyperparameter table only
    best_hyper_score = experiment_tracker.get_best_score(
        target_metric="accuracy", table="hyperparameter_experiments"
    )
    assert best_hyper_score == 0.95

    # Get best score from evaluation table only
    best_eval_score = experiment_tracker.get_best_score(
        target_metric="accuracy", table="evaluation_experiments"
    )
    assert best_eval_score == 0.98


def test_get_best_score_lower_is_better(experiment_tracker, sample_training_config):
    """Test getting best score for lower-is-better metric."""
    # Add multiple results with lower-is-better metric
    experiment_tracker.add_results(
        1, "cv_score", sample_training_config, {"param1": 1}, 0.15, "loss", False
    )
    experiment_tracker.add_results(
        2, "cv_score", sample_training_config, {"param1": 2}, 0.12, "loss", False
    )
    experiment_tracker.add_results(
        3, "oof_score", sample_training_config, {"param1": 3}, 0.08, "loss", False
    )

    # Get best (lowest) score
    best_score = experiment_tracker.get_best_score(target_metric="loss")
    assert best_score == 0.08


def test_get_experiment_summary(experiment_tracker, sample_training_config):
    """Test getting experiment summary statistics."""
    # Add various results
    experiment_tracker.add_results(
        1, "cv_score", sample_training_config, {"param1": 1}, 0.95, "accuracy", True
    )
    experiment_tracker.add_results(
        1, "cv_score", sample_training_config, {"param1": 2}, 0.92, "precision", True
    )
    experiment_tracker.add_results(
        2, "oof_score", sample_training_config, {"param1": 3}, 0.88, "accuracy", True
    )

    summary = experiment_tracker.get_experiment_summary()

    assert summary["total_hyperparameter_experiments"] == 2
    assert summary["total_evaluation_experiments"] == 1
    assert summary["unique_experiments"] == 2
    assert "accuracy" in summary["metrics_used"]
    assert "precision" in summary["metrics_used"]


def test_legacy_compatibility(experiment_tracker, sample_training_config):
    """Test legacy compatibility properties."""
    # Add some results
    experiment_tracker.add_results(
        1, "cv_score", sample_training_config, {"param1": 1}, 0.95, "accuracy", True
    )
    experiment_tracker.add_results(
        2, "oof_score", sample_training_config, {"param1": 2}, 0.88, "f1_score", True
    )

    # Test legacy property
    experiment_ids = experiment_tracker.experiment_id
    assert len(experiment_ids) == 2
    assert 1 in experiment_ids
    assert 2 in experiment_ids


def test_database_persistence():
    """Test that data persists across tracker instances."""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "persistence_test.duckdb")

    # Create first tracker and add data
    tracker1 = ExperimentTracker(db_path=db_path)
    config = TrainingConfig()
    tracker1.add_results(1, "cv_score", config, {"param1": 1}, 0.95, "accuracy", True)
    tracker1.close()

    # Create second tracker with same database
    tracker2 = ExperimentTracker(db_path=db_path)
    results = tracker2.retrieve_results_as_df()
    assert len(results) == 1
    assert results["eval_scores"].iloc[0] == 0.95
    tracker2.close()

    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)


def test_temporary_database():
    """Test that temporary database is created when no path provided."""
    tracker = ExperimentTracker()  # No db_path provided
    config = TrainingConfig()

    # Should work with temporary database
    tracker.add_results(1, "cv_score", config, {"param1": 1}, 0.95, "accuracy", True)

    results = tracker.retrieve_results_as_df()
    assert len(results) == 1

    # Database should be cleaned up automatically
    db_path = tracker.db_path
    tracker.close()

    # Temporary file should be cleaned up
    assert not os.path.exists(db_path)
