import pytest

from bluecast.config.training_config import TrainingConfig
from bluecast.experimentation.tracking import ExperimentTracker


@pytest.fixture
def experiment_tracker():
    return ExperimentTracker()


def test_add_results(experiment_tracker):
    # Add some sample data
    experiment_id = 1
    score_category = "cv_score"
    training_config = (
        TrainingConfig()
    )  # You may need to create a valid TrainingConfig instance
    model_parameters = {"param1": 1, "param2": "abc"}
    eval_scores = 0.95
    metric_used = "accuracy"
    metric_higher_is_better = True

    experiment_tracker.add_results(
        experiment_id,
        score_category,
        training_config,
        model_parameters,
        eval_scores,
        metric_used,
        metric_higher_is_better,
    )

    # Check if the data was added correctly
    assert experiment_tracker.experiment_id == [1]
    assert experiment_tracker.score_category == ["cv_score"]
    assert experiment_tracker.training_configs == [
        training_config.model_dump(mode="json")
    ]
    assert experiment_tracker.model_parameters == [model_parameters]
    assert experiment_tracker.eval_scores == [0.95]
    assert experiment_tracker.metric_used == ["accuracy"]
    assert experiment_tracker.metric_higher_is_better == [True]


def test_retrieve_results_as_df(experiment_tracker):
    # Add some sample data
    experiment_id = 1
    score_category = "cv_score"
    training_config = (
        TrainingConfig()
    )  # You may need to create a valid TrainingConfig instance
    model_parameters = {"param1": 1, "param2": "abc"}
    eval_scores = 0.95
    metric_used = "accuracy"
    metric_higher_is_better = True

    experiment_tracker.add_results(
        experiment_id,
        score_category,
        training_config,
        model_parameters,
        eval_scores,
        metric_used,
        metric_higher_is_better,
    )

    # Retrieve the results as a DataFrame
    results_df = experiment_tracker.retrieve_results_as_df()

    # Check if the retrieved data matches the added data
    assert results_df["experiment_id"].tolist() == [1]
    assert results_df["score_category"].tolist() == ["cv_score"]
    assert results_df["eval_scores"].tolist() == [0.95]
    assert results_df["metric_used"].tolist() == ["accuracy"]
    assert results_df["metric_higher_is_better"].tolist() == [True]


def test_get_best_score_empty(experiment_tracker):
    # Ensure it raises an exception when no results have been added
    with pytest.raises(
        ValueError, match="No results have been found in experiment tracker"
    ):
        experiment_tracker.get_best_score(target_metric="loss")


def test_get_best_score_higher_is_better(experiment_tracker):
    # Add some sample data with a higher-is-better metric
    experiment_id = 1
    score_category = "cv_score"
    training_config = (
        TrainingConfig()
    )  # You may need to create a valid TrainingConfig instance
    model_parameters = {"param1": 1, "param2": "abc"}
    eval_scores = 0.95
    metric_used = "accuracy"
    metric_higher_is_better = True

    experiment_tracker.add_results(
        experiment_id,
        score_category,
        training_config,
        model_parameters,
        eval_scores,
        metric_used,
        metric_higher_is_better,
    )

    # Ensure the best score is correctly computed
    best_score = experiment_tracker.get_best_score(target_metric="accuracy")
    assert best_score == 0.95


def test_get_best_score_lower_is_better(experiment_tracker):
    # Add some sample data with a lower-is-better metric
    experiment_id = 1
    score_category = "cv_score"
    training_config = (
        TrainingConfig()
    )  # You may need to create a valid TrainingConfig instance
    model_parameters = {"param1": 1, "param2": "abc"}
    eval_scores = 0.95
    metric_used = "loss"  # Assuming "loss" is a metric that is lower-is-better
    metric_higher_is_better = False

    experiment_tracker.add_results(
        experiment_id,
        score_category,
        training_config,
        model_parameters,
        eval_scores,
        metric_used,
        metric_higher_is_better,
    )

    # Ensure the best score is correctly computed
    best_score = experiment_tracker.get_best_score(target_metric="loss")
    assert best_score == 0.95
