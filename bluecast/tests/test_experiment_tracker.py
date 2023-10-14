import pytest

from bluecast.config.training_config import TrainingConfig
from bluecast.experimentation.tracking import ExperimentTracker


@pytest.fixture
def experiment_tracker():
    return ExperimentTracker()


def test_add_results(experiment_tracker):
    # Add some sample data
    experiment_id = 1
    score_category = "hyperparameter_tuning"
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
    assert experiment_tracker.score_category == ["hyperparameter_tuning"]
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
    score_category = "hyperparameter_tuning"
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
    assert results_df["score_category"].tolist() == ["hyperparameter_tuning"]
    assert results_df["eval_scores"].tolist() == [0.95]
    assert results_df["metric_used"].tolist() == ["accuracy"]
    assert results_df["metric_higher_is_better"].tolist() == [True]
