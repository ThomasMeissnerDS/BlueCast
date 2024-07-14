from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    matthews_corrcoef,
    mean_squared_error,
    root_mean_squared_error,
)

from bluecast.evaluation.eval_metrics import (
    ClassificationEvalWrapper,
    RegressionEvalWrapper,
    eval_classifier,
    eval_regressor,
    mean_squared_error_diff_sklearn_versions,
    plot_probability_distribution,
    root_mean_squared_error_diff_sklearn_versions,
)


@pytest.fixture
def sample_data():
    y_true = np.array([0, 1, 1, 0])
    y_probs = np.array([[0.8, 0.2], [0.2, 0.8], [0.4, 0.6], [0.9, 0.1]])
    return y_true, y_probs


@pytest.fixture
def sample_data_regression():
    y_true = np.array([1.0, 5.0, 25.0])
    y_hat = np.array([3.0, 3.0, 28.0])
    return y_true, y_hat


def test_init():
    wrapper = ClassificationEvalWrapper()
    assert wrapper.higher_is_better
    assert wrapper.eval_against == "classes"
    assert wrapper.metric_func == matthews_corrcoef


def test_probas_all_classes(sample_data):
    y_true, y_probs = sample_data
    wrapper = ClassificationEvalWrapper(
        eval_against="probas_all_classes",
        metric_func=log_loss,
        higher_is_better=False,
        metric_name="logloss",
    )
    score = wrapper.classification_eval_func_wrapper(y_true, y_probs)
    expected_score = log_loss(y_true.tolist(), y_probs.tolist())
    assert score == expected_score


def test_probas_target_class(sample_data):
    y_true, y_probs = sample_data
    wrapper = ClassificationEvalWrapper(
        eval_against="probas_target_class", metric_func=log_loss, higher_is_better=False
    )
    score = wrapper.classification_eval_func_wrapper(y_true, y_probs)
    y_probs_best_class = np.asarray([line[1] for line in y_probs])
    expected_score = log_loss(y_true, y_probs_best_class)
    assert score == expected_score


def test_classes(sample_data):
    y_true, y_probs = sample_data
    wrapper = ClassificationEvalWrapper(
        eval_against="classes", metric_func=accuracy_score, higher_is_better=True
    )
    score = wrapper.classification_eval_func_wrapper(y_true, y_probs)
    y_classes = np.asarray([np.argmax(line) for line in y_probs])
    expected_score = accuracy_score(y_true, y_classes)
    assert score == -expected_score


def test_higher_is_better_false(sample_data):
    y_true, y_probs = sample_data
    wrapper = ClassificationEvalWrapper(
        higher_is_better=True, eval_against="classes", metric_func=accuracy_score
    )
    score = wrapper.classification_eval_func_wrapper(y_true, y_probs)
    y_classes = np.asarray([np.argmax(line) for line in y_probs])
    expected_score = accuracy_score(y_true, y_classes)
    assert score == -expected_score


def test_invalid_eval_against():
    with pytest.raises(ValueError):
        ClassificationEvalWrapper(eval_against="invalid_value")


def test_regression_init():
    wrapper = RegressionEvalWrapper()
    assert not wrapper.higher_is_better
    assert wrapper.metric_func == root_mean_squared_error


def test_regression_run_with_args(sample_data_regression):
    y_true, y_hat = sample_data_regression
    wrapper = RegressionEvalWrapper(
        higher_is_better=False, metric_func=mean_squared_error, **{"squared": False}
    )
    score = wrapper.regression_eval_func_wrapper(y_true, y_hat)
    expected_score = mean_squared_error(y_true, y_hat, squared=False)
    assert score == expected_score


def test_regression_run_witouth_args(sample_data_regression):
    y_true, y_hat = sample_data_regression
    wrapper = RegressionEvalWrapper(
        higher_is_better=False, metric_func=mean_squared_error
    )
    score = wrapper.regression_eval_func_wrapper(y_true, y_hat)
    expected_score = mean_squared_error(y_true, y_hat)
    assert score == expected_score


def test_root_mean_squared_error_import():
    with patch.dict("sys.modules", {"sklearn.metrics.root_mean_squared_error": None}):
        with patch("sklearn.metrics.mean_squared_error") as mock_mse:
            from sklearn.metrics import mean_squared_error as root_mean_squared_error

            assert root_mean_squared_error is mock_mse


def test_root_mean_squared_error_direct_import():
    with patch.dict(
        "sys.modules", {"sklearn.metrics.root_mean_squared_error": object()}
    ):
        from sklearn.metrics import root_mean_squared_error

        assert root_mean_squared_error is not None
        assert callable(
            root_mean_squared_error
        )  # assuming root_mean_squared_error is callable


def test_classification_eval_func_wrapper_invalid_eval_against():
    # Instantiate the class with an invalid eval_against value
    def dummy_metric_func(y_true, y_pred, **kwargs):
        # Dummy metric function for testing purposes
        return 0.5

    class TestClass(ClassificationEvalWrapper):
        def __init__(self, eval_against, metric_func):
            super().__init__(eval_against=eval_against, metric_func=metric_func)

    test_obj = TestClass(
        eval_against="probas_all_classes", metric_func=dummy_metric_func
    )
    test_obj.eval_against = "invalid_value"

    # Dummy data for testing
    y_true = [1, 0, 1]
    y_probs = [[0.4, 0.6], [0.7, 0.3], [0.2, 0.8]]

    # Use pytest to check if ValueError is raised
    with pytest.raises(
        ValueError,
        match=r"Unknown value for eval_against: invalid_value\. Possible values are 'probas' or 'classes'",
    ):
        test_obj.classification_eval_func_wrapper(y_true, y_probs)


def test_eval_classifier_binary():
    y_true = np.array([0, 1, 0, 1])
    y_probs = np.array([0.1, 0.9, 0.2, 0.8])
    y_classes = np.array([0, 1, 0, 1])

    result = eval_classifier(y_true, y_probs, y_classes)

    assert "matthews" in result
    assert "accuracy" in result
    assert "recall" in result
    assert "f1_score_macro" in result
    assert "f1_score_micro" in result
    assert "f1_score_weighted" in result
    assert "log_loss" in result
    assert "balanced_logloss" in result
    assert "roc_auc" in result
    assert "classfication_report" in result
    assert "confusion_matrix" in result


def test_eval_classifier_multiclass():
    y_true = np.array([0, 1, 2, 1])
    y_probs = np.array(
        [[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.2, 0.2, 0.6], [0.3, 0.4, 0.3]]
    )
    y_classes = np.array([0, 1, 2, 1])

    result = eval_classifier(y_true, y_probs, y_classes)

    assert "matthews" in result
    assert "accuracy" in result
    assert "recall" in result
    assert "f1_score_macro" in result
    assert "f1_score_micro" in result
    assert "f1_score_weighted" in result
    assert "log_loss" in result
    assert "balanced_logloss" in result
    assert "roc_auc" in result
    assert "classfication_report" in result
    assert "confusion_matrix" in result


def test_eval_regressor():
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_preds = np.array([2.5, 0.0, 2.1, 7.8])

    result = eval_regressor(y_true, y_preds)

    assert "mae" in result
    assert "r2_score" in result
    assert "MSE" in result
    assert "RMSE" in result
    assert "median_absolute_error" in result


def test_eval_classifier_exceptions():
    y_true = np.array([0, 0, 0, 1])
    y_probs = np.array([0.0, 0.0, 0.0, 0.0])
    y_classes = np.array([0, 0, 0, 0])  # Introduce an error here

    result = eval_classifier(y_true, y_probs, y_classes)

    assert result["matthews"] == 0  # Error handling should set matthews to 0


def test_eval_regressor_exceptions():
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_preds = np.array([2.5, 0.0, np.nan, 7.8])  # Introduce a NaN value

    with pytest.raises(ValueError):
        eval_regressor(y_true, y_preds)


def test_plot_probability_distribution_multiclass():
    # Generate synthetic data for testing
    np.random.seed(0)
    n_samples = 100
    n_classes = 3
    probs = np.random.rand(n_samples, n_classes)
    probs /= probs.sum(
        axis=1, keepdims=True
    )  # Normalize to ensure they are valid probabilities
    y_classes = np.random.randint(0, n_classes, n_samples)  # TODO: Fix random_seed

    # Call the function and catch the plot
    plt.figure()
    plot_probability_distribution(probs, y_classes, opacity=0.7)

    # Since the function generates a plot, we can use plt.gcf() to get the current figure and perform assertions
    fig = plt.gcf()
    assert fig is not None, "The plot was not created."

    # Cleanup the plot to avoid side effects
    plt.close(fig)


def test_mean_squared_error_diff_sklearn_versions():
    # Test case 1: Simple example with integers
    y_true_1 = [1, 2, 3, 4, 5]
    y_preds_1 = [1, 2, 3, 4, 5]
    expected_mse_1 = 0.0
    mse_1 = mean_squared_error_diff_sklearn_versions(y_true_1, y_preds_1)
    assert (
        mse_1 == expected_mse_1
    ), f"Test Case 1 Failed: Expected {expected_mse_1}, got {mse_1}"

    # Test case 2: Example with slight differences
    y_true_2 = [1, 2, 3, 4, 5]
    y_preds_2 = [1, 2, 3, 4, 6]
    expected_mse_2 = 0.2
    mse_2 = mean_squared_error_diff_sklearn_versions(y_true_2, y_preds_2)
    assert (
        mse_2 == expected_mse_2
    ), f"Test Case 2 Failed: Expected {expected_mse_2}, got {mse_2}"

    # Test case 3: Example with negative values
    y_true_3 = [-1, -2, -3, -4, -5]
    y_preds_3 = [-1, -2, -3, -4, -5]
    expected_mse_3 = 0.0
    mse_3 = mean_squared_error_diff_sklearn_versions(y_true_3, y_preds_3)
    assert (
        mse_3 == expected_mse_3
    ), f"Test Case 3 Failed: Expected {expected_mse_3}, got {mse_3}"


def mock_root_mean_squared_error(y_true, y_preds):
    raise AttributeError("root_mean_squared_error is not available")


def test_root_mean_squared_error_diff_sklearn_versions_correct():
    y_true = [1, 2, 3, 4, 5]
    y_preds = [1, 2, 3, 4, 6]
    expected_rmse = mean_squared_error(y_true, y_preds, squared=False)
    result = root_mean_squared_error_diff_sklearn_versions(y_true, y_preds)
    assert np.isclose(result, expected_rmse), f"Expected {expected_rmse}, got {result}"


def test_root_mean_squared_error_diff_sklearn_versions_fallback(monkeypatch):
    monkeypatch.setattr(
        "bluecast.evaluation.eval_metrics.root_mean_squared_error",
        mock_root_mean_squared_error,
    )
    y_true = [1, 2, 3, 4, 5]
    y_preds = [1, 2, 3, 4, 6]
    expected_rmse = mean_squared_error(y_true, y_preds, squared=False)
    result = root_mean_squared_error_diff_sklearn_versions(y_true, y_preds)
    assert np.isclose(result, expected_rmse), f"Expected {expected_rmse}, got {result}"


def test_root_mean_squared_error_diff_sklearn_versions_empty():
    y_true = []
    y_preds = []
    with pytest.raises(ValueError):
        root_mean_squared_error_diff_sklearn_versions(y_true, y_preds)


def test_root_mean_squared_error_diff_sklearn_versions_single_element():
    y_true = [1]
    y_preds = [1]
    expected_rmse = mean_squared_error(y_true, y_preds, squared=False)
    result = root_mean_squared_error_diff_sklearn_versions(y_true, y_preds)
    assert np.isclose(result, expected_rmse), f"Expected {expected_rmse}, got {result}"


def test_root_mean_squared_error_diff_sklearn_versions_mismatched_lengths():
    y_true = [1, 2, 3]
    y_preds = [1, 2]
    with pytest.raises(ValueError):
        root_mean_squared_error_diff_sklearn_versions(y_true, y_preds)
