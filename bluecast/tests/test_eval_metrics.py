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
