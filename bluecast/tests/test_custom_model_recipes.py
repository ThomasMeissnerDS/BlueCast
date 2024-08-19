import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from bluecast.blueprints.custom_model_recipes import (
    LinearRegressionModel,
    LogisticRegressionModel,
)


@pytest.fixture
def data():
    # Generate a synthetic binary classification dataset
    X, y = make_classification(
        n_samples=100, n_features=20, n_informative=2, n_redundant=10, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return (
        pd.DataFrame(X_train),
        pd.DataFrame(X_test),
        pd.Series(y_train),
        pd.Series(y_test),
    )


@pytest.fixture
def model():
    return LogisticRegressionModel(random_state=42)


def test_autotune(data, model):
    X_train, X_test, y_train, y_test = data
    model.autotune(X_train, X_test, y_train, y_test)

    # Check if the model has been fitted by inspecting the attributes
    assert hasattr(
        model.model.best_estimator_, "coef_"
    ), "Model should have been fitted and have coefficients."


def test_fit(data, model):
    X_train, X_test, y_train, y_test = data
    model.fit(X_train, X_test, y_train, y_test)

    # Again, check if the model has been fitted
    assert hasattr(
        model.model.best_estimator_, "coef_"
    ), "Model should have been fitted after calling fit method."


def test_predict(data, model):
    X_train, X_test, y_train, y_test = data
    model.fit(X_train, X_test, y_train, y_test)

    probas, classes = model.predict(X_test)

    # Check the types of the returned values
    assert isinstance(
        probas, np.ndarray
    ), "Predicted probabilities should be a numpy array."
    assert isinstance(classes, np.ndarray), "Predicted classes should be a numpy array."

    # Check the shape of the returned values
    assert probas.shape == (
        X_test.shape[0],
    ), "Predicted probabilities should have the correct shape."
    assert classes.shape == (
        X_test.shape[0],
    ), "Predicted classes should have the correct shape."

    # Check if values are within the expected range
    assert np.all(
        (probas >= 0) & (probas <= 1)
    ), "Predicted probabilities should be between 0 and 1."
    assert np.all(
        (classes == 0) | (classes == 1)
    ), "Predicted classes should be either 0 or 1."


@pytest.fixture
def regression_data():
    # Generate a synthetic regression dataset
    X, y = make_regression(
        n_samples=100, n_features=20, n_informative=10, noise=0.1, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return (
        pd.DataFrame(X_train),
        pd.DataFrame(X_test),
        pd.Series(y_train),
        pd.Series(y_test),
    )


## TEST LINEAR REGRESSION MODEL


@pytest.fixture
def linear_model():
    return LinearRegressionModel()


def test_linear_autotune(regression_data, linear_model):
    X_train, X_test, y_train, y_test = regression_data
    linear_model.autotune(X_train, X_test, y_train, y_test)

    # Check if the model has been fitted by inspecting the attributes
    assert hasattr(
        linear_model.model, "coef_"
    ), "Model should have been fitted and have coefficients."


def test_linear_fit(regression_data, linear_model):
    X_train, X_test, y_train, y_test = regression_data
    linear_model.fit(X_train, X_test, y_train, y_test)

    # Again, check if the model has been fitted
    assert hasattr(
        linear_model.model, "coef_"
    ), "Model should have been fitted after calling fit method."


def test_linear_predict(regression_data, linear_model):
    X_train, X_test, y_train, y_test = regression_data
    linear_model.fit(X_train, X_test, y_train, y_test)

    predictions = linear_model.predict(X_test)

    # Check the types of the returned values
    assert isinstance(predictions, np.ndarray), "Predictions should be a numpy array."

    # Check the shape of the returned values
    assert predictions.shape == (
        X_test.shape[0],
    ), "Predictions should have the correct shape."

    # Optionally, check the values of predictions are finite (not NaN or inf)
    assert np.all(np.isfinite(predictions)), "Predictions should be finite values."


def test_linear_predict_range(regression_data, linear_model):
    X_train, X_test, y_train, y_test = regression_data
    linear_model.fit(X_train, X_test, y_train, y_test)

    predictions = linear_model.predict(X_test)

    # Check if predictions are within a reasonable range
    assert np.all(
        predictions >= y_train.min() - 10
    ), "Predictions should not be too low."
    assert np.all(
        predictions <= y_train.max() + 10
    ), "Predictions should not be too high."
