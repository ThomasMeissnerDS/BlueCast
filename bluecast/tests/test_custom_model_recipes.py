import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from bluecast.blueprints.custom_model_recipes import LogisticRegressionModel


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
        model.model, "coef_"
    ), "Model should have been fitted and have coefficients."


def test_fit(data, model):
    X_train, X_test, y_train, y_test = data
    model.fit(X_train, X_test, y_train, y_test)

    # Again, check if the model has been fitted
    assert hasattr(
        model.model, "coef_"
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
