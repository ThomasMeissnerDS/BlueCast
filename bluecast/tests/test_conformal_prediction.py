import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from bluecast.conformal_prediction.conformal_prediction import (
    ConformalPredictionWrapper,
)


def test_calibrate():
    # Generate some random data
    X, y = make_classification(
        n_samples=100, n_features=5, random_state=42, n_classes=2
    )
    X_train, X_calibrate, y_train, y_calibrate = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train a logistic regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Create a ConformalPredictionWrapper instance
    wrapper = ConformalPredictionWrapper(model)

    # Calibrate the wrapper
    wrapper.calibrate(X_calibrate, y_calibrate)

    # Check that the nonconformity scores have been calculated
    assert isinstance(wrapper.nonconformity_scores, np.ndarray)
    assert len(wrapper.nonconformity_scores) == len(y_calibrate)

    # Check that the nonconformity scores are floats
    for score in wrapper.nonconformity_scores:
        assert isinstance(score, float)


def test_predict():
    # Generate some random data
    X, y = make_classification(
        n_samples=100, n_features=5, random_state=42, n_classes=2
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train a logistic regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Create a ConformalPredictionWrapper instance
    wrapper = ConformalPredictionWrapper(model)

    # Make predictions
    y_pred = wrapper.predict(X_test)

    # Check that the predictions have the correct shape
    assert y_pred.shape == (len(X_test),)


def test_predict_proba():
    # Generate some random data
    X, y = make_classification(
        n_samples=100, n_features=5, random_state=42, n_classes=2
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train a logistic regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Create a ConformalPredictionWrapper instance
    wrapper = ConformalPredictionWrapper(model)

    # Make predictions
    y_pred_proba = wrapper.predict_proba(X_test)

    # Check that the predictions have the correct shape
    assert y_pred_proba.shape == (len(X_test), 2)


def test_predict_interval():
    # Generate some random data
    X, y = make_classification(
        n_samples=100, n_features=5, random_state=42, n_classes=2
    )
    X_train, X_calibrate, y_train, y_calibrate = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train a logistic regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Create a ConformalPredictionWrapper instance
    wrapper = ConformalPredictionWrapper(model)

    # Calibrate the wrapper
    wrapper.calibrate(X_calibrate, y_calibrate)

    # Make predictions
    y_pred_interval = wrapper.predict_interval(X_calibrate)

    # Check that the predictions have the correct shape
    assert y_pred_interval.shape == (len(X_calibrate), 2)


def test_predict_sets():
    # Generate some random data
    X, y = make_classification(
        n_samples=400, n_features=5, random_state=42, n_classes=2
    )
    X_train, X_calibrate, y_train, y_calibrate = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Train a logistic regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Create a ConformalPredictionWrapper instance
    wrapper = ConformalPredictionWrapper(model)

    # Calibrate the wrapper
    wrapper.calibrate(X_calibrate, y_calibrate)

    # Make predictions
    alpha = 0.05
    y_pred_sets = wrapper.predict_sets(X_test, alpha=alpha)

    # Check that the predictions have the correct shape
    assert len(y_pred_sets) == len(X_test)

    # Check that each prediction set is a set
    for prediction_set in y_pred_sets:
        assert isinstance(prediction_set, set)

        # Check that each element in the prediction set is a tuple
        for element in prediction_set:
            # Check that each tuple has one or two elements
            assert element in [0, 1]

            # Check that each element in the tuple is an integer
            assert isinstance(element, int)

    # Count correct predictions
    correct_predictions = sum(
        1 for pred_set, true_value in zip(y_pred_sets, y_test) if true_value in pred_set
    )

    # Calculate percentage
    assert correct_predictions / len(y_test) >= 1 - alpha

    # Make predictions
    alpha = 0.01
    y_pred_sets = wrapper.predict_sets(X_test, alpha=alpha)
    correct_predictions = sum(
        1 for pred_set, true_value in zip(y_pred_sets, y_test) if true_value in pred_set
    )

    assert correct_predictions / len(y_test) >= 1 - alpha
