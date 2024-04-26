import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from bluecast.conformal_prediction.conformal_prediction_regression import (
    ConformalPredictionRegressionWrapper,
)


def test_calibrate():
    # Generate some random data
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    X_train, X_calibrate, y_train, y_calibrate = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Create a ConformalPredictionRegressionWrapper instance
    wrapper = ConformalPredictionRegressionWrapper(model)

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
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Create a ConformalPredictionRegressionWrapper instance
    wrapper = ConformalPredictionRegressionWrapper(model)

    # Make predictions
    y_pred = wrapper.predict(X_test)

    # Check that the predictions have the correct shape
    assert y_pred.shape == (len(X_test),)


def test_predict_interval():
    # Generate some random data
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    X_train, X_calibrate, y_train, y_calibrate = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Create a ConformalPredictionRegressionWrapper instance
    wrapper = ConformalPredictionRegressionWrapper(model)

    # Calibrate the wrapper
    wrapper.calibrate(X_calibrate, y_calibrate)

    # Make predictions
    alphas = [0.05, 0.01]
    y_pred_interval = wrapper.predict_interval(X_calibrate, alphas)

    # Check that the predictions have the correct shape
    assert y_pred_interval.shape == (len(X_calibrate), len(alphas) * 2)


def test_predict_interval_with_single_data_point():
    # Generate some random data
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Create a ConformalPredictionRegressionWrapper instance
    wrapper = ConformalPredictionRegressionWrapper(model)

    # Calibrate the wrapper with a single data point calibration set
    wrapper.calibrate(X_train, y_train)

    # Make predictions
    alphas = [0.05, 0.01]
    y_pred_interval = wrapper.predict_interval(X_train, alphas)

    # Check that the predictions have the correct shape
    assert y_pred_interval.shape == (len(X_train), len(alphas) * 2)


def test_predict_interval_with_single_quantile():
    # Generate some random data
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    X_train, X_calibrate, y_train, y_calibrate = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Create a ConformalPredictionRegressionWrapper instance
    wrapper = ConformalPredictionRegressionWrapper(model)

    # Calibrate the wrapper
    wrapper.calibrate(X_calibrate, y_calibrate)

    # Make predictions
    alphas = [0.05]
    y_pred_interval = wrapper.predict_interval(X_calibrate, alphas)

    # Check that the predictions have the correct shape
    assert y_pred_interval.shape == (len(X_calibrate), len(alphas) * 2)
    assert np.sum(y_pred_interval.iloc[:, 0] <= y_pred_interval.iloc[:, -1]) == len(
        X_calibrate
    )

    # Make predictions
    alphas = [0.05, 0.1, 0.5]
    y_pred_interval = wrapper.predict_interval(X_calibrate, alphas)
    assert y_pred_interval.shape == (len(X_calibrate), len(alphas) * 2)
