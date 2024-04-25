import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from bluecast.blueprints.cast import BlueCast
from bluecast.blueprints.cast_regression import BlueCastRegression
from bluecast.config.training_config import TrainingConfig
from bluecast.conformal_prediction.effectiveness_nonconformity_measures import (
    prediction_interval_spans,
)
from bluecast.conformal_prediction.evaluation import (
    prediction_interval_coverage,
    prediction_set_coverage,
)


def test_prediction_set_coverage():
    X, y = make_classification(
        n_samples=10000, n_features=5, random_state=42, n_classes=2
    )
    X_train, X_calibrate, y_train, y_calibrate = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_test, X_calibrate, y_test, y_calibrate = train_test_split(
        X_calibrate, y_calibrate, test_size=0.5, random_state=42
    )

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    X_calibrate = pd.DataFrame(X_calibrate)

    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)
    X_calibrate.columns = X_calibrate.columns.astype(str)

    # Create a custom training config and adjust general training parameters
    train_config = TrainingConfig()
    train_config.hyperparameter_tuning_rounds = 10
    train_config.autotune_model = (
        False  # we want to run just normal training, no hyperparameter tuning
    )

    automl = BlueCast(
        class_problem="binary",
        conf_training=train_config,
    )

    X_train["target"] = y_train
    automl.fit(X_train, target_col="target")

    # make use of calibration
    automl.calibrate(X_calibrate, y_calibrate)

    # prediction sets given a certain confidence interval alpha
    alpha = 0.05
    pred_sets = automl.predict_sets(X_test, alpha=alpha)

    assert prediction_set_coverage(y_test, pred_sets.values) > 1 - alpha


def test_prediction_interval_coverage():
    X, y = make_regression(
        n_samples=10000,
        n_features=5,
        random_state=42,
    )
    X_train, X_calibrate, y_train, y_calibrate = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_test, X_calibrate, y_test, y_calibrate = train_test_split(
        X_calibrate, y_calibrate, test_size=0.5, random_state=42
    )
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    X_calibrate = pd.DataFrame(X_calibrate)

    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)
    X_calibrate.columns = X_calibrate.columns.astype(str)

    # Create a custom training config and adjust general training parameters
    train_config_reg = TrainingConfig()
    train_config_reg.hyperparameter_tuning_rounds = 10
    train_config_reg.autotune_model = (
        True  # we want to run just normal training, no hyperparameter tuning
    )

    automl_reg = BlueCastRegression(
        class_problem="regression",
        conf_training=train_config_reg,
    )

    X_train["target"] = y_train
    automl_reg.fit(X_train, target_col="target")

    # make use of calibration
    automl_reg.calibrate(X_calibrate, y_calibrate)

    # p-values for each class being the correct one
    pred_intervals = automl_reg.predict_interval(X_test, alphas=[0.01, 0.05, 0.1])

    val_results = prediction_interval_coverage(
        y_test, pred_intervals, alphas=[0.01, 0.05, 0.1]
    )
    assert val_results[0.01] > val_results[0.05] > val_results[0.1]

    band_sizes = prediction_interval_spans(pred_intervals, alphas=[0.01, 0.05, 0.1])
    assert band_sizes[0.01] > band_sizes[0.05] > band_sizes[0.1]
