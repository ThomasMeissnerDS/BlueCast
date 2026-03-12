"""Tests for the unified BlueCastAuto interface."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from bluecast.blueprints.unified import BlueCastAuto
from bluecast.config.training_config import TrainingConfig
from bluecast.ensemble.ensemble_config import EnsembleConfig


@pytest.fixture
def fast_config():
    return TrainingConfig(
        hyperparameter_tuning_rounds=2,
        hyperparameter_tuning_max_runtime_secs=10,
        enable_feature_selection=False,
        calculate_shap_values=False,
        plot_hyperparameter_tuning_overview=False,
        hypertuning_cv_folds=2,
        train_size=0.8,
        bluecast_cv_train_n_model=(2, 1),
    )


@pytest.fixture
def binary_data():
    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(5)])
    df["target"] = y
    return df


@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=200, n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(5)])
    df["target"] = y
    return df


def test_binary_single_model(binary_data, fast_config):
    automl = BlueCastAuto(
        class_problem="binary",
        use_cross_validation=False,
        conf_training=fast_config,
    )
    automl.fit(binary_data, target_col="target")
    preds = automl.predict(binary_data.drop("target", axis=1))
    assert isinstance(preds, tuple)
    assert len(preds[0]) == len(binary_data)


def test_binary_cv(binary_data, fast_config):
    automl = BlueCastAuto(
        class_problem="binary",
        use_cross_validation=True,
        conf_training=fast_config,
    )
    result = automl.fit_eval(binary_data, target_col="target")
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_regression_single_model(regression_data, fast_config):
    from sklearn.model_selection import train_test_split

    df_train, df_eval = train_test_split(regression_data, test_size=0.2, random_state=42)
    y_eval = df_eval.pop("target")

    automl = BlueCastAuto(
        class_problem="regression",
        use_cross_validation=False,
        conf_training=fast_config,
    )
    metrics = automl.fit_eval(
        df_train, target_col="target", df_eval=df_eval, y_eval=y_eval
    )
    assert isinstance(metrics, dict)
    assert "r2_score" in metrics


def test_regression_cv(regression_data, fast_config):
    automl = BlueCastAuto(
        class_problem="regression",
        use_cross_validation=True,
        conf_training=fast_config,
    )
    result = automl.fit_eval(regression_data, target_col="target")
    assert isinstance(result, tuple)


def test_inner_model_property(binary_data, fast_config):
    automl = BlueCastAuto(
        class_problem="binary",
        use_cross_validation=False,
        conf_training=fast_config,
    )
    assert automl.inner_model is not None
    assert automl.class_problem == "binary"


def test_predict_proba_classification(binary_data, fast_config):
    automl = BlueCastAuto(
        class_problem="binary",
        use_cross_validation=False,
        conf_training=fast_config,
    )
    automl.fit(binary_data, target_col="target")
    probs = automl.predict_proba(binary_data.drop("target", axis=1))
    assert len(probs) == len(binary_data)


def test_predict_proba_regression_raises(regression_data, fast_config):
    automl = BlueCastAuto(
        class_problem="regression",
        use_cross_validation=False,
        conf_training=fast_config,
    )
    with pytest.raises(AttributeError, match="not available for regression"):
        automl.predict_proba(regression_data.drop("target", axis=1))


def test_predict_sets_regression_raises(fast_config):
    automl = BlueCastAuto(
        class_problem="regression",
        use_cross_validation=False,
        conf_training=fast_config,
    )
    with pytest.raises(AttributeError, match="predict_interval"):
        automl.predict_sets(pd.DataFrame())


def test_predict_interval_classification_raises(fast_config):
    automl = BlueCastAuto(
        class_problem="binary",
        use_cross_validation=False,
        conf_training=fast_config,
    )
    with pytest.raises(AttributeError, match="predict_sets"):
        automl.predict_interval(pd.DataFrame())


def test_transform_new_data_cv_raises(binary_data, fast_config):
    automl = BlueCastAuto(
        class_problem="binary",
        use_cross_validation=True,
        conf_training=fast_config,
    )
    with pytest.raises(AttributeError, match="not available for CV"):
        automl.transform_new_data(pd.DataFrame())


def test_show_oof_scores_non_cv_raises(fast_config):
    automl = BlueCastAuto(
        class_problem="binary",
        use_cross_validation=False,
        conf_training=fast_config,
    )
    with pytest.raises(AttributeError, match="only available for CV"):
        automl.show_oof_scores()


def test_fit_eval_non_cv_no_eval_raises(binary_data, fast_config):
    automl = BlueCastAuto(
        class_problem="binary",
        use_cross_validation=False,
        conf_training=fast_config,
    )
    with pytest.raises(ValueError, match="df_eval and y_eval are required"):
        automl.fit_eval(binary_data, target_col="target")
