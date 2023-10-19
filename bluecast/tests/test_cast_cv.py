from typing import Tuple

import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from bluecast.blueprints.cast import BlueCast
from bluecast.blueprints.cast_cv import BlueCastCV
from bluecast.config.training_config import TrainingConfig, XgboostTuneParamsConfig
from bluecast.tests.make_data.create_data import create_synthetic_dataframe


@pytest.fixture
def synthetic_train_test_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = create_synthetic_dataframe(2000, random_state=20)
    df_val = create_synthetic_dataframe(2000, random_state=200)
    return df_train, df_val


def test_blueprint_cv_xgboost(synthetic_train_test_data):
    """Test that tests the BlueCast cv class"""
    df_train = synthetic_train_test_data[0]
    df_val = synthetic_train_test_data[1]
    xgboost_param_config = XgboostTuneParamsConfig()
    xgboost_param_config.steps_max = 100
    train_config = TrainingConfig()
    train_config.hyperparameter_tuning_rounds = 10

    nb_models = 3

    skf = StratifiedKFold(
        n_splits=nb_models,
        shuffle=True,
        random_state=5,
    )

    automl_cv = BlueCastCV(
        conf_xgboost=xgboost_param_config, conf_training=train_config, stratifier=skf
    )
    automl_cv.fit_eval(
        df_train,
        target_col="target",
    )
    print(automl_cv.experiment_tracker.experiment_id)
    assert (
        len(automl_cv.experiment_tracker.experiment_id)
        <= train_config.hyperparameter_tuning_rounds * nb_models
        + nb_models * 7  # 7 metrics stored in fit_eval
    )
    assert automl_cv.experiment_tracker.experiment_id[-1] == 22
    print("Autotuning successful.")
    y_probs, y_classes = automl_cv.predict(df_val.drop("target", axis=1))
    print("Predicting successful.")
    assert len(y_probs) == len(df_val.index)
    assert len(y_classes) == len(df_val.index)
    y_probs, y_classes = automl_cv.predict(
        df_val.drop("target", axis=1), return_sub_models_preds=True
    )
    assert isinstance(y_probs, pd.DataFrame)
    assert isinstance(y_classes, pd.DataFrame)

    # Assert that the bluecast_models attribute is updated
    assert len(automl_cv.bluecast_models) == nb_models

    automl_cv = BlueCastCV(
        conf_xgboost=xgboost_param_config, conf_training=train_config, stratifier=None
    )
    automl_cv.fit_eval(
        df_train,
        target_col="target",
    )
    assert automl_cv.stratifier
    assert (
        len(automl_cv.experiment_tracker.experiment_id)
        <= train_config.hyperparameter_tuning_rounds * 5
        + 5 * 7  # 7 metrics stored in fit_eval, 5 = default split
    )  # due to Optuna pruning

    # Assert that the bluecast_models attribute is updated
    assert len(automl_cv.bluecast_models) == 5

    # Check if each model in bluecast_models is an instance of BlueCast
    for model in automl_cv.bluecast_models:
        assert isinstance(model, BlueCast)
