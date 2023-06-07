from typing import Tuple

import pandas as pd
import pytest

from bluecast.blueprints.cast import BlueCast
from bluecast.config.training_config import TrainingConfig, XgboostTuneParamsConfig
from bluecast.tests.make_data.create_data import create_synthetic_dataframe


@pytest.fixture
def synthetic_train_test_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = create_synthetic_dataframe(1000, random_state=20)
    df_val = create_synthetic_dataframe(100, random_state=200)
    return df_train, df_val


def test_blueprint_xgboost(synthetic_train_test_data):
    """Test that tests the BlueCast class"""
    df_train = synthetic_train_test_data[0]
    df_val = synthetic_train_test_data[1]
    xgboost_param_config = XgboostTuneParamsConfig()
    xgboost_param_config.steps_max = 100
    xgboost_param_config.num_leaves_max = 16
    train_config = TrainingConfig()
    train_config.hyperparameter_tuning_rounds = 10

    automl = BlueCast(
        class_problem="binary",
        target_column="target",
        conf_training=train_config,
        conf_xgboost=xgboost_param_config,
    )
    automl.fit(df_train, target_col="target")
    print("Autotuning successful.")
    y_probs, y_classes = automl.predict(df_val.drop("target", axis=1))
    print("Predicting successful.")
    assert len(y_probs) == len(df_val.index)
    assert len(y_classes) == len(df_val.index)
