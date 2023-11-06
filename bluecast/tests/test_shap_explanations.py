from unittest import mock

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from bluecast.blueprints.cast import BlueCast
from bluecast.config.training_config import TrainingConfig, XgboostTuneParamsConfig
from bluecast.evaluation.eval_metrics import matthews_corrcoef
from bluecast.evaluation.shap_values import shap_explanations
from bluecast.tests.make_data.create_data import create_synthetic_dataframe


@pytest.fixture
def mock_model():
    return mock.Mock()


@pytest.fixture
def test_data():
    return pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})


def test_shap_explanations():
    """Test that tests the BlueCast class"""
    df_train = create_synthetic_dataframe(200, random_state=20)
    df_val = create_synthetic_dataframe(100, random_state=21)
    xgboost_param_config = XgboostTuneParamsConfig()
    xgboost_param_config.steps_max = 100
    train_config = TrainingConfig()
    train_config.hyperparameter_tuning_rounds = 10
    train_config.enable_feature_selection = False
    train_config.hypertuning_cv_folds = 2

    automl = BlueCast(
        class_problem="binary",
        target_column="target",
        conf_training=train_config,
        conf_xgboost=xgboost_param_config,
    )
    eval_dict = automl.fit_eval(
        df_train, df_val.drop("target", axis=1), df_val["target"], target_col="target"
    )
    print(eval_dict)
    assert isinstance(eval_dict, dict)

    targets = df_train.pop("target")

    # test that non-tree based models are supported as well
    model = LogisticRegression()
    model.fit(
        df_train.loc[
            :, ["numerical_feature_1", "numerical_feature_2", "numerical_feature_3"]
        ],
        targets,
    )
    model_shap_values = shap_explanations(
        model,
        df_val.loc[
            :, ["numerical_feature_1", "numerical_feature_2", "numerical_feature_3"]
        ],
    )
    assert model_shap_values.any()


def test_eval_classifier_except():
    y_true = np.array([0, 0, 0, 0])
    y_classes = np.array([1, 1, 1, 1])
    result = matthews_corrcoef(y_true, y_classes)
    assert result == 0
