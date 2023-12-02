from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from bluecast.blueprints.cast_cv_regression import BlueCastCVRegression
from bluecast.blueprints.cast_regression import BlueCastRegression
from bluecast.config.training_config import TrainingConfig
from bluecast.config.training_config import (
    XgboostTuneParamsRegressionConfig as XgboostTuneParamsConfig,
)
from bluecast.ml_modelling.base_classes import BaseClassMlRegressionModel
from bluecast.tests.make_data.create_data import create_synthetic_dataframe_regression


@pytest.fixture
def synthetic_train_test_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = create_synthetic_dataframe_regression(2000, random_state=20)
    df_val = create_synthetic_dataframe_regression(2000, random_state=200)
    return df_train, df_val


def test_blueprint_cv_xgboost(synthetic_train_test_data):
    """Test that tests the BlueCast cv class"""
    df_train = synthetic_train_test_data[0]
    df_val = synthetic_train_test_data[1]
    xgboost_param_config = XgboostTuneParamsConfig()
    xgboost_param_config.steps_max = 100
    xgboost_param_config.max_depth_max = 3
    train_config = TrainingConfig()
    train_config.hyperparameter_tuning_rounds = 10

    nb_models = 3

    skf = KFold(
        n_splits=nb_models,
        shuffle=True,
        random_state=5,
    )

    automl_cv = BlueCastCVRegression(
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
    assert automl_cv.experiment_tracker.experiment_id[-1] < 50
    print("Autotuning successful.")
    preds = automl_cv.predict(df_val.drop("target", axis=1))
    print("Predicting successful.")
    assert len(preds) == len(df_val.index)
    preds = automl_cv.predict(
        df_val.drop("target", axis=1), return_sub_models_preds=True
    )
    assert isinstance(preds, pd.DataFrame)

    # Assert that the bluecast_models attribute is updated
    assert len(automl_cv.bluecast_models) == nb_models

    train_config = TrainingConfig()
    train_config.hyperparameter_tuning_rounds = 3
    train_config.enable_feature_selection = True
    train_config.hypertuning_cv_folds = 2
    train_config.enable_grid_search_fine_tuning = True
    train_config.gridsearch_nb_parameters_per_grid = 2
    train_config.precise_cv_tuning = False

    automl_cv = BlueCastCVRegression(
        conf_xgboost=xgboost_param_config, conf_training=train_config, stratifier=None
    )
    automl_cv.fit_eval(
        df_train,
        target_col="target",
    )
    assert automl_cv.stratifier
    assert (
        len(automl_cv.experiment_tracker.experiment_id)
        <= 5 * (train_config.hyperparameter_tuning_rounds * 2 + 5 * 7)
        # 7 metrics stored in fit_eval, 5 = default split, cv tuning rounds
    )  # due to Optuna pruning

    # Assert that the bluecast_models attribute is updated
    assert len(automl_cv.bluecast_models) == 5

    # Check if each model in bluecast_models is an instance of BlueCast
    for model in automl_cv.bluecast_models:
        assert isinstance(model, BlueCastRegression)


class CustomLRModel(BaseClassMlRegressionModel):
    def __init__(self):
        self.model = None

    def fit(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        self.model = RandomForestClassifier()
        self.model.fit(x_train, y_train)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(df)
        return preds


def test_bluecast_cv_fit_eval_with_custom_model():
    custom_model = CustomLRModel()

    # Create an instance of the BlueCast class with the custom model
    bluecast = BlueCastRegression(
        class_problem="regression",
        target_column="target",
        ml_model=custom_model,
    )

    # Create some sample data for testing
    x_train = pd.DataFrame(
        {
            "feature1": [i for i in range(10)],
            "feature2": [i for i in range(10)],
            "feature3": [i for i in range(10)],
            "feature4": [i for i in range(10)],
            "feature5": [i for i in range(10)],
            "feature6": [i for i in range(10)],
        }
    )
    y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    x_test = pd.DataFrame(
        {
            "feature1": [i for i in range(10)],
            "feature2": [i for i in range(10)],
            "feature3": [i for i in range(10)],
            "feature4": [i for i in range(10)],
            "feature5": [i for i in range(10)],
            "feature6": [i for i in range(10)],
        }
    )
    y_test = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    x_train["target"] = y_train

    # Fit the BlueCast model using the custom model
    bluecast.fit_eval(x_train, x_test, y_test, "target")

    # Predict on the test data using the custom model
    preds = bluecast.predict(x_test)

    # Assert the expected results
    assert isinstance(preds, np.ndarray)
    assert isinstance(preds, np.ndarray)
    print(bluecast.experiment_tracker.experiment_id)
    assert len(bluecast.experiment_tracker.experiment_id) == 8  # due to custom model