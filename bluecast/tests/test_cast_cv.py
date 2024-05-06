from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from bluecast.blueprints.cast import BlueCast
from bluecast.blueprints.cast_cv import BlueCastCV
from bluecast.config.training_config import TrainingConfig, XgboostTuneParamsConfig
from bluecast.ml_modelling.base_classes import (
    BaseClassMlModel,
    PredictedClasses,
    PredictedProbas,
)
from bluecast.tests.make_data.create_data import create_synthetic_dataframe


@pytest.fixture
def synthetic_train_test_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = create_synthetic_dataframe(2000, random_state=20)
    df_val = create_synthetic_dataframe(2000, random_state=200)
    return df_train, df_val


@pytest.fixture
def synthetic_calibration_data() -> pd.DataFrame:
    df_calibration = create_synthetic_dataframe(2000, random_state=2000)
    return df_calibration


def test_blueprint_cv_xgboost(synthetic_train_test_data, synthetic_calibration_data):
    """Test that tests the BlueCast cv class"""
    df_train = synthetic_train_test_data[0]
    df_val = synthetic_train_test_data[1]
    df_calibration = synthetic_calibration_data
    xgboost_param_config = XgboostTuneParamsConfig()
    xgboost_param_config.steps_max = 100
    xgboost_param_config.max_depth_max = 3
    train_config = TrainingConfig()
    train_config.hyperparameter_tuning_rounds = 10
    train_config.sample_data_during_tuning = True

    nb_models = 3

    skf = StratifiedKFold(
        n_splits=nb_models,
        shuffle=True,
        random_state=5,
    )

    automl_cv = BlueCastCV(
        conf_xgboost=xgboost_param_config, conf_training=train_config, stratifier=skf
    )
    oof_mean, oof_std = automl_cv.fit_eval(
        df_train,
        target_col="target",
    )
    assert isinstance(oof_mean, float)
    assert isinstance(oof_std, float)
    print(automl_cv.experiment_tracker.experiment_id)
    assert (
        len(automl_cv.experiment_tracker.experiment_id)
        <= train_config.hyperparameter_tuning_rounds * nb_models
        + nb_models * 7  # 7 metrics stored in fit_eval
    )
    assert automl_cv.experiment_tracker.experiment_id[-1] < 50
    print("Autotuning successful.")
    y_probs, y_classes = automl_cv.predict(df_val.drop("target", axis=1))
    print("Predicting successful.")
    assert len(y_probs) == len(df_val.index)
    assert len(y_classes) == len(df_val.index)

    y_probs = automl_cv.predict_proba(df_val.drop("target", axis=1))
    print("Predicting  class scores successful.")
    assert len(y_probs) == len(df_val.index)

    y_probs = automl_cv.predict_proba(
        df_val.drop("target", axis=1), return_sub_models_preds=True
    )
    print("Predicting class scores for all classes successful.")
    assert len(y_probs) == len(df_val.index)
    assert y_probs.shape[1] > 1  # even in binary cases this should be 2

    y_probs, y_classes = automl_cv.predict(
        df_val.drop("target", axis=1), return_sub_models_preds=True
    )
    assert isinstance(y_probs, pd.DataFrame)
    assert isinstance(y_classes, pd.DataFrame)

    # Assert that the bluecast_models attribute is updated
    assert len(automl_cv.bluecast_models) == nb_models

    # test conformal prediction
    automl_cv.calibrate(df_calibration.drop("target", axis=1), df_calibration["target"])
    pred_intervals = automl_cv.predict_p_values(df_val.drop("target", axis=1))
    pred_sets = automl_cv.predict_sets(df_val.drop("target", axis=1))
    assert isinstance(pred_intervals, np.ndarray)
    assert isinstance(pred_sets, pd.DataFrame)

    train_config = TrainingConfig()
    train_config.hyperparameter_tuning_rounds = 3
    train_config.enable_feature_selection = True
    train_config.hypertuning_cv_folds = 2
    train_config.enable_grid_search_fine_tuning = True
    train_config.gridsearch_nb_parameters_per_grid = 2
    train_config.precise_cv_tuning = False

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
        <= 5 * (train_config.hyperparameter_tuning_rounds * 2 + 5 * 7)
        # 7 metrics stored in fit_eval, 5 = default split, cv tuning rounds
    )  # due to Optuna pruning

    # Assert that the bluecast_models attribute is updated
    assert len(automl_cv.bluecast_models) == 5

    # Check if each model in bluecast_models is an instance of BlueCast
    for model in automl_cv.bluecast_models:
        assert isinstance(model, BlueCast)


class CustomLRModel(BaseClassMlModel):
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

    def predict(self, df: pd.DataFrame) -> Tuple[PredictedProbas, PredictedClasses]:
        predicted_probas = self.model.predict_proba(df)[:, 1]
        predicted_classes = self.model.predict(df)
        return predicted_probas, predicted_classes


def test_bluecast_cv_fit_eval_with_custom_model():
    custom_model = CustomLRModel()

    # Create an instance of the BlueCast class with the custom model
    bluecast = BlueCast(
        class_problem="binary",
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
    predicted_probas, predicted_classes = bluecast.predict(x_test)

    # Assert the expected results
    assert isinstance(predicted_probas, np.ndarray)
    assert isinstance(predicted_classes, np.ndarray)
    print(bluecast.experiment_tracker.experiment_id)
    assert len(bluecast.experiment_tracker.experiment_id) == 8  # due to custom model
