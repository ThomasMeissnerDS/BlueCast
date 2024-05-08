from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pytest
import xgboost
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import KFold

from bluecast.blueprints.cast_cv_regression import BlueCastCVRegression
from bluecast.blueprints.cast_regression import BlueCastRegression
from bluecast.config.training_config import TrainingConfig
from bluecast.config.training_config import (
    XgboostTuneParamsRegressionConfig as XgboostTuneParamsConfig,
)
from bluecast.ml_modelling.base_classes import BaseClassMlRegressionModel
from bluecast.preprocessing.custom import CustomPreprocessing
from bluecast.tests.make_data.create_data import create_synthetic_dataframe_regression


@pytest.fixture
def synthetic_train_test_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = create_synthetic_dataframe_regression(2000, random_state=20)
    df_val = create_synthetic_dataframe_regression(2000, random_state=200)
    return df_train, df_val


@pytest.fixture
def synthetic_calibration_data() -> pd.DataFrame:
    df_calibration = create_synthetic_dataframe_regression(2000, random_state=200)
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

    skf = KFold(
        n_splits=nb_models,
        shuffle=True,
        random_state=5,
    )

    automl_cv = BlueCastCVRegression(
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
    preds = automl_cv.predict(df_val.drop("target", axis=1), mean_type="arithmetic")
    print("Predicting successful.")

    preds_geom = automl_cv.predict(df_val.drop("target", axis=1), mean_type="geometric")
    assert isinstance(preds_geom, pd.Series)

    preds_harm = automl_cv.predict(df_val.drop("target", axis=1), mean_type="harmonic")
    assert isinstance(preds_harm, pd.Series)

    preds_median = automl_cv.predict(df_val.drop("target", axis=1), mean_type="median")
    assert isinstance(preds_median, pd.Series)

    preds_wrong_val = automl_cv.predict(
        df_val.drop("target", axis=1), mean_type="wrong_value"
    )
    assert isinstance(preds_wrong_val, pd.Series)

    assert len(preds) == len(df_val.index)
    preds = automl_cv.predict(
        df_val.drop("target", axis=1), return_sub_models_preds=True
    )
    assert isinstance(preds, pd.DataFrame)

    # Assert that the bluecast_models attribute is updated
    assert len(automl_cv.bluecast_models) == nb_models

    # test conformal prediction
    automl_cv.calibrate(df_calibration.drop("target", axis=1), df_calibration["target"])
    pred_intervals = automl_cv.predict_interval(
        df_val.drop("target", axis=1), alphas=[0.01, 0.05]
    )
    assert isinstance(pred_intervals, pd.DataFrame)

    train_config = TrainingConfig()
    train_config.hyperparameter_tuning_rounds = 3
    train_config.enable_feature_selection = True
    train_config.hypertuning_cv_folds = 2
    train_config.enable_grid_search_fine_tuning = True
    train_config.gridsearch_nb_parameters_per_grid = 2
    train_config.precise_cv_tuning = True

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

    # test fine tune
    train_config.precise_cv_tuning = True
    automl_cv = BlueCastCVRegression(
        conf_xgboost=xgboost_param_config, conf_training=train_config, stratifier=None
    )
    automl_cv.fit_eval(
        df_train,
        target_col="target",
    )
    assert True
    assert automl_cv.class_problem == "regression"
    preds = automl_cv.predict(
        df_val.drop("target", axis=1), return_sub_models_preds=True
    )
    assert isinstance(preds, pd.DataFrame)


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
        self.model = RandomForestRegressor()
        self.model.fit(x_train, y_train)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(df)
        return preds


class MyCustomLastMilePreprocessing(CustomPreprocessing):
    def custom_function(self, df: pd.DataFrame) -> pd.DataFrame:
        df["custom_col"] = 5
        return df

    def fit_transform(
        self, df: pd.DataFrame, target: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        df = self.custom_function(df)
        df = df.head(1000)
        target = target.head(1000)
        return df, target

    def transform(
        self,
        df: pd.DataFrame,
        target: Optional[pd.Series] = None,
        predicton_mode: bool = False,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        df = self.custom_function(df)
        if not predicton_mode and isinstance(target, pd.Series):
            df = df.head(100)
            target = target.head(100)
        return df, target


class CustomModel(BaseClassMlRegressionModel):
    def __init__(self):
        self.model = None

    def fit(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        self.model = RandomForestRegressor()
        self.model.fit(x_train, y_train)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(df)
        return preds


def test_bluecast_cv_fit_eval_with_custom_model():
    custom_model = CustomLRModel()

    # Create an instance of the BlueCast class with the custom model
    bluecast = BlueCastRegression(
        class_problem="regression",
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
    y_train = pd.Series([0.0, 1.0, 0.0, 10.0, 0.0, 1.0, 0.0, 1.0, 0.0, 45.0])
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
    y_test = pd.Series([0.0, 1.0, 0.0, 10.0, 0.0, 1.0, 0.0, 1.0, 0.0, 45.0])

    x_train["target"] = y_train

    # Fit the BlueCast model using the custom model
    bluecast.fit_eval(x_train, x_test, y_test, "target")

    # Predict on the test data using the custom model
    preds = bluecast.predict(x_test)

    # Assert the expected results
    assert isinstance(preds, np.ndarray)
    assert isinstance(preds, np.ndarray)
    print(bluecast.experiment_tracker.experiment_id)
    assert len(bluecast.experiment_tracker.experiment_id) == 6  # due to custom model


def test_bluecast_cv_with_custom_objects():
    custom_model = CustomModel()
    train_config = TrainingConfig()
    train_config.hyperparameter_tuning_rounds = 10
    train_config.enable_feature_selection = True
    train_config.hypertuning_cv_folds = 2
    train_config.enable_grid_search_fine_tuning = False
    train_config.gridsearch_nb_parameters_per_grid = 2
    train_config.use_full_data_for_final_model = True

    xgboost_param_config = XgboostTuneParamsConfig()
    xgboost_param_config.steps_max = 100
    xgboost_param_config.max_depth_max = 3

    # add custom feature selection
    class RFECVSelector(CustomPreprocessing):
        def __init__(self, random_state: int = 0):
            super().__init__()
            self.selected_features = None
            self.random_state = random_state
            self.selection_strategy: RFECV = RFECV(
                estimator=xgboost.XGBRegressor(),
                step=1,
                cv=KFold(2, random_state=random_state, shuffle=True),
                min_features_to_select=1,
                scoring=make_scorer(mean_absolute_error),
                n_jobs=2,
            )

        def fit_transform(
            self, df: pd.DataFrame, target: pd.Series
        ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
            self.selection_strategy.fit(df, target)
            self.selected_features = self.selection_strategy.support_
            df = df.loc[:, self.selected_features]
            return df, target

        def transform(
            self,
            df: pd.DataFrame,
            target: Optional[pd.Series] = None,
            predicton_mode: bool = False,
        ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
            df = df.loc[:, self.selected_features]
            return df, target

    class MyCustomPreprocessor(CustomPreprocessing):
        def __init__(self, random_state: int = 0):
            super().__init__()
            self.selected_features = None
            self.random_state = random_state
            self.selection_strategy: RFECV = RFECV(
                estimator=xgboost.XGBRegressor(),
                step=1,
                cv=KFold(2, random_state=random_state, shuffle=True),
                min_features_to_select=1,
                scoring=make_scorer(mean_absolute_error),
                n_jobs=2,
            )

        def fit_transform(
            self, df: pd.DataFrame, target: pd.Series
        ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
            return df, target

        def transform(
            self,
            df: pd.DataFrame,
            target: Optional[pd.Series] = None,
            predicton_mode: bool = False,
        ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
            return df, target

    class MyCustomInFoldPreprocessor(CustomPreprocessing):
        def __init__(self):
            super().__init__()

        def fit_transform(
            self, df: pd.DataFrame, target: pd.Series
        ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
            df["leakage"] = target
            return df, target

        def transform(
            self,
            df: pd.DataFrame,
            target: Optional[pd.Series] = None,
            predicton_mode: bool = False,
        ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
            df["leakage"] = 0
            return df, target

    custom_feature_selector = RFECVSelector()
    custum_preproc = MyCustomPreprocessor()
    custom_infold_preproc = MyCustomInFoldPreprocessor()

    # Create an instance of the BlueCast class with the custom model
    bluecast = BlueCastRegression(
        class_problem="regression",
        ml_model=custom_model,
        conf_xgboost=xgboost_param_config,
        conf_training=train_config,
        custom_feature_selector=custom_feature_selector,
        custom_preprocessor=custum_preproc,
        custom_in_fold_preprocessor=custom_infold_preproc,
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
    y_train = pd.Series([0.0, 1.0, 0.0, 10.0, 0.0, 1.0, 0.0, 1.0, 0.0, 45.0])
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

    x_train["target"] = y_train

    # Fit the BlueCast model using the custom model
    bluecast.fit(x_train, "target")

    # Predict on the test data using the custom model
    preds = bluecast.predict(x_test.copy())

    # Assert the expected results
    assert isinstance(preds, np.ndarray)
    print(bluecast.experiment_tracker.experiment_id)
    assert (
        len(bluecast.experiment_tracker.experiment_id) == 0
    )  # due to custom model and fit method
