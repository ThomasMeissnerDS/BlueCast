from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import KFold

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
    df_calibration = create_synthetic_dataframe_regression(2000, random_state=2000)
    return df_calibration


def test_blueprint_xgboost(synthetic_train_test_data, synthetic_calibration_data):
    """Test that tests the BlueCast class"""
    df_train = synthetic_train_test_data[0]
    df_val = synthetic_train_test_data[1]
    df_calibration = synthetic_calibration_data
    xgboost_param_config = XgboostTuneParamsConfig()
    xgboost_param_config.steps_max = 100
    xgboost_param_config.max_depth_max = 3

    # add custom last mile computation
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

    custom_last_mile_computation = MyCustomLastMilePreprocessing()

    automl = BlueCastRegression(
        class_problem="regression",
        conf_xgboost=xgboost_param_config,
        custom_last_mile_computation=custom_last_mile_computation,
    )
    automl.fit_eval(
        df_train,
        df_train.drop("target", axis=1),
        df_train["target"],
        target_col="target",
    )
    print("Autotuning successful.")
    y_preds = automl.predict(df_val.drop("target", axis=1), save_shap_values=True)
    print("Predicting successful.")
    assert len(y_preds) == len(df_val.index)

    # test conformal prediction
    automl.calibrate(df_calibration.drop("target", axis=1), df_calibration["target"])
    pred_intervals = automl.predict_interval(
        df_val.drop("target", axis=1), alphas=[0.01, 0.05]
    )
    assert isinstance(pred_intervals, pd.DataFrame)


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


def test_bluecast_with_custom_model():
    # Create an instance of the custom model
    custom_model = CustomModel()
    train_config = TrainingConfig()
    train_config.hyperparameter_tuning_rounds = 10
    train_config.enable_feature_selection = True
    train_config.hypertuning_cv_folds = 2
    train_config.enable_grid_search_fine_tuning = True
    train_config.gridsearch_nb_parameters_per_grid = 2
    train_config.precise_cv_tuning = True

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
                estimator=xgb.XGBRegressor(),
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
                estimator=xgb.XGBRegressor(),
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

    x_train["target"] = y_train

    # Fit the BlueCast model using the custom model
    bluecast.fit(x_train, "target")

    # Predict on the test data using the custom model
    preds = bluecast.predict(x_test)

    # Assert the expected results
    assert isinstance(preds, np.ndarray)
    print(bluecast.experiment_tracker.experiment_id)
    assert (
        len(bluecast.experiment_tracker.experiment_id) == 0
    )  # due to custom model and fit method

    train_config.precise_cv_tuning = False
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

    x_train["target"] = y_train

    # Fit the BlueCast model using the custom model
    bluecast.fit(x_train, "target")

    # Predict on the test data using the custom model
    preds = bluecast.predict(x_test)

    # Assert the expected results
    assert isinstance(preds, np.ndarray)
    print(bluecast.experiment_tracker.experiment_id)
    assert (
        len(bluecast.experiment_tracker.experiment_id) == 0
    )  # due to custom model and fit method


@pytest.fixture
def bluecast_instance():
    custom_config = TrainingConfig()
    # Create a fixture to instantiate the BlueCast class with default values for testing
    bluecast_instance = BlueCastRegression(
        class_problem="regression", conf_training=custom_config
    )
    bluecast_instance.target_column = "target"
    return bluecast_instance


def test_enable_feature_selection_warning(bluecast_instance):
    # Test if a warning is raised when feature selection is disabled
    df = pd.DataFrame({"feature1": [1, 2, 3], "target": [0, 1, 0]})
    with pytest.warns(UserWarning, match="Feature selection is disabled."):
        bluecast_instance.initial_checks(df)


def test_hypertuning_cv_folds_warning(bluecast_instance):
    # Test if a warning is raised when hypertuning_cv_folds is set to 1
    df = pd.DataFrame({"feature1": [1, 2, 3], "target": [0, 1, 0]})
    with pytest.warns(UserWarning, match="Cross validation is disabled."):
        bluecast_instance.conf_training.hypertuning_cv_folds = 1
        bluecast_instance.initial_checks(df)


def test_missing_feature_selector_warning(bluecast_instance):
    # Test if a warning is raised when feature selection is enabled but no feature selector is provided
    df = pd.DataFrame({"feature1": [1, 2, 3], "target": [0, 1, 0]})
    bluecast_instance.conf_training.enable_feature_selection = True
    with pytest.warns(
        UserWarning,
        match="Feature selection is enabled but no feature selector has been provided.",
    ):
        bluecast_instance.initial_checks(df)


def test_missing_xgboost_tune_params_config_warning(bluecast_instance):
    # Test if a warning is raised when XgboostTuneParamsConfig is not provided
    df = pd.DataFrame({"feature1": [1, 2, 3], "target": [0, 1, 0]})
    bluecast_instance.conf_xgboost = None
    with pytest.warns(
        UserWarning, match="No XgboostTuneParamsConfig has been provided."
    ):
        bluecast_instance.initial_checks(df)


def test_min_features_to_select_warning(bluecast_instance):
    # Test if a warning is raised when min_features_to_select is greater than or equal to the number of features
    df = pd.DataFrame({"feature1": [1, 2, 3], "target": [0, 1, 0]})
    bluecast_instance.conf_training.enable_feature_selection = True
    bluecast_instance.conf_training.min_features_to_select = 3
    message = """The minimum number of features to select is greater or equal to the number of features in
            the dataset while feature selection is enabled. Consider reducing the minimum number of features to
            select or disabling feature selection via TrainingConfig."""
    with pytest.warns(
        UserWarning,
        match=message,
    ):
        bluecast_instance.initial_checks(df)


def test_shap_values_and_ml_algorithm_warning(bluecast_instance):
    # Test if a warning is raised when calculate_shap_values is True and cat_encoding_via_ml_algorithm is True
    df = pd.DataFrame({"feature1": [1, 2, 3], "target": [0, 1, 0]})
    bluecast_instance.conf_training.calculate_shap_values = True
    bluecast_instance.conf_training.cat_encoding_via_ml_algorithm = True
    with pytest.warns(
        UserWarning,
        match="SHAP values cannot be calculated when categorical encoding via ML algorithm is enabled.",
    ):
        bluecast_instance.initial_checks(df)


def test_cat_encoding_via_ml_algorithm_and_ml_model_warning():
    # Test if a warning is raised when cat_encoding_via_ml_algorithm is True and ml_model is provided
    df = pd.DataFrame({"feature1": [1, 2, 3], "target": [0, 1, 0]})
    custom_config = TrainingConfig()
    custom_config.cat_encoding_via_ml_algorithm = True
    bluecast_instance = BlueCastRegression(
        class_problem="regression", conf_training=custom_config
    )
    from sklearn.linear_model import LinearRegression

    bluecast_instance.ml_model = LinearRegression()
    message = """Categorical encoding via ML algorithm is enabled. Make sure to handle categorical features
            within the provided ml model or consider disabling categorical encoding via ML algorithm in the
            TrainingConfig alternatively."""
    with pytest.warns(
        UserWarning,
        match=message,
    ):
        bluecast_instance.initial_checks(df)


def test_precise_cv_tuning_warnings(bluecast_instance):
    # Test if warnings are raised for precise_cv_tuning conditions
    df = pd.DataFrame({"feature1": [1, 2, 3], "target": [0, 1, 0]})
    bluecast_instance.conf_training.precise_cv_tuning = True
    with pytest.warns(UserWarning, match="Precise fine tuning has been enabled."):
        bluecast_instance.initial_checks(df)
    with pytest.warns(
        UserWarning,
        match="Precise fine tuning has been enabled, but no custom_in_fold_preprocessor has been provided.",
    ):
        bluecast_instance.initial_checks(df)
    with pytest.warns(
        UserWarning,
        match="Precise fine tuning has been enabled, but number of hypertuning_cv_folds is less than 2.",
    ):
        bluecast_instance.conf_training.hypertuning_cv_folds = 1
        bluecast_instance.initial_checks(df)
