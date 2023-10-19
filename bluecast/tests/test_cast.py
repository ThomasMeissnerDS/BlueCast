from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

from bluecast.blueprints.cast import BlueCast
from bluecast.config.training_config import TrainingConfig, XgboostTuneParamsConfig
from bluecast.ml_modelling.base_classes import (
    BaseClassMlModel,
    PredictedClasses,
    PredictedProbas,
)
from bluecast.preprocessing.custom import CustomPreprocessing
from bluecast.tests.make_data.create_data import (
    create_synthetic_dataframe,
    create_synthetic_multiclass_dataframe,
)


@pytest.fixture
def synthetic_train_test_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = create_synthetic_dataframe(2000, random_state=20)
    df_val = create_synthetic_dataframe(2000, random_state=200)
    return df_train, df_val


@pytest.fixture
def synthetic_multiclass_train_test_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = create_synthetic_multiclass_dataframe(2000, random_state=20)
    df_val = create_synthetic_multiclass_dataframe(2000, random_state=200)
    return df_train, df_val


def test_blueprint_xgboost(
    synthetic_train_test_data, synthetic_multiclass_train_test_data
):
    """Test that tests the BlueCast class"""
    df_train = synthetic_train_test_data[0]
    df_val = synthetic_train_test_data[1]
    xgboost_param_config = XgboostTuneParamsConfig()
    xgboost_param_config.steps_max = 100

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

    automl = BlueCast(
        class_problem="binary",
        target_column="target",
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
    y_probs, y_classes = automl.predict(df_val.drop("target", axis=1))
    print("Predicting successful.")
    assert len(y_probs) == len(df_val.index)
    assert len(y_classes) == len(df_val.index)

    # Test multiclass pipeline
    df_train = synthetic_multiclass_train_test_data[0]
    df_val = synthetic_multiclass_train_test_data[1]

    automl = BlueCast(
        class_problem="multiclass",
        target_column="target",
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
    y_probs, y_classes = automl.predict(df_val.drop("target", axis=1))
    print("Predicting successful.")
    assert len(y_probs) == len(df_val.index)
    assert len(y_classes) == len(df_val.index)
    assert (
        len(automl.experiment_tracker.experiment_id)
        <= automl.conf_training.hyperparameter_tuning_rounds
    )


class CustomModel(BaseClassMlModel):
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
        predicted_probas = self.model.predict_proba(df)
        predicted_classes = self.model.predict(df)
        return predicted_probas, predicted_classes


def test_bluecast_with_custom_model():
    # Create an instance of the custom model
    custom_model = CustomModel()
    train_config = TrainingConfig()
    train_config.hyperparameter_tuning_rounds = 10
    train_config.enable_feature_selection = True
    train_config.hypertuning_cv_folds = 2
    train_config.enable_grid_search_fine_tuning = True

    # add custom feature selection
    class RFECVSelector(CustomPreprocessing):
        def __init__(self, random_state: int = 0):
            super().__init__()
            self.selected_features = None
            self.random_state = random_state
            self.selection_strategy: RFECV = RFECV(
                estimator=xgb.XGBClassifier(),
                step=1,
                cv=StratifiedKFold(2, random_state=random_state, shuffle=True),
                min_features_to_select=1,
                scoring=make_scorer(matthews_corrcoef),
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
                estimator=xgb.XGBClassifier(),
                step=1,
                cv=StratifiedKFold(2, random_state=random_state, shuffle=True),
                min_features_to_select=1,
                scoring=make_scorer(matthews_corrcoef),
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

    custom_feature_selector = RFECVSelector()
    custum_preproc = MyCustomPreprocessor()

    # Create an instance of the BlueCast class with the custom model
    bluecast = BlueCast(
        class_problem="binary",
        target_column="target",
        ml_model=custom_model,
        conf_training=train_config,
        custom_feature_selector=custom_feature_selector,
        custom_preprocessor=custum_preproc,
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
    predicted_probas, predicted_classes = bluecast.predict(x_test)

    # Assert the expected results
    assert isinstance(predicted_probas, np.ndarray)
    assert isinstance(predicted_classes, np.ndarray)
    print(bluecast.experiment_tracker.experiment_id)
    assert len(bluecast.experiment_tracker.experiment_id) == 0  # due to custom model
