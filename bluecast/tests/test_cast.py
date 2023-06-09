from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from bluecast.blueprints.cast import BlueCast
from bluecast.config.training_config import TrainingConfig, XgboostTuneParamsConfig
from bluecast.ml_modelling.base_classes import (
    BaseClassMlModel,
    PredictedClasses,
    PredictedProbas,
)
from bluecast.preprocessing.custom import CustomPreprocessing
from bluecast.tests.make_data.create_data import create_synthetic_dataframe


@pytest.fixture
def synthetic_train_test_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = create_synthetic_dataframe(2000, random_state=20)
    df_val = create_synthetic_dataframe(2000, random_state=200)
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

    # add custom last mile computation
    class MyCustomPreprocessing(CustomPreprocessing):
        def custom_function(self, df: pd.DataFrame) -> pd.DataFrame:
            df = df / 2
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

    custom_preprocessor = MyCustomPreprocessing()

    automl = BlueCast(
        class_problem="binary",
        target_column="target",
        conf_training=train_config,
        conf_xgboost=xgboost_param_config,
        custom_preprocessor=custom_preprocessor,
    )
    automl.fit(df_train, target_col="target")
    print("Autotuning successful.")
    y_probs, y_classes = automl.predict(df_val.drop("target", axis=1))
    print("Predicting successful.")
    assert len(y_probs) == len(df_val.index)
    assert len(y_classes) == len(df_val.index)


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
        self.model = LogisticRegression()
        self.model.fit(x_train, y_train)

    def predict(self, df: pd.DataFrame) -> Tuple[PredictedProbas, PredictedClasses]:
        predicted_probas = self.model.predict_proba(df)
        predicted_classes = self.model.predict(df)
        return predicted_probas, predicted_classes


def test_bluecast_with_custom_model():
    # Create an instance of the custom model
    custom_model = CustomModel()

    # Create an instance of the BlueCast class with the custom model
    bluecast = BlueCast(
        class_problem="binary",
        target_column="target",
        ml_model=custom_model,
    )

    # Create some sample data for testing
    x_train = pd.DataFrame(
        {"feature1": [i for i in range(10)], "feature2": [i for i in range(10)]}
    )
    y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    x_test = pd.DataFrame(
        {"feature1": [i for i in range(10)], "feature2": [i for i in range(10)]}
    )

    x_train["target"] = y_train

    # Fit the BlueCast model using the custom model
    bluecast.fit(x_train, "target")

    # Predict on the test data using the custom model
    predicted_probas, predicted_classes = bluecast.predict(x_test)

    # Assert the expected results
    assert isinstance(predicted_probas, np.ndarray)
    assert isinstance(predicted_classes, np.ndarray)
