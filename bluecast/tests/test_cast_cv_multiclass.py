from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from bluecast.blueprints.cast_cv import BlueCastCV
from bluecast.ml_modelling.base_classes import (
    BaseClassMlModel,
    PredictedClasses,
    PredictedProbas,
)


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
        predicted_probas = self.model.predict_proba(df)
        predicted_classes = np.asarray([np.argmax(line) for line in predicted_probas])
        return predicted_probas, predicted_classes


def test_bluecast_cv_fit_eval_multiclass_with_custom_model():
    custom_model = CustomLRModel()

    # Create an instance of the BlueCast class with the custom model
    bluecast = BlueCastCV(
        class_problem="multiclass",
        ml_model=custom_model,
    )

    # Create some sample data for testing
    x_train = pd.DataFrame(
        {
            "feature1": [i for i in range(20)],
            "feature2": [i for i in range(20)],
            "feature3": [i for i in range(20)],
            "feature4": [i for i in range(20)],
            "feature5": [i for i in range(20)],
            "feature6": [i + 4 for i in range(20)],
        }
    )
    y_train = pd.Series([0, 1, 0, 1, 0, 2, 0, 2, 0, 2, 0, 1, 0, 1, 0, 2, 0, 2, 0, 2])
    x_test = pd.DataFrame(
        {
            "feature1": [i for i in range(20)],
            "feature2": [i for i in range(20)],
            "feature3": [i for i in range(20)],
            "feature4": [i for i in range(20)],
            "feature5": [i for i in range(20)],
            "feature6": [i + 4 for i in range(20)],
        }
    )
    y_test = pd.Series([0, 1, 0, 1, 0, 2, 0, 2, 0, 2, 0, 1, 0, 1, 0, 2, 0, 2, 0, 2])

    x_train["target"] = y_train

    train_all = pd.concat([x_train, x_test], axis=0)
    train_all["target"] = pd.concat([y_train, y_test], axis=0)

    # Fit the BlueCast model using the custom model
    bluecast.fit_eval(train_all, "target")

    # Predict on the test data using the custom model
    predicted_probas, predicted_classes = bluecast.predict(x_test)

    # Assert the expected results
    assert isinstance(predicted_probas, type(predicted_classes))
    print(bluecast.experiment_tracker.experiment_id)
    assert len(bluecast.experiment_tracker.experiment_id) == 36  # due to custom model


def test_bluecast_cv_fit_eval_multiclass_without_custom_model():
    # Create an instance of the BlueCast class with the custom model
    bluecast = BlueCastCV(
        class_problem="multiclass",
    )

    # Create some sample data for testing
    x_train = pd.DataFrame(
        {
            "feature1": [i for i in range(20)],
            "feature2": [i for i in range(20)],
            "feature3": [i for i in range(20)],
            "feature4": [i for i in range(20)],
            "feature5": [i for i in range(20)],
            "feature6": [i + 4 for i in range(20)],
        }
    )
    y_train = pd.Series([0, 1, 0, 1, 0, 2, 0, 2, 0, 2, 0, 1, 0, 1, 0, 2, 0, 2, 0, 2])
    y_train = y_train.replace({0: "zero", 1: "one", 2: "two"})

    x_test = pd.DataFrame(
        {
            "feature1": [i for i in range(20)],
            "feature2": [i for i in range(20)],
            "feature3": [i for i in range(20)],
            "feature4": [i for i in range(20)],
            "feature5": [i for i in range(20)],
            "feature6": [i + 4 for i in range(20)],
        }
    )
    y_test = pd.Series([0, 1, 0, 1, 0, 2, 0, 2, 0, 2, 0, 1, 0, 1, 0, 2, 0, 2, 0, 2])
    y_test = y_test.replace({0: "zero", 1: "one", 2: "two"})

    x_train["target"] = y_train

    train_all = pd.concat([x_train, x_test], axis=0)
    train_all["target"] = pd.concat([y_train, y_test], axis=0)

    # Fit the BlueCast model using the custom model
    bluecast.fit_eval(train_all, "target")

    # Predict on the test data using the custom model
    predicted_probas, predicted_classes = bluecast.predict(x_test)

    # Assert the expected results
    assert isinstance(predicted_probas, pd.Series)
