import numpy as np
import pandas as pd

from bluecast.blueprints.cast_cv import BlueCastCV
from bluecast.config.training_config import TrainingConfig
from bluecast.conformal_prediction.nonconformity_measures import (
    brier_score,
    margin_nonconformity_measure,
)


def test_bluecast_cv_fit_eval_multiclass_without_custom_model():
    # Create an instance of the BlueCast class with the custom model
    train_config = TrainingConfig()
    train_config.calculate_shap_values = False

    bluecast = BlueCastCV(
        class_problem="multiclass",
        conf_training=train_config,
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

    x_calibration = pd.DataFrame(
        {
            "feature1": [i for i in range(20)],
            "feature2": [i for i in range(20)],
            "feature3": [i for i in range(20)],
            "feature4": [i for i in range(20)],
            "feature5": [i for i in range(20)],
            "feature6": [i + 4 for i in range(20)],
        }
    )
    y_calibration = pd.Series(
        [1, 1, 0, 1, 0, 2, 0, 2, 0, 2, 0, 2, 0, 1, 0, 2, 0, 2, 0, 2]
    )
    y_calibration = y_calibration.replace({0: "zero", 1: "one", 2: "two"})

    x_train["target"] = y_train

    train_all = pd.concat([x_train, x_test], axis=0)
    train_all["target"] = pd.concat([y_train, y_test], axis=0)

    # Fit the BlueCast model using the custom model
    bluecast.fit_eval(train_all, "target")

    # Predict on the test data using the custom model
    predicted_probas, predicted_classes = bluecast.predict(x_test)
    print("++++++++")
    print("MULTICLASS PREDICTED PROBAS")
    print(predicted_probas)

    # Assert the expected results
    assert predicted_probas.shape[1] == 3

    probas = bluecast.predict_proba(x_test)
    print("--------")
    print("MULTICLASS PREDICTED PROBAS from predict_proba")
    print(probas)

    # test conformal prediction
    bluecast.calibrate(x_calibration, y_calibration)
    pred_intervals = bluecast.predict_p_values(x_test)
    pred_sets = bluecast.predict_sets(x_test)
    assert isinstance(pred_intervals, np.ndarray)
    assert isinstance(pred_sets, pd.DataFrame)
    print(f"MULTICLASS pred_intervals: {pred_intervals}")
    print(f"MULTICLASS pred_sets: {pred_sets}")

    # test passing brier score
    bluecast.calibrate(
        x_calibration, y_calibration, **{"nonconformity_measure_scorer": brier_score}
    )
    pred_intervals = bluecast.predict_p_values(x_test)
    pred_sets = bluecast.predict_sets(x_test)
    assert isinstance(pred_intervals, np.ndarray)
    assert isinstance(pred_sets, pd.DataFrame)
    print(f"MULTICLASS pred_intervals: {pred_intervals}")
    print(f"MULTICLASS pred_sets: {pred_sets}")

    # test passing MNM score
    bluecast.calibrate(
        x_calibration,
        y_calibration,
        **{"nonconformity_measure_scorer": margin_nonconformity_measure},
    )
    pred_intervals = bluecast.predict_p_values(x_test)
    pred_sets = bluecast.predict_sets(x_test)
    assert isinstance(pred_intervals, np.ndarray)
    assert isinstance(pred_sets, pd.DataFrame)
    print(f"MULTICLASS pred_intervals: {pred_intervals}")
    print(f"MULTICLASS pred_sets: {pred_sets}")
