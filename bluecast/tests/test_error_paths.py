"""Tests for error paths and edge cases.

Covers predict-before-fit, calibrate-before-fit, invalid inputs, and other
error handling that was previously untested.
"""

import numpy as np
import pandas as pd
import pytest

from bluecast.blueprints.cast import BlueCast
from bluecast.blueprints.cast_regression import BlueCastRegression


class TestPredictBeforeFit:
    def test_bluecast_predict_before_fit(self):
        automl = BlueCast(class_problem="binary")
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(RuntimeError, match="ML model could not be found"):
            automl.predict(df)

    def test_bluecast_predict_proba_before_fit(self):
        automl = BlueCast(class_problem="binary")
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(RuntimeError, match="ML model could not be found"):
            automl.predict_proba(df)

    def test_bluecast_regression_predict_before_fit(self):
        automl = BlueCastRegression(class_problem="regression")
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(RuntimeError):
            automl.predict(df)

    def test_bluecast_transform_new_data_before_fit(self):
        automl = BlueCast(class_problem="binary")
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(RuntimeError, match="Feature type converter"):
            automl.transform_new_data(df)

    def test_bluecast_regression_transform_before_fit(self):
        automl = BlueCastRegression(class_problem="regression")
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises((RuntimeError, Exception)):
            automl.transform_new_data(df)


class TestCalibrateBeforeFit:
    def test_bluecast_predict_sets_before_calibrate(
        self, trained_bluecast_binary, synthetic_binary_data
    ):
        automl = trained_bluecast_binary
        df_test = synthetic_binary_data.drop("target", axis=1).head(5)
        with pytest.raises(ValueError, match="not been calibrated"):
            automl.predict_sets(df_test)

    def test_bluecast_predict_p_values_before_calibrate(
        self, trained_bluecast_binary, synthetic_binary_data
    ):
        automl = trained_bluecast_binary
        df_test = synthetic_binary_data.drop("target", axis=1).head(5)
        with pytest.raises(ValueError, match="not been calibrated"):
            automl.predict_p_values(df_test)

    def test_bluecast_regression_predict_interval_before_calibrate(
        self, trained_bluecast_regression, synthetic_regression_data
    ):
        automl = trained_bluecast_regression
        df_test = synthetic_regression_data.drop("target", axis=1).head(5)
        with pytest.raises(ValueError, match="not been calibrated"):
            automl.predict_interval(df_test, alphas=[0.05])


class TestTransformNewData:
    def test_transform_new_data_binary(
        self, trained_bluecast_binary, synthetic_binary_data
    ):
        automl = trained_bluecast_binary
        df_test = synthetic_binary_data.drop("target", axis=1).head(10)
        transformed = automl.transform_new_data(df_test)
        assert isinstance(transformed, pd.DataFrame)
        assert len(transformed) == 10
        assert "target" not in transformed.columns

    def test_transform_new_data_regression(
        self, trained_bluecast_regression, synthetic_regression_data
    ):
        automl = trained_bluecast_regression
        df_test = synthetic_regression_data.drop("target", axis=1).head(10)
        transformed = automl.transform_new_data(df_test)
        assert isinstance(transformed, pd.DataFrame)
        assert len(transformed) == 10


class TestSingleRowPrediction:
    def test_single_row_predict_binary(
        self, trained_bluecast_binary, synthetic_binary_data
    ):
        automl = trained_bluecast_binary
        df_single = synthetic_binary_data.drop("target", axis=1).head(1)
        y_probs, y_classes = automl.predict(df_single)
        assert len(y_probs) == 1
        assert len(y_classes) == 1

    def test_single_row_predict_regression(
        self, trained_bluecast_regression, synthetic_regression_data
    ):
        automl = trained_bluecast_regression
        df_single = synthetic_regression_data.drop("target", axis=1).head(1)
        y_preds = automl.predict(df_single)
        assert len(y_preds) == 1


class TestInputWithMissingValues:
    def test_predict_with_nan_values(
        self, trained_bluecast_binary, synthetic_binary_data
    ):
        automl = trained_bluecast_binary
        df_test = synthetic_binary_data.drop("target", axis=1).head(5).copy()
        df_test.iloc[0, 2] = np.nan
        df_test.iloc[2, 3] = np.nan
        y_probs, y_classes = automl.predict(df_test)
        assert len(y_probs) == 5
        assert not np.isnan(y_classes).any()
