"""Tests for group-conditional conformal prediction and conformal fairness."""

import numpy as np
import pandas as pd
import pytest

from bluecast.conformal_prediction.conformal_prediction import (
    ConformalPredictionWrapper,
)
from bluecast.conformal_prediction.conformal_prediction_regression import (
    ConformalPredictionRegressionWrapper,
)
from bluecast.conformal_prediction.evaluation import (
    conformal_fairness_check,
    prediction_interval_coverage_by_group,
    prediction_interval_spans_by_group,
)


class MockClassifier:
    def predict_proba(self, x):
        rng = np.random.default_rng(42)
        n = len(x)
        p = rng.random(n)
        return np.column_stack([1 - p, p])


class MockRegressor:
    def predict(self, x):
        rng = np.random.default_rng(42)
        return rng.normal(50, 10, len(x))


# --- Regression Group-Conditional ---
class TestRegressionGroupConditional:
    def test_calibrate_with_groups(self):
        model = MockRegressor()
        wrapper = ConformalPredictionRegressionWrapper(model, min_group_size=5)

        rng = np.random.default_rng(42)
        cal_x = pd.DataFrame(
            {"feat": rng.random(100), "group": np.repeat(["A", "B"], 50)}
        )
        cal_y = pd.Series(rng.normal(50, 10, 100))

        wrapper.calibrate(cal_x, cal_y, group_columns=["group"])
        assert wrapper.nonconformity_scores_by_group is not None
        assert len(wrapper.nonconformity_scores_by_group) > 0

    def test_predict_interval_with_groups(self):
        model = MockRegressor()
        wrapper = ConformalPredictionRegressionWrapper(model, min_group_size=5)

        rng = np.random.default_rng(42)
        cal_x = pd.DataFrame(
            {"feat": rng.random(100), "group": np.repeat(["A", "B"], 50)}
        )
        cal_y = pd.Series(rng.normal(50, 10, 100))
        wrapper.calibrate(cal_x, cal_y, group_columns=["group"])

        test_x = pd.DataFrame(
            {"feat": rng.random(20), "group": np.repeat(["A", "B"], 10)}
        )
        intervals = wrapper.predict_interval(test_x, alphas=[0.1])
        assert intervals.shape == (20, 2)
        assert "0.1_low" in intervals.columns
        assert "0.9_high" in intervals.columns

    def test_predict_interval_without_groups(self):
        model = MockRegressor()
        wrapper = ConformalPredictionRegressionWrapper(model)

        rng = np.random.default_rng(42)
        cal_x = pd.DataFrame({"feat": rng.random(100)})
        cal_y = pd.Series(rng.normal(50, 10, 100))
        wrapper.calibrate(cal_x, cal_y)

        test_x = pd.DataFrame({"feat": rng.random(20)})
        intervals = wrapper.predict_interval(test_x, alphas=[0.05, 0.1])
        assert intervals.shape[0] == 20

    def test_small_group_falls_back(self):
        model = MockRegressor()
        wrapper = ConformalPredictionRegressionWrapper(model, min_group_size=80)

        rng = np.random.default_rng(42)
        # Group B has only 10 samples, below min_group_size of 80
        groups = ["A"] * 90 + ["B"] * 10
        cal_x = pd.DataFrame({"feat": rng.random(100), "group": groups})
        cal_y = pd.Series(rng.normal(50, 10, 100))
        wrapper.calibrate(cal_x, cal_y, group_columns=["group"])

        assert ("B",) not in wrapper.nonconformity_scores_by_group


# --- Classification Group-Conditional ---
class TestClassificationGroupConditional:
    def test_calibrate_with_groups(self):
        model = MockClassifier()
        wrapper = ConformalPredictionWrapper(model, min_group_size=5)

        rng = np.random.default_rng(42)
        cal_x = pd.DataFrame(
            {"feat": rng.random(100), "tier": np.repeat(["gold", "silver"], 50)}
        )
        cal_y = pd.Series(rng.choice([0, 1], 100))

        wrapper.calibrate(cal_x, cal_y, group_columns=["tier"])
        assert wrapper.nonconformity_scores_by_group is not None
        assert len(wrapper.nonconformity_scores_by_group) > 0

    def test_predict_sets_with_groups(self):
        model = MockClassifier()
        wrapper = ConformalPredictionWrapper(model, min_group_size=5)

        rng = np.random.default_rng(42)
        cal_x = pd.DataFrame(
            {"feat": rng.random(100), "tier": np.repeat(["gold", "silver"], 50)}
        )
        cal_y = pd.Series(rng.choice([0, 1], 100))
        wrapper.calibrate(cal_x, cal_y, group_columns=["tier"])

        test_x = pd.DataFrame(
            {"feat": rng.random(20), "tier": np.repeat(["gold", "silver"], 10)}
        )
        pred_sets = wrapper.predict_sets(test_x, alpha=0.1, group_columns=["tier"])
        assert pred_sets.shape[0] == 20


# --- Per-Group Evaluation ---
class TestGroupEvaluation:
    def test_coverage_by_group(self):
        y_true = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        intervals = pd.DataFrame(
            {
                "0.1_low": [0, 1, 2, 3, 4, 5, 6, 7],
                "0.9_high": [2, 3, 4, 5, 6, 7, 8, 9],
            }
        )
        groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        result = prediction_interval_coverage_by_group(
            y_true, intervals, [0.1], groups
        )
        assert "A" in result
        assert "B" in result
        assert 0.1 in result["A"]

    def test_spans_by_group(self):
        intervals = pd.DataFrame(
            {
                "0.1_low": [0, 0, 0, 0, 10, 10, 10, 10],
                "0.9_high": [2, 2, 2, 2, 20, 20, 20, 20],
            }
        )
        groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        result = prediction_interval_spans_by_group(intervals, [0.1], groups)
        assert result["A"][0.1] == 2.0
        assert result["B"][0.1] == 10.0


# --- Conformal Fairness Check ---
class TestConformalFairness:
    def test_all_fair(self):
        coverages = {
            "urban": {0.1: 0.91},
            "rural": {0.1: 0.89},
        }
        result = conformal_fairness_check(coverages, target_coverage=0.9, tolerance=0.05)
        assert result["is_fair"] is True

    def test_unfair(self):
        coverages = {
            "urban": {0.1: 0.92},
            "rural": {0.1: 0.78},
        }
        result = conformal_fairness_check(coverages, target_coverage=0.9, tolerance=0.05)
        assert result["is_fair"] is False
        assert result["coverage_range"] == (0.78, 0.92)

    def test_result_structure(self):
        coverages = {"A": {0.1: 0.90}, "B": {0.1: 0.85}}
        result = conformal_fairness_check(coverages, target_coverage=0.9)
        assert "is_fair" in result
        assert "target_coverage" in result
        assert "tolerance" in result
        assert "coverage_range" in result
        assert "group_results" in result
        assert "A" in result["group_results"]
