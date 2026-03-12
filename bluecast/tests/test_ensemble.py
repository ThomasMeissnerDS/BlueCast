"""Tests for the ensemble module (hill climbing, stacking, mean blending)."""

import numpy as np
import pandas as pd
import pytest

from bluecast.ensemble.ensemble_config import EnsembleConfig
from bluecast.ensemble.hill_climbing import (
    HillClimbingEnsemble,
    _convert_to_ranks,
    _default_classification_metric,
    _default_regression_metric,
)
from bluecast.ensemble.mean_blending import blend_predictions_mean
from bluecast.ensemble.stacking import StackingEnsemble


# --- Mean Blending ---
class TestMeanBlending:
    def setup_method(self):
        rng = np.random.default_rng(42)
        self.df = pd.DataFrame(
            {"a": rng.random(100), "b": rng.random(100), "c": rng.random(100)}
        )
        self.cols = ["a", "b", "c"]

    def test_arithmetic(self):
        result = blend_predictions_mean(self.df, self.cols, "arithmetic")
        expected = self.df[self.cols].mean(axis=1)
        pd.testing.assert_series_equal(result, expected)

    def test_median(self):
        result = blend_predictions_mean(self.df, self.cols, "median")
        expected = self.df[self.cols].median(axis=1)
        pd.testing.assert_series_equal(result, expected)

    def test_geometric(self):
        result = blend_predictions_mean(self.df, self.cols, "geometric")
        assert len(result) == 100
        assert result.min() > 0

    def test_harmonic(self):
        result = blend_predictions_mean(self.df, self.cols, "harmonic")
        assert len(result) == 100
        assert result.min() > 0

    def test_unknown_falls_back(self):
        result = blend_predictions_mean(self.df, self.cols, "unknown_type")
        expected = self.df[self.cols].mean(axis=1)
        pd.testing.assert_series_equal(result, expected)


# --- Stacking ---
class TestStacking:
    def test_fit_predict(self):
        rng = np.random.default_rng(42)
        oof = rng.random((200, 5))
        y = (rng.random(200) > 0.5).astype(float)
        stacker = StackingEnsemble(use_ranks=True)
        stacker.fit(oof, y)
        assert stacker.is_fitted

        test_preds = rng.random((50, 5))
        result = stacker.predict(test_preds)
        assert result.shape == (50,)
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_fit_predict_no_ranks(self):
        rng = np.random.default_rng(42)
        oof = rng.random((200, 5))
        y = (rng.random(200) > 0.5).astype(float)
        stacker = StackingEnsemble(use_ranks=False)
        stacker.fit(oof, y)
        result = stacker.predict(rng.random((50, 5)))
        assert result.shape == (50,)

    def test_predict_before_fit_raises(self):
        stacker = StackingEnsemble()
        with pytest.raises(RuntimeError, match="not been fitted"):
            stacker.predict(np.random.rand(10, 3))

    def test_convert_to_ranks(self):
        data = np.array([10.0, 30.0, 20.0, 40.0])
        ranks = StackingEnsemble._convert_to_ranks(data)
        assert ranks.shape == (4,)
        assert np.isclose(ranks.min(), 0.0)
        assert np.isclose(ranks.max(), 1.0)


# --- Hill Climbing ---
class TestHillClimbing:
    def test_fit_predict(self):
        rng = np.random.default_rng(42)
        n = 200
        y = (rng.random(n) > 0.5).astype(float)
        preds = [rng.random(n) for _ in range(8)]

        hc = HillClimbingEnsemble(
            weight_min=0.0, weight_max=0.5, weight_step=0.1, tolerance=1e-5
        )
        hc.fit(preds, y)
        assert hc.is_fitted
        assert len(hc.selected_indices) > 0
        assert len(hc.history) > 0
        assert len(hc.weights_map) > 0

        result = hc.predict(preds)
        assert result.shape == (n,)
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_fit_with_negative_weights(self):
        rng = np.random.default_rng(42)
        n = 200
        y = (rng.random(n) > 0.5).astype(float)
        preds = [rng.random(n) for _ in range(5)]

        hc = HillClimbingEnsemble(
            weight_min=-0.3, weight_max=0.5, weight_step=0.1, tolerance=1e-5
        )
        hc.fit(preds, y)
        assert hc.is_fitted

    def test_predict_before_fit_raises(self):
        hc = HillClimbingEnsemble()
        with pytest.raises(RuntimeError, match="not been fitted"):
            hc.predict([np.random.rand(10)])

    def test_get_selected_model_info(self):
        rng = np.random.default_rng(42)
        preds = [rng.random(100) for _ in range(5)]
        y = (rng.random(100) > 0.5).astype(float)

        hc = HillClimbingEnsemble(weight_step=0.1)
        hc.fit(preds, y)
        info = hc.get_selected_model_info()
        assert isinstance(info, list)
        assert len(info) > 0
        assert "model_idx" in info[0]
        assert "weight" in info[0]

    def test_custom_eval_metric(self):
        rng = np.random.default_rng(42)
        preds = [rng.random(100) for _ in range(5)]
        y = rng.random(100)

        hc = HillClimbingEnsemble(
            eval_metric=_default_regression_metric,
            weight_step=0.1,
        )
        hc.fit(preds, y)
        assert hc.is_fitted

    def test_model_names(self):
        rng = np.random.default_rng(42)
        preds = [rng.random(100) for _ in range(3)]
        y = (rng.random(100) > 0.5).astype(float)
        names = ["lgbm", "xgb", "catboost"]

        hc = HillClimbingEnsemble(weight_step=0.1)
        hc.fit(preds, y, model_names=names)
        for entry in hc.history:
            assert entry["model"] in names


def test_convert_to_ranks():
    data = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
    ranks = _convert_to_ranks(data)
    assert ranks.shape == (5,)
    assert np.isclose(ranks.min(), 0.0)
    assert np.isclose(ranks.max(), 1.0)


def test_default_metrics():
    y = np.array([0, 1, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.8, 0.2, 0.7])
    auc = _default_classification_metric(y, p)
    assert 0 <= auc <= 1

    y_reg = np.array([1.0, 2.0, 3.0])
    p_reg = np.array([1.1, 2.2, 2.9])
    neg_rmse = _default_regression_metric(y_reg, p_reg)
    assert neg_rmse <= 0


# --- EnsembleConfig ---
class TestEnsembleConfig:
    def test_defaults(self):
        config = EnsembleConfig()
        assert config.ensemble_strategy == "mean"
        assert config.mean_type == "arithmetic"

    def test_no_negative_weights(self):
        config = EnsembleConfig(hc_allow_negative_weights=False)
        assert config.hc_weight_min == 0.0

    def test_repr(self):
        config = EnsembleConfig(ensemble_strategy="stacking")
        r = repr(config)
        assert "stacking" in r
