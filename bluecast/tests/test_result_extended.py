"""Extended tests for BlueCastAIResult — predict, predict_proba, show_report paths."""

import os
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from bluecast.ai.result import BlueCastAIResult


class TestPredict:
    def test_predict_single_pipeline(self):
        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = np.array([0, 1, 0])
        result = BlueCastAIResult(pipeline=mock_pipeline)
        preds = result.predict(pd.DataFrame({"a": [1, 2, 3]}))
        assert len(preds) == 3

    def test_predict_tuple_return(self):
        """BlueCast classification returns (probas, classes) tuple."""
        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = (
            np.array([0.2, 0.8, 0.3]),
            np.array([0, 1, 0]),
        )
        result = BlueCastAIResult(pipeline=mock_pipeline)
        preds = result.predict(pd.DataFrame({"a": [1, 2, 3]}))
        np.testing.assert_array_equal(preds, [0, 1, 0])

    def test_predict_multiple_pipelines_mean(self):
        p1 = MagicMock()
        p1.predict.return_value = np.array([1.0, 2.0])
        p2 = MagicMock()
        p2.predict.return_value = np.array([3.0, 4.0])
        result = BlueCastAIResult(pipelines=[p1, p2])
        preds = result.predict(pd.DataFrame({"a": [1, 2]}))
        np.testing.assert_array_almost_equal(preds, [2.0, 3.0])

    def test_predict_multiple_pipelines_tuple(self):
        p1 = MagicMock()
        p1.predict.return_value = (np.array([0.1, 0.9]), np.array([0, 1]))
        result = BlueCastAIResult(pipelines=[p1])
        preds = result.predict(pd.DataFrame({"a": [1, 2]}))
        np.testing.assert_array_equal(preds, [0, 1])

    def test_predict_hill_climbing_ensemble(self):
        hc = MagicMock()
        hc.predict.return_value = np.array([0.5, 0.6])
        p1 = MagicMock()
        p1.predict.return_value = np.array([0.4, 0.5])
        result = BlueCastAIResult(pipelines=[p1], hill_climbing_ensemble=hc)
        result.predict(pd.DataFrame({"a": [1, 2]}))
        hc.predict.assert_called_once()

    def test_predict_hc_tuple_return(self):
        hc = MagicMock()
        hc.predict.return_value = np.array([0.5, 0.6])
        p1 = MagicMock()
        p1.predict.return_value = (np.array([0.1, 0.9]), np.array([0, 1]))
        result = BlueCastAIResult(pipelines=[p1], hill_climbing_ensemble=hc)
        result.predict(pd.DataFrame({"a": [1, 2]}))
        hc.predict.assert_called_once()

    def test_predict_hc_series_return(self):
        hc = MagicMock()
        hc.predict.return_value = np.array([0.5, 0.6])
        p1 = MagicMock()
        p1.predict.return_value = pd.Series([0.4, 0.5])
        result = BlueCastAIResult(pipelines=[p1], hill_climbing_ensemble=hc)
        result.predict(pd.DataFrame({"a": [1, 2]}))
        hc.predict.assert_called_once()


class TestPredictProba:
    def test_predict_proba_single_pipeline_method(self):
        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.return_value = np.array([[0.2, 0.8], [0.7, 0.3]])
        result = BlueCastAIResult(pipeline=mock_pipeline)
        proba = result.predict_proba(pd.DataFrame({"a": [1, 2]}))
        assert proba.shape == (2, 2)

    def test_predict_proba_tuple_fallback(self):
        mock_pipeline = MagicMock(spec=[])
        mock_pipeline.predict = MagicMock(
            return_value=(np.array([0.2, 0.8]), np.array([0, 1]))
        )
        result = BlueCastAIResult(pipeline=mock_pipeline)
        proba = result.predict_proba(pd.DataFrame({"a": [1, 2]}))
        np.testing.assert_array_equal(proba, [0.2, 0.8])

    def test_predict_proba_no_pipeline_raises(self):
        result = BlueCastAIResult()
        with pytest.raises(RuntimeError, match="No trained pipeline"):
            result.predict_proba(pd.DataFrame({"a": [1]}))

    def test_predict_proba_no_method_raises(self):
        mock_pipeline = MagicMock(spec=[])
        mock_pipeline.predict = MagicMock(return_value=np.array([0.5]))
        result = BlueCastAIResult(pipeline=mock_pipeline)
        with pytest.raises(AttributeError):
            result.predict_proba(pd.DataFrame({"a": [1]}))

    def test_predict_proba_multi_pipelines(self):
        p1 = MagicMock()
        p1.predict_proba = MagicMock(return_value=np.array([0.2, 0.8]))
        p2 = MagicMock()
        p2.predict_proba = MagicMock(return_value=np.array([0.4, 0.6]))
        result = BlueCastAIResult(pipelines=[p1, p2])
        proba = result.predict_proba(pd.DataFrame({"a": [1, 2]}))
        assert proba is not None

    def test_predict_proba_multi_pipelines_no_proba(self):
        p1 = MagicMock(spec=[])
        p1.predict = MagicMock(return_value=np.array([0.5]))
        result = BlueCastAIResult(pipelines=[p1])
        with pytest.raises(AttributeError):
            result.predict_proba(pd.DataFrame({"a": [1]}))

    def test_predict_proba_hc_ensemble(self):
        hc = MagicMock()
        hc.predict.return_value = np.array([0.5, 0.6])
        p1 = MagicMock()
        p1.predict_proba = MagicMock(return_value=np.array([0.2, 0.8]))
        result = BlueCastAIResult(pipelines=[p1], hill_climbing_ensemble=hc)
        result.predict_proba(pd.DataFrame({"a": [1, 2]}))
        hc.predict.assert_called_once()

    def test_predict_proba_hc_no_proba_fallback(self):
        hc = MagicMock()
        hc.predict.return_value = np.array([0.5])
        p1 = MagicMock(spec=[])
        p1.predict = MagicMock(return_value=(np.array([0.2]), np.array([0])))
        result = BlueCastAIResult(pipelines=[p1], hill_climbing_ensemble=hc)
        result.predict_proba(pd.DataFrame({"a": [1]}))

    def test_predict_proba_hc_no_method_raises(self):
        hc = MagicMock()
        p1 = MagicMock(spec=[])
        p1.predict = MagicMock(return_value=np.array([0.5]))
        result = BlueCastAIResult(pipelines=[p1], hill_climbing_ensemble=hc)
        with pytest.raises(AttributeError):
            result.predict_proba(pd.DataFrame({"a": [1]}))


class TestShowReport:
    def test_show_report_full_no_markdown(self, capsys):
        result = BlueCastAIResult(
            metrics={"roc_auc": 0.85, "accuracy": 0.9, "count": 100},
            run_history=[
                {"success": True, "metrics": {"roc_auc": 0.85}},
                {"success": False, "metrics": {}},
            ],
            class_problem="binary",
            feature_engineering_code="df['new'] = 1",
            pipeline_code="pipeline.fit(df)",
            agent_log=[f"entry_{i}" for i in range(15)],
        )
        result.show_report()
        captured = capsys.readouterr()
        assert "binary" in captured.out
        assert "0.85" in captured.out
        assert "Feature engineering applied: Yes" in captured.out
        assert (
            "pipeline.fit" in captured.out.lower() or "pipeline" in captured.out.lower()
        )
        assert "... and 5 more" in captured.out

    def test_show_report_no_fe(self, capsys):
        result = BlueCastAIResult(class_problem="regression")
        result.show_report()
        captured = capsys.readouterr()
        assert "Feature engineering applied: No" in captured.out

    def test_show_report_int_metric(self, capsys):
        result = BlueCastAIResult(metrics={"count": 42, "label": "test"})
        result.show_report()
        captured = capsys.readouterr()
        assert "42" in captured.out
        assert "test" in captured.out


class TestSaveReport:
    def test_save_report_empty(self):
        result = BlueCastAIResult(report_markdown="")
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            path = f.name
        try:
            result.save_report(path)
            with open(path) as f:
                assert f.read() == ""
        finally:
            os.unlink(path)


class TestRepr:
    def test_repr_with_pipeline(self):
        result = BlueCastAIResult(
            pipeline=MagicMock(),
            run_history=[{"success": True}],
            metrics={"auc": 0.9},
        )
        r = repr(result)
        assert "trained" in r
        assert "runs=1" in r
