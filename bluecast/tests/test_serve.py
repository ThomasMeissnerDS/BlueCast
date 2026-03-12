"""Tests for the serve module (schemas, app factory, exporter)."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from bluecast.blueprints.cast import BlueCast
from bluecast.config.training_config import TrainingConfig
from bluecast.serve.schemas import (
    _extract_column_info,
    _get_class_problem,
    _has_conformal,
    build_schema_response,
)

try:
    import pydantic  # noqa: F401

    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False

try:
    import fastapi  # noqa: F401

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


@pytest.fixture
def fast_config():
    return TrainingConfig(
        hyperparameter_tuning_rounds=2,
        hyperparameter_tuning_max_runtime_secs=10,
        enable_feature_selection=False,
        calculate_shap_values=False,
        plot_hyperparameter_tuning_overview=False,
        hypertuning_cv_folds=2,
        train_size=0.8,
    )


@pytest.fixture
def trained_pipeline(fast_config):
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
    df["cat_col"] = np.random.choice(["A", "B"], size=100)
    df["target"] = y
    pipeline = BlueCast(class_problem="binary", conf_training=fast_config)
    pipeline.fit(df, target_col="target")
    return pipeline


# --- Schema tests (no optional deps needed) ---
class TestSchemas:
    def test_extract_column_info(self, trained_pipeline):
        cols = _extract_column_info(trained_pipeline)
        assert isinstance(cols, list)
        assert len(cols) > 0
        col_names = {c["name"] for c in cols}
        assert "target" not in col_names

    def test_get_class_problem(self, trained_pipeline):
        assert _get_class_problem(trained_pipeline) == "binary"

    def test_has_conformal_false(self, trained_pipeline):
        assert _has_conformal(trained_pipeline) is False

    def test_build_schema_response(self, trained_pipeline):
        schema = build_schema_response(trained_pipeline)
        assert schema["class_problem"] == "binary"
        assert "columns" in schema
        assert schema["n_columns"] > 0
        assert schema["target_column"] == "target"

    @pytest.mark.skipif(not HAS_PYDANTIC, reason="pydantic not installed")
    def test_build_request_model(self, trained_pipeline):
        from bluecast.serve.schemas import build_request_model

        Model = build_request_model(trained_pipeline)
        assert Model is not None
        instance = Model()
        d = instance.model_dump()
        assert isinstance(d, dict)

    @pytest.mark.skipif(not HAS_PYDANTIC, reason="pydantic not installed")
    def test_build_request_model_no_schema(self):
        from bluecast.serve.schemas import build_request_model

        class FakePipeline:
            class_problem = "binary"
            target_column = "y"

        Model = build_request_model(FakePipeline())
        assert Model is not None


# --- App factory tests ---
class TestAppFactory:
    @pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")
    def test_create_app(self, trained_pipeline):
        from bluecast.serve.app import create_app

        app = create_app(trained_pipeline)
        assert app is not None
        assert app.title == "BlueCast Model API"

    def test_format_prediction_binary(self):
        from bluecast.serve.app import _format_prediction

        result = (np.array([0.8]), np.array([1]))
        formatted = _format_prediction(result, "binary")
        assert "probabilities" in formatted
        assert "predicted_class" in formatted
        assert formatted["predicted_class"] == 1

    def test_format_prediction_regression(self):
        from bluecast.serve.app import _format_prediction

        result = np.array([42.5])
        formatted = _format_prediction(result, "regression")
        assert "prediction" in formatted
        assert formatted["prediction"] == 42.5

    def test_format_batch_prediction(self):
        from bluecast.serve.app import _format_batch_prediction

        result = (np.array([0.8, 0.3]), np.array([1, 0]))
        formatted = _format_batch_prediction(result, "binary")
        assert "probabilities" in formatted
        assert "predicted_classes" in formatted
        assert formatted["count"] == 2

    def test_format_prediction_series(self):
        from bluecast.serve.app import _format_prediction

        result = pd.Series([42.5])
        formatted = _format_prediction(result, "regression")
        assert "prediction" in formatted


# --- Exporter tests ---
class TestExporter:
    def test_export_api(self, trained_pipeline):
        from bluecast.serve.exporter import export_api

        with tempfile.TemporaryDirectory() as tmpdir:
            output = export_api(trained_pipeline, tmpdir)
            assert os.path.isfile(os.path.join(output, "app.py"))
            assert os.path.isfile(os.path.join(output, "model.pkl"))
            assert os.path.isfile(os.path.join(output, "requirements.txt"))
            assert os.path.isfile(os.path.join(output, "Dockerfile"))
            assert os.path.isfile(os.path.join(output, "README.md"))

    def test_export_api_no_docker(self, trained_pipeline):
        from bluecast.serve.exporter import export_api

        with tempfile.TemporaryDirectory() as tmpdir:
            output = export_api(trained_pipeline, tmpdir, include_docker=False)
            assert os.path.isfile(os.path.join(output, "app.py"))
            assert not os.path.isfile(os.path.join(output, "Dockerfile"))

    def test_generated_app_content(self, trained_pipeline):
        from bluecast.serve.exporter import export_api

        with tempfile.TemporaryDirectory() as tmpdir:
            export_api(trained_pipeline, tmpdir)
            with open(os.path.join(tmpdir, "app.py")) as f:
                code = f.read()
            assert "FastAPI" in code
            assert "/predict" in code
            assert "/health" in code

    def test_generated_requirements(self, trained_pipeline):
        from bluecast.serve.exporter import export_api

        with tempfile.TemporaryDirectory() as tmpdir:
            export_api(trained_pipeline, tmpdir)
            with open(os.path.join(tmpdir, "requirements.txt")) as f:
                reqs = f.read()
            assert "bluecast" in reqs
            assert "fastapi" in reqs
            assert "uvicorn" in reqs

    def test_generated_readme(self, trained_pipeline):
        from bluecast.serve.exporter import export_api

        with tempfile.TemporaryDirectory() as tmpdir:
            export_api(trained_pipeline, tmpdir)
            with open(os.path.join(tmpdir, "README.md")) as f:
                readme = f.read()
            assert "binary" in readme
            assert "/predict" in readme
