from unittest.mock import MagicMock

import numpy as np
import pytest

from bluecast.serve.app import _format_batch_prediction, _format_prediction, create_app

try:
    from fastapi.testclient import TestClient

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")


@pytest.fixture
def mock_pipeline():
    pipeline = MagicMock()
    pipeline.class_problem = "binary"
    pipeline.conformal_prediction_enabled = False
    pipeline.eval_metrics = {"accuracy": 0.9, "auc": 0.95}
    del pipeline._inner
    del pipeline.bluecast_models

    # Mock schema detector
    ftd = MagicMock()
    ftd.num_columns = ["feat1", "feat2"]
    ftd.cat_columns = ["feat3"]
    ftd.date_columns = []
    ftd.detected_col_types = {"feat1": "float64", "feat2": "int64", "feat3": "object"}
    pipeline.feat_type_detector = ftd
    pipeline.target_column = "target"

    # Mock predict
    pipeline.predict.return_value = (np.array([0.9]), np.array([1]))

    return pipeline


def test_format_prediction():
    res = _format_prediction((np.array([0.9]), np.array([1])), "binary")
    assert res == {"probabilities": 0.9, "predicted_class": 1}

    res = _format_prediction(np.array([42.0]), "regression")
    assert res == {"prediction": 42.0}


def test_format_batch_prediction():
    res = _format_batch_prediction((np.array([0.9, 0.1]), np.array([1, 0])), "binary")
    assert res["count"] == 2
    assert res["probabilities"] == [0.9, 0.1]
    assert res["predicted_classes"] == [1, 0]

    res = _format_batch_prediction(np.array([42.0, 43.0]), "regression")
    assert res["count"] == 2
    assert res["predictions"] == [42.0, 43.0]


def test_create_app_health(mock_pipeline):
    app = create_app(mock_pipeline)
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_create_app_metrics(mock_pipeline):
    app = create_app(mock_pipeline)
    client = TestClient(app)

    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.json() == {"accuracy": 0.9, "auc": 0.95}


def test_create_app_schema(mock_pipeline):
    app = create_app(mock_pipeline)
    client = TestClient(app)

    response = client.get("/schema")
    assert response.status_code == 200
    assert "columns" in response.json()
    assert len(response.json()["columns"]) == 3


def test_create_app_predict(mock_pipeline):
    app = create_app(mock_pipeline)
    client = TestClient(app)

    response = client.post("/predict", json={"feat1": 1.0, "feat2": 2, "feat3": "a"})
    assert response.status_code == 200
    assert response.json() == {"probabilities": 0.9, "predicted_class": 1}


def test_create_app_predict_batch(mock_pipeline):
    mock_pipeline.predict.return_value = (np.array([0.9, 0.1]), np.array([1, 0]))
    app = create_app(mock_pipeline)
    client = TestClient(app)

    response = client.post(
        "/predict/batch",
        json=[
            {"feat1": 1.0, "feat2": 2, "feat3": "a"},
            {"feat1": 2.0, "feat2": 3, "feat3": "b"},
        ],
    )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert data["probabilities"] == [0.9, 0.1]
