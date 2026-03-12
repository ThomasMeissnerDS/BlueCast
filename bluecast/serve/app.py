"""FastAPI app factory for serving BlueCast pipelines."""

import logging
import traceback
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from bluecast.serve.schemas import (
    _get_class_problem,
    _has_conformal,
    build_request_model,
    build_schema_response,
)

logger = logging.getLogger(__name__)


def _format_prediction(raw_result: Any, class_problem: str) -> Dict[str, Any]:
    """Format a single prediction result into a JSON-friendly dict."""
    if class_problem in ("binary", "multiclass"):
        if isinstance(raw_result, tuple) and len(raw_result) == 2:
            probs, classes = raw_result
            prob_val = probs.tolist() if hasattr(probs, "tolist") else probs
            class_val = classes.tolist() if hasattr(classes, "tolist") else classes
            if isinstance(prob_val, list) and len(prob_val) == 1:
                prob_val = prob_val[0]
            if isinstance(class_val, list) and len(class_val) == 1:
                class_val = class_val[0]
            return {"probabilities": prob_val, "predicted_class": class_val}
        else:
            val = raw_result.tolist() if hasattr(raw_result, "tolist") else raw_result
            return {"prediction": val}
    else:
        if isinstance(raw_result, (np.ndarray, pd.Series)):
            val = raw_result.tolist()
            if isinstance(val, list) and len(val) == 1:
                val = val[0]
            return {"prediction": val}
        return {"prediction": raw_result}


def _format_batch_prediction(raw_result: Any, class_problem: str) -> Dict[str, Any]:
    """Format batch predictions."""
    if class_problem in ("binary", "multiclass"):
        if isinstance(raw_result, tuple) and len(raw_result) == 2:
            probs, classes = raw_result
            return {
                "probabilities": probs.tolist() if hasattr(probs, "tolist") else list(probs),
                "predicted_classes": (
                    classes.tolist() if hasattr(classes, "tolist") else list(classes)
                ),
                "count": len(probs) if hasattr(probs, "__len__") else 1,
            }
    if isinstance(raw_result, (np.ndarray, pd.Series)):
        return {"predictions": raw_result.tolist(), "count": len(raw_result)}
    return {"predictions": raw_result}


def create_app(pipeline: Any) -> Any:
    """Create a FastAPI application from a trained BlueCast pipeline.

    The app auto-generates request schemas from the pipeline's column metadata
    and provides prediction, health, schema, and metrics endpoints.

    :param pipeline: A trained BlueCast pipeline (any variant).
    :returns: A FastAPI application instance.
    """
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError:
        raise ImportError(
            "FastAPI is required for bluecast.serve. "
            "Install with: pip install 'bluecast[serve]'"
        )

    class_problem = _get_class_problem(pipeline)
    has_conformal = _has_conformal(pipeline)

    RequestModel = build_request_model(pipeline)

    app = FastAPI(
        title="BlueCast Model API",
        description=(
            f"Auto-generated API for a BlueCast {class_problem} model. "
            f"Conformal prediction: {'enabled' if has_conformal else 'disabled'}."
        ),
        version="1.0.0",
    )

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "healthy",
            "model_type": class_problem,
            "conformal_prediction": has_conformal,
        }

    @app.get("/schema")
    def schema() -> Dict[str, Any]:
        return build_schema_response(pipeline)

    @app.get("/metrics")
    def metrics() -> Dict[str, Any]:
        eval_metrics = getattr(pipeline, "eval_metrics", None)
        if hasattr(pipeline, "_inner"):
            eval_metrics = getattr(pipeline._inner, "eval_metrics", eval_metrics)
        if eval_metrics is None:
            return {"message": "No evaluation metrics available. Use fit_eval() to generate."}
        serializable = {}
        for k, v in eval_metrics.items():
            if isinstance(v, (int, float, str, bool)):
                serializable[k] = v
            elif isinstance(v, (np.floating, np.integer)):
                serializable[k] = float(v)
        return serializable

    @app.post("/predict")
    def predict(request: RequestModel) -> Dict[str, Any]:  # type: ignore[valid-type]
        try:
            data = request.model_dump()
            data = {k: v for k, v in data.items() if k != "_placeholder"}
            df = pd.DataFrame([data])
            result = pipeline.predict(df)
            return _format_prediction(result, class_problem)
        except Exception as e:
            logger.error(f"Prediction failed: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/predict/batch")
    def predict_batch(requests: List[RequestModel]) -> Dict[str, Any]:  # type: ignore[valid-type]
        try:
            data_list = []
            for req in requests:
                d = req.model_dump()
                d = {k: v for k, v in d.items() if k != "_placeholder"}
                data_list.append(d)
            df = pd.DataFrame(data_list)
            result = pipeline.predict(df)
            return _format_batch_prediction(result, class_problem)
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=400, detail=str(e))

    return app
