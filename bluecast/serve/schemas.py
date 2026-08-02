"""Auto-generate Pydantic request/response models from a trained BlueCast pipeline."""

import logging
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


def _extract_column_info(pipeline: Any) -> List[Dict[str, Any]]:
    """Extract column names and types from a trained pipeline.

    Works with BlueCast, BlueCastRegression, BlueCastCV, BlueCastCVRegression,
    and BlueCastAuto by inspecting the inner model's schema detector and
    feature type detector.
    """
    inner = pipeline
    if hasattr(pipeline, "_inner"):
        inner = pipeline._inner
    if hasattr(inner, "bluecast_models") and inner.bluecast_models:
        inner = inner.bluecast_models[0]

    columns: List[Dict[str, Any]] = []
    target_col = getattr(inner, "target_column", None) or ""

    if hasattr(inner, "feat_type_detector") and inner.feat_type_detector:
        ftd = inner.feat_type_detector
        num_cols = set(getattr(ftd, "num_columns", []) or [])
        cat_cols = set(getattr(ftd, "cat_columns", []) or [])
        date_cols = set(getattr(ftd, "date_columns", []) or [])
        detected = getattr(ftd, "detected_col_types", {}) or {}

        all_cols = list(num_cols | cat_cols | date_cols | set(detected.keys()))
        for col in all_cols:
            if col == target_col:
                continue
            dtype = detected.get(col, "")
            if col in num_cols or "float" in str(dtype) or "int" in str(dtype):
                columns.append(
                    {
                        "name": col,
                        "type": "number",
                        "python_type": "float",
                        "nullable": True,
                    }
                )
            elif col in date_cols or "datetime" in str(dtype):
                columns.append(
                    {
                        "name": col,
                        "type": "string",
                        "python_type": "str",
                        "nullable": True,
                    }
                )
            else:
                columns.append(
                    {
                        "name": col,
                        "type": "string",
                        "python_type": "str",
                        "nullable": True,
                    }
                )

    if not columns and hasattr(inner, "schema_detector") and inner.schema_detector:
        schema = inner.schema_detector.train_schema
        for col in schema:
            if col == target_col:
                continue
            columns.append(
                {
                    "name": col,
                    "type": "number",
                    "python_type": "float",
                    "nullable": True,
                }
            )

    return columns


def _get_class_problem(pipeline: Any) -> str:
    """Extract the class_problem from any pipeline type."""
    if hasattr(pipeline, "class_problem"):
        return pipeline.class_problem
    if hasattr(pipeline, "_inner") and hasattr(pipeline._inner, "class_problem"):
        return pipeline._inner.class_problem
    return "binary"


def _has_conformal(pipeline: Any) -> bool:
    """Check if the pipeline has been calibrated for conformal prediction."""
    inner = pipeline
    if hasattr(pipeline, "_inner"):
        inner = pipeline._inner
    return getattr(inner, "conformal_prediction_wrapper", None) is not None


def build_request_model(pipeline: Any) -> "Type":
    """Build a Pydantic BaseModel class from the pipeline's schema.

    :param pipeline: A trained BlueCast pipeline.
    :returns: A dynamically created Pydantic model class.
    """
    try:
        from pydantic import create_model
    except ImportError:
        raise ImportError(
            "pydantic is required for schema generation. "
            "Install with: pip install 'bluecast[serve]'"
        )

    columns = _extract_column_info(pipeline)

    fields = {}
    for col in columns:
        if col["python_type"] == "float":
            fields[col["name"]] = (Optional[float], None)
        else:
            fields[col["name"]] = (Optional[str], None)

    if not fields:
        fields["_placeholder"] = (Optional[str], None)
        logger.warning(
            "Could not detect schema from pipeline. "
            "The /predict endpoint will accept any JSON."
        )

    return create_model("PredictionRequest", **fields)


def build_schema_response(pipeline: Any) -> Dict[str, Any]:
    """Build a JSON-serializable schema description for the /schema endpoint."""
    columns = _extract_column_info(pipeline)
    class_problem = _get_class_problem(pipeline)
    has_conf = _has_conformal(pipeline)

    return {
        "class_problem": class_problem,
        "has_conformal_prediction": has_conf,
        "target_column": getattr(pipeline, "target_column", None)
        or getattr(getattr(pipeline, "_inner", None), "target_column", None)
        or "",
        "columns": columns,
        "n_columns": len(columns),
    }
