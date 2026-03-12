"""Tool definitions and implementations for BlueCastAI agents.

Each tool is a concrete Python function that agents call via the LLM's
tool-use / function-calling interface. Tools provide deterministic,
safe operations on data and pipelines.
"""

import json
import logging
import traceback
from io import StringIO
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from bluecast.ai.providers.base import ToolDefinition

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data analysis tools
# ---------------------------------------------------------------------------

def tool_describe_data(df: pd.DataFrame, target_col: str) -> str:
    """Generate a comprehensive data profile."""
    buf = StringIO()
    buf.write(f"Shape: {df.shape}\n\n")
    buf.write("Dtypes:\n")
    buf.write(df.dtypes.to_string())
    buf.write(f"\n\nNull counts:\n{df.isnull().sum().to_string()}")
    buf.write(f"\n\nNull percentages:\n{(df.isnull().mean() * 100).round(2).to_string()}")

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if num_cols:
        buf.write(f"\n\nNumeric describe:\n{df[num_cols].describe().round(4).to_string()}")

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols[:10]:
        buf.write(f"\n\n'{col}' value counts (top 10):\n{df[col].value_counts().head(10).to_string()}")

    if target_col in df.columns:
        buf.write(f"\n\nTarget '{target_col}' distribution:\n{df[target_col].value_counts().to_string()}")
        n_unique = df[target_col].nunique()
        if n_unique <= 2:
            buf.write("\n\nDetected problem: binary classification")
        elif n_unique <= 20:
            buf.write(f"\n\nDetected problem: multiclass classification ({n_unique} classes)")
        else:
            buf.write("\n\nDetected problem: regression")

    return buf.getvalue()


def tool_check_correlations(df: pd.DataFrame, target_col: str, threshold: float = 0.8) -> str:
    """Check for high correlations among features and with target."""
    num_df = df.select_dtypes(include=["number"])
    if num_df.empty:
        return "No numeric columns found."

    corr = num_df.corr(numeric_only=True)
    lines = []

    if target_col in corr.columns:
        target_corr = corr[target_col].drop(target_col).abs().sort_values(ascending=False)
        lines.append(f"Top correlations with target '{target_col}':")
        for col, val in target_corr.head(10).items():
            lines.append(f"  {col}: {val:.4f}")

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    high_corr_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if abs(corr.iloc[i, j]) >= threshold:
                high_corr_pairs.append(
                    (corr.columns[i], corr.columns[j], corr.iloc[i, j])
                )

    if high_corr_pairs:
        lines.append(f"\nHighly correlated feature pairs (|r| >= {threshold}):")
        for c1, c2, val in sorted(high_corr_pairs, key=lambda x: -abs(x[2])):
            lines.append(f"  {c1} <-> {c2}: {val:.4f}")
    else:
        lines.append(f"\nNo feature pairs with |r| >= {threshold}")

    return "\n".join(lines)


def tool_check_leakage(df: pd.DataFrame, target_col: str) -> str:
    """Check for potential target leakage."""
    from bluecast.eda.data_leakage_checks import (
        detect_categorical_leakage,
        detect_leakage_via_correlation,
    )

    results = []
    try:
        num_leaky = detect_leakage_via_correlation(df, target_col, threshold=0.95)
        if num_leaky:
            results.append(f"Correlation leakage suspects: {num_leaky}")
        else:
            results.append("No correlation-based leakage detected.")
    except Exception as e:
        results.append(f"Correlation leakage check failed: {e}")

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols and target_col in df.columns:
        try:
            cat_leaky = detect_categorical_leakage(df[cat_cols + [target_col]], target_col, threshold=0.95)
            if cat_leaky:
                results.append(f"Categorical leakage suspects: {cat_leaky}")
            else:
                results.append("No categorical leakage detected.")
        except Exception as e:
            results.append(f"Categorical leakage check failed: {e}")

    return "\n".join(results)


# ---------------------------------------------------------------------------
# Feature engineering tools
# ---------------------------------------------------------------------------

def tool_create_feature(
    df: pd.DataFrame,
    feature_code: str,
) -> Dict[str, Any]:
    """Execute feature engineering code and return the modified DataFrame.

    The code should modify 'df' in-place or assign new columns.
    Returns dict with 'success', 'new_columns', 'error'.
    """
    original_cols = set(df.columns)
    try:
        local_vars = {"df": df, "np": np, "pd": pd}
        exec(feature_code, {}, local_vars)
        df_result = local_vars.get("df", df)
        new_cols = list(set(df_result.columns) - original_cols)
        return {
            "success": True,
            "new_columns": new_cols,
            "shape": list(df_result.shape),
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "new_columns": [],
            "shape": list(df.shape),
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Pipeline tools
# ---------------------------------------------------------------------------

def tool_build_and_run_pipeline(
    df: pd.DataFrame,
    target_col: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Build and evaluate a BlueCast pipeline from a config dict.

    Config keys:
        class_problem: "binary" | "multiclass" | "regression"
        use_cv: bool (default True)
        ensemble_strategy: "mean" | "stacking" | "hill_climbing"
        n_folds: int
        n_repeats: int
        tuning_rounds: int
        tuning_max_runtime: int
        autotune_on_device: "cpu" | "gpu"
    """
    from bluecast.blueprints.unified import BlueCastAuto
    from bluecast.config.training_config import TrainingConfig
    from bluecast.ensemble.ensemble_config import EnsembleConfig

    class_problem = config.get("class_problem", "binary")
    use_cv = config.get("use_cv", True)

    training_config = TrainingConfig(
        hyperparameter_tuning_rounds=config.get("tuning_rounds", 50),
        hyperparameter_tuning_max_runtime_secs=config.get("tuning_max_runtime", 120),
        enable_feature_selection=config.get("enable_feature_selection", False),
        calculate_shap_values=False,
        plot_hyperparameter_tuning_overview=False,
        hypertuning_cv_folds=config.get("hypertuning_cv_folds", 3),
        autotune_on_device=config.get("autotune_on_device", "cpu"),
        bluecast_cv_train_n_model=(
            config.get("n_folds", 5),
            config.get("n_repeats", 1),
        ),
    )

    ensemble_config = None
    if use_cv:
        strategy = config.get("ensemble_strategy", "mean")
        ensemble_config = EnsembleConfig(ensemble_strategy=strategy)
        if strategy == "hill_climbing":
            ensemble_config.hc_weight_min = config.get("hc_weight_min", -0.3)
            ensemble_config.hc_weight_max = config.get("hc_weight_max", 0.5)
            ensemble_config.hc_weight_step = config.get("hc_weight_step", 0.01)

    try:
        pipeline = BlueCastAuto(
            class_problem=class_problem,
            use_cross_validation=use_cv,
            conf_training=training_config,
            ensemble_config=ensemble_config,
        )

        if use_cv:
            result = pipeline.fit_eval(df, target_col=target_col)
            if isinstance(result, tuple):
                oof_mean, oof_std = result
                metrics = {"oof_mean": oof_mean, "oof_std": oof_std}
            else:
                metrics = result
        else:
            from sklearn.model_selection import train_test_split
            df_train, df_eval = train_test_split(df, test_size=0.2, random_state=42)
            y_eval = df_eval.pop(target_col)
            metrics = pipeline.fit_eval(
                df_train, target_col=target_col,
                df_eval=df_eval, y_eval=y_eval,
            )

        return {
            "success": True,
            "metrics": _serialize_metrics(metrics),
            "pipeline": pipeline,
            "config_used": config,
            "error": None,
        }
    except Exception as e:
        logger.error(f"Pipeline build failed: {e}\n{traceback.format_exc()}")
        return {
            "success": False,
            "metrics": {},
            "pipeline": None,
            "config_used": config,
            "error": str(e),
        }


def _serialize_metrics(metrics) -> Dict[str, Any]:
    """Convert metrics to JSON-serializable format."""
    if isinstance(metrics, dict):
        result = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float, str, bool)):
                result[k] = v
            elif isinstance(v, np.floating):
                result[k] = float(v)
            elif isinstance(v, np.integer):
                result[k] = int(v)
            elif k == "fairness":
                result[k] = str(v)[:200]
        return result
    elif isinstance(metrics, tuple) and len(metrics) == 2:
        return {"oof_mean": float(metrics[0]), "oof_std": float(metrics[1])}
    return {"raw": str(metrics)[:500]}


# ---------------------------------------------------------------------------
# Web search tool
# ---------------------------------------------------------------------------

def tool_web_search(query: str) -> str:
    """Search the web for data science techniques and domain knowledge."""
    try:
        import requests
        response = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={"q": query, "num": 3},
            timeout=10,
        )
        if response.ok:
            results = response.json().get("items", [])
            return "\n\n".join(
                f"**{r['title']}**\n{r.get('snippet', '')}\nURL: {r['link']}"
                for r in results[:3]
            )
    except Exception:
        pass
    return f"Web search for '{query}' did not return results. Use domain knowledge instead."


# ---------------------------------------------------------------------------
# Tool registry - definitions for LLM function calling
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: Dict[str, ToolDefinition] = {
    "describe_data": ToolDefinition(
        name="describe_data",
        description="Generate a comprehensive profile of the dataset including dtypes, nulls, distributions, and target analysis.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    "check_correlations": ToolDefinition(
        name="check_correlations",
        description="Check for high correlations among features and with the target column.",
        parameters={
            "type": "object",
            "properties": {
                "threshold": {
                    "type": "number",
                    "description": "Correlation threshold to flag pairs. Default 0.8.",
                }
            },
            "required": [],
        },
    ),
    "check_leakage": ToolDefinition(
        name="check_leakage",
        description="Check for potential target leakage via high correlation or Theil's U.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    "create_feature": ToolDefinition(
        name="create_feature",
        description="Execute Python feature engineering code that modifies the DataFrame 'df'. "
                    "The code can use 'df', 'np', and 'pd'. Example: df['ratio'] = df['a'] / (df['b'] + 1)",
        parameters={
            "type": "object",
            "properties": {
                "feature_code": {
                    "type": "string",
                    "description": "Python code to create new features on 'df'.",
                },
                "description": {
                    "type": "string",
                    "description": "Brief description of what features are created.",
                },
            },
            "required": ["feature_code"],
        },
    ),
    "build_and_run_pipeline": ToolDefinition(
        name="build_and_run_pipeline",
        description="Build and evaluate a BlueCast ML pipeline. Returns metrics.",
        parameters={
            "type": "object",
            "properties": {
                "class_problem": {
                    "type": "string",
                    "enum": ["binary", "multiclass", "regression"],
                    "description": "Type of ML problem.",
                },
                "use_cv": {
                    "type": "boolean",
                    "description": "Whether to use cross-validation. Default true.",
                },
                "ensemble_strategy": {
                    "type": "string",
                    "enum": ["mean", "stacking", "hill_climbing"],
                    "description": "How to combine CV fold predictions.",
                },
                "n_folds": {"type": "integer", "description": "Number of CV folds. Default 5."},
                "n_repeats": {"type": "integer", "description": "Number of CV repeats. Default 1."},
                "tuning_rounds": {"type": "integer", "description": "Hyperparameter tuning rounds. Default 50."},
                "tuning_max_runtime": {"type": "integer", "description": "Max tuning time in seconds. Default 120."},
                "autotune_on_device": {
                    "type": "string",
                    "enum": ["cpu", "gpu"],
                    "description": "Device for training. Default cpu.",
                },
            },
            "required": ["class_problem"],
        },
    ),
    "web_search": ToolDefinition(
        name="web_search",
        description="Search the web for data science techniques, domain knowledge, or Kaggle solutions.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query.",
                },
            },
            "required": ["query"],
        },
    ),
}
