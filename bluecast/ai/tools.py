"""Tool definitions and implementations for BlueCastAI agents.

Each tool is a concrete Python function that agents call via the LLM's
tool-use / function-calling interface. Tools provide deterministic,
safe operations on data and pipelines.
"""

import logging
import traceback
from io import StringIO
from typing import Any, Dict

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
    buf.write(
        f"\n\nNull percentages:\n{(df.isnull().mean() * 100).round(2).to_string()}"
    )

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if num_cols:
        buf.write(
            f"\n\nNumeric describe:\n{df[num_cols].describe().round(4).to_string()}"
        )

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols[:10]:
        buf.write(
            f"\n\n'{col}' value counts (top 10):\n{df[col].value_counts().head(10).to_string()}"
        )

    if target_col in df.columns:
        buf.write(
            f"\n\nTarget '{target_col}' distribution:\n{df[target_col].value_counts().to_string()}"
        )
        n_unique = df[target_col].nunique()
        if n_unique <= 2:
            buf.write("\n\nDetected problem: binary classification")
        elif n_unique <= 20:
            buf.write(
                f"\n\nDetected problem: multiclass classification ({n_unique} classes)"
            )
        else:
            buf.write("\n\nDetected problem: regression")

    return buf.getvalue()


def tool_check_correlations(
    df: pd.DataFrame, target_col: str, threshold: float = 0.8
) -> str:
    """Check for high correlations among features and with target."""
    num_df = df.select_dtypes(include=["number"])
    if num_df.empty:
        return "No numeric columns found."

    corr = num_df.corr(numeric_only=True)
    lines = []

    if target_col in corr.columns:
        target_corr = (
            corr[target_col].drop(target_col).abs().sort_values(ascending=False)
        )
        lines.append(f"Top correlations with target '{target_col}':")
        for col, val in target_corr.head(10).items():
            lines.append(f"  {col}: {val:.4f}")

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
            cat_leaky = detect_categorical_leakage(
                df[cat_cols + [target_col]], target_col, threshold=0.95
            )
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


def tool_create_tfidf_features(
    df: pd.DataFrame, text_col: str, max_features: int = 50
) -> Dict[str, Any]:
    """Apply TF-IDF to a text column, adding top-N features to df."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    if text_col not in df.columns:
        return {
            "success": False,
            "new_columns": [],
            "error": f"Column '{text_col}' not found.",
        }

    try:
        vec = TfidfVectorizer(max_features=max_features, stop_words="english")
        tfidf_matrix = vec.fit_transform(df[text_col].fillna("").astype(str))
        feature_names = [f"tfidf_{text_col}_{w}" for w in vec.get_feature_names_out()]
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(), columns=feature_names, index=df.index
        )
        for col in tfidf_df.columns:
            df[col] = tfidf_df[col]
        return {
            "success": True,
            "new_columns": feature_names,
            "shape": list(df.shape),
            "error": None,
        }
    except Exception as e:
        return {"success": False, "new_columns": [], "error": str(e)}


# ---------------------------------------------------------------------------
# Advanced data analysis tools
# ---------------------------------------------------------------------------


def tool_check_uniqueness(df: pd.DataFrame) -> str:
    """Cardinality analysis for every column."""
    lines = ["Column cardinality analysis:\n"]
    for col in df.columns:
        n = df[col].nunique()
        pct = n / max(len(df), 1) * 100
        dtype = str(df[col].dtype)
        is_id = pct > 95 and n > 100
        flag = " ⚠️ LIKELY ID/KEY" if is_id else ""
        lines.append(f"  {col}: {n} unique ({pct:.1f}%), dtype={dtype}{flag}")
    return "\n".join(lines)


def tool_check_outliers(
    df: pd.DataFrame, n_show: int = 5, contamination: float = 0.05
) -> str:
    """Detect outliers via IsolationForest on numeric features."""
    from sklearn.ensemble import IsolationForest

    num_df = df.select_dtypes(include=["number"]).dropna(axis=1)
    if num_df.empty or len(num_df) < 10:
        return "Not enough numeric data for outlier detection."

    try:
        iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        scores = iso.fit_predict(num_df)
        outlier_mask = scores == -1
        n_outliers = int(outlier_mask.sum())

        lines = [
            f"IsolationForest detected {n_outliers} outliers "
            f"({n_outliers / len(df) * 100:.1f}% of rows, "
            f"contamination={contamination}).\n"
        ]

        if n_outliers > 0:
            outlier_idx = num_df.index[outlier_mask]
            sample_idx = outlier_idx[:n_show]
            lines.append(f"Sample outlier rows (first {len(sample_idx)}):\n")
            lines.append(df.loc[sample_idx].to_string())

            # Show which features differ most for outliers vs normal
            normal_means = num_df.loc[~outlier_mask].mean()
            outlier_means = num_df.loc[outlier_mask].mean()
            diff = (
                ((outlier_means - normal_means) / normal_means.replace(0, np.nan))
                .dropna()
                .abs()
            )
            top_diff = diff.sort_values(ascending=False).head(5)
            lines.append("\nFeatures with largest outlier deviation:")
            for col_name, val in top_diff.items():
                lines.append(f"  {col_name}: {val:.2%} deviation from normal mean")

        return "\n".join(lines)
    except Exception as e:
        return f"Outlier detection failed: {e}"


def tool_inspect_rows(df: pd.DataFrame, indices: str = "", condition: str = "") -> str:
    """Inspect specific rows by index list or pandas query condition."""
    try:
        if condition:
            subset = df.query(condition)
        elif indices:
            idx_list = [int(i.strip()) for i in indices.split(",")]
            subset = df.iloc[idx_list]
        else:
            return "Provide either 'indices' (comma-separated) or 'condition' (pandas query)."

        if len(subset) > 20:
            return (
                f"Query returned {len(subset)} rows (showing first 20):\n"
                + subset.head(20).to_string()
            )
        return f"Query returned {len(subset)} rows:\n" + subset.to_string()
    except Exception as e:
        return f"Row inspection failed: {e}"


def tool_run_sql_query(df: pd.DataFrame, query: str) -> str:
    """Run SQL against the DataFrame using pandasql."""
    try:
        import pandasql
    except ImportError:
        # Fallback: use pandas operations
        return (
            "pandasql not installed. Use tool_inspect_rows with a pandas query "
            "condition instead, or ask the user to install pandasql."
        )

    try:
        result = pandasql.sqldf(query, {"df": df})
        if len(result) > 50:
            return (
                f"Query returned {len(result)} rows (showing first 50):\n"
                + result.head(50).to_string()
            )
        return f"Query returned {len(result)} rows:\n" + result.to_string()
    except Exception as e:
        return f"SQL query failed: {e}"


def tool_check_temporal_patterns(df: pd.DataFrame, target_col: str) -> str:
    """Detect datetime columns and check for temporal patterns."""
    dt_cols = df.select_dtypes(include=["datetime", "datetime64"]).columns.tolist()

    # Also try to parse object columns that look like dates
    for col in df.select_dtypes(include=["object"]).columns:
        sample = df[col].dropna().head(20)
        try:
            pd.to_datetime(sample)
            dt_cols.append(col)
        except (ValueError, TypeError):
            pass

    if not dt_cols:
        return "No datetime or date-like columns detected."

    lines = [f"Datetime columns found: {dt_cols}\n"]

    for col in dt_cols[:3]:  # Limit to first 3
        try:
            dt_series = pd.to_datetime(df[col])
            lines.append(f"\n--- {col} ---")
            lines.append(f"  Range: {dt_series.min()} to {dt_series.max()}")
            lines.append(f"  Nulls: {dt_series.isna().sum()}")

            # Check for gaps
            sorted_dt = dt_series.dropna().sort_values()
            if len(sorted_dt) > 1:
                diffs = sorted_dt.diff().dropna()
                lines.append(
                    f"  Median interval: {diffs.median()}, " f"Max gap: {diffs.max()}"
                )

            # Target drift over time (if numeric target)
            if target_col in df.columns and df[target_col].dtype in [
                "float64",
                "int64",
            ]:
                temp_df = df[[col, target_col]].dropna()
                temp_df["_dt"] = pd.to_datetime(temp_df[col])
                temp_df["_month"] = temp_df["_dt"].dt.to_period("M")
                monthly = temp_df.groupby("_month")[target_col].mean()
                if len(monthly) > 1:
                    lines.append("  Target mean by month (last 6):")
                    for period, val in monthly.tail(6).items():
                        lines.append(f"    {period}: {val:.4f}")
        except Exception as e:
            lines.append(f"  Error analyzing {col}: {e}")

    return "\n".join(lines)


def tool_check_group_statistics(df: pd.DataFrame, group_col: str, agg_col: str) -> str:
    """Group-by statistics for a categorical × numeric pair."""
    if group_col not in df.columns:
        return f"Column '{group_col}' not found."
    if agg_col not in df.columns:
        return f"Column '{agg_col}' not found."

    try:
        grouped = df.groupby(group_col)[agg_col].agg(
            ["count", "mean", "std", "min", "max"]
        )
        grouped = grouped.sort_values("count", ascending=False)
        if len(grouped) > 30:
            return (
                f"Group statistics for {agg_col} by {group_col} "
                f"({len(grouped)} groups, showing top 30):\n"
                + grouped.head(30).to_string()
            )
        return f"Group statistics for {agg_col} by {group_col}:\n" + grouped.to_string()
    except Exception as e:
        return f"Group statistics failed: {e}"


# ---------------------------------------------------------------------------
# Pipeline tools
# ---------------------------------------------------------------------------


def tool_build_and_run_pipeline(
    df: pd.DataFrame,
    target_col: str,
    config: Dict[str, Any],
    custom_preprocessor=None,
    ml_model=None,
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
        hyperparameter_tuning_rounds=int(config.get("tuning_rounds", 50)),
        hyperparameter_tuning_max_runtime_secs=int(
            config.get("tuning_max_runtime", 120)
        ),
        enable_feature_selection=bool(config.get("enable_feature_selection", False)),
        calculate_shap_values=False,
        plot_hyperparameter_tuning_overview=False,
        hypertuning_cv_folds=int(config.get("hypertuning_cv_folds", 3)),
        autotune_on_device=config.get("autotune_on_device", "cpu"),
        bluecast_cv_train_n_model=(
            int(config.get("n_folds", 5)),
            int(config.get("n_repeats", 1)),
        ),
    )
    
    if "out_of_fold_dataset_store_path" in config:
        training_config.out_of_fold_dataset_store_path = config["out_of_fold_dataset_store_path"]

    if "cat_encoding_via_ml_algorithm" in config:
        training_config.cat_encoding_via_ml_algorithm = config["cat_encoding_via_ml_algorithm"]

    ensemble_config = None
    if use_cv:
        strategy = config.get("ensemble_strategy", "mean")
        ensemble_config = EnsembleConfig(ensemble_strategy=strategy)
        if strategy == "hill_climbing":
            ensemble_config.hc_weight_min = config.get("hc_weight_min", -0.3)
            ensemble_config.hc_weight_max = config.get("hc_weight_max", 0.5)
            ensemble_config.hc_weight_step = config.get("hc_weight_step", 0.01)
            if class_problem == "regression":
                ensemble_config.hc_blending_method = "rank"  # regression shouldn't use probability

    try:
        pipeline = BlueCastAuto(
            class_problem=class_problem,
            use_cross_validation=use_cv,
            conf_training=training_config,
            ensemble_config=ensemble_config,
            custom_preprocessor=custom_preprocessor,
            ml_model=ml_model,
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
            metrics = pipeline.fit_eval(  # type: ignore[assignment]
                df_train,
                target_col=target_col,
                df_eval=df_eval,
                y_eval=y_eval,
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
            params={"q": query, "num": 3},  # type: ignore[arg-type]
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
        description="Execute Python feature engineering code that modifies or re-assigns the DataFrame 'df'. "
        "The code has access to 'df', 'np', 'pd', and can import from 'bluecast.preprocessing'. "
        "Example: \nfrom bluecast.preprocessing.feature_creation import add_binned_features\ndf = add_binned_features(df, ['a'])",
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
                "n_folds": {
                    "type": "integer",
                    "description": "Number of CV folds. Default 5.",
                },
                "n_repeats": {
                    "type": "integer",
                    "description": "Number of CV repeats. Default 1.",
                },
                "tuning_rounds": {
                    "type": "integer",
                    "description": "Hyperparameter tuning rounds. Default 50.",
                },
                "tuning_max_runtime": {
                    "type": "integer",
                    "description": "Max tuning time in seconds. Default 120.",
                },
                "autotune_on_device": {
                    "type": "string",
                    "enum": ["cpu", "gpu"],
                    "description": "Device for training. Default cpu.",
                },
                "out_of_fold_dataset_store_path": {
                    "type": "string",
                    "description": "Path to save out-of-fold predictions. Omit to not save.",
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
    "check_uniqueness": ToolDefinition(
        name="check_uniqueness",
        description="Analyze the cardinality (number of unique values) of every column. Flags likely ID columns. Use this to understand feature types and detect columns that should be dropped.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    "check_outliers": ToolDefinition(
        name="check_outliers",
        description="Detect outliers using IsolationForest on all numeric columns. Returns outlier rows and most-deviating features. Useful for understanding data quality.",
        parameters={
            "type": "object",
            "properties": {
                "n_show": {
                    "type": "integer",
                    "description": "Number of outlier rows to display. Default 5.",
                },
                "contamination": {
                    "type": "number",
                    "description": "Expected proportion of outliers (0.01-0.2). Default 0.05.",
                },
            },
            "required": [],
        },
    ),
    "inspect_rows": ToolDefinition(
        name="inspect_rows",
        description="Inspect specific rows by index or condition. Use this to drill into suspicious rows identified by other tools.",
        parameters={
            "type": "object",
            "properties": {
                "indices": {
                    "type": "string",
                    "description": "Comma-separated row indices to inspect (e.g. '0,5,10').",
                },
                "condition": {
                    "type": "string",
                    "description": "Pandas query condition (e.g. 'age > 100').",
                },
            },
            "required": [],
        },
    ),
    "run_sql_query": ToolDefinition(
        name="run_sql_query",
        description="Run a SQL query against the DataFrame (table name is 'df'). Use for complex aggregations, joins, or custom analysis.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "SQL query. The DataFrame is available as table 'df'.",
                },
            },
            "required": ["query"],
        },
    ),
    "check_temporal_patterns": ToolDefinition(
        name="check_temporal_patterns",
        description="Auto-detect datetime columns and analyze temporal patterns: date ranges, gaps, target drift over time.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    "check_group_statistics": ToolDefinition(
        name="check_group_statistics",
        description="Compute group-by statistics (count, mean, std, min, max) for a numeric column grouped by a categorical column.",
        parameters={
            "type": "object",
            "properties": {
                "group_col": {
                    "type": "string",
                    "description": "Categorical column to group by.",
                },
                "agg_col": {
                    "type": "string",
                    "description": "Numeric column to aggregate.",
                },
            },
            "required": ["group_col", "agg_col"],
        },
    ),
    "create_tfidf_features": ToolDefinition(
        name="create_tfidf_features",
        description="Apply TF-IDF vectorization to a text column, adding the top-N most important word features to the DataFrame.",
        parameters={
            "type": "object",
            "properties": {
                "text_col": {
                    "type": "string",
                    "description": "Name of the text column to vectorize.",
                },
                "max_features": {
                    "type": "integer",
                    "description": "Maximum number of TF-IDF features. Default 50.",
                },
            },
            "required": ["text_col"],
        },
    ),
}
