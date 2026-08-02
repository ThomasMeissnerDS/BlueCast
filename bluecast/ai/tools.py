"""Tool definitions and implementations for BlueCastAI agents.

Each tool is a concrete Python function that agents call via the LLM's
tool-use / function-calling interface. Tools provide deterministic,
safe operations on data and pipelines.
"""

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


def tool_evaluate_imputations(  # noqa: C901
    df: pd.DataFrame, target_col: str, fill_value: float = -999.0
) -> str:  # noqa: C901
    """Test imputation strategies for numerical columns with NaNs or sentinel values.

    Auto-detects common sentinel values (999, -999, 9999, -9999, etc.) that
    likely represent hidden missing data. For each affected column, tests 8
    strategies (raw, mean, median, 0, fill_value, 1st-percentile,
    99th-percentile, mode) and returns a per-column recommendation.

    Downsamples to 10 000 rows when the DataFrame is larger to keep MI fast.
    """
    if target_col not in df.columns:
        return f"Target column '{target_col}' not found."

    # Downsample if too large to make Mutual Information computation snappy
    if len(df) > 10_000:
        df = df.sample(n=10_000, random_state=42)

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)

    # ------------------------------------------------------------------
    # 1. Detect sentinel values
    # ------------------------------------------------------------------
    SENTINEL_CANDIDATES = [
        999.0,
        -999.0,
        999,
        -999,
        9999.0,
        -9999.0,
        9999,
        -9999,
        -1.0,
        -1,
    ]

    sentinel_info: Dict[str, List[float]] = {}  # col -> list of detected sentinels
    for col in num_cols:
        col_sentinels = []
        col_series = df[col].dropna()
        if col_series.empty:
            continue
        col_mean = col_series.mean()
        col_std = col_series.std()
        for sv in SENTINEL_CANDIDATES:
            count = (df[col] == sv).sum()
            pct = count / len(df)
            # Heuristic: appears in >2% of rows AND is >2 stdev from mean
            if pct > 0.02 and col_std > 0 and abs(sv - col_mean) > 2 * col_std:
                col_sentinels.append(sv)
        if col_sentinels:
            sentinel_info[col] = col_sentinels

    # Columns to evaluate: those with NaN or detected sentinels
    cols_with_nans = [c for c in num_cols if df[c].isnull().any()]
    cols_to_eval = sorted(set(cols_with_nans) | set(sentinel_info.keys()))

    if not cols_to_eval:
        return "No numerical columns with missing values or sentinel values found."

    import warnings

    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    is_classification = df[target_col].nunique() <= 20

    results = []
    recommendations: Dict[str, str] = {}

    buf = StringIO()

    # Report detected sentinels
    if sentinel_info:
        buf.write("### Detected Sentinel Values (Likely Hidden Missing Data)\n\n")
        for col, svs in sentinel_info.items():
            for sv in svs:
                cnt = int((df[col] == sv).sum())
                pct = cnt / len(df) * 100
                buf.write(
                    f"- **{col}**: value `{sv}` appears {cnt} times ({pct:.1f}% of rows)\n"
                )
        buf.write("\n")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for col in cols_to_eval:
            # Create a working copy where sentinels are replaced with NaN
            work = df[col].copy()
            sentinels_for_col = sentinel_info.get(col, [])
            for sv in sentinels_for_col:
                work = work.replace(sv, np.nan)

            # Compute statistics on clean (non-sentinel, non-NaN) values
            clean = work.dropna()
            if len(clean) < 10:
                continue

            col_mean = clean.mean()
            col_median = clean.median()
            col_mode = clean.mode().iloc[0] if len(clean.mode()) > 0 else col_median
            col_p1 = clean.quantile(0.01)
            col_p99 = clean.quantile(0.99)

            strategies = {
                "raw (drop missing)": clean,
                "mean": work.fillna(col_mean),
                "median": work.fillna(col_median),
                "zero": work.fillna(0),
                f"static ({fill_value})": work.fillna(fill_value),
                "1st percentile": work.fillna(col_p1),
                "99th percentile": work.fillna(col_p99),
                "mode": work.fillna(col_mode),
            }

            best_mi = -1.0
            best_strat = "raw (drop missing)"

            for strat_name, series in strategies.items():
                if strat_name == "raw (drop missing)":
                    mask = series.index
                    y = df.loc[mask, target_col].dropna()
                    x_vals = series.loc[y.index]
                    if len(y) < 10:
                        continue
                    x = x_vals.values.reshape(-1, 1)
                else:
                    valid_mask = df[target_col].notnull()
                    y = df.loc[valid_mask, target_col]
                    x = series[valid_mask].values.reshape(-1, 1)

                if len(y) < 10:
                    continue

                # Correlation
                corr = np.corrcoef(x.flatten(), y.values)[0, 1]
                if np.isnan(corr):
                    corr = 0.0

                # Mutual Information
                if is_classification:
                    mi = mutual_info_classif(x, y, random_state=42)[0]
                else:
                    mi = mutual_info_regression(x, y, random_state=42)[0]

                results.append(
                    {
                        "Feature": col,
                        "Strategy": strat_name,
                        "Correlation (abs)": round(abs(corr), 4),
                        "Mutual Info": round(mi, 4),
                    }
                )

                if mi > best_mi:
                    best_mi = mi
                    best_strat = strat_name

            recommendations[col] = best_strat

    if not results:
        return "Evaluation failed: not enough valid target data."

    res_df = pd.DataFrame(results)

    buf.write("### Imputation Strategy Evaluation\n\n")

    # Sort by Mutual Info
    buf.write("#### Top strategies by Mutual Information (non-linear signal):\n")
    buf.write(
        res_df.sort_values("Mutual Info", ascending=False)
        .head(20)
        .to_string(index=False)
    )

    # Sort by Correlation
    buf.write("\n\n#### Top strategies by Pearson Correlation (linear signal):\n")
    buf.write(
        res_df.sort_values("Correlation (abs)", ascending=False)
        .head(20)
        .to_string(index=False)
    )

    # Per-column recommendations
    if recommendations:
        buf.write("\n\n### Per-Column Imputation Recommendation\n\n")
        buf.write("Based on highest Mutual Information with the target:\n\n")
        for col, strat in recommendations.items():
            sentinel_note = ""
            if col in sentinel_info:
                sentinel_note = f" ⚠️ Sentinel values detected: {sentinel_info[col]}"
            buf.write(f"- **{col}**: Best strategy = **{strat}**{sentinel_note}\n")

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Feature engineering tools
# ---------------------------------------------------------------------------


def tool_check_feature_quality(
    df: pd.DataFrame, target_col: str, feature_cols: List[str]
) -> str:
    """Check signal quality of newly created features via correlation and MI.

    This is a lightweight check (<1s) that helps the LLM decide whether a
    feature is worth keeping before running a full pipeline.

    :param df: DataFrame containing both the features and target column.
    :param target_col: Name of the target column.
    :param feature_cols: List of feature column names to evaluate.
    :returns: Per-feature quality assessment string.
    """
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.preprocessing import LabelEncoder

    if target_col not in df.columns:
        return f"Target column '{target_col}' not found."

    y_raw = df[target_col]
    if (
        y_raw.dtype == "object"
        or y_raw.dtype.name == "category"
        or y_raw.dtype == "string"
    ):
        try:
            y = LabelEncoder().fit_transform(y_raw.astype(str)).astype(np.float64)
        except Exception:
            y = pd.to_numeric(y_raw, errors="coerce").values
    else:
        y = pd.to_numeric(y_raw, errors="coerce").values

    results = []
    for col in feature_cols:
        if col not in df.columns:
            results.append(f"{col}: MISSING (not found in DataFrame)")
            continue

        x_raw = df[col]
        if (
            x_raw.dtype == "object"
            or x_raw.dtype.name == "category"
            or x_raw.dtype == "string"
        ):
            try:
                x = LabelEncoder().fit_transform(x_raw.astype(str)).astype(np.float64)
            except Exception:
                x = pd.to_numeric(x_raw, errors="coerce").values
        else:
            x = pd.to_numeric(x_raw, errors="coerce").values

        mask = ~(np.isnan(x) | np.isinf(x) | np.isnan(y) | np.isinf(y))
        if mask.sum() < 30:
            results.append(f"{col}: insufficient non-null data ({mask.sum()} rows)")
            continue

        x_clean = x[mask].reshape(-1, 1)
        y_clean = y[mask]

        try:
            corr = float(np.corrcoef(x_clean.ravel(), y_clean)[0, 1])
        except Exception:
            corr = 0.0

        try:
            mi = float(mutual_info_regression(x_clean, y_clean, random_state=42)[0])
        except Exception:
            mi = 0.0

        if abs(corr) > 0.3 or mi > 0.1:
            quality = "HIGH"
        elif abs(corr) > 0.1 or mi > 0.05:
            quality = "MEDIUM"
        else:
            quality = "LOW"

        results.append(
            f"{col}: correlation={corr:.4f}, mutual_info={mi:.4f} → {quality} signal"
        )

    return "\n".join(results)


def tool_create_feature(
    df: pd.DataFrame,
    feature_code: str,
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute feature engineering code and return the modified DataFrame.

    The code should modify 'df' in-place or assign new columns.
    Returns dict with 'success', 'new_columns', 'error'.
    """
    original_cols = set(df.columns)
    if state is None:
        state = {}
    try:
        local_vars = {
            "df": df,
            "np": np,
            "pd": pd,
            "state": state,
            "is_fit": True,
        }
        import contextlib
        import io

        stdout_capture = io.StringIO()
        with contextlib.redirect_stdout(stdout_capture):
            exec(feature_code, local_vars)

        captured_out = stdout_capture.getvalue()
        if captured_out.strip():
            # Log the captured print statements to standard python logger instead of notebook stdout
            logger.info(f"Captured output from feature code: {captured_out.strip()}")

        df_result = local_vars.get("df", df)
        new_cols = list(set(df_result.columns) - original_cols)
        return {
            "success": True,
            "new_columns": new_cols,
            "shape": list(df_result.shape),
            "error": None,
            "df": df_result,
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
            "df": df,
        }
    except Exception as e:
        return {"success": False, "new_columns": [], "error": str(e)}


def tool_drop_collinear_features(
    df: pd.DataFrame, threshold: float = 0.9, target_col: Optional[str] = None
) -> Dict[str, Any]:
    """Identify highly correlated numerical features and drop one from each pair.

    If target_col is provided, it prefers dropping the feature less correlated with target.

    Returns a dict with 'success', 'dropped_columns', 'error'.
    """
    try:
        num_df = df.select_dtypes(include=["number"])
        if target_col and target_col in num_df.columns:
            target_corr = num_df.corrwith(num_df[target_col]).abs()
        else:
            target_corr = pd.Series(index=num_df.columns, data=1.0)

        corr_matrix = num_df.corr(numeric_only=True).abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop = set()
        for i in range(len(upper.columns)):
            col1 = upper.columns[i]
            for j in range(i):
                col2 = upper.columns[j]
                if upper.iloc[j, i] > threshold:
                    if target_corr.get(col1, 0) < target_corr.get(col2, 0):
                        to_drop.add(col1)
                    else:
                        to_drop.add(col2)

        to_drop_list = list(to_drop)
        for c in to_drop_list:
            if c in df.columns:
                df.drop(columns=[c], inplace=True)

        return {"success": True, "dropped_columns": to_drop_list, "error": None}
    except Exception as e:
        return {"success": False, "dropped_columns": [], "error": str(e)}


def tool_l1_feature_selection(
    df: pd.DataFrame,
    target_col: str,
    class_problem: str = "regression",
    alpha: float = 0.01,
) -> Dict[str, Any]:
    """Perform L1 regularization to drop uninformative features for linear models.

    Returns a dict with 'success', 'dropped_columns', 'error'.
    """
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Lasso, LogisticRegression
        from sklearn.preprocessing import StandardScaler

        if target_col not in df.columns:
            return {
                "success": False,
                "dropped_columns": [],
                "error": f"Target column {target_col} missing.",
            }

        y = df[target_col]
        num_cols = (
            df.drop(columns=[target_col], errors="ignore")
            .select_dtypes(include=["number"])
            .columns.tolist()
        )

        if not num_cols:
            return {
                "success": False,
                "dropped_columns": [],
                "error": "No numeric columns.",
            }

        X = df[num_cols].copy()

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(imputer.fit_transform(X))

        if class_problem == "regression":
            model = Lasso(alpha=alpha, random_state=300)
        else:
            model = LogisticRegression(
                penalty="l1", solver="liblinear", C=1 / (alpha + 1e-5), random_state=300
            )

        model.fit(X_scaled, y)
        coef = model.coef_
        if len(coef.shape) > 1:
            coef = np.max(np.abs(coef), axis=0)

        dropped = []
        for c, c_val in zip(num_cols, coef):
            if abs(c_val) < 1e-5:
                dropped.append(c)
                df.drop(columns=[c], inplace=True)

        return {"success": True, "dropped_columns": dropped, "error": None}
    except Exception as e:
        return {"success": False, "dropped_columns": [], "error": str(e)}


# ---------------------------------------------------------------------------
# Advanced data analysis tools
# ---------------------------------------------------------------------------


def tool_check_adversarial_validation(df: pd.DataFrame, test_condition: str) -> str:
    """Run adversarial validation between train and test datasets defined by a condition."""
    try:
        from bluecast.monitoring.data_monitoring import DataDrift

        subset_test = df.query(test_condition)
        subset_train = df.drop(subset_test.index)

        if len(subset_test) < 10 or len(subset_train) < 10:
            return "Condition resulted in too few rows for either train or test split (need >= 10)."

        drift = DataDrift()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        auc_raw = drift.adversarial_validation(
            subset_train, subset_test, cat_columns=cat_cols
        )
        import numpy as np

        auc = float(np.ravel(auc_raw)[0])

        res = f"Adversarial Validation AUC: {auc:.4f}\n"
        if auc > 0.6:
            res += "WARNING: High AUC indicates significant covariate shift between splits.\n"
        else:
            res += "Low AUC indicates train and test distributions are similar.\n"

        top_features = drift.adversarial_feature_importance[:5]
        if top_features:
            res += "Top features driving the drift:\n"
            for f, s in top_features:
                s_val = float(np.ravel(s)[0])
                res += f"  {f}: {s_val:.4f}\n"
        return res
    except Exception as e:
        return f"Adversarial validation failed: {e}"


def tool_check_mutual_information(
    df: pd.DataFrame, target_col: str, task_type: str = "classification"
) -> str:
    """Calculate mutual information between features and target."""

    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OrdinalEncoder

    try:
        if target_col not in df.columns:
            return f"Target '{target_col}' not found."

        df_clean = df.copy()
        y = df_clean.pop(target_col)

        df_clean = df_clean.dropna(thresh=int(len(df_clean) * 0.1), axis=1)

        cat_cols = df_clean.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        if cat_cols:
            df_clean[cat_cols] = OrdinalEncoder().fit_transform(
                df_clean[cat_cols].astype(str)
            )

        num_cols = df_clean.select_dtypes(include=["number"]).columns.tolist()
        df_clean[num_cols] = SimpleImputer(strategy="median").fit_transform(
            df_clean[num_cols]
        )

        if task_type == "classification":
            mi = mutual_info_classif(df_clean, y, random_state=42)
        else:
            mi = mutual_info_regression(df_clean, y, random_state=42)

        mi_series = pd.Series(mi, index=df_clean.columns).sort_values(ascending=False)

        res = f"Top 15 Mutual Information scores (task: {task_type}):\n"
        for col, val in mi_series.head(15).items():
            res += f"  {col}: {val:.4f}\n"
        return res
    except Exception as e:
        return f"Mutual information failed: {e}"


def tool_target_distribution_test(df: pd.DataFrame, target_col: str) -> str:
    """Check normality of target distribution using Shapiro-Wilk test."""
    try:
        from scipy.stats import shapiro, skew

        if target_col not in df.columns:
            return f"Target '{target_col}' not found."

        y = df[target_col].dropna()
        if not pd.api.types.is_numeric_dtype(y):
            return "Target is not numeric, cannot test for normality."

        if len(y) > 5000:
            y = y.sample(5000, random_state=42)

        stat, p = shapiro(y)
        sk = skew(y)

        stat = float(np.ravel(stat)[0])
        p = float(np.ravel(p)[0])
        sk = float(np.ravel(sk)[0])

        res = f"Shapiro-Wilk Test on '{target_col}':\n"
        res += f"  Statistic: {stat:.4f}, p-value: {p:.4e}\n"
        res += f"  Skewness: {sk:.4f}\n"

        if p < 0.05:
            res += "  Result: Distribution is NOT normal (reject H0).\n"
            if abs(sk) > 1:
                res += "  Recommendation: High skew detected. Consider a log, Box-Cox, or Yeo-Johnson transformation."
        else:
            res += "  Result: Distribution appears normal (fail to reject H0)."

        return res
    except Exception as e:
        return f"Target distribution test failed: {e}"


def tool_nlp_profiling(df: pd.DataFrame, text_col: str) -> str:
    """Analyze a text column for length, vocabulary richness, etc."""
    try:
        if text_col not in df.columns:
            return f"Column '{text_col}' not found."

        texts = df[text_col].dropna().astype(str)
        if len(texts) == 0:
            return "No valid text found."

        lengths = texts.str.len()
        words = texts.str.split()
        word_counts = words.str.len()

        res = f"NLP Profiling for '{text_col}' ({len(texts)} non-null rows):\n"
        res += f"  Character Length: mean={lengths.mean():.1f}, median={lengths.median():.1f}, max={lengths.max()}\n"
        res += f"  Word Count: mean={word_counts.mean():.1f}, median={word_counts.median():.1f}, max={word_counts.max()}\n"

        all_words = pd.Series([w.lower() for wordlist in words for w in wordlist])
        vocab = all_words.nunique()
        res += f"  Total vocabulary size (unique words): {vocab}\n"

        res += "  Top 10 most common words:\n"
        for word, count in all_words.value_counts().head(10).items():
            res += f"    '{word}': {count}\n"

        return res
    except Exception as e:
        return f"NLP profiling failed: {e}"


def tool_apply_target_encoding(
    df: pd.DataFrame, target_col: str, cat_cols: str
) -> dict:
    """Apply Out-of-fold target encoding to categorical columns."""
    try:
        from bluecast.preprocessing.target_encoding import (
            BinaryClassTargetEncoder,
            MultiClassTargetEncoder,
        )

        cols_to_encode = [c.strip() for c in cat_cols.split(",")]
        missing = [c for c in cols_to_encode if c not in df.columns]
        if missing:
            return {
                "success": False,
                "new_columns": [],
                "error": f"Columns not found: {missing}",
            }

        n_unique = df[target_col].nunique()
        if n_unique <= 2:
            bin_encoder = BinaryClassTargetEncoder(
                cat_columns=cols_to_encode  # type: ignore
            )
            df_encoded = bin_encoder.fit_target_encode_binary_class(
                df.copy(), df[target_col]
            )
        else:
            multi_encoder = MultiClassTargetEncoder(
                cat_columns=cols_to_encode, target_col=target_col  # type: ignore
            )
            df_encoded = multi_encoder.fit_target_encode_multiclass(
                df.copy(), df[target_col]
            )

        new_columns = []
        for c in df_encoded.columns:
            if c not in df.columns:
                df[c] = df_encoded[c]
                new_columns.append(c)
            elif c in cols_to_encode:
                new_col = f"{c}_te"
                df[new_col] = df_encoded[c]
                new_columns.append(new_col)

        return {"success": True, "new_columns": new_columns, "error": None}
    except Exception as e:
        return {"success": False, "new_columns": [], "error": str(e)}


def tool_automated_numeric_interactions(df: pd.DataFrame, num_cols: str) -> dict:
    """Create basic numeric interactions (multiplication, ratio) between top numeric columns."""
    try:
        import numpy as np

        cols = [c.strip() for c in num_cols.split(",") if c.strip() in df.columns]
        if len(cols) < 2:
            return {
                "success": False,
                "new_columns": [],
                "error": "Provide at least 2 valid numeric columns.",
            }

        new_columns = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c1, c2 = cols[i], cols[j]

                # Multiplication
                m_col = f"{c1}_mult_{c2}"
                df[m_col] = df[c1] * df[c2]
                new_columns.append(m_col)

                # Ratio (with stable epsilon)
                r_col = f"{c1}_div_{c2}"
                eps = 1e-6
                df[r_col] = df[c1] / (df[c2].fillna(0) + eps)
                new_columns.append(r_col)

                # Log of ratio (very common in financial/fit data)
                l_col = f"log_{c1}_div_{c2}"
                df[l_col] = np.log1p(df[c1].clip(0)) - np.log1p(df[c2].clip(0))
                new_columns.append(l_col)

        return {"success": True, "new_columns": new_columns, "error": None}
    except Exception as e:
        return {"success": False, "new_columns": [], "error": str(e)}


def tool_create_groupby_aggregations(
    df: pd.DataFrame, group_col: str, agg_cols: str, aggregations: str
) -> dict:
    """Create group-by aggregate features."""
    try:
        if group_col not in df.columns:
            return {
                "success": False,
                "new_columns": [],
                "error": f"Group column '{group_col}' not found.",
            }

        target_agg_cols = [
            c.strip() for c in agg_cols.split(",") if c.strip() in df.columns
        ]
        aggs = [a.strip() for a in aggregations.split(",")]

        new_columns = []
        for agg_col in target_agg_cols:
            grouped = df.groupby(group_col)[agg_col].agg(aggs)
            for agg in aggs:
                new_col = f"{agg_col}_{agg}_by_{group_col}"
                df[new_col] = df[group_col].map(grouped[agg])
                new_columns.append(new_col)

        return {"success": True, "new_columns": new_columns, "error": None}
    except Exception as e:
        return {"success": False, "new_columns": [], "error": str(e)}


def tool_inspect_residuals(
    df: pd.DataFrame,
    target_col: str,
    pred_col: str,
    task_type: str = "regression",
    n_rows: int = 20,
) -> str:
    """Inspect rows with the highest loss/residuals between target and predictions."""
    try:
        import numpy as np

        if target_col not in df.columns or pred_col not in df.columns:
            return "Target or prediction column not found in df."

        df_res = df.copy()
        if task_type == "regression":
            df_res["_residual_loss"] = (df_res[target_col] - df_res[pred_col]).abs()
        else:
            y = df_res[target_col].astype(float)
            p = df_res[pred_col].astype(float).clip(1e-15, 1 - 1e-15)
            df_res["_residual_loss"] = -(y * np.log(p) + (1 - y) * np.log(1 - p))

        highest_loss = df_res.sort_values("_residual_loss", ascending=False).head(
            n_rows
        )

        res = f"Top {n_rows} rows with highest residuals/loss:\n"
        res += highest_loss.to_string()
        return res
    except Exception as e:
        return f"Residual analysis failed: {e}"


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


def tool_apply_pseudo_labeling(
    df: pd.DataFrame,
    target_col: str,
    task_type: str = "binary",
    confidence_threshold: float = 0.9,
) -> str:
    """Implement pseudo-labeling for semi-supervised augmentation.
    Trains a model on labeled rows, predicts unlabeled rows, and assigns high-confidence predictions.
    """
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import LabelEncoder

    if target_col not in df.columns:
        return f"Error: '{target_col}' not found."

    labeled_mask = df[target_col].notna()
    if labeled_mask.all():
        return "No unlabeled rows (NaN targets) found for pseudo-labeling."

    df_labeled = df[labeled_mask].copy()
    df_unlabeled = df[~labeled_mask].copy()

    X_train = df_labeled.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y_train = df_labeled[target_col]
    X_unlabeled = df_unlabeled.drop(columns=[target_col]).select_dtypes(
        include=[np.number]
    )

    if X_train.empty:
        return "No numeric features available for pseudo-labeling model."

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_unlabeled_imp = imputer.transform(X_unlabeled)

    pseudo_labels_added = 0

    if task_type in ["binary", "multiclass"]:
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train_imp, y_train_enc)
        probs = model.predict_proba(X_unlabeled_imp)
        max_probs = np.max(probs, axis=1)
        preds = np.argmax(probs, axis=1)

        high_conf_mask = max_probs >= confidence_threshold
        pseudo_labels_added = int(high_conf_mask.sum())
        if pseudo_labels_added > 0:
            assigned_classes = le.inverse_transform(preds[high_conf_mask])
            df.loc[df_unlabeled.index[high_conf_mask], target_col] = assigned_classes

    else:
        model_reg = RandomForestRegressor(n_estimators=50, random_state=42)
        model_reg.fit(X_train_imp, y_train)
        preds = model_reg.predict(X_unlabeled_imp)
        df.loc[df_unlabeled.index, target_col] = preds
        pseudo_labels_added = len(preds)

    return f"Pseudo-labeling completed. Added {pseudo_labels_added} pseudo-labeled rows to the dataset."


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


def tool_build_and_run_pipeline(  # noqa: C901
    df: pd.DataFrame,
    target_col: str,
    config: Dict[str, Any],
    custom_preprocessor=None,
    ml_model=None,
    conf_training: Optional[Any] = None,
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

    if conf_training is not None:
        import copy

        training_config = copy.deepcopy(conf_training)
    else:
        training_config = TrainingConfig()

    training_config.hyperparameter_tuning_rounds = int(
        config.get("tuning_rounds", training_config.hyperparameter_tuning_rounds)
    )
    training_config.hyperparameter_tuning_max_runtime_secs = int(
        config.get(
            "tuning_max_runtime", training_config.hyperparameter_tuning_max_runtime_secs
        )
    )
    training_config.hypertuning_cv_folds = int(
        config.get("hypertuning_cv_folds", training_config.hypertuning_cv_folds)
    )
    if "autotune_on_device" in config:
        training_config.autotune_on_device = config.get("autotune_on_device")

    if "n_folds" in config or "n_repeats" in config:
        # Apply agent-recommended n_folds/n_repeats.  Always clamp n_folds
        # to >= 2 because sklearn's cross-validation requires at least 2 splits.
        n_folds = max(
            2,
            int(config.get("n_folds", training_config.bluecast_cv_train_n_model[0])),
        )
        n_repeats = int(
            config.get("n_repeats", training_config.bluecast_cv_train_n_model[1])
        )
        training_config.bluecast_cv_train_n_model = (n_folds, n_repeats)

    # Safety net: ensure n_folds >= 2 even when no agent override was provided
    if use_cv and training_config.bluecast_cv_train_n_model[0] < 2:
        training_config.bluecast_cv_train_n_model = (
            2,
            training_config.bluecast_cv_train_n_model[1],
        )

    # Apply standard fallbacks if not explicitly provided
    if "enable_feature_selection" in config:
        training_config.enable_feature_selection = bool(
            config.get("enable_feature_selection")
        )

    if "out_of_fold_dataset_store_path" in config:
        training_config.out_of_fold_dataset_store_path = config[
            "out_of_fold_dataset_store_path"
        ]

    if "cat_encoding_via_ml_algorithm" in config:
        training_config.cat_encoding_via_ml_algorithm = config[
            "cat_encoding_via_ml_algorithm"
        ]

    if "enable_feature_selection" in config:
        training_config.enable_feature_selection = config["enable_feature_selection"]

    ensemble_config = None
    if use_cv:
        strategy = config.get("ensemble_strategy", "hill_climbing")
        reg_metric = config.get("regression_eval_metric", "rmse")
        ensemble_config = EnsembleConfig(
            ensemble_strategy=strategy, regression_eval_metric=reg_metric
        )
        if strategy == "hill_climbing":
            ensemble_config.hc_weight_min = config.get("hc_weight_min", -0.3)
            ensemble_config.hc_weight_max = config.get("hc_weight_max", 0.5)
            ensemble_config.hc_weight_step = config.get("hc_weight_step", 0.01)
            if class_problem == "regression":
                ensemble_config.hc_blending_method = (
                    "probability"  # regression must use raw values
                )

    if class_problem == "regression" and ensemble_config:
        ensemble_config.stacking_use_ranks = False

    # Handle custom config overrides for regression metrics
    conf_tuning = None
    single_fold_eval_metric_func = None

    if class_problem == "regression" and config.get("regression_eval_metric"):
        from bluecast.ai.metrics import (
            get_bluecast_eval_wrapper,
            get_regression_metric_config,
        )
        from bluecast.config.training_config import CatboostTuneParamsRegressionConfig

        metric_name = str(config.get("regression_eval_metric", "mae"))
        metric_config = get_regression_metric_config(metric_name)

        conf_tuning = CatboostTuneParamsRegressionConfig()
        conf_tuning.catboost_loss_function = metric_config["catboost_loss"]
        conf_tuning.catboost_eval_metric = metric_config["catboost_loss"]

        single_fold_eval_metric_func = get_bluecast_eval_wrapper(metric_name)

    # Create a default CatBoost tuning config if none was provided and
    # apply mode-specific overrides (e.g. catboost_depth_max) from the
    # config dict.  This ensures non-ultimate modes use tighter search
    # spaces without requiring users to manually configure them.
    _has_catboost_overrides = any(k.startswith("catboost_") for k in config)
    if conf_tuning is None and _has_catboost_overrides:
        if class_problem == "regression":
            from bluecast.config.training_config import (
                CatboostTuneParamsRegressionConfig,
            )

            conf_tuning = CatboostTuneParamsRegressionConfig()
        else:
            from bluecast.config.training_config import CatboostTuneParamsConfig

            conf_tuning = CatboostTuneParamsConfig()

    if conf_tuning is not None and _has_catboost_overrides:
        for attr in [
            "depth_max",
            "border_count_max",
            "learning_rate_min",
            "iterations_max",
        ]:
            key = f"catboost_{attr}"
            if key in config:
                setattr(conf_tuning, attr, config[key])

    if ml_model is not None:
        # Inject tuning budget into custom models so they respect the
        # orchestrator's tuning_rounds / tuning_max_runtime settings.
        # Custom PyTorch models (RegularizedRegressionModel, MLPRegressionModel,
        # SO1DCNNRegressionModel) read from self.conf_tuning as a dict.
        if not hasattr(ml_model, "conf_tuning") or ml_model.conf_tuning is None:
            ml_model.conf_tuning = {}

        ml_model.conf_tuning.update(
            {
                "tuning_rounds": int(config.get("tuning_rounds", 15)),
                "tuning_max_runtime": int(config.get("tuning_max_runtime", 120)),
                "nn_max_iter": int(config.get("nn_max_iter", 200)),
            }
        )

        for k, v in config.items():
            if k.startswith(("nn_", "histgb_", "rf_")):
                ml_model.conf_tuning[k] = v

        if hasattr(ml_model, "cv_folds"):
            ml_model.cv_folds = int(
                config.get("hypertuning_cv_folds", training_config.hypertuning_cv_folds)
            )

        # Custom models do their own internal CV during autotune() and
        # completely ignore the x_test/y_test from cast_regression's inner
        # train/test split.  When BlueCastAI uses CV (outer fold provides
        # holdout), that inner split wastes ~20% of each fold's training data.
        # Setting train_size=0.99 effectively eliminates the waste without
        # modifying BlueCastCVRegression's default behaviour for other users.
        if use_cv:
            training_config.train_size = 0.99

    try:
        pipeline = BlueCastAuto(
            class_problem=class_problem,
            use_cross_validation=use_cv,
            conf_training=training_config,
            ensemble_config=ensemble_config,
            conf_tuning=conf_tuning,
            custom_preprocessor=custom_preprocessor,
            ml_model=ml_model,
        )

        if single_fold_eval_metric_func:
            # Inject dynamic metric if using default BlueCast pipeline for native models
            pipeline.single_fold_eval_metric_func = single_fold_eval_metric_func  # type: ignore

        if "columns_to_drop" in config and isinstance(config["columns_to_drop"], list):
            df = df.drop(columns=config["columns_to_drop"], errors="ignore")

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
        if isinstance(metrics, dict):
            ml_model_obj = getattr(pipeline._inner, "ml_model", None)
            if ml_model_obj is not None:
                if hasattr(ml_model_obj, "best_tuning_score_"):
                    metrics["tuning_score"] = ml_model_obj.best_tuning_score_
                elif hasattr(ml_model_obj, "best_score"):
                    metrics["tuning_score"] = ml_model_obj.best_score
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
    "check_adversarial_validation": ToolDefinition(
        name="check_adversarial_validation",
        description="Run adversarial validation between train and test distributions to detect covariate shift/data drift. Requires a df.query() condition to split the data into test vs train sets.",
        parameters={
            "type": "object",
            "properties": {
                "test_condition": {
                    "type": "string",
                    "description": "Pandas query condition defining the test set (e.g. 'is_test == 1' or 'date > 2023'). The remaining data becomes the train set.",
                },
            },
            "required": ["test_condition"],
        },
    ),
    "check_mutual_information": ToolDefinition(
        name="check_mutual_information",
        description="Calculate mutual information between features and target to assess predictive power prior to training.",
        parameters={
            "type": "object",
            "properties": {
                "target_col": {
                    "type": "string",
                    "description": "Target column name.",
                },
                "task_type": {
                    "type": "string",
                    "enum": ["classification", "regression"],
                    "description": "Type of prediction task.",
                },
            },
            "required": ["target_col", "task_type"],
        },
    ),
    "target_distribution_test": ToolDefinition(
        name="target_distribution_test",
        description="Test the target distribution for normality (Shapiro-Wilk) and check its skewness.",
        parameters={
            "type": "object",
            "properties": {
                "target_col": {
                    "type": "string",
                    "description": "Target column name.",
                },
            },
            "required": ["target_col"],
        },
    ),
    "nlp_profiling": ToolDefinition(
        name="nlp_profiling",
        description="Profile a text column in depth (character counts, word counts, vocabulary size).",
        parameters={
            "type": "object",
            "properties": {
                "text_col": {
                    "type": "string",
                    "description": "Text column name.",
                },
            },
            "required": ["text_col"],
        },
    ),
    "apply_target_encoding": ToolDefinition(
        name="apply_target_encoding",
        description="Apply Out-of-fold target encoding to specific categorical columns.",
        parameters={
            "type": "object",
            "properties": {
                "target_col": {
                    "type": "string",
                    "description": "Target column name for encoding.",
                },
                "cat_cols": {
                    "type": "string",
                    "description": "Comma-separated list of categorical columns to encode.",
                },
            },
            "required": ["target_col", "cat_cols"],
        },
    ),
    "automated_numeric_interactions": ToolDefinition(
        name="automated_numeric_interactions",
        description="Automatically create interaction features (multiplications and ratios) for a given list of numeric features.",
        parameters={
            "type": "object",
            "properties": {
                "num_cols": {
                    "type": "string",
                    "description": "Comma-separated list of numeric columns to interact.",
                },
            },
            "required": ["num_cols"],
        },
    ),
    "create_groupby_aggregations": ToolDefinition(
        name="create_groupby_aggregations",
        description="Create groupby aggregate features mapping a group to numeric aggregations.",
        parameters={
            "type": "object",
            "properties": {
                "group_col": {
                    "type": "string",
                    "description": "Categorical column to group by.",
                },
                "agg_cols": {
                    "type": "string",
                    "description": "Comma-separated list of numeric columns to aggregate.",
                },
                "aggregations": {
                    "type": "string",
                    "description": "Comma-separated list of pandas aggregations (e.g., 'mean,std,max,min').",
                },
            },
            "required": ["group_col", "agg_cols", "aggregations"],
        },
    ),
    "inspect_residuals": ToolDefinition(
        name="inspect_residuals",
        description="Inspect the highest loss/residual rows to see where the model fails most.",
        parameters={
            "type": "object",
            "properties": {
                "target_col": {
                    "type": "string",
                    "description": "Target column name.",
                },
                "pred_col": {
                    "type": "string",
                    "description": "Prediction column name inside the dataframe.",
                },
                "task_type": {
                    "type": "string",
                    "enum": ["classification", "regression"],
                    "description": "Task type affects how residuals are computed.",
                },
                "n_rows": {
                    "type": "integer",
                    "description": "Number of rows to display (default 20).",
                },
            },
            "required": ["target_col", "pred_col"],
        },
    ),
    "describe_data": ToolDefinition(
        name="describe_data",
        description="Generate a comprehensive profile of the dataset including dtypes, nulls, distributions, and target analysis.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    "evaluate_imputations": ToolDefinition(
        name="evaluate_imputations",
        description="Test imputation strategies for numerical columns with NaNs or sentinel values (999, -999, etc.). Auto-detects hidden missing data encoded as round-number sentinels. Tests 8 strategies (raw, mean, median, 0, static, 1st-percentile, 99th-percentile, mode) and returns per-column recommendations based on Mutual Information with the target.",  # noqa: E501
        parameters={
            "type": "object",
            "properties": {
                "fill_value": {
                    "type": "number",
                    "description": "Static value to use for static imputation. Default -999.0.",
                }
            },
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
        description="Check for simple correlation or categorical target leakage.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    "drop_collinear_features": ToolDefinition(
        name="drop_collinear_features",
        description="Identify highly correlated numerical features and drop one from each correlated pair. Essential for linear architectures.",
        parameters={
            "type": "object",
            "properties": {
                "threshold": {
                    "type": "number",
                    "description": "Pearson correlation threshold to drop pairs (e.g. 0.9).",
                }
            },
            "required": [],
        },
    ),
    "l1_feature_selection": ToolDefinition(
        name="l1_feature_selection",
        description="Perform Embedded L1 feature selection using Lasso/LogisticRegression to drop non-essential features natively for linear models.",
        parameters={
            "type": "object",
            "properties": {
                "alpha": {
                    "type": "number",
                    "description": "Regularization strength parameter. Higher drops more features (default: 0.01).",
                }
            },
            "required": [],
        },
    ),
    "create_feature": ToolDefinition(
        name="create_feature",
        description="Execute Python feature engineering code that modifies or re-assigns the DataFrame 'df'. "
        "The code has access to 'df', 'np', 'pd', and can import from 'bluecast.preprocessing'. "
        "Example: \nfrom bluecast.preprocessing.feature_creation import add_binned_features\ndf = add_binned_features(df, ['a'], state=state, is_fit=is_fit)",
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
    "check_feature_quality": ToolDefinition(
        name="check_feature_quality",
        description="Check the predictive quality of newly created features by computing their correlation and mutual information with the target. "
        "Returns a per-feature quality rating (HIGH/MEDIUM/LOW). Use this AFTER create_feature to validate whether new features are worth keeping. "
        "This is a lightweight check (<1 second) that does NOT train a model.",
        parameters={
            "type": "object",
            "properties": {
                "feature_cols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of feature column names to evaluate (typically the new_columns from a recent create_feature call).",
                },
            },
            "required": ["feature_cols"],
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
                "enable_feature_selection": {
                    "type": "boolean",
                    "description": "Enable automatic feature selection. Default false.",
                },
                "columns_to_drop": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of explicit column names to drop.",
                },
                "rf_max_depth_min": {
                    "type": "integer",
                    "description": "RandomForest Min Depth. Default 3.",
                },
                "rf_max_depth_max": {
                    "type": "integer",
                    "description": "RandomForest Max Depth. Default 15.",
                },
                "rf_estimators_min": {
                    "type": "integer",
                    "description": "RandomForest Min Estimators. Default 50.",
                },
                "rf_estimators_max": {
                    "type": "integer",
                    "description": "RandomForest Max Estimators. Default 300.",
                },
                "histgb_max_iter_max": {
                    "type": "integer",
                    "description": "HistGradientBoosting Max Iterations. Default 500.",
                },
                "histgb_depth_max": {
                    "type": "integer",
                    "description": "HistGradientBoosting Max Depth. Default 9.",
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
    "apply_pseudo_labeling": ToolDefinition(
        name="apply_pseudo_labeling",
        description="Implement pseudo-labeling for semi-supervised augmentation. Predicts NaN targets and fills them if confident.",
        parameters={
            "type": "object",
            "properties": {
                "task_type": {
                    "type": "string",
                    "description": "'binary', 'multiclass', or 'regression'. Default 'binary'.",
                },
                "confidence_threshold": {
                    "type": "number",
                    "description": "Threshold for assigning pseudo logic (e.g. 0.90).",
                },
            },
            "required": [],
        },
    ),
}
