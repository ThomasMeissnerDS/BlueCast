"""
Fairness evaluation for machine learning models.

Provides metrics to detect and quantify bias across sensitive groups
(e.g. gender, age group, ethnicity). Works standalone or integrated
into the BlueCast fit_eval pipeline.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


@dataclass
class GroupMetrics:
    """Performance metrics for a single group."""

    group_name: Any
    count: int
    positive_rate: float = 0.0
    true_positive_rate: float = 0.0
    false_positive_rate: float = 0.0
    precision: float = 0.0
    accuracy: float = 0.0
    f1: float = 0.0
    auc: Optional[float] = None
    # Regression-specific
    mae: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None


@dataclass
class FairnessReport:
    """Structured fairness audit report.

    Contains per-group metrics, pairwise ratios, and an overall summary.
    Ratios close to 1.0 indicate fairness; deviation indicates bias.
    The commonly used "four-fifths rule" flags ratios below 0.8.
    """

    sensitive_column: str
    group_metrics: Dict[Any, GroupMetrics] = field(default_factory=dict)
    demographic_parity: Dict[str, float] = field(default_factory=dict)
    equalized_odds_tpr: Dict[str, float] = field(default_factory=dict)
    equalized_odds_fpr: Dict[str, float] = field(default_factory=dict)
    equal_opportunity: Dict[str, float] = field(default_factory=dict)
    predictive_parity: Dict[str, float] = field(default_factory=dict)
    auc_parity: Dict[str, float] = field(default_factory=dict)
    # Regression-specific
    mae_ratio: Dict[str, float] = field(default_factory=dict)
    rmse_ratio: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to a flat dictionary for logging or storage."""
        result: Dict[str, Any] = {"sensitive_column": self.sensitive_column}
        for name, group in self.group_metrics.items():
            prefix = f"group_{name}"
            result[f"{prefix}_count"] = group.count
            result[f"{prefix}_positive_rate"] = group.positive_rate
            result[f"{prefix}_tpr"] = group.true_positive_rate
            result[f"{prefix}_fpr"] = group.false_positive_rate
            result[f"{prefix}_precision"] = group.precision
            result[f"{prefix}_accuracy"] = group.accuracy
            result[f"{prefix}_f1"] = group.f1
            if group.auc is not None:
                result[f"{prefix}_auc"] = group.auc
            if group.mae is not None:
                result[f"{prefix}_mae"] = group.mae
            if group.rmse is not None:
                result[f"{prefix}_rmse"] = group.rmse

        result["demographic_parity"] = self.demographic_parity
        result["equalized_odds_tpr"] = self.equalized_odds_tpr
        result["equalized_odds_fpr"] = self.equalized_odds_fpr
        result["equal_opportunity"] = self.equal_opportunity
        result["predictive_parity"] = self.predictive_parity
        result["auc_parity"] = self.auc_parity
        result["mae_ratio"] = self.mae_ratio
        result["rmse_ratio"] = self.rmse_ratio
        return result

    def summary_df(self) -> pd.DataFrame:
        """Return a DataFrame summarising per-group metrics."""
        rows = []
        for name, gm in self.group_metrics.items():
            row = {
                "group": name,
                "count": gm.count,
                "positive_rate": round(gm.positive_rate, 4),
                "TPR": round(gm.true_positive_rate, 4),
                "FPR": round(gm.false_positive_rate, 4),
                "precision": round(gm.precision, 4),
                "accuracy": round(gm.accuracy, 4),
                "F1": round(gm.f1, 4),
            }
            if gm.auc is not None:
                row["AUC"] = round(gm.auc, 4)
            if gm.mae is not None:
                row["MAE"] = round(gm.mae, 4)
            if gm.rmse is not None:
                row["RMSE"] = round(gm.rmse, 4)
            if gm.r2 is not None:
                row["R2"] = round(gm.r2, 4)
            rows.append(row)
        return pd.DataFrame(rows).set_index("group")

    def ratios_df(self) -> pd.DataFrame:
        """Return a DataFrame of all pairwise fairness ratios."""
        all_ratios: Dict[str, Dict[str, float]] = {}
        for label, ratios in [
            ("demographic_parity", self.demographic_parity),
            ("equal_opportunity (TPR)", self.equal_opportunity),
            ("equalized_odds_TPR", self.equalized_odds_tpr),
            ("equalized_odds_FPR", self.equalized_odds_fpr),
            ("predictive_parity", self.predictive_parity),
            ("AUC_parity", self.auc_parity),
            ("MAE_ratio", self.mae_ratio),
            ("RMSE_ratio", self.rmse_ratio),
        ]:
            if ratios:
                all_ratios[label] = ratios
        if not all_ratios:
            return pd.DataFrame()
        return pd.DataFrame(all_ratios).T

    def __repr__(self) -> str:
        lines = [f"FairnessReport(column='{self.sensitive_column}')"]
        lines.append("\nPer-group metrics:")
        lines.append(self.summary_df().to_string())
        rdf = self.ratios_df()
        if not rdf.empty:
            lines.append("\nFairness ratios (1.0 = perfect parity):")
            lines.append(rdf.to_string())
        return "\n".join(lines)


def _safe_ratio(a: float, b: float) -> float:
    """Compute a/b, returning 0.0 when either value is zero."""
    if b == 0 or a == 0:
        return 0.0
    return a / b


class FairnessAuditor:
    """Audit model fairness across one or more sensitive attributes.

    Computes standard fairness metrics (demographic parity, equalized odds,
    equal opportunity, predictive parity, AUC parity) for classification, and
    MAE/RMSE ratios for regression. All ratios are pairwise against a reference
    group; ratios close to 1.0 indicate equity.

    :param sensitive_columns: List of column names in the evaluation DataFrame
        that identify sensitive groups (e.g. ["gender", "age_group"]).
    :param reference_group: Value within the sensitive column to use as the
        reference (denominator) for ratio calculations. If None, the group
        with the most samples is used.
    :param four_fifths_threshold: Threshold below which a ratio is flagged
        as potentially unfair (default: 0.8, the "four-fifths rule").

    Usage::

        from bluecast.evaluation.fairness import FairnessAuditor

        auditor = FairnessAuditor(sensitive_columns=["gender"])
        report = auditor.audit_classification(
            y_true, y_pred_classes, y_pred_probs, df_eval
        )
        print(report)
        auditor.plot_fairness_dashboard(report)
    """

    def __init__(
        self,
        sensitive_columns: List[str],
        reference_group: Optional[Any] = None,
        four_fifths_threshold: float = 0.8,
    ):
        self.sensitive_columns = sensitive_columns
        self.reference_group = reference_group
        self.four_fifths_threshold = four_fifths_threshold

    def _resolve_reference(self, groups: pd.Series) -> Any:
        """Determine the reference group (largest group if not specified)."""
        if self.reference_group is not None:
            return self.reference_group
        return groups.value_counts().idxmax()

    def _compute_classification_group_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: Optional[np.ndarray],
        mask: np.ndarray,
        group_name: Any,
    ) -> GroupMetrics:
        yt = y_true[mask]
        yp = y_pred[mask]

        n = int(mask.sum())
        positive_rate = float(yp.mean()) if n > 0 else 0.0

        positives = yt == 1
        negatives = yt == 0

        tp = int(((yp == 1) & positives).sum())
        fp = int(((yp == 1) & negatives).sum())
        fn = int(((yp == 0) & positives).sum())
        tn = int(((yp == 0) & negatives).sum())

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        prec = precision_score(yt, yp, zero_division=0)
        acc = accuracy_score(yt, yp)
        f1 = f1_score(yt, yp, zero_division=0)

        auc = None
        if y_probs is not None and len(np.unique(yt)) > 1:
            try:
                probs_group = y_probs[mask]
                auc = roc_auc_score(yt, probs_group)
            except ValueError:
                auc = None

        return GroupMetrics(
            group_name=group_name,
            count=n,
            positive_rate=positive_rate,
            true_positive_rate=tpr,
            false_positive_rate=fpr,
            precision=prec,
            accuracy=acc,
            f1=f1,
            auc=auc,
        )

    def _compute_regression_group_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        mask: np.ndarray,
        group_name: Any,
    ) -> GroupMetrics:
        yt = y_true[mask]
        yp = y_pred[mask]
        n = int(mask.sum())

        mae = float(mean_absolute_error(yt, yp)) if n > 0 else 0.0
        rmse = float(np.sqrt(mean_squared_error(yt, yp))) if n > 0 else 0.0
        r2 = float(r2_score(yt, yp)) if n > 1 else 0.0

        return GroupMetrics(
            group_name=group_name,
            count=n,
            mae=mae,
            rmse=rmse,
            r2=r2,
        )

    def _compute_pairwise_ratios(
        self,
        group_metrics: Dict[Any, GroupMetrics],
        reference: Any,
        attr: str,
    ) -> Dict[str, float]:
        """Compute ratio of each group's metric to the reference group's metric."""
        ref_val = getattr(group_metrics[reference], attr, 0.0) or 0.0
        ratios = {}
        for name, gm in group_metrics.items():
            if name == reference:
                continue
            val = getattr(gm, attr, 0.0) or 0.0
            ratios[f"{name}_vs_{reference}"] = _safe_ratio(val, ref_val)
        return ratios

    def audit_classification(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred_classes: Union[np.ndarray, pd.Series],
        y_pred_probs: Optional[Union[np.ndarray, pd.Series]],
        df_sensitive: pd.DataFrame,
    ) -> List[FairnessReport]:
        """Run a full classification fairness audit.

        :param y_true: True binary labels (0/1).
        :param y_pred_classes: Predicted binary labels.
        :param y_pred_probs: Predicted probabilities for the positive class (optional).
        :param df_sensitive: DataFrame containing the sensitive columns.
        :returns: List of FairnessReport, one per sensitive column.
        """
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred_classes).ravel()
        y_probs = np.asarray(y_pred_probs).ravel() if y_pred_probs is not None else None

        reports = []
        for col in self.sensitive_columns:
            if col not in df_sensitive.columns:
                logger.warning(f"Sensitive column '{col}' not found in DataFrame, skipping.")
                continue

            groups = df_sensitive[col]
            reference = self._resolve_reference(groups)
            unique_groups = groups.unique()

            group_metrics: Dict[Any, GroupMetrics] = {}
            for g in unique_groups:
                mask = (groups == g).values
                group_metrics[g] = self._compute_classification_group_metrics(
                    y_true, y_pred, y_probs, mask, g
                )

            report = FairnessReport(
                sensitive_column=col,
                group_metrics=group_metrics,
                demographic_parity=self._compute_pairwise_ratios(
                    group_metrics, reference, "positive_rate"
                ),
                equalized_odds_tpr=self._compute_pairwise_ratios(
                    group_metrics, reference, "true_positive_rate"
                ),
                equalized_odds_fpr=self._compute_pairwise_ratios(
                    group_metrics, reference, "false_positive_rate"
                ),
                equal_opportunity=self._compute_pairwise_ratios(
                    group_metrics, reference, "true_positive_rate"
                ),
                predictive_parity=self._compute_pairwise_ratios(
                    group_metrics, reference, "precision"
                ),
                auc_parity=self._compute_pairwise_ratios(
                    group_metrics, reference, "auc"
                ),
            )

            self._log_flagged_ratios(report)
            reports.append(report)

        return reports

    def audit_regression(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        df_sensitive: pd.DataFrame,
    ) -> List[FairnessReport]:
        """Run a regression fairness audit.

        :param y_true: True target values.
        :param y_pred: Predicted values.
        :param df_sensitive: DataFrame containing the sensitive columns.
        :returns: List of FairnessReport, one per sensitive column.
        """
        y_true = np.asarray(y_true).ravel()
        y_pred_arr = np.asarray(y_pred).ravel()

        reports = []
        for col in self.sensitive_columns:
            if col not in df_sensitive.columns:
                logger.warning(f"Sensitive column '{col}' not found in DataFrame, skipping.")
                continue

            groups = df_sensitive[col]
            reference = self._resolve_reference(groups)
            unique_groups = groups.unique()

            group_metrics: Dict[Any, GroupMetrics] = {}
            for g in unique_groups:
                mask = (groups == g).values
                group_metrics[g] = self._compute_regression_group_metrics(
                    y_true, y_pred_arr, mask, g
                )

            report = FairnessReport(
                sensitive_column=col,
                group_metrics=group_metrics,
                mae_ratio=self._compute_pairwise_ratios(
                    group_metrics, reference, "mae"
                ),
                rmse_ratio=self._compute_pairwise_ratios(
                    group_metrics, reference, "rmse"
                ),
            )
            reports.append(report)

        return reports

    def _log_flagged_ratios(self, report: FairnessReport) -> None:
        """Log ratios that fall below the four-fifths threshold."""
        for label, ratios in [
            ("demographic_parity", report.demographic_parity),
            ("equal_opportunity", report.equal_opportunity),
            ("predictive_parity", report.predictive_parity),
            ("AUC_parity", report.auc_parity),
        ]:
            for pair, ratio in ratios.items():
                if 0 < ratio < self.four_fifths_threshold:
                    logger.warning(
                        f"Fairness flag [{report.sensitive_column}]: "
                        f"{label} ratio {pair} = {ratio:.3f} "
                        f"(below {self.four_fifths_threshold} threshold)"
                    )

    def plot_fairness_dashboard(
        self,
        report: FairnessReport,
        figsize: tuple = (14, 8),
    ) -> None:
        """Plot an interactive fairness dashboard using plotly.

        :param report: A FairnessReport from audit_classification or audit_regression.
        :param figsize: Tuple of (width, height) in pixels (approximate).
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            logger.warning("plotly is required for fairness dashboard. Install with: pip install plotly")
            return

        summary = report.summary_df()
        groups = summary.index.tolist()

        is_regression = "MAE" in summary.columns

        if is_regression:
            metric_cols = [c for c in ["MAE", "RMSE", "R2"] if c in summary.columns]
            n_metrics = len(metric_cols)
            fig = make_subplots(
                rows=1, cols=n_metrics,
                subplot_titles=metric_cols,
            )
            for i, metric in enumerate(metric_cols, 1):
                fig.add_trace(
                    go.Bar(
                        x=groups,
                        y=summary[metric].values,
                        name=metric,
                        text=[f"{v:.4f}" for v in summary[metric].values],
                        textposition="auto",
                    ),
                    row=1, col=i,
                )
        else:
            metric_cols = ["positive_rate", "TPR", "FPR", "precision", "F1"]
            if "AUC" in summary.columns:
                metric_cols.append("AUC")

            n_metrics = len(metric_cols)
            n_cols = min(3, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=metric_cols,
            )
            for idx, metric in enumerate(metric_cols):
                r = idx // n_cols + 1
                c = idx % n_cols + 1
                fig.add_trace(
                    go.Bar(
                        x=groups,
                        y=summary[metric].values,
                        name=metric,
                        text=[f"{v:.4f}" for v in summary[metric].values],
                        textposition="auto",
                    ),
                    row=r, col=c,
                )

        fig.update_layout(
            title_text=f"Fairness Dashboard: {report.sensitive_column}",
            showlegend=False,
            width=figsize[0] * 70,
            height=figsize[1] * 70,
        )
        fig.show()

    def plot_ratio_heatmap(self, report: FairnessReport) -> None:
        """Plot a heatmap of fairness ratios with the four-fifths threshold line.

        :param report: A FairnessReport from audit_classification or audit_regression.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.warning("plotly is required for ratio heatmap.")
            return

        rdf = report.ratios_df()
        if rdf.empty:
            logger.info("No ratios to plot.")
            return

        fig = go.Figure(
            data=go.Heatmap(
                z=rdf.values,
                x=rdf.columns.tolist(),
                y=rdf.index.tolist(),
                colorscale="RdYlGn",
                zmin=0,
                zmax=1.5,
                text=np.round(rdf.values, 3).astype(str),
                texttemplate="%{text}",
                colorbar_title="Ratio",
            )
        )
        fig.update_layout(
            title=f"Fairness Ratios: {report.sensitive_column} (1.0 = parity)",
            xaxis_title="Group Pair",
            yaxis_title="Metric",
        )
        fig.show()
