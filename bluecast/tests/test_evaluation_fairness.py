import numpy as np
import pandas as pd

from bluecast.evaluation.fairness import FairnessAuditor


def test_fairness_classification():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])
    y_probs = np.array([0.1, 0.9, 0.2, 0.4, 0.3, 0.8, 0.6, 0.7])
    df_sensitive = pd.DataFrame({"gender": ["M", "F", "M", "F", "M", "F", "M", "F"]})

    auditor = FairnessAuditor(sensitive_columns=["gender"])
    reports = auditor.audit_classification(y_true, y_pred, y_probs, df_sensitive)

    assert len(reports) == 1
    report = reports[0]
    assert report.sensitive_column == "gender"
    assert "M" in report.group_metrics
    assert "F" in report.group_metrics

    # Check representations
    d = report.to_dict()
    assert "demographic_parity" in d

    summary = report.summary_df()
    assert "M" in summary.index

    ratios = report.ratios_df()
    assert not ratios.empty


def test_fairness_regression():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.5, 4.2])
    df_sensitive = pd.DataFrame({"age": ["young", "old", "young", "old"]})

    auditor = FairnessAuditor(sensitive_columns=["age"])
    reports = auditor.audit_regression(y_true, y_pred, df_sensitive)

    assert len(reports) == 1
    report = reports[0]
    assert report.sensitive_column == "age"
    assert "young" in report.group_metrics
    assert "old" in report.group_metrics


def test_missing_column():
    auditor = FairnessAuditor(sensitive_columns=["missing_col"])
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.1, 1.9])
    df_sensitive = pd.DataFrame({"age": ["young", "old"]})
    reports = auditor.audit_regression(y_true, y_pred, df_sensitive)
    assert len(reports) == 0


def test_plots():
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.1, 1.9])
    df_sensitive = pd.DataFrame({"age": ["young", "old"]})
    auditor = FairnessAuditor(sensitive_columns=["age"])
    reports = auditor.audit_regression(y_true, y_pred, df_sensitive)
    report = reports[0]

    # We just call them to ensure they don't crash if plotly is missing/present
    auditor.plot_fairness_dashboard(report)
    auditor.plot_ratio_heatmap(report)
