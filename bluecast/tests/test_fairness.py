"""Tests for the fairness evaluation module."""

import numpy as np
import pandas as pd
import pytest

from bluecast.evaluation.fairness import (
    FairnessAuditor,
    FairnessReport,
    GroupMetrics,
    _safe_ratio,
)


@pytest.fixture
def binary_data():
    rng = np.random.default_rng(42)
    n = 500
    y_true = rng.choice([0, 1], size=n, p=[0.6, 0.4])
    y_pred = y_true.copy()
    noise_idx = rng.choice(n, size=50, replace=False)
    y_pred[noise_idx] = 1 - y_pred[noise_idx]
    y_probs = y_pred.astype(float) + rng.normal(0, 0.05, n)
    y_probs = np.clip(y_probs, 0, 1)
    df = pd.DataFrame(
        {
            "gender": rng.choice(["male", "female"], size=n),
            "age_group": rng.choice(["young", "middle", "old"], size=n),
        }
    )
    return y_true, y_pred, y_probs, df


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(42)
    n = 400
    y_true = rng.normal(100, 20, n)
    y_pred = y_true + rng.normal(0, 5, n)
    df = pd.DataFrame({"region": rng.choice(["urban", "rural"], size=n)})
    # Make rural predictions worse
    rural_mask = df["region"] == "rural"
    y_pred[rural_mask.values] += rng.normal(0, 15, rural_mask.sum())
    return y_true, y_pred, df


def test_safe_ratio():
    assert _safe_ratio(0.8, 0.4) == 2.0
    assert _safe_ratio(0.0, 0.5) == 0.0
    assert _safe_ratio(0.5, 0.0) == 0.0
    assert _safe_ratio(0.0, 0.0) == 0.0


def test_fairness_auditor_classification_single_column(binary_data):
    y_true, y_pred, y_probs, df = binary_data
    auditor = FairnessAuditor(sensitive_columns=["gender"])
    reports = auditor.audit_classification(y_true, y_pred, y_probs, df)
    assert len(reports) == 1
    report = reports[0]
    assert report.sensitive_column == "gender"
    assert len(report.group_metrics) == 2
    assert "male" in report.group_metrics or "female" in report.group_metrics


def test_fairness_auditor_classification_multi_column(binary_data):
    y_true, y_pred, y_probs, df = binary_data
    auditor = FairnessAuditor(sensitive_columns=["gender", "age_group"])
    reports = auditor.audit_classification(y_true, y_pred, y_probs, df)
    assert len(reports) == 2
    assert reports[0].sensitive_column == "gender"
    assert reports[1].sensitive_column == "age_group"
    assert len(reports[1].group_metrics) == 3


def test_fairness_auditor_classification_metrics(binary_data):
    y_true, y_pred, y_probs, df = binary_data
    auditor = FairnessAuditor(sensitive_columns=["gender"])
    reports = auditor.audit_classification(y_true, y_pred, y_probs, df)
    report = reports[0]

    for gm in report.group_metrics.values():
        assert 0 <= gm.positive_rate <= 1
        assert 0 <= gm.true_positive_rate <= 1
        assert 0 <= gm.false_positive_rate <= 1
        assert 0 <= gm.precision <= 1
        assert 0 <= gm.accuracy <= 1
        assert 0 <= gm.f1 <= 1
        assert gm.auc is None or 0 <= gm.auc <= 1
        assert gm.count > 0


def test_fairness_auditor_classification_ratios(binary_data):
    y_true, y_pred, y_probs, df = binary_data
    auditor = FairnessAuditor(sensitive_columns=["gender"])
    reports = auditor.audit_classification(y_true, y_pred, y_probs, df)
    report = reports[0]

    assert len(report.demographic_parity) > 0
    assert len(report.equalized_odds_tpr) > 0
    assert len(report.equalized_odds_fpr) > 0
    assert len(report.equal_opportunity) > 0
    assert len(report.predictive_parity) > 0
    assert len(report.auc_parity) > 0


def test_fairness_auditor_regression(regression_data):
    y_true, y_pred, df = regression_data
    auditor = FairnessAuditor(sensitive_columns=["region"])
    reports = auditor.audit_regression(y_true, y_pred, df)
    assert len(reports) == 1
    report = reports[0]
    assert report.sensitive_column == "region"
    assert len(report.group_metrics) == 2

    for gm in report.group_metrics.values():
        assert gm.mae is not None and gm.mae >= 0
        assert gm.rmse is not None and gm.rmse >= 0
        assert gm.r2 is not None
        assert gm.count > 0

    assert len(report.mae_ratio) > 0
    assert len(report.rmse_ratio) > 0


def test_fairness_auditor_no_probs(binary_data):
    y_true, y_pred, _, df = binary_data
    auditor = FairnessAuditor(sensitive_columns=["gender"])
    reports = auditor.audit_classification(y_true, y_pred, None, df)
    assert len(reports) == 1
    for gm in reports[0].group_metrics.values():
        assert gm.auc is None


def test_fairness_auditor_missing_column(binary_data):
    y_true, y_pred, y_probs, df = binary_data
    auditor = FairnessAuditor(sensitive_columns=["nonexistent"])
    reports = auditor.audit_classification(y_true, y_pred, y_probs, df)
    assert len(reports) == 0


def test_fairness_auditor_reference_group(binary_data):
    y_true, y_pred, y_probs, df = binary_data
    auditor = FairnessAuditor(
        sensitive_columns=["gender"], reference_group="female"
    )
    reports = auditor.audit_classification(y_true, y_pred, y_probs, df)
    report = reports[0]
    for key in report.demographic_parity:
        assert "female" in key


def test_fairness_report_to_dict(binary_data):
    y_true, y_pred, y_probs, df = binary_data
    auditor = FairnessAuditor(sensitive_columns=["gender"])
    reports = auditor.audit_classification(y_true, y_pred, y_probs, df)
    d = reports[0].to_dict()
    assert "sensitive_column" in d
    assert d["sensitive_column"] == "gender"
    assert "demographic_parity" in d


def test_fairness_report_summary_df(binary_data):
    y_true, y_pred, y_probs, df = binary_data
    auditor = FairnessAuditor(sensitive_columns=["gender"])
    reports = auditor.audit_classification(y_true, y_pred, y_probs, df)
    summary = reports[0].summary_df()
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == 2
    assert "positive_rate" in summary.columns


def test_fairness_report_ratios_df(binary_data):
    y_true, y_pred, y_probs, df = binary_data
    auditor = FairnessAuditor(sensitive_columns=["gender"])
    reports = auditor.audit_classification(y_true, y_pred, y_probs, df)
    rdf = reports[0].ratios_df()
    assert isinstance(rdf, pd.DataFrame)
    assert len(rdf) > 0


def test_fairness_report_repr(binary_data):
    y_true, y_pred, y_probs, df = binary_data
    auditor = FairnessAuditor(sensitive_columns=["gender"])
    reports = auditor.audit_classification(y_true, y_pred, y_probs, df)
    r = repr(reports[0])
    assert "gender" in r
    assert "FairnessReport" in r


def test_fairness_regression_report_ratios(regression_data):
    y_true, y_pred, df = regression_data
    auditor = FairnessAuditor(sensitive_columns=["region"])
    reports = auditor.audit_regression(y_true, y_pred, df)
    rdf = reports[0].ratios_df()
    assert "MAE_ratio" in rdf.index or "RMSE_ratio" in rdf.index
