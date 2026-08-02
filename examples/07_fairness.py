"""
Fairness Evaluation
====================

BlueCast provides tools to measure whether models treat different demographic
or categorical groups equitably. This example demonstrates:

1. Standalone FairnessAuditor for classification
2. Standalone FairnessAuditor for regression
3. Automatic fairness evaluation via fit_eval (TrainingConfig integration)
4. Conformal prediction fairness check (equal coverage across groups)
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from bluecast.blueprints.cast import BlueCast
from bluecast.blueprints.cast_regression import BlueCastRegression
from bluecast.config.training_config import TrainingConfig
from bluecast.evaluation.fairness import FairnessAuditor

fast_config = TrainingConfig(
    hyperparameter_tuning_rounds=10,
    hyperparameter_tuning_max_runtime_secs=30,
    enable_feature_selection=False,
    calculate_shap_values=False,
    plot_hyperparameter_tuning_overview=False,
    hypertuning_cv_folds=2,
    train_size=0.8,
)


def make_biased_classification_data(n=3000, seed=42):
    """Create data where the model might behave differently across groups."""
    rng = np.random.default_rng(seed)
    X, y = make_classification(
        n_samples=n,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        random_state=seed,
        flip_y=0.1,
    )
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)])

    # Sensitive attributes
    df["gender"] = rng.choice(["male", "female"], size=n, p=[0.6, 0.4])
    df["age_group"] = rng.choice(["18-30", "31-50", "51+"], size=n, p=[0.3, 0.45, 0.25])

    # Inject subtle bias: shift feature distributions by group
    female_mask = df["gender"] == "female"
    df.loc[female_mask, "feat_0"] += 0.5
    df.loc[df["age_group"] == "51+", "feat_1"] -= 0.3

    df["target"] = y
    return df


def make_biased_regression_data(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    X, y = make_regression(
        n_samples=n,
        n_features=10,
        n_informative=6,
        noise=10.0,
        random_state=seed,
    )
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)])
    df["region"] = rng.choice(["urban", "suburban", "rural"], size=n)

    # Different noise levels per region (model will be less accurate for rural)
    for region, noise_scale in [("urban", 1), ("suburban", 2), ("rural", 5)]:
        mask = df["region"] == region
        y[mask] += rng.normal(0, noise_scale * 15, mask.sum())

    df["target"] = y
    return df


# =========================================================
# 1. Standalone Fairness Audit (Classification)
# =========================================================
print("=" * 60)
print("1. STANDALONE FAIRNESS AUDIT (CLASSIFICATION)")
print("=" * 60)

df = make_biased_classification_data()
df_train, df_eval = train_test_split(df, test_size=0.3, random_state=42)
y_eval = df_eval.pop("target")

automl = BlueCast(class_problem="binary", conf_training=fast_config)
automl.fit(df_train, target_col="target")
y_probs, y_classes = automl.predict(df_eval)

auditor = FairnessAuditor(sensitive_columns=["gender", "age_group"])
reports = auditor.audit_classification(y_eval, y_classes, y_probs, df_eval)

for report in reports:
    print(f"\n--- {report.sensitive_column} ---")
    print(report.summary_df())
    print("\nFairness ratios (1.0 = perfect parity):")
    rdf = report.ratios_df()
    if not rdf.empty:
        print(rdf.to_string())
print()


# =========================================================
# 2. Standalone Fairness Audit (Regression)
# =========================================================
print("=" * 60)
print("2. STANDALONE FAIRNESS AUDIT (REGRESSION)")
print("=" * 60)

df_reg = make_biased_regression_data()
df_train, df_eval = train_test_split(df_reg, test_size=0.3, random_state=42)
y_eval = df_eval.pop("target")

automl_reg = BlueCastRegression(class_problem="regression", conf_training=fast_config)
automl_reg.fit(df_train, target_col="target")
y_preds = automl_reg.predict(df_eval)

auditor_reg = FairnessAuditor(sensitive_columns=["region"])
reports_reg = auditor_reg.audit_regression(y_eval, y_preds, df_eval)

for report in reports_reg:
    print(f"\n--- {report.sensitive_column} ---")
    print(report.summary_df())
    print("\nError ratios (1.0 = equal error across groups):")
    rdf = report.ratios_df()
    if not rdf.empty:
        print(rdf.to_string())
print()


# =========================================================
# 3. Automatic Fairness via fit_eval
# =========================================================
print("=" * 60)
print("3. AUTOMATIC FAIRNESS VIA FIT_EVAL")
print("=" * 60)

config_with_fairness = TrainingConfig(
    hyperparameter_tuning_rounds=10,
    hyperparameter_tuning_max_runtime_secs=30,
    enable_feature_selection=False,
    calculate_shap_values=False,
    plot_hyperparameter_tuning_overview=False,
    hypertuning_cv_folds=2,
    fairness_sensitive_columns=["gender", "age_group"],  # just set this!
)

df = make_biased_classification_data()
df_train, df_eval = train_test_split(df, test_size=0.3, random_state=42)
y_eval = df_eval.pop("target")

automl_fair = BlueCast(class_problem="binary", conf_training=config_with_fairness)
metrics = automl_fair.fit_eval(df_train, df_eval, y_eval, target_col="target")

print(f"Standard metrics: ROC AUC = {metrics.get('roc_auc', 'N/A'):.4f}")
if "fairness" in metrics:
    print(f"\nFairness reports included: {len(metrics['fairness'])} attribute(s)")
    for fr in metrics["fairness"]:
        col = fr["sensitive_column"]
        dp = fr.get("demographic_parity", {})
        print(f"  {col}:")
        for pair, ratio in dp.items():
            flag = " <-- FLAGGED" if 0 < ratio < 0.8 else ""
            print(f"    demographic_parity {pair}: {ratio:.3f}{flag}")
print()


# =========================================================
# 4. Conformal Prediction Fairness Check
# =========================================================
print("=" * 60)
print("4. CONFORMAL PREDICTION FAIRNESS CHECK")
print("=" * 60)

from bluecast.conformal_prediction.conformal_prediction_regression import (  # noqa: E402
    ConformalPredictionRegressionWrapper,
)
from bluecast.conformal_prediction.evaluation import (  # noqa: E402
    conformal_fairness_check,
    prediction_interval_coverage_by_group,
)

df_reg = make_biased_regression_data()
df_train, df_temp = train_test_split(df_reg, test_size=0.4, random_state=42)
df_cal, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)

y_cal = df_cal.pop("target")
y_test = df_test.pop("target")

automl_cp = BlueCastRegression(class_problem="regression", conf_training=fast_config)
automl_cp.fit(df_train, target_col="target")

# Calibrate with group-conditional conformal prediction
wrapper = ConformalPredictionRegressionWrapper(automl_cp, min_group_size=20)
wrapper.calibrate(df_cal, y_cal, group_columns=["region"])

# Predict group-conditional intervals
intervals = wrapper.predict_interval(df_test, alphas=[0.1])

# Check fairness of coverage across groups
test_groups = df_test["region"].values
group_coverages = prediction_interval_coverage_by_group(
    y_test, intervals, [0.1], test_groups
)

fairness_result = conformal_fairness_check(
    group_coverages, target_coverage=0.9, tolerance=0.05
)

print(f"Coverage target: {fairness_result['target_coverage']}")
print(f"Coverage range: {fairness_result['coverage_range']}")
print(f"All groups fair: {fairness_result['is_fair']}")
print("\nPer-group results:")
for group, info in fairness_result["group_results"].items():
    cov = info.get("alpha_0.1_coverage", 0)
    fair = info.get("alpha_0.1_fair", False)
    status = "PASS" if fair else "FAIL"
    print(f"  {group}: coverage={cov:.3f} [{status}]")
print()

print("All fairness examples completed successfully!")
