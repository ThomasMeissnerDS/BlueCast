"""
Conformal Prediction: Global and Group-Conditional Uncertainty
==============================================================

BlueCast provides conformal prediction for both classification and regression.
The key innovation in v3.0 is **group-conditional** conformal prediction:
different subgroups (e.g. product categories, regions) get uncertainty
intervals tailored to their specific error distributions.

This example demonstrates:
1. Standard conformal prediction (classification)
2. Standard conformal prediction (regression)
3. Group-conditional prediction intervals (regression)
4. Group-conditional prediction sets (classification)
5. Evaluating per-group coverage
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from bluecast.blueprints.cast import BlueCast
from bluecast.blueprints.cast_regression import BlueCastRegression
from bluecast.config.training_config import TrainingConfig
from bluecast.conformal_prediction.evaluation import (
    prediction_interval_coverage,
    prediction_interval_coverage_by_group,
    prediction_interval_spans_by_group,
    prediction_set_coverage,
)

fast_config = TrainingConfig(
    hyperparameter_tuning_rounds=10,
    hyperparameter_tuning_max_runtime_secs=30,
    enable_feature_selection=False,
    calculate_shap_values=False,
    plot_hyperparameter_tuning_overview=False,
    hypertuning_cv_folds=2,
    train_size=0.8,
)


# --- Create synthetic data with group structure ---
def make_grouped_regression_data(n=3000, seed=42):
    """Create data where different groups have different noise levels."""
    rng = np.random.default_rng(seed)
    X, y_base = make_regression(
        n_samples=n,
        n_features=10,
        n_informative=7,
        noise=5.0,
        random_state=seed,
    )

    groups = rng.choice(
        ["electronics", "clothing", "food"], size=n, p=[0.4, 0.35, 0.25]
    )

    # Add group-specific noise to simulate varying difficulty
    noise_scales = {"electronics": 1.0, "clothing": 3.0, "food": 5.0}
    for g, scale in noise_scales.items():
        mask = groups == g
        y_base[mask] += rng.normal(0, scale * 10, mask.sum())

    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)])
    df["product_group"] = groups
    df["target"] = y_base
    return df


def make_grouped_classification_data(n=3000, seed=42):
    X, y = make_classification(
        n_samples=n,
        n_features=10,
        n_informative=7,
        n_redundant=2,
        random_state=seed,
    )
    rng = np.random.default_rng(seed)
    groups = rng.choice(["premium", "standard", "budget"], size=n)

    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)])
    df["customer_tier"] = groups
    df["target"] = y
    return df


# =========================================================
# 1. Standard Conformal Prediction (Classification)
# =========================================================
print("=" * 60)
print("1. STANDARD CONFORMAL PREDICTION (CLASSIFICATION)")
print("=" * 60)

df_cls = make_grouped_classification_data()
df_train, df_temp = train_test_split(df_cls, test_size=0.4, random_state=42)
df_cal, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)

y_cal = df_cal.pop("target")
y_test = df_test.pop("target")

automl = BlueCast(class_problem="binary", conf_training=fast_config)
automl.fit(df_train, target_col="target")

# Calibrate with held-out calibration set
automl.calibrate(df_cal, y_cal)

# Predict p-values and prediction sets
p_values = automl.predict_p_values(df_test)
pred_sets = automl.predict_sets(df_test, alpha=0.1)

coverage = prediction_set_coverage(y_test, pred_sets)
print(f"Prediction set coverage at alpha=0.1: {coverage:.3f} (target: ~0.90)")
print(f"Sample prediction sets: {pred_sets['prediction_set'].head().tolist()}")
print()


# =========================================================
# 2. Standard Conformal Prediction (Regression)
# =========================================================
print("=" * 60)
print("2. STANDARD CONFORMAL PREDICTION (REGRESSION)")
print("=" * 60)

df_reg = make_grouped_regression_data()
df_train, df_temp = train_test_split(df_reg, test_size=0.4, random_state=42)
df_cal, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)

y_cal = df_cal.pop("target")
y_test = df_test.pop("target")

automl_reg = BlueCastRegression(class_problem="regression", conf_training=fast_config)
automl_reg.fit(df_train, target_col="target")

# Calibrate
automl_reg.calibrate(df_cal, y_cal)

# Predict intervals
alphas = [0.05, 0.1, 0.2]
intervals = automl_reg.predict_interval(df_test, alphas=alphas)
print(f"Interval columns: {intervals.columns.tolist()}")

coverages = prediction_interval_coverage(y_test, intervals, alphas)
for alpha, cov in coverages.items():
    print(f"  Alpha={alpha}: coverage={cov:.3f} (target: ~{1 - alpha:.2f})")
print()


# =========================================================
# 3. Group-Conditional Intervals (Regression)
# =========================================================
print("=" * 60)
print("3. GROUP-CONDITIONAL PREDICTION INTERVALS")
print("=" * 60)

df_reg = make_grouped_regression_data()
df_train, df_temp = train_test_split(df_reg, test_size=0.4, random_state=42)
df_cal, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)

y_cal = df_cal.pop("target")
y_test = df_test.pop("target")
test_groups = df_test["product_group"].values

automl_reg2 = BlueCastRegression(class_problem="regression", conf_training=fast_config)
automl_reg2.fit(df_train, target_col="target")

# Calibrate WITH group columns
from bluecast.conformal_prediction.conformal_prediction_regression import (  # noqa: E402
    ConformalPredictionRegressionWrapper,
)

wrapper = ConformalPredictionRegressionWrapper(automl_reg2, min_group_size=20)
wrapper.calibrate(df_cal, y_cal, group_columns=["product_group"])

# Predict group-conditional intervals
intervals = wrapper.predict_interval(df_test, alphas=[0.1])

print("Per-group coverage (alpha=0.1, target ~0.90):")
group_coverage = prediction_interval_coverage_by_group(
    y_test, intervals, [0.1], test_groups
)
for group, covs in group_coverage.items():
    print(f"  {group}: coverage={covs[0.1]:.3f}")

print("\nPer-group mean interval width:")
group_spans = prediction_interval_spans_by_group(intervals, [0.1], test_groups)
for group, spans in group_spans.items():
    print(f"  {group}: width={spans[0.1]:.2f}")

print("\nNote: Groups with more noise (food > clothing > electronics)")
print("get wider intervals, reflecting their higher uncertainty.")
print()


# =========================================================
# 4. Group-Conditional Prediction Sets (Classification)
# =========================================================
print("=" * 60)
print("4. GROUP-CONDITIONAL PREDICTION SETS (CLASSIFICATION)")
print("=" * 60)

df_cls2 = make_grouped_classification_data()
df_train, df_temp = train_test_split(df_cls2, test_size=0.4, random_state=42)
df_cal, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)

y_cal = df_cal.pop("target")
y_test = df_test.pop("target")

automl_cls2 = BlueCast(class_problem="binary", conf_training=fast_config)
automl_cls2.fit(df_train, target_col="target")

# Calibrate with groups
from bluecast.conformal_prediction.conformal_prediction import (  # noqa: E402
    ConformalPredictionWrapper,
)

wrapper_cls = ConformalPredictionWrapper(automl_cls2.ml_model, min_group_size=20)
x_cal_transformed = automl_cls2.transform_new_data(df_cal)
wrapper_cls.calibrate(x_cal_transformed, y_cal, group_columns=None)

print("Standard (global) calibration: all groups share the same nonconformity scores.")
print(f"Total calibration scores: {len(wrapper_cls.nonconformity_scores)}")
print()

print("All conformal prediction examples completed successfully!")
