"""
BlueCast Full Showcase: End-to-End AutoML Workflow
===================================================

This single script walks through BlueCast's complete feature set on a
synthetic credit-risk dataset, from raw data to production-ready model:

    1. EDA & data quality
    2. Unified interface (BlueCastAuto)
    3. Fit & evaluate (holdout + CV)
    4. Ensemble strategies (stacking, hill climbing)
    5. Conformal prediction (global + group-conditional)
    6. Fairness auditing
    7. Linear model baseline
    8. Experiment tracking
    9. Save & load pipeline
   10. Data drift monitoring
"""

import logging
import os
import tempfile

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from bluecast.config.training_config import TrainingConfig

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Synthetic dataset: credit risk with realistic structure
# ---------------------------------------------------------------------------


def make_credit_data(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    X, y = make_classification(
        n_samples=n,
        n_features=10,
        n_informative=7,
        n_redundant=2,
        random_state=seed,
        flip_y=0.08,
    )
    df = pd.DataFrame(
        X,
        columns=[
            "income",
            "credit_score",
            "age",
            "debt_ratio",
            "num_accounts",
            "utilization",
            "payment_history",
            "inquiries",
            "credit_age",
            "monthly_balance",
        ],
    )

    df["income"] = (df["income"] * 15000 + 55000).clip(15000, 250000).round(0)
    df["credit_score"] = (df["credit_score"] * 50 + 700).clip(300, 850).round(0)
    df["age"] = (df["age"] * 10 + 42).clip(18, 80).round(0)
    df["debt_ratio"] = (df["debt_ratio"] * 0.12 + 0.35).clip(0, 1).round(3)

    df["employment"] = rng.choice(
        ["employed", "self_employed", "unemployed", "retired"],
        size=n,
        p=[0.55, 0.25, 0.1, 0.1],
    )
    df["region"] = rng.choice(["north", "south", "east", "west"], size=n)
    df["gender"] = rng.choice(["male", "female"], size=n, p=[0.55, 0.45])

    for col in ["income", "credit_score"]:
        mask = rng.random(n) < 0.03
        df.loc[mask, col] = np.nan

    df["default"] = y
    return df


fast_config = TrainingConfig(
    hyperparameter_tuning_rounds=10,
    hyperparameter_tuning_max_runtime_secs=30,
    enable_feature_selection=False,
    calculate_shap_values=False,
    plot_hyperparameter_tuning_overview=False,
    hypertuning_cv_folds=2,
    train_size=0.8,
)


print("=" * 70)
print(" BlueCast Full Showcase")
print("=" * 70)

df = make_credit_data()
print(f"\nDataset: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Target ('default'): {df['default'].value_counts().to_dict()}\n")


# =====================================================================
# 1. EDA & Data Quality
# =====================================================================
print("-" * 70)
print("1. EDA & DATA QUALITY")
print("-" * 70)

from bluecast.eda import (  # noqa: E402
    correlation_to_target,
    detect_leakage_via_correlation,
    mutual_info_to_target,
    plot_null_percentage,
    univariate_plots,
)

num_cols = df.select_dtypes(include="number").columns.drop("default").tolist()

figs = univariate_plots(df[num_cols], show=False)
print(f"  Univariate plots generated: {len(figs)}")

fig_nulls = plot_null_percentage(df, show=False)
print(f"  Null percentage plot: OK ({df.isnull().sum().sum()} total nulls)")

fig_corr = correlation_to_target(
    df[num_cols + ["default"]], target="default", show=False
)
print("  Correlation-to-target plot: OK")

fig_mi = mutual_info_to_target(
    df[num_cols + ["default"]].dropna(),
    target="default",
    class_problem="binary",
    show=False,
)
print("  Mutual information plot: OK")

leaky = detect_leakage_via_correlation(
    df[num_cols + ["default"]], "default", threshold=0.95
)
print(f"  Leakage check: {'NONE detected' if not leaky else leaky}")


# =====================================================================
# 2. Unified Interface — Single-Model Binary Classification
# =====================================================================
print("\n" + "-" * 70)
print("2. UNIFIED INTERFACE (BlueCastAuto)")
print("-" * 70)

from bluecast.blueprints.unified import BlueCastAuto  # noqa: E402

df_train, df_test = train_test_split(df, test_size=0.25, random_state=42)
y_test = df_test.pop("default")

automl = BlueCastAuto(
    class_problem="binary",
    use_cross_validation=False,
    conf_training=fast_config,
)

metrics = automl.fit_eval(
    df_train,
    target_col="default",
    df_eval=df_test,
    y_eval=y_test,
)
print(f"  ROC AUC:  {metrics.get('roc_auc', 'N/A'):.4f}")  # type: ignore[union-attr]
print(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")  # type: ignore[union-attr]
print(f"  F1 (weighted): {metrics.get('f1_score_weighted', 'N/A'):.4f}")  # type: ignore[union-attr]


# =====================================================================
# 3. Cross-Validation with Mean Blending
# =====================================================================
print("\n" + "-" * 70)
print("3. CROSS-VALIDATION (MEAN BLENDING)")
print("-" * 70)

from bluecast.ensemble.ensemble_config import EnsembleConfig  # noqa: E402

cv_config = TrainingConfig(
    hyperparameter_tuning_rounds=10,
    hyperparameter_tuning_max_runtime_secs=30,
    enable_feature_selection=False,
    calculate_shap_values=False,
    plot_hyperparameter_tuning_overview=False,
    hypertuning_cv_folds=2,
    train_size=0.8,
    bluecast_cv_train_n_model=(3, 1),
)

automl_cv = BlueCastAuto(
    class_problem="binary",
    use_cross_validation=True,
    conf_training=cv_config,
    ensemble_config=EnsembleConfig(ensemble_strategy="mean"),
)
oof_mean, oof_std = automl_cv.fit_eval(df, target_col="default")
print(f"  OOF Matthews: {oof_mean:.4f} +/- {oof_std:.4f}")

y_probs, y_classes = automl_cv.predict(df_test)
print(f"  Test predictions: {y_probs.shape[0]} samples")


# =====================================================================
# 4. Stacking Ensemble
# =====================================================================
print("\n" + "-" * 70)
print("4. STACKING ENSEMBLE")
print("-" * 70)

automl_stack = BlueCastAuto(
    class_problem="binary",
    use_cross_validation=True,
    conf_training=cv_config,
    ensemble_config=EnsembleConfig(
        ensemble_strategy="stacking", stacking_use_ranks=True
    ),
)
oof_mean, oof_std = automl_stack.fit_eval(df, target_col="default")
print(f"  OOF Matthews (stacking): {oof_mean:.4f} +/- {oof_std:.4f}")


# =====================================================================
# 5. Hill Climbing Ensemble
# =====================================================================
print("\n" + "-" * 70)
print("5. HILL CLIMBING ENSEMBLE")
print("-" * 70)

automl_hc = BlueCastAuto(
    class_problem="binary",
    use_cross_validation=True,
    conf_training=cv_config,
    ensemble_config=EnsembleConfig(
        ensemble_strategy="hill_climbing",
        hc_blending_method="rank",
        hc_weight_min=0.0,
        hc_weight_max=0.5,
        hc_weight_step=0.05,
    ),
)
oof_mean, oof_std = automl_hc.fit_eval(df, target_col="default")
print(f"  OOF Matthews (hill climbing): {oof_mean:.4f} +/- {oof_std:.4f}")

hc = automl_hc.inner_model.hill_climbing_ensemble
if hc:
    print(
        f"  Models selected: {len(hc.selected_indices)} / {len(automl_hc.bluecast_models)}"  # type: ignore[arg-type]
    )


# =====================================================================
# 6. Conformal Prediction (Global + Group-Conditional)
# =====================================================================
print("\n" + "-" * 70)
print("6. CONFORMAL PREDICTION")
print("-" * 70)

from bluecast.blueprints.cast import BlueCast  # noqa: E402
from bluecast.conformal_prediction.evaluation import (  # noqa: E402
    prediction_set_coverage,
)

df_train_cp, df_temp = train_test_split(df, test_size=0.4, random_state=42)
df_cal, df_test_cp = train_test_split(df_temp, test_size=0.5, random_state=42)
y_cal = df_cal.pop("default")
y_test_cp = df_test_cp.pop("default")

model_cp = BlueCast(class_problem="binary", conf_training=fast_config)
model_cp.fit(df_train_cp, target_col="default")

model_cp.calibrate(df_cal, y_cal)

pred_sets = model_cp.predict_sets(df_test_cp, alpha=0.1)
coverage = prediction_set_coverage(y_test_cp, pred_sets)
print(f"  Global coverage at alpha=0.1: {coverage:.3f} (target ~0.90)")
print(f"  Sample sets: {pred_sets['prediction_set'].head(5).tolist()}")


# =====================================================================
# 7. Fairness Auditing
# =====================================================================
print("\n" + "-" * 70)
print("7. FAIRNESS AUDITING")
print("-" * 70)

from bluecast.evaluation.fairness import FairnessAuditor  # noqa: E402

y_probs_fair, y_classes_fair = model_cp.predict(df_test_cp)

auditor = FairnessAuditor(sensitive_columns=["gender", "region"])
reports = auditor.audit_classification(
    y_test_cp, y_classes_fair, y_probs_fair, df_test_cp
)

for report in reports:
    print(f"\n  --- {report.sensitive_column} ---")
    summary = report.summary_df()
    for col in ["positive_rate", "TPR", "precision"]:
        if col in summary.columns:
            vals = {str(g): f"{v:.3f}" for g, v in summary[col].items()}
            print(f"    {col}: {vals}")

    rdf = report.ratios_df()
    if not rdf.empty and "demographic_parity" in rdf.index:
        dp = rdf.loc["demographic_parity"]
        for pair, ratio in dp.items():
            flag = " <-- below 4/5 rule" if 0 < ratio < 0.8 else ""
            print(f"    DP ratio ({pair}): {ratio:.3f}{flag}")


# =====================================================================
# 8. Automatic Fairness via fit_eval
# =====================================================================
print("\n" + "-" * 70)
print("8. AUTOMATIC FAIRNESS IN FIT_EVAL")
print("-" * 70)

fair_config = TrainingConfig(
    hyperparameter_tuning_rounds=10,
    hyperparameter_tuning_max_runtime_secs=30,
    enable_feature_selection=False,
    calculate_shap_values=False,
    plot_hyperparameter_tuning_overview=False,
    hypertuning_cv_folds=2,
    train_size=0.8,
    fairness_sensitive_columns=["gender"],
)

df_train_f, df_eval_f = train_test_split(df, test_size=0.25, random_state=42)
y_eval_f = df_eval_f.pop("default")

automl_fair = BlueCast(class_problem="binary", conf_training=fair_config)
metrics_fair = automl_fair.fit_eval(
    df_train_f, df_eval_f, y_eval_f, target_col="default"
)

if "fairness" in metrics_fair:
    fr = metrics_fair["fairness"][0]
    print(f"  Fairness auto-computed for: {fr['sensitive_column']}")
    dp = fr.get("demographic_parity", {})
    for pair, ratio in dp.items():
        print(f"    DP {pair}: {ratio:.3f}")
else:
    print("  Fairness metrics not included (column may not be in eval data)")


# =====================================================================
# 9. Linear Model Baseline
# =====================================================================
print("\n" + "-" * 70)
print("9. LINEAR MODEL BASELINE")
print("-" * 70)

from bluecast.blueprints.custom_model_recipes import (  # noqa: E402
    LogisticRegressionModel,
)
from bluecast.blueprints.preprocessing_recipes import (  # noqa: E402
    LinearModelPreprocessingConfig,
    PreprocessingForLinearModels,
)

preproc = PreprocessingForLinearModels(
    config=LinearModelPreprocessingConfig(
        scaler="standard",
        imputation_strategy="median",
        collinearity_threshold=0.9,
    ),
)
lr_model = LogisticRegressionModel(scoring="roc_auc", cv_folds=3)

lr_config = TrainingConfig(
    hyperparameter_tuning_rounds=10,
    hyperparameter_tuning_max_runtime_secs=30,
    enable_feature_selection=False,
    calculate_shap_values=False,
    plot_hyperparameter_tuning_overview=False,
    hypertuning_cv_folds=2,
    train_size=0.8,
    cat_encoding_via_ml_algorithm=False,
)

df_train_lr, df_eval_lr = train_test_split(df, test_size=0.25, random_state=42)
y_eval_lr = df_eval_lr.pop("default")

automl_lr = BlueCast(
    class_problem="binary",
    conf_training=lr_config,
    ml_model=lr_model,
    custom_preprocessor=preproc,
)
metrics_lr = automl_lr.fit_eval(
    df_train_lr, df_eval_lr, y_eval_lr, target_col="default"
)
print(f"  Logistic Regression ROC AUC: {metrics_lr.get('roc_auc', 'N/A'):.4f}")


# =====================================================================
# 10. Experiment Tracking
# =====================================================================
print("\n" + "-" * 70)
print("10. EXPERIMENT TRACKING")
print("-" * 70)

from bluecast.experimentation.tracking import ExperimentTracker  # noqa: E402

tracker = ExperimentTracker()
tracker.add_results(
    experiment_id=0,
    score_category="oof_score",
    training_config=fast_config,
    model_parameters={"model": "CatBoost"},
    eval_scores=0.87,
    metric_used="roc_auc",
    metric_higher_is_better=True,
)
tracker.add_results(
    experiment_id=1,
    score_category="oof_score",
    training_config=fast_config,
    model_parameters={"model": "LogisticRegression"},
    eval_scores=0.82,
    metric_used="roc_auc",
    metric_higher_is_better=True,
)

results_df = tracker.retrieve_results_as_df()
if results_df is not None:
    print(f"  Tracked experiments: {len(results_df)} rows")

best = tracker.get_best_score(target_metric="roc_auc")
print(f"  Best ROC AUC: {best}")


# =====================================================================
# 11. Save & Load Pipeline
# =====================================================================
print("\n" + "-" * 70)
print("11. SAVE & LOAD PIPELINE")
print("-" * 70)

from bluecast.general_utils.general_utils import (  # noqa: E402
    load_for_production,
    save_to_production,
)

with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, "model.p")
    save_to_production(model_cp, path)
    loaded = load_for_production(path)

    y_before, _ = model_cp.predict(df_test_cp)
    y_after, _ = loaded.predict(df_test_cp)
    match = np.allclose(y_before, y_after, atol=1e-10)
    print(f"  Saved and loaded: predictions match = {match}")


# =====================================================================
# 12. Data Drift Monitoring
# =====================================================================
print("\n" + "-" * 70)
print("12. DATA DRIFT MONITORING")
print("-" * 70)

from bluecast.monitoring.data_monitoring import DataDrift  # noqa: E402

rng = np.random.default_rng(42)
baseline = pd.DataFrame(
    {
        "income": rng.normal(55000, 15000, 500),
        "credit_score": rng.normal(700, 50, 500),
    }
)
shifted = pd.DataFrame(
    {
        "income": rng.normal(60000, 18000, 500),
        "credit_score": rng.normal(700, 50, 500),
    }
)

drift = DataDrift()
drift.kolmogorov_smirnov_test(baseline, shifted)
for col, drifted in drift.kolmogorov_smirnov_flags.items():
    print(f"  K-S {col}: {'DRIFT' if drifted else 'stable'}")


# =====================================================================
# Regression Example
# =====================================================================
print("\n" + "-" * 70)
print("BONUS: REGRESSION")
print("-" * 70)

from bluecast.blueprints.cast_regression import BlueCastRegression  # noqa: E402

X_r, y_r = make_regression(
    n_samples=1000, n_features=8, n_informative=5, noise=10, random_state=42
)
df_r = pd.DataFrame(X_r, columns=[f"f_{i}" for i in range(8)])
df_r["target"] = y_r

df_train_r, df_eval_r = train_test_split(df_r, test_size=0.2, random_state=42)
y_eval_r = df_eval_r.pop("target")

reg = BlueCastRegression(class_problem="regression", conf_training=fast_config)
m_r = reg.fit_eval(df_train_r, df_eval_r, y_eval_r, target_col="target")
print(f"  R2:   {m_r.get('r2_score', 'N/A'):.4f}")
print(f"  RMSE: {m_r.get('RMSE', 'N/A'):.4f}")

# Conformal intervals for regression
reg.calibrate(df_eval_r, y_eval_r)
intervals = reg.predict_interval(df_eval_r, alphas=[0.1])
from bluecast.conformal_prediction.evaluation import (  # noqa: E402
    prediction_interval_coverage,
)

cov = prediction_interval_coverage(y_eval_r, intervals, [0.1])
print(f"  Regression interval coverage at alpha=0.1: {cov[0.1]:.3f}")


# =====================================================================
print("\n" + "=" * 70)
print(" ALL SHOWCASE SECTIONS COMPLETED SUCCESSFULLY")
print("=" * 70)
print("""
Features demonstrated:
  1.  EDA (univariate, correlation, mutual info, leakage detection)
  2.  Unified interface (BlueCastAuto)
  3.  Cross-validation with mean blending
  4.  Stacking ensemble
  5.  Hill climbing ensemble
  6.  Conformal prediction (global)
  7.  Fairness auditing (standalone + automatic)
  8.  Automatic fairness in fit_eval
  9.  Linear model baseline (Logistic Regression + preprocessing)
  10. Experiment tracking (DuckDB)
  11. Save & load pipeline
  12. Data drift monitoring (K-S test)
  +   Regression with conformal intervals
""")
