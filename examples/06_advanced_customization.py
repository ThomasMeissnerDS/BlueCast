"""
Advanced Customization: Custom Models, EDA, Monitoring, Experiment Tracking
============================================================================

BlueCast is designed to be customizable at every level. This example covers:
1. Custom preprocessing pipelines
2. Custom ML models (XGBoost backend)
3. EDA utilities
4. Data drift monitoring
5. Experiment tracking with DuckDB
6. Custom evaluation metrics
7. Saving and loading trained pipelines
"""

import logging
import os
import tempfile
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from bluecast.blueprints.cast import BlueCast
from bluecast.config.training_config import (
    TrainingConfig,
    XgboostFinalParamConfig,
    XgboostTuneParamsConfig,
)
from bluecast.evaluation.eval_metrics import ClassificationEvalWrapper
from bluecast.experimentation.tracking import ExperimentTracker
from bluecast.monitoring.data_monitoring import DataDrift
from bluecast.preprocessing.custom import CustomPreprocessing

logging.basicConfig(level=logging.WARNING)

fast_config = TrainingConfig(
    hyperparameter_tuning_rounds=10,
    hyperparameter_tuning_max_runtime_secs=30,
    enable_feature_selection=False,
    calculate_shap_values=False,
    plot_hyperparameter_tuning_overview=False,
    hypertuning_cv_folds=2,
    train_size=0.8,
)


def make_data(n=1500, seed=42):
    X, y = make_classification(
        n_samples=n,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        random_state=seed,
    )
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(12)])
    df["region"] = rng.choice(["US", "EU", "APAC"], size=n)
    df["timestamp"] = pd.date_range("2023-01-01", periods=n, freq="h")
    df["target"] = y
    return df


# =========================================================
# 1. Custom Preprocessing Pipeline
# =========================================================
print("=" * 60)
print("1. CUSTOM PREPROCESSING PIPELINE")
print("=" * 60)


class RatioFeatureCreator(CustomPreprocessing):
    """Create ratio features from numerical columns."""

    def __init__(self):
        super().__init__()
        self.ratio_pairs = []

    def fit_transform(
        self, df: pd.DataFrame, target: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.ratio_pairs = []
        for i in range(min(3, len(num_cols))):
            for j in range(i + 1, min(4, len(num_cols))):
                col_a, col_b = num_cols[i], num_cols[j]
                ratio_name = f"ratio_{col_a}_{col_b}"
                df[ratio_name] = df[col_a] / (df[col_b].abs() + 1e-8)
                self.ratio_pairs.append((col_a, col_b, ratio_name))
        return df, target

    def transform(
        self,
        df: pd.DataFrame,
        target: Optional[pd.Series] = None,
        prediction_mode: bool = False,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        for col_a, col_b, ratio_name in self.ratio_pairs:
            df[ratio_name] = df[col_a] / (df[col_b].abs() + 1e-8)
        return df, target


df = make_data()
df_train, df_eval = train_test_split(df, test_size=0.2, random_state=42)
y_eval = df_eval.pop("target")

custom_preproc = RatioFeatureCreator()

automl = BlueCast(
    class_problem="binary",
    conf_training=fast_config,
    custom_preprocessor=custom_preproc,
    date_columns=["timestamp"],
)
metrics = automl.fit_eval(df_train, df_eval, y_eval, target_col="target")
print(f"ROC AUC with custom ratio features: {metrics.get('roc_auc', 0):.4f}")
print(f"Ratio features created: {len(custom_preproc.ratio_pairs)}")
print()


# =========================================================
# 2. XGBoost Backend with Custom Config
# =========================================================
print("=" * 60)
print("2. XGBOOST BACKEND WITH CUSTOM CONFIG")
print("=" * 60)

from bluecast.ml_modelling.xgboost import XgboostModel  # noqa: E402

xgb_tune = XgboostTuneParamsConfig(
    max_depth_min=2,
    max_depth_max=6,
    eta_min=0.01,
    eta_max=0.3,
    steps_min=100,
    steps_max=500,
    xgboost_objective="binary:logistic",
    xgboost_eval_metric="logloss",
)

xgb_params = XgboostFinalParamConfig()
xgb_params.params["objective"] = "binary:logistic"
xgb_params.params["eval_metric"] = "logloss"

xgb_model = XgboostModel(
    class_problem="binary",
    conf_training=fast_config,
    conf_xgboost=xgb_tune,
    conf_params_xgboost=xgb_params,
)

df = make_data()
df_train, df_eval = train_test_split(df, test_size=0.2, random_state=42)
y_eval = df_eval.pop("target")

automl_xgb = BlueCast(
    class_problem="binary",
    conf_training=fast_config,
    conf_tuning=xgb_tune,
    conf_params=xgb_params,
    ml_model=xgb_model,
    date_columns=["timestamp"],
)
metrics = automl_xgb.fit_eval(df_train, df_eval, y_eval, target_col="target")
print(f"XGBoost ROC AUC: {metrics.get('roc_auc', 0):.4f}")
print()


# =========================================================
# 3. Data Drift Monitoring
# =========================================================
print("=" * 60)
print("3. DATA DRIFT MONITORING")
print("=" * 60)

rng = np.random.default_rng(42)
df_baseline = pd.DataFrame(
    {
        "revenue": rng.normal(1000, 200, 500),
        "clicks": rng.poisson(50, 500),
        "conversion_rate": rng.beta(2, 8, 500),
    }
)

df_new = pd.DataFrame(
    {
        "revenue": rng.normal(1050, 250, 500),  # shifted distribution
        "clicks": rng.poisson(55, 500),  # slightly shifted
        "conversion_rate": rng.beta(2, 8, 500),  # same distribution
    }
)

drift_monitor = DataDrift()

# Kolmogorov-Smirnov test
drift_monitor.kolmogorov_smirnov_test(df_baseline, df_new, threshold=0.05)
print("K-S Test drift flags:")
for col, flag in drift_monitor.kolmogorov_smirnov_flags.items():
    status = "DRIFT DETECTED" if flag else "no drift"
    print(f"  {col}: {status}")

# Population Stability Index
psi_flags = drift_monitor.population_stability_index(df_baseline, df_new)
print("\nPSI drift flags:")
for col, flag in psi_flags.items():
    psi_val = drift_monitor.population_stability_index_values.get(col, 0)
    print(f"  {col}: PSI={psi_val:.4f} ({'DRIFT' if flag else 'stable'})")
print()


# =========================================================
# 4. Experiment Tracking
# =========================================================
print("=" * 60)
print("4. EXPERIMENT TRACKING WITH DUCKDB")
print("=" * 60)

tracker = ExperimentTracker()

tracker.add_results(
    experiment_id=0,
    score_category="oof_score",
    training_config=fast_config,
    model_parameters={"max_depth": 5, "eta": 0.1},
    eval_scores=0.85,
    metric_used="roc_auc",
    metric_higher_is_better=True,
)

tracker.add_results(
    experiment_id=1,
    score_category="oof_score",
    training_config=fast_config,
    model_parameters={"max_depth": 8, "eta": 0.05},
    eval_scores=0.87,
    metric_used="roc_auc",
    metric_higher_is_better=True,
)

results_df = tracker.retrieve_results_as_df()
if results_df is not None:
    print(f"Tracked experiments: {len(results_df)} rows")
    print(f"Columns: {results_df.columns.tolist()}")

best = tracker.get_best_score(target_metric="roc_auc")
print(f"Best score: {best}")
print()


# =========================================================
# 5. Custom Evaluation Metric
# =========================================================
print("=" * 60)
print("5. CUSTOM EVALUATION METRIC")
print("=" * 60)

from sklearn.metrics import f1_score  # noqa: E402

custom_metric = ClassificationEvalWrapper(
    higher_is_better=True,
    eval_against="classes",
    metric_func=f1_score,
    metric_name="F1 Score (macro)",
    average="macro",
    zero_division=0,
)

df = make_data()
df_train, df_eval = train_test_split(df, test_size=0.2, random_state=42)

automl_custom_eval = BlueCast(
    class_problem="binary",
    conf_training=fast_config,
    single_fold_eval_metric_func=custom_metric,
    date_columns=["timestamp"],
)
automl_custom_eval.fit(df_train, target_col="target")
print("Model trained with custom F1 metric for hyperparameter tuning")
print()


# =========================================================
# 6. Save and Load Pipelines
# =========================================================
print("=" * 60)
print("6. SAVE AND LOAD PIPELINES")
print("=" * 60)

df = make_data()
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

automl_save = BlueCast(
    class_problem="binary",
    conf_training=fast_config,
    date_columns=["timestamp"],
)
automl_save.fit(df_train, target_col="target")

y_probs_before, y_classes_before = automl_save.predict(df_test.drop("target", axis=1))

with tempfile.TemporaryDirectory() as tmpdir:
    filepath = os.path.join(tmpdir, "bluecast_model.p")

    from bluecast.general_utils.general_utils import (  # noqa: E402
        load_for_production,
        save_to_production,
    )

    save_to_production(automl_save, filepath)
    print(f"Model saved to {filepath}")

    loaded_model = load_for_production(filepath)
    y_probs_after, y_classes_after = loaded_model.predict(
        df_test.drop("target", axis=1)
    )

    match = np.allclose(y_probs_before, y_probs_after, atol=1e-10)
    print(f"Predictions match after load: {match}")

print()
print("All advanced customization examples completed successfully!")
