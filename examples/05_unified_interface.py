"""
Unified Interface: BlueCastAuto
================================

BlueCastAuto is a single entry point for all BlueCast functionality.
Instead of choosing between BlueCast, BlueCastCV, BlueCastRegression,
and BlueCastCVRegression, just set `class_problem` and `use_cross_validation`.

This example demonstrates:
1. Binary classification (single model)
2. Binary classification (CV with stacking)
3. Regression (single model)
4. Regression (CV with hill climbing)
5. Switching between problem types with minimal code changes
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from bluecast.blueprints.unified import BlueCastAuto
from bluecast.config.training_config import TrainingConfig
from bluecast.ensemble.ensemble_config import EnsembleConfig

fast_config = TrainingConfig(
    hyperparameter_tuning_rounds=10,
    hyperparameter_tuning_max_runtime_secs=30,
    enable_feature_selection=False,
    calculate_shap_values=False,
    plot_hyperparameter_tuning_overview=False,
    hypertuning_cv_folds=2,
    bluecast_cv_train_n_model=(3, 1),
)


def make_data(problem="binary", n=1500, seed=42):
    rng = np.random.default_rng(seed)
    if problem == "regression":
        X, y = make_regression(
            n_samples=n, n_features=10, n_informative=7, noise=10.0, random_state=seed
        )
    else:
        n_classes = 4 if problem == "multiclass" else 2
        X, y = make_classification(
            n_samples=n,
            n_features=10,
            n_informative=7,
            n_redundant=2,
            n_classes=n_classes,
            n_clusters_per_class=1,
            random_state=seed,
        )
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)])
    df["category"] = rng.choice(["A", "B", "C"], size=n)
    df["target"] = y
    return df


# =========================================================
# 1. Binary Classification - Single Model
# =========================================================
print("=" * 60)
print("1. BINARY CLASSIFICATION (SINGLE MODEL)")
print("=" * 60)

df = make_data("binary")
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

automl = BlueCastAuto(
    class_problem="binary",
    use_cross_validation=False,
    conf_training=fast_config,
)
automl.fit(df_train, target_col="target")
y_probs, y_classes = automl.predict(df_test.drop("target", axis=1))
print(f"Predictions: {y_probs.shape[0]} samples")
print(f"Backend: {type(automl.inner_model).__name__}")
print()


# =========================================================
# 2. Binary Classification - CV with Stacking
# =========================================================
print("=" * 60)
print("2. BINARY CLASSIFICATION (CV + STACKING)")
print("=" * 60)

df = make_data("binary")

automl_cv = BlueCastAuto(
    class_problem="binary",
    use_cross_validation=True,
    conf_training=fast_config,
    ensemble_config=EnsembleConfig(ensemble_strategy="stacking"),
)
oof_mean, oof_std = automl_cv.fit_eval(df, target_col="target")
print(f"OOF score: {oof_mean:.4f} +/- {oof_std:.4f}")
print(f"Backend: {type(automl_cv.inner_model).__name__}")

y_probs, y_classes = automl_cv.predict(df.drop("target", axis=1))
print(f"Predictions: {y_probs.shape[0]} samples")
print()


# =========================================================
# 3. Regression - Single Model
# =========================================================
print("=" * 60)
print("3. REGRESSION (SINGLE MODEL)")
print("=" * 60)

df = make_data("regression")
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
y_test = df_test.pop("target")

automl_reg = BlueCastAuto(
    class_problem="regression",
    use_cross_validation=False,
    conf_training=fast_config,
)
metrics = automl_reg.fit_eval(
    df_train,
    target_col="target",
    df_eval=df_test,
    y_eval=y_test,
)
print(f"R2: {metrics.get('r2_score', 'N/A'):.4f}")  # type: ignore[union-attr]
print(f"Backend: {type(automl_reg.inner_model).__name__}")
print()


# =========================================================
# 4. Regression - CV with Hill Climbing
# =========================================================
print("=" * 60)
print("4. REGRESSION (CV + HILL CLIMBING)")
print("=" * 60)

df = make_data("regression")

automl_reg_cv = BlueCastAuto(
    class_problem="regression",
    use_cross_validation=True,
    conf_training=fast_config,
    ensemble_config=EnsembleConfig(
        ensemble_strategy="hill_climbing",
        hc_weight_min=0.0,
        hc_weight_max=0.5,
        hc_weight_step=0.05,
    ),
)
oof_mean, oof_std = automl_reg_cv.fit_eval(df, target_col="target")
print(f"OOF score: {oof_mean:.4f} +/- {oof_std:.4f}")
print(f"Backend: {type(automl_reg_cv.inner_model).__name__}")
print()


# =========================================================
# 5. Multiclass Classification
# =========================================================
print("=" * 60)
print("5. MULTICLASS CLASSIFICATION (SINGLE MODEL)")
print("=" * 60)

df = make_data("multiclass")
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

automl_mc = BlueCastAuto(
    class_problem="multiclass",
    use_cross_validation=False,
    conf_training=fast_config,
)
automl_mc.fit(df_train, target_col="target")
y_probs, y_classes = automl_mc.predict(df_test.drop("target", axis=1))
print(f"Predictions: {y_probs.shape}")
print(f"Unique classes: {np.unique(y_classes)}")
print(f"Backend: {type(automl_mc.inner_model).__name__}")
print()

print("All unified interface examples completed successfully!")
