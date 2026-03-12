"""
BlueCast Quick Start: Binary, Multiclass, and Regression
========================================================

This example shows the simplest way to use BlueCast for all three problem types
using synthetic data. No configuration is needed -- sensible defaults are applied.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from bluecast.blueprints.cast import BlueCast
from bluecast.blueprints.cast_regression import BlueCastRegression
from bluecast.config.training_config import TrainingConfig


def make_binary_data(n=2000, seed=42):
    X, y = make_classification(
        n_samples=n, n_features=12, n_informative=8,
        n_redundant=2, random_state=seed,
    )
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(12)])
    df["category"] = np.random.default_rng(seed).choice(
        ["retail", "tech", "health"], size=n
    )
    df["target"] = y
    return df


def make_multiclass_data(n=2000, seed=42):
    X, y = make_classification(
        n_samples=n, n_features=12, n_informative=10,
        n_redundant=2, n_classes=4, n_clusters_per_class=1,
        random_state=seed,
    )
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(12)])
    df["region"] = np.random.default_rng(seed).choice(
        ["north", "south", "east", "west"], size=n
    )
    df["target"] = y
    return df


def make_regression_data(n=2000, seed=42):
    X, y = make_regression(
        n_samples=n, n_features=12, n_informative=8,
        noise=10.0, random_state=seed,
    )
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(12)])
    df["size_category"] = np.random.default_rng(seed).choice(
        ["small", "medium", "large"], size=n
    )
    df["target"] = y
    return df


# Use fast training settings for examples
fast_config = TrainingConfig(
    hyperparameter_tuning_rounds=10,
    hyperparameter_tuning_max_runtime_secs=30,
    enable_feature_selection=False,
    calculate_shap_values=False,
    plot_hyperparameter_tuning_overview=False,
    hypertuning_cv_folds=2,
    train_size=0.8,
)


# --- 1. Binary Classification ---
print("=" * 60)
print("1. BINARY CLASSIFICATION")
print("=" * 60)

df_binary = make_binary_data()
df_train, df_eval = train_test_split(df_binary, test_size=0.2, random_state=42)

automl = BlueCast(class_problem="binary", conf_training=fast_config)
automl.fit(df_train, target_col="target")

y_probs, y_classes = automl.predict(df_eval.drop("target", axis=1))
print(f"Predictions shape: {y_probs.shape}")
print(f"Sample probabilities: {y_probs[:5]}")
print(f"Sample classes:       {y_classes[:5]}")
print()


# --- 2. Multiclass Classification ---
print("=" * 60)
print("2. MULTICLASS CLASSIFICATION")
print("=" * 60)

df_multi = make_multiclass_data()
df_train, df_eval = train_test_split(df_multi, test_size=0.2, random_state=42)

automl_multi = BlueCast(class_problem="multiclass", conf_training=fast_config)
automl_multi.fit(df_train, target_col="target")

y_probs, y_classes = automl_multi.predict(df_eval.drop("target", axis=1))
print(f"Predictions shape: probabilities={y_probs.shape}, classes={y_classes.shape}")
print(f"Unique classes predicted: {np.unique(y_classes)}")
print()


# --- 3. Regression ---
print("=" * 60)
print("3. REGRESSION")
print("=" * 60)

df_reg = make_regression_data()
df_train, df_eval = train_test_split(df_reg, test_size=0.2, random_state=42)

automl_reg = BlueCastRegression(class_problem="regression", conf_training=fast_config)
automl_reg.fit(df_train, target_col="target")

y_preds = automl_reg.predict(df_eval.drop("target", axis=1))
print(f"Predictions shape: {y_preds.shape}")
print(f"Sample predictions: {y_preds[:5]}")
print()


# --- 4. fit_eval for production-like evaluation ---
print("=" * 60)
print("4. FIT_EVAL (HOLDOUT EVALUATION)")
print("=" * 60)

df_binary = make_binary_data()
df_train, df_eval = train_test_split(df_binary, test_size=0.2, random_state=42)
y_eval = df_eval.pop("target")

automl_eval = BlueCast(class_problem="binary", conf_training=fast_config)
metrics = automl_eval.fit_eval(df_train, df_eval, y_eval, target_col="target")
print(f"Evaluation metrics: {list(metrics.keys())}")
print(f"ROC AUC: {metrics.get('roc_auc', 'N/A'):.4f}")
print(f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
print()

print("All quick start examples completed successfully!")
