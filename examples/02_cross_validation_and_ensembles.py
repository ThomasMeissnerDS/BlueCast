"""
Cross-Validation and Ensemble Strategies
=========================================

BlueCast supports three ensemble strategies for CV-based training:
1. Mean blending (arithmetic, geometric, harmonic, median)
2. Stacking (Ridge meta-learner on rank-transformed OOF predictions)
3. Hill climbing (greedy forward selection with optimal weights)

This example demonstrates all three approaches on a binary classification task.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from bluecast.blueprints.cast_cv import BlueCastCV
from bluecast.config.training_config import TrainingConfig
from bluecast.ensemble.ensemble_config import EnsembleConfig


def make_data(n=2000, seed=42):
    X, y = make_classification(
        n_samples=n, n_features=15, n_informative=10,
        n_redundant=3, random_state=seed,
    )
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(15)])
    df["sector"] = np.random.default_rng(seed).choice(
        ["finance", "healthcare", "tech", "retail"], size=n
    )
    df["target"] = y
    return df


fast_config = TrainingConfig(
    hyperparameter_tuning_rounds=10,
    hyperparameter_tuning_max_runtime_secs=30,
    enable_feature_selection=False,
    calculate_shap_values=False,
    plot_hyperparameter_tuning_overview=False,
    hypertuning_cv_folds=2,
    bluecast_cv_train_n_model=(3, 1),  # 3-fold, 1 repeat for speed
)

df = make_data()


# --- 1. Mean Blending (Default) ---
print("=" * 60)
print("1. MEAN BLENDING (DEFAULT)")
print("=" * 60)

ensemble_mean = EnsembleConfig(
    ensemble_strategy="mean",
    mean_type="arithmetic",
)

automl_mean = BlueCastCV(
    class_problem="binary",
    conf_training=fast_config,
    ensemble_config=ensemble_mean,
)
oof_mean, oof_std = automl_mean.fit_eval(df, target_col="target")
print(f"Mean blending OOF score: {oof_mean:.4f} +/- {oof_std:.4f}")
print()


# --- 2. Stacking Ensemble ---
print("=" * 60)
print("2. STACKING ENSEMBLE (RIDGE META-LEARNER)")
print("=" * 60)

ensemble_stacking = EnsembleConfig(
    ensemble_strategy="stacking",
    stacking_use_ranks=True,       # rank-transform before stacking
    stacking_meta_learner=None,    # default: Ridge(alpha=10.0)
)

automl_stack = BlueCastCV(
    class_problem="binary",
    conf_training=fast_config,
    ensemble_config=ensemble_stacking,
)
oof_mean, oof_std = automl_stack.fit_eval(df, target_col="target")
print(f"Stacking OOF score: {oof_mean:.4f} +/- {oof_std:.4f}")

# Predict with stacking
y_probs, y_classes = automl_stack.predict(df.drop("target", axis=1))
print(f"Stacking predictions shape: {y_probs.shape}")
print()


# --- 3. Hill Climbing Ensemble ---
print("=" * 60)
print("3. HILL CLIMBING ENSEMBLE")
print("=" * 60)

ensemble_hc = EnsembleConfig(
    ensemble_strategy="hill_climbing",
    hc_blending_method="rank",
    hc_weight_min=0.0,       # no negative weights for simplicity
    hc_weight_max=0.5,
    hc_weight_step=0.05,
    hc_tolerance=1e-6,
)

automl_hc = BlueCastCV(
    class_problem="binary",
    conf_training=fast_config,
    ensemble_config=ensemble_hc,
)
oof_mean, oof_std = automl_hc.fit_eval(df, target_col="target")
print(f"Hill climbing OOF score: {oof_mean:.4f} +/- {oof_std:.4f}")

if automl_hc.hill_climbing_ensemble:
    print(f"Models selected: {len(automl_hc.hill_climbing_ensemble.selected_indices)}")
    for entry in automl_hc.hill_climbing_ensemble.history:
        print(f"  Step {entry['iteration']}: {entry['model']} "
              f"(weight={entry['weight']:+.3f}, score={entry['score']:.6f})")
print()


# --- 4. Custom Meta-Learner for Stacking ---
print("=" * 60)
print("4. STACKING WITH CUSTOM META-LEARNER")
print("=" * 60)

from sklearn.linear_model import LogisticRegression

ensemble_custom = EnsembleConfig(
    ensemble_strategy="stacking",
    stacking_use_ranks=True,
    stacking_meta_learner=LogisticRegression(C=1.0, max_iter=1000),
)

automl_custom = BlueCastCV(
    class_problem="binary",
    conf_training=fast_config,
    ensemble_config=ensemble_custom,
)
oof_mean, oof_std = automl_custom.fit_eval(df, target_col="target")
print(f"Custom stacking OOF score: {oof_mean:.4f} +/- {oof_std:.4f}")
print()

print("All ensemble examples completed successfully!")
