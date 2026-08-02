"""
Linear Models with Advanced Preprocessing
==========================================

BlueCast supports linear/logistic regression as custom model backends.
This example shows:
1. Logistic regression with L1/L2/ElasticNet tuning
2. Regularized regression (Ridge/Lasso/ElasticNet auto-selection)
3. Configurable preprocessing (scaler, imputation, polynomial features)
4. Combining preprocessing and model in the BlueCast pipeline
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from bluecast.blueprints.cast import BlueCast
from bluecast.blueprints.cast_regression import BlueCastRegression
from bluecast.blueprints.custom_model_recipes import (
    LogisticRegressionModel,
    RegularizedRegressionModel,
)
from bluecast.blueprints.preprocessing_recipes import (
    LinearModelPreprocessingConfig,
    PreprocessingForLinearModels,
)
from bluecast.config.training_config import TrainingConfig

fast_config = TrainingConfig(
    hyperparameter_tuning_rounds=10,
    hyperparameter_tuning_max_runtime_secs=30,
    enable_feature_selection=False,
    calculate_shap_values=False,
    plot_hyperparameter_tuning_overview=False,
    cat_encoding_via_ml_algorithm=False,  # use target encoding for linear models
)


# --- Create synthetic data ---
def make_cls_data(n=1500, seed=42):
    X, y = make_classification(
        n_samples=n,
        n_features=10,
        n_informative=6,
        n_redundant=3,
        random_state=seed,
    )
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)])
    rng = np.random.default_rng(seed)
    df["channel"] = rng.choice(["online", "store", "partner"], size=n)
    # Inject some missing values
    mask = rng.random(n) < 0.05
    df.loc[mask, "feat_0"] = np.nan
    df["target"] = y
    return df


def make_reg_data(n=1500, seed=42):
    X, y = make_regression(
        n_samples=n,
        n_features=10,
        n_informative=6,
        noise=15.0,
        random_state=seed,
    )
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)])
    rng = np.random.default_rng(seed)
    df["material"] = rng.choice(["steel", "aluminum", "plastic"], size=n)
    mask = rng.random(n) < 0.05
    df.loc[mask, "feat_2"] = np.nan
    df["target"] = y
    return df


# =========================================================
# 1. Logistic Regression
# =========================================================
print("=" * 60)
print("1. LOGISTIC REGRESSION WITH AUTO-TUNING")
print("=" * 60)

df_cls = make_cls_data()
df_train, df_eval = train_test_split(df_cls, test_size=0.2, random_state=42)
y_eval = df_eval.pop("target")

num_cols = [f"feat_{i}" for i in range(10)]

preproc_config = LinearModelPreprocessingConfig(
    scaler="standard",
    imputation_strategy="median",
    collinearity_threshold=0.9,
    add_polynomial_features=False,
)
preproc = PreprocessingForLinearModels(num_columns=num_cols, config=preproc_config)

logistic_model = LogisticRegressionModel(
    scoring="roc_auc",
    cv_folds=3,
)

automl = BlueCast(
    class_problem="binary",
    conf_training=fast_config,
    ml_model=logistic_model,
    custom_preprocessor=preproc,
)
metrics = automl.fit_eval(df_train, df_eval, y_eval, target_col="target")
print(f"ROC AUC: {metrics.get('roc_auc', 'N/A'):.4f}")
print(f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
print()


# =========================================================
# 2. Regularized Regression (Ridge/Lasso/ElasticNet)
# =========================================================
print("=" * 60)
print("2. REGULARIZED REGRESSION (AUTO-SELECTS BEST MODEL)")
print("=" * 60)

df_reg = make_reg_data()
df_train, df_eval = train_test_split(df_reg, test_size=0.2, random_state=42)
y_eval = df_eval.pop("target")

preproc_reg = PreprocessingForLinearModels(
    num_columns=num_cols,
    config=LinearModelPreprocessingConfig(
        scaler="robust",
        imputation_strategy="median",
        collinearity_threshold=0.85,
    ),
)

reg_model = RegularizedRegressionModel(
    scoring="neg_mean_squared_error",
    cv_folds=3,
)

automl_reg = BlueCastRegression(
    class_problem="regression",
    conf_training=fast_config,
    ml_model=reg_model,
    custom_preprocessor=preproc_reg,
)
metrics = automl_reg.fit_eval(df_train, df_eval, y_eval, target_col="target")
print(f"Best model type: {reg_model.best_model_type}")
print(f"R2 score: {metrics.get('r2_score', 'N/A'):.4f}")
print(f"RMSE: {metrics.get('RMSE', 'N/A'):.4f}")
print()


# =========================================================
# 3. Polynomial Features for Non-Linear Patterns
# =========================================================
print("=" * 60)
print("3. POLYNOMIAL FEATURES FOR NON-LINEAR RELATIONSHIPS")
print("=" * 60)

preproc_poly = PreprocessingForLinearModels(
    num_columns=num_cols,
    config=LinearModelPreprocessingConfig(
        scaler="standard",
        imputation_strategy="median",
        add_polynomial_features=True,
        polynomial_degree=2,
        polynomial_interaction_only=True,
        max_polynomial_features=8,  # limit to top 8 features
    ),
)

logistic_poly = LogisticRegressionModel(scoring="roc_auc", cv_folds=3)

df_cls = make_cls_data()
df_train, df_eval = train_test_split(df_cls, test_size=0.2, random_state=42)
y_eval = df_eval.pop("target")

automl_poly = BlueCast(
    class_problem="binary",
    conf_training=fast_config,
    ml_model=logistic_poly,
    custom_preprocessor=preproc_poly,
)
metrics = automl_poly.fit_eval(df_train, df_eval, y_eval, target_col="target")
print(f"ROC AUC with poly features: {metrics.get('roc_auc', 'N/A'):.4f}")
print()


# =========================================================
# 4. Comparing Scaler Types
# =========================================================
print("=" * 60)
print("4. SCALER COMPARISON")
print("=" * 60)

df_cls = make_cls_data()

for scaler_name in ["standard", "power", "robust", "minmax"]:
    df_train, df_eval = train_test_split(df_cls, test_size=0.2, random_state=42)
    y_eval = df_eval.pop("target")

    preproc_s = PreprocessingForLinearModels(
        num_columns=num_cols,
        config=LinearModelPreprocessingConfig(scaler=scaler_name),  # type: ignore[arg-type]
    )
    lr = LogisticRegressionModel(scoring="roc_auc", cv_folds=3)
    automl_s = BlueCast(
        class_problem="binary",
        conf_training=fast_config,
        ml_model=lr,
        custom_preprocessor=preproc_s,
    )
    m = automl_s.fit_eval(df_train, df_eval, y_eval, target_col="target")
    print(f"  {scaler_name:10s}: ROC AUC = {m.get('roc_auc', 0):.4f}")

print()
print("All linear model examples completed successfully!")
