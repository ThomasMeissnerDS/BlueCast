"""
Exploratory Data Analysis (EDA)
================================

BlueCast includes a rich EDA library with interactive Plotly-based visualizations.
All plot functions return figure objects and accept a `show=True` parameter, so
they work in notebooks (interactive) and scripts (collect figures for reports).

This example demonstrates:
1. Univariate analysis (histograms, box plots)
2. Bivariate analysis (violin plots by target)
3. Correlation analysis (heatmap, target correlation)
4. Dimensionality reduction (PCA, t-SNE)
5. Categorical analysis (Theil's U, target distribution)
6. Data quality checks (nulls, Benford's law, leakage detection)
7. Distribution comparison (train vs test)
8. Mutual information
9. ECDF and Andrews curves
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# All EDA functions can be imported from bluecast.eda directly
from bluecast.eda import (
    bi_variate_plots,
    correlation_heatmap,
    correlation_to_target,
    detect_categorical_leakage,
    detect_leakage_via_correlation,
    mutual_info_to_target,
    plot_andrews_curve,
    plot_benfords_law,
    plot_classification_target_distribution_within_categories,
    plot_ecdf,
    plot_missing_values_matrix,
    plot_null_percentage,
    plot_pca,
    plot_pca_cumulative_variance,
    plot_pie_chart,
    plot_theil_u_heatmap,
    plot_tsne,
    univariate_plots,
)


def make_eda_dataset(n=1000, seed=42):
    """Create a realistic synthetic dataset with mixed types and data quality issues."""
    rng = np.random.default_rng(seed)
    X, y = make_classification(
        n_samples=n,
        n_features=8,
        n_informative=5,
        n_redundant=2,
        random_state=seed,
        flip_y=0.05,
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
        ],
    )

    # Scale to realistic ranges
    df["income"] = (df["income"] * 15000 + 60000).clip(20000, 200000).round(0)
    df["credit_score"] = (df["credit_score"] * 50 + 700).clip(300, 850).round(0)
    df["age"] = (df["age"] * 8 + 40).clip(18, 80).round(0)
    df["debt_ratio"] = (df["debt_ratio"] * 0.15 + 0.3).clip(0, 1).round(3)

    # Categorical features
    df["employment"] = rng.choice(
        ["employed", "self_employed", "unemployed", "retired"],
        size=n,
        p=[0.6, 0.2, 0.1, 0.1],
    )
    df["region"] = rng.choice(["north", "south", "east", "west"], size=n)
    df["loan_type"] = rng.choice(["mortgage", "auto", "personal", "student"], size=n)

    # Inject some missing values
    for col in ["income", "credit_score", "debt_ratio"]:
        mask = rng.random(n) < 0.03
        df.loc[mask, col] = np.nan

    df["target"] = y
    return df


# All plots use show=False so this script doesn't block.
# In a notebook, use show=True (the default) or call fig.show() on the returned figure.
SHOW = False

df = make_eda_dataset()
num_cols = [
    "income",
    "credit_score",
    "age",
    "debt_ratio",
    "num_accounts",
    "utilization",
    "payment_history",
    "inquiries",
]
cat_cols = ["employment", "region", "loan_type"]

print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Target distribution:\n{df['target'].value_counts().to_dict()}\n")


# =========================================================
# 1. Univariate Analysis
# =========================================================
print("=" * 60)
print("1. UNIVARIATE ANALYSIS")
print("=" * 60)

# Returns a list of figures, one per column
figs = univariate_plots(df[num_cols], show=SHOW)
print(f"Generated {len(figs)} univariate plots (histogram + box plot each)")

# Pie chart for a categorical column
fig = plot_pie_chart(df, "employment", show=SHOW)
print(f"Pie chart for 'employment': {type(fig).__name__}")
print()


# =========================================================
# 2. Bivariate Analysis
# =========================================================
print("=" * 60)
print("2. BIVARIATE ANALYSIS (VIOLIN PLOTS BY TARGET)")
print("=" * 60)

fig = bi_variate_plots(
    df[num_cols + ["target"]], target="target", num_cols_grid=4, show=SHOW
)
print(f"Bivariate plot: {type(fig).__name__}")
print()


# =========================================================
# 3. Correlation Analysis
# =========================================================
print("=" * 60)
print("3. CORRELATION ANALYSIS")
print("=" * 60)

fig_heatmap = correlation_heatmap(df[num_cols], show=SHOW)
print(f"Correlation heatmap: {type(fig_heatmap).__name__}")

fig_target = correlation_to_target(
    df[num_cols + ["target"]], target="target", show=SHOW
)
print(f"Correlation to target: {type(fig_target).__name__}")
print()


# =========================================================
# 4. Dimensionality Reduction
# =========================================================
print("=" * 60)
print("4. DIMENSIONALITY REDUCTION (PCA + t-SNE)")
print("=" * 60)

fig_pca = plot_pca(df[num_cols + ["target"]], target="target", show=SHOW)
print(f"PCA scatter: {type(fig_pca).__name__}")

fig_var = plot_pca_cumulative_variance(df[num_cols], n_components=8, show=SHOW)
print(f"PCA cumulative variance: {type(fig_var).__name__}")

# t-SNE (handles NaN rows automatically)
fig_tsne = plot_tsne(
    df[num_cols + ["target"]].dropna(),
    target="target",
    perplexity=30,
    show=SHOW,
)
print(f"t-SNE: {type(fig_tsne).__name__}")
print()


# =========================================================
# 5. Categorical Analysis
# =========================================================
print("=" * 60)
print("5. CATEGORICAL ANALYSIS")
print("=" * 60)

# Theil's U heatmap (association between categorical variables)
fig_theil, theil_matrix = plot_theil_u_heatmap(df, cat_cols, show=SHOW)
print(f"Theil's U heatmap: {type(fig_theil).__name__}")
print(f"Theil's U matrix shape: {theil_matrix.shape}")

# Target distribution within categories
figs = plot_classification_target_distribution_within_categories(
    df, cat_cols, "target", show=SHOW
)
print(f"Target distribution plots: {len(figs)} figures")
print()


# =========================================================
# 6. Data Quality Checks
# =========================================================
print("=" * 60)
print("6. DATA QUALITY CHECKS")
print("=" * 60)

# Null values
fig_nulls = plot_null_percentage(df, show=SHOW)
print(f"Null percentage plot: {type(fig_nulls).__name__}")
null_pct = df.isnull().mean()
for col in null_pct[null_pct > 0].index:
    print(f"  {col}: {null_pct[col]:.1%} missing")

# Missing values matrix
fig_matrix = plot_missing_values_matrix(df, show=SHOW)
print(f"Missing values matrix: {type(fig_matrix).__name__}")

# Benford's Law (fraud/data quality check)
fig_benford = plot_benfords_law(df, "income", show=SHOW)
print(f"Benford's Law plot: {type(fig_benford).__name__}")

# Leakage detection
print("\nLeakage checks:")
leaky_cols = detect_leakage_via_correlation(
    df[num_cols + ["target"]], target_column="target", threshold=0.9
)
print(f"  Correlation-based leakage: {leaky_cols if leaky_cols else 'none detected'}")

leaky_cat = detect_categorical_leakage(
    df[cat_cols + ["target"]], target_column="target", threshold=0.9
)
print(
    f"  Categorical leakage (Theil's U): {leaky_cat if leaky_cat else 'none detected'}"
)
print()


# =========================================================
# 7. Distribution Comparison (Train vs Test)
# =========================================================
print("=" * 60)
print("7. MUTUAL INFORMATION")
print("=" * 60)

fig_mi = mutual_info_to_target(
    df[num_cols + ["target"]].dropna(),
    target="target",
    class_problem="binary",
    show=SHOW,
)
print(f"Mutual information plot: {type(fig_mi).__name__}")
print()


# =========================================================
# 8. ECDF
# =========================================================
print("=" * 60)
print("8. ECDF (EMPIRICAL CUMULATIVE DISTRIBUTION)")
print("=" * 60)

figs_ecdf = plot_ecdf(df, ["income", "credit_score", "age"], show=SHOW)
if isinstance(figs_ecdf, list):
    print(f"ECDF plots: {len(figs_ecdf)} figures")
else:
    print(f"ECDF plot: {type(figs_ecdf).__name__}")
print()


# =========================================================
# 9. Andrews Curves
# =========================================================
print("=" * 60)
print("9. ANDREWS CURVES")
print("=" * 60)

fig_andrews = plot_andrews_curve(
    df[num_cols[:4] + ["target"]],
    target="target",
    n_samples=100,
    show=SHOW,
)
print(f"Andrews curve: {type(fig_andrews).__name__}")
print()


# =========================================================
# Summary
# =========================================================
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
All EDA functions:
  - Return plotly Figure objects (can be saved, embedded, or displayed)
  - Accept show=True/False to control interactive display
  - Work with both numerical and categorical data
  - Handle missing values gracefully

For interactive dashboards, use:
  from bluecast.eda import create_eda_dashboard

  # Auto-detects regression vs classification from the target column
  create_eda_dashboard(df, target_col="target")

  # Or use the specific variants directly:
  from bluecast.eda.analyse import create_eda_dashboard_classification
  from bluecast.eda.analyse import create_eda_dashboard_regression

This launches a full Dash-based EDA dashboard in your browser
with 14 interactive plot types, data filtering, and outlier detection.
""")

print("All EDA examples completed successfully!")
