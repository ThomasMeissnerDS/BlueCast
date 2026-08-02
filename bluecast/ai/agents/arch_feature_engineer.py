"""Architecture-specific feature engineering agent.

This agent creates features tailored to a specific model architecture.
Unlike the base FeatureEngineerAgent (which creates shared, additive
features), this agent knows the target architecture's constraints and
can leverage feedback from previous iterations (feature importances,
OOF metrics) to refine its feature strategy.
"""

from typing import Dict, List

from bluecast.ai.agents.base import BaseAgent
from bluecast.ai.providers.base import ToolDefinition
from bluecast.ai.tools import (
    TOOL_DEFINITIONS,
    tool_check_feature_quality,
    tool_create_feature,
    tool_create_tfidf_features,
)

# Architecture-specific FE guidance — split into preprocessing (bare minimum)
# and feature engineering (optional, iteration-dependent).
ARCH_FE_GUIDELINES: Dict[str, str] = {
    "catboost": (
        "## Preprocessing Best Practices (MUST DO)\n"
        "- CatBoost handles raw categorical columns natively — do NOT encode them.\n"
        "- CatBoost handles missing values natively — NO imputation needed.\n"
        "- Leave the data as-is for categoricals and NaNs.\n"
        "\n## Feature Engineering (OPTIONAL — adapt to iteration)\n"
        "- WARNING: Avoid using StateAwareGroupbyAggregator on HIGH-CARDINALITY categoricals (e.g., granular IDs). This causes severe target leakage on StratifiedKFold validation splits and fails on unseen test data.\n"
        "- Group aggregations using StateAwareGroupbyAggregator (Only on LOW-cardinality categorical columns like City, Region).\n"
        "  Example: Group by low-cardinality columns and compute mean/std of numerical columns.\n"
        "- Numerical interactions, polynomial terms, binning.\n"
        "- WARNING: If creating ratio features, NEVER divide blindly. You MUST clip the denominator away from zero (e.g. `df['col_b'].clip(lower=0.1)`) to prevent exploding infinities on unseen test data.\n"
        "- Create features that capture non-linear relationships between numericals.\n"
        "- CatBoost is already strong on raw data — only add features you believe are truly informative."
    ),
    "xgboost": (
        "## Preprocessing Best Practices (MUST DO)\n"
        "- XGBoost requires all features to be numeric.\n"
        "- BlueCast's native Target Encoding handles categorical columns automatically — leave categorical text columns UNENCODED.\n"
        "- Impute ALL missing values with simple constants (e.g., fillna(0) or fillna(-999)).\n"
        "- Create missing value indicator columns (df['col_is_missing'] = df['col'].isna().astype(int)) for columns with >5% missing.\n"
        "\n## Feature Engineering (OPTIONAL — adapt to iteration)\n"
        "- Interaction features (products, differences between related columns).\n"
        "- WARNING: If creating ratio features, NEVER divide blindly. You MUST clip the denominator away from zero (e.g. `df['col_b'].clip(lower=0.1)`).\n"
        "- Polynomial terms for suspected non-linear relationships.\n"
        "- Group aggregations (mean/median of target-correlated features grouped by LOW-cardinality categorical columns. Avoid high-cardinality!)."
    ),
    "histgb": (
        "## Preprocessing Best Practices (MUST DO)\n"
        "- HistGradientBoosting requires numeric features.\n"
        "- BlueCast handles categorical encoding automatically — leave text columns unencoded.\n"
        "- HistGB handles missing values natively (via dedicated NaN bin) — NO imputation needed.\n"
        "- However, creating explicit missing value indicator columns can still help.\n"
        "\n## Feature Engineering (OPTIONAL — adapt to iteration)\n"
        "- Interaction features, polynomial terms.\n"
        "- WARNING: If creating ratio features, NEVER divide blindly. You MUST clip the denominator away from zero (e.g. `df['col_b'].clip(lower=0.1)`).\n"
        "- Group-based aggregations using StateAwareGroupbyAggregator (LOW-cardinality columns only).\n"
        "- HistGB is fast — you can create more features without major speed penalties."
    ),
    "randomforest": (
        "## Preprocessing Best Practices (MUST DO)\n"
        "- RandomForest requires all features to be numeric.\n"
        "- BlueCast handles categorical encoding automatically — leave text columns unencoded.\n"
        "- Impute ALL missing values with constants (e.g., fillna(0) or fillna(-999)). RandomForest cannot handle NaN.\n"
        "- Create missing value indicator columns for columns with >5% missing.\n"
        "- RandomForest is NOT sensitive to feature scale — do NOT scale features.\n"
        "\n## Feature Engineering (OPTIONAL — adapt to iteration)\n"
        "- Binned features (RandomForest handles these particularly well).\n"
        "- Interaction features and polynomial terms.\n"
        "- RandomForest benefits from diverse feature types (binary indicators, bins, ratios)."
    ),
    "linear": (
        "## Preprocessing Best Practices (MUST DO)\n"
        "- Linear models are EXTREMELY sensitive to feature scale, outliers, and multicollinearity.\n"
        "- BlueCast handles categorical encoding automatically — leave text columns unencoded.\n"
        "- CRITICAL: Scale ALL numerical features statelessly:\n"
        "  Use df['col'] = df['col'] / HARDCODED_MAX (a constant you choose from the data profile, NOT df['col'].max()).\n"
        "  Alternatively, use the pipeline's built-in RobustScaler (already applied for linear models).\n"
        "- Impute ALL missing values with constants (e.g., fillna(0)). Linear models crash on NaN.\n"
        "- Handle outliers: clip extreme values statelessly (e.g., df['col'] = df['col'].clip(-1000, 1000)).\n"
        "- Avoid highly multicollinear features — use drop_collinear_features tool if needed.\n"
        "\n## Target Scaling (automatic)\n"
        "- The pipeline automatically wraps the linear model in TransformedTargetRegressor(transformer=StandardScaler()).\n"
        "- This means the target is scaled during training and predictions are automatically inverse-transformed.\n"
        "- You do NOT need to scale the target manually.\n"
        "\n## Feature Engineering (OPTIONAL — adapt to iteration)\n"
        "- Polynomial and interaction features to capture non-linearity.\n"
        "- Log-transform skewed features statelessly (e.g., df['col_log'] = np.log1p(df['col'].clip(0))).\n"
        "- Try different feature scaling approaches across iterations:\n"
        "  * Iteration 1: Use raw features (pipeline handles RobustScaler).\n"
        "  * Iteration 2: Add log-transformed versions of skewed features (np.log1p).\n"
        "  * Iteration 3: Add sqrt-transformed versions (np.sqrt(df['col'].clip(0))).\n"
        "- Use l1_feature_selection tool to drop uninformative features.\n"
        "- Linear models benefit most from well-scaled, low-noise features."
    ),
    "mlp": (
        "## Preprocessing Best Practices (MUST DO)\n"
        "- MLP Neural Networks require all features to be numeric.\n"
        "- BlueCast handles categorical encoding automatically — leave text columns unencoded.\n"
        "- The pipeline applies SimpleImputer + StandardScaler automatically for MLP.\n"
        "- Impute ALL missing values with constants (e.g., fillna(0)). MLP cannot handle NaN.\n"
        "- MLP is sensitive to feature scale — but the pipeline handles scaling for you.\n"
        "- MLP is sensitive to multicollinearity and high-dimensional noise.\n"
        "  Use drop_collinear_features to remove redundant features.\n"
        "  Use l1_feature_selection to drop uninformative features.\n"
        "\n## Target Scaling (automatic)\n"
        "- The pipeline automatically wraps MLP in TransformedTargetRegressor(transformer=StandardScaler()).\n"
        "- This means the target is scaled during training and predictions are automatically inverse-transformed.\n"
        "- You do NOT need to scale the target manually.\n"
        "\n## Feature Engineering (OPTIONAL — adapt to iteration)\n"
        "- Interaction features and polynomial terms.\n"
        "- MLP can learn non-linear relationships, so focus on providing informative raw features.\n"
        "- Try different feature transformations across iterations:\n"
        "  * Add log-transformed versions of skewed features (np.log1p).\n"
        "  * Add sqrt-transformed versions for features with large ranges.\n"
        "- Avoid creating too many features — MLP can overfit on high-dimensional sparse data."
    ),
}


class ArchFeatureEngineerAgent(BaseAgent):
    """Feature engineer that creates features specifically for one architecture.

    The agent writes FE snippets to ``context.arch_feature_snippets[arch_name]``
    instead of the shared ``context.feature_code_snippets``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._arch_name: str = ""
        self._arch_display_name: str = ""
        self.feature_states: Dict[str, dict] = {}
        self.register_tool_impl(
            "create_feature",
            self._create_feature_wrapper,
        )
        self.register_tool_impl(
            "create_tfidf_features",
            self._create_tfidf_wrapper,
        )
        self.register_tool_impl(
            "drop_collinear_features",
            self._drop_collinear_wrapper,
        )
        self.register_tool_impl(
            "l1_feature_selection",
            self._l1_selection_wrapper,
        )
        self.register_tool_impl(
            "check_feature_quality",
            self._check_feature_quality_wrapper,
        )

    def set_architecture(self, arch_name: str, display_name: str) -> None:
        """Set the target architecture for the next run."""
        self._arch_name = arch_name
        self._arch_display_name = display_name

    def _create_feature_wrapper(self, feature_code: str, description: str = "", **kw):
        # Work on a copy of the training data to prevent in-place corruption if snippets fail or are rejected
        if self.context.engineered_df is not None:
            df = self.context.engineered_df.copy()
        elif self.context.df_train is not None:
            df = self.context.df_train.copy()
        else:
            return {
                "success": False,
                "new_columns": [],
                "shape": [],
                "error": "No training data available.",
            }

        # Hide target column from feature engineering to prevent schema mismatches on inference
        target_col = getattr(self.context, "target_col", None)
        if target_col and target_col in df.columns:
            df = df.drop(columns=[target_col])

        state = self.feature_states.setdefault(self._arch_name, {})
        result = tool_create_feature(df, feature_code, state=state)
        if result["success"]:
            df = result.pop("df")
            new_cols = result.get("new_columns", [])
            valid_cols = []
            for col in new_cols:
                if col in df.columns:
                    # Drop constant columns which add noise and cause tree models to overfit
                    if df[col].nunique() > 1:
                        valid_cols.append(col)
                    else:
                        df.drop(columns=[col], inplace=True)

            # If all newly created features were constant, reject the entire snippet
            if new_cols and not valid_cols:
                return {
                    "success": False,
                    "new_columns": [],
                    "error": "Snippet rejected: All generated features were constant or had zero variance.",
                }

            self.context.engineered_df = df
            # Store in arch-specific snippet list
            snippets = self.context.arch_feature_snippets.setdefault(
                self._arch_name, []
            )
            if feature_code not in snippets:
                snippets.append(feature_code)
            result["new_columns"] = valid_cols

        return result

    def _create_tfidf_wrapper(self, text_col: str, max_features: int = 50, **kw):
        if self.context.engineered_df is not None:
            df = self.context.engineered_df
        elif self.context.df_train is not None:
            df = self.context.df_train.copy()
        else:
            return {
                "success": False,
                "new_columns": [],
                "error": "No training data available.",
            }

        # Hide target column from feature engineering to prevent schema mismatches on inference
        target_col = getattr(self.context, "target_col", None)
        if target_col and target_col in df.columns:
            df = df.drop(columns=[target_col])

        result = tool_create_tfidf_features(df, text_col, max_features)
        if result["success"]:
            df = result.pop("df", df)
            self.context.engineered_df = df
            tfidf_code = (
                f"from bluecast.preprocessing.feature_creation import TfIdfTextEncoder\n"
                f"if 'tfidf_{text_col}' not in state:\n"
                f"    state['tfidf_{text_col}'] = TfIdfTextEncoder(max_features={max_features})\n"
                f"if is_fit:\n"
                f"    df = state['tfidf_{text_col}'].fit_transform(df, '{text_col}')\n"
                f"else:\n"
                f"    df = state['tfidf_{text_col}'].transform(df, '{text_col}')\n"
            )
            snippets = self.context.arch_feature_snippets.setdefault(
                self._arch_name, []
            )
            snippets.append(tfidf_code)
        return result

    def _drop_collinear_wrapper(self, threshold: float = 0.9, **kw):
        from bluecast.ai.tools import tool_drop_collinear_features

        if self.context.engineered_df is not None:
            df = self.context.engineered_df
        elif self.context.df_train is not None:
            df = self.context.df_train.copy()
        else:
            return {"success": False, "error": "No training data available."}

        target_col = getattr(self.context, "target_col", None)
        result = tool_drop_collinear_features(df, threshold, target_col)

        if result.get("success"):
            self.context.engineered_df = df
            dropped = result["dropped_columns"]
            if dropped:
                code = (
                    f"to_drop = {dropped}\n"
                    f"df = df.drop(columns=[c for c in to_drop if c in df.columns])\n"
                )
                snippets = self.context.arch_feature_snippets.setdefault(
                    self._arch_name, []
                )
                snippets.append(code)
                result["message"] = (
                    f"Dropped {len(dropped)} collinear columns: {dropped}"
                )
            else:
                result["message"] = "No collinear columns exceeded the threshold."

        return result

    def _l1_selection_wrapper(self, alpha: float = 0.01, **kw):
        from bluecast.ai.tools import tool_l1_feature_selection

        if self.context.engineered_df is not None:
            df = self.context.engineered_df
        elif self.context.df_train is not None:
            df = self.context.df_train.copy()
        else:
            return {"success": False, "error": "No training data available."}

        target_col = getattr(self.context, "target_col", None)
        class_problem = getattr(self.context, "class_problem", "regression")
        if not target_col:
            return {"success": False, "error": "Context missing target_col."}

        result = tool_l1_feature_selection(df, target_col, class_problem, alpha)

        if result.get("success"):
            self.context.engineered_df = df
            dropped = result["dropped_columns"]
            if dropped:
                code = (
                    f"to_drop = {dropped}\n"
                    f"df = df.drop(columns=[c for c in to_drop if c in df.columns])\n"
                )
                snippets = self.context.arch_feature_snippets.setdefault(
                    self._arch_name, []
                )
                snippets.append(code)
                result["message"] = (
                    f"Dropped {len(dropped)} uninformative columns using L1 regularization."
                )
            else:
                result["message"] = "No columns were dropped."

        return result

    @property
    def name(self) -> str:
        return "ArchFeatureEngineer"

    def system_prompt(self) -> str:
        data_summary = self.context.get_data_summary()
        profile_data = self.context.data_profile or "Not yet profiled."
        if isinstance(profile_data, dict):
            profile = profile_data.get("summary", str(profile_data))
        else:
            profile = str(profile_data)

        arch_guidelines = ARCH_FE_GUIDELINES.get(
            self._arch_name, "Use your best judgment for this architecture."
        )

        # Include feature importances from previous iteration if available
        importance_section = ""
        importances = self.context.arch_feature_importances.get(self._arch_name)
        if importances:
            sorted_feats = sorted(
                importances.items(), key=lambda x: abs(x[1]), reverse=True
            )
            top_feats = sorted_feats[:20]
            bottom_feats = sorted_feats[-10:] if len(sorted_feats) > 20 else []
            importance_section = "\n\nFeature importances from previous iteration:\n"
            importance_section += "Top features:\n"
            for feat, imp in top_feats:
                importance_section += f"  {feat}: {imp:.4f}\n"
            if bottom_feats:
                importance_section += (
                    "Lowest importance features (consider dropping):\n"
                )
                for feat, imp in bottom_feats:
                    importance_section += f"  {feat}: {imp:.4f}\n"

        hints = ""
        if self.context.data_warnings:
            hints = "\nData warnings:\n" + "\n".join(
                f"- {w}" for w in self.context.data_warnings
            )

        return f"""You are an architecture-specific feature engineer for the BlueCast AutoML framework.

You are creating features specifically for the **{self._arch_display_name}** model.

## Architecture-Specific Guidelines
{arch_guidelines}

## General Guidelines
- Call create_feature SEPARATELY for each feature (one snippet per call)
- DO NOT drop existing columns — use the tool to ADD features
- Keep feature names descriptive and unique
- STRICT: DO NOT use stateful transformations (StandardScaler, Target Encoding, global means via groupby). These leak validation targets during CV.
- Use the 'state' dictionary to store and retrieve data-dependent parameters (like means, counts, or scalers) between 'fit' and 'transform' phases.
- The code runs inside a framework that provides 'is_fit' (bool) and 'state' (dict) in the local scope.
- DO NOT reference columns with ZERO VARIANCE (only 1 unique value) — they are dropped by the pipeline.

## Iterative Strategy
You will be called multiple times across iterations. **Adapt your approach based on the iteration:**
- **Early iterations:** Focus on essential fixes only — missing value indicators, basic imputation, dropping constant columns. Create 1-3 simple, high-signal features. The goal is a fast baseline.
- **Middle iterations:** Add interaction features and ratios between the top predictors identified via feature importances from previous iterations. Create 3-5 targeted features.
- **Later iterations:** Use advanced techniques — group-level aggregations, polynomial features, binned features. Focus on the error analysis rows where the model struggles most.

Do NOT use all available tools at once. Choose the most appropriate ones for your current iteration.

## Available Framework Tools
The following tools are available if you need them. Use your judgment about which are appropriate for the current iteration:
- `from bluecast.preprocessing.feature_creation import add_polynomial_features`
  Signature: add_polynomial_features(df, cols=['...'], degree=2)
- `from bluecast.preprocessing.feature_creation import add_interaction_features`
  Signature: add_interaction_features(df, cols_a=['...'], cols_b=['...'], operations=['mul', 'div', 'add', 'sub'])
- `from bluecast.preprocessing.feature_creation import add_binned_features`
  Signature: add_binned_features(df, cols=['...'], num_bins=5, state=state, is_fit=is_fit)
  IMPORTANT: Always pass state=state and is_fit=is_fit to ensure train/test bin edge consistency.
- `from bluecast.preprocessing.feature_creation import StateAwareGroupbyAggregator`
  Usage:
  if 'my_agg' not in state:
      state['my_agg'] = StateAwareGroupbyAggregator(groupby_cols=['cat_col'], agg_cols=['num_col'], aggregations=['mean'])
  if is_fit:
      df = state['my_agg'].fit_transform(df)
  else:
      df = state['my_agg'].transform(df)
- `from bluecast.preprocessing.feature_creation import add_pca_features`
  Signature: add_pca_features(df, cols=['num1', 'num2', 'num3'], n_components=3, state=state, is_fit=is_fit)
  IMPORTANT: Always pass state=state and is_fit=is_fit to ensure train/test PCA consistency.
  Creates n_components new 'pca_1', 'pca_2', ... columns capturing the principal components of the input features.
  Useful for reducing dimensionality of correlated numeric features.
- `check_feature_quality`: After calling create_feature, use this tool to instantly check if your new features
  have meaningful correlation or mutual information with the target. Pass the new_columns list from create_feature.
  Returns a quality rating (HIGH/MEDIUM/LOW) per feature. Use this to decide if the feature is worth keeping.
- `drop_collinear_features` (Linear Only): Tool to drop highly correlated columns.
- `l1_feature_selection` (Linear Only): Tool to drop uninformative features via L1.
{importance_section}

Dataset overview:
{data_summary}

Analysis findings:
{profile}
{hints}"""

    def _check_feature_quality_wrapper(self, feature_cols: list, **kw):
        """Check signal quality of newly created features."""
        if self.context.engineered_df is not None:
            df = self.context.engineered_df
        elif self.context.df_train is not None:
            df = self.context.df_train.copy()
        else:
            return "No training data available."

        target_col = getattr(self.context, "target_col", None)
        if not target_col:
            return "No target column set in context."

        # Need target column in the df for quality check
        full_df = self.context.df_train
        if full_df is not None and target_col in full_df.columns:
            df_with_target = df.copy()
            # Align indices for target assignment
            common_idx = df_with_target.index.intersection(full_df.index)
            df_with_target.loc[common_idx, target_col] = full_df.loc[
                common_idx, target_col
            ]
        else:
            return "Cannot check quality: target column not available."

        return tool_check_feature_quality(df_with_target, target_col, feature_cols)

    def get_tools(self) -> List[ToolDefinition]:
        tools = [
            TOOL_DEFINITIONS["create_feature"],
            TOOL_DEFINITIONS["create_tfidf_features"],
            TOOL_DEFINITIONS["check_feature_quality"],
        ]
        if self._arch_name in ["linear", "logistic", "mlp"]:
            if "drop_collinear_features" in TOOL_DEFINITIONS:
                tools.append(TOOL_DEFINITIONS["drop_collinear_features"])
            if "l1_feature_selection" in TOOL_DEFINITIONS:
                tools.append(TOOL_DEFINITIONS["l1_feature_selection"])
        return tools
