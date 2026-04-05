"""Architecture-specific feature engineering agent.

This agent creates features tailored to a specific model architecture.
Unlike the base FeatureEngineerAgent (which creates shared, additive
features), this agent knows the target architecture's constraints and
can leverage feedback from previous iterations (feature importances,
OOF metrics) to refine its feature strategy.
"""

from typing import Dict, List, Optional

from bluecast.ai.agents.base import BaseAgent
from bluecast.ai.providers.base import ToolDefinition
from bluecast.ai.tools import (
    TOOL_DEFINITIONS,
    tool_create_feature,
    tool_create_tfidf_features,
)

# Architecture-specific FE guidance
ARCH_FE_GUIDELINES: Dict[str, str] = {
    "catboost": (
        "CatBoost handles raw categorical columns natively — do NOT encode them.\n"
        "Focus on: numerical interactions, polynomial terms, ratios, binning.\n"
        "CatBoost also handles missing values natively — no imputation needed.\n"
        "Create features that capture non-linear relationships between numericals."
    ),
    "xgboost": (
        "XGBoost requires all features to be numeric. However, BlueCast's native\n"
        "Target Encoding will handle categorical columns automatically — leave\n"
        "categorical text columns unencoded! Focus on: interaction features,\n"
        "polynomial terms, missing value indicators, group aggregations.\n"
        "Impute missing values with simple constants (e.g., fillna(0))."
    ),
    "histgb": (
        "HistGradientBoosting requires numeric features. BlueCast handles\n"
        "categorical encoding automatically. Focus on: interaction features,\n"
        "polynomial terms, ratio features, group-based aggregations.\n"
        "HistGB handles missing values natively — no imputation needed."
    ),
    "randomforest": (
        "RandomForest requires all features to be numeric. BlueCast handles\n"
        "categorical encoding automatically. Focus on: binned features,\n"
        "interaction features, polynomial terms.\n"
        "Impute ALL missing values with constants (e.g., fillna(0)).\n"
        "RandomForest benefits from diverse feature types."
    ),
    "linear": (
        "Linear models are extremely sensitive to feature scale and encoding.\n"
        "BlueCast handles categorical encoding automatically — leave text as-is.\n"
        "CRITICAL: Scale all numerical features statelessly (e.g.,\n"
        "df['col'] = df['col'] / HARDCODED_MAX — NOT df['col'].max()).\n"
        "Create polynomial and interaction features for non-linearity.\n"
        "Remove or don't create features that are highly multicollinear.\n"
        "Impute ALL missing values with constants."
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
        self.register_tool_impl(
            "create_feature",
            self._create_feature_wrapper,
        )
        self.register_tool_impl(
            "create_tfidf_features",
            self._create_tfidf_wrapper,
        )

    def set_architecture(self, arch_name: str, display_name: str) -> None:
        """Set the target architecture for the next run."""
        self._arch_name = arch_name
        self._arch_display_name = display_name

    def _create_feature_wrapper(self, feature_code: str, description: str = "", **kw):
        # Work on a copy of the training data
        if self.context.engineered_df is not None:
            df = self.context.engineered_df
        elif self.context.df_train is not None:
            df = self.context.df_train.copy()
        else:
            return {
                "success": False,
                "new_columns": [],
                "shape": [],
                "error": "No training data available.",
            }

        result = tool_create_feature(df, feature_code)
        if result["success"]:
            self.context.engineered_df = df
            # Store in arch-specific snippet list
            snippets = self.context.arch_feature_snippets.setdefault(
                self._arch_name, []
            )
            if feature_code not in snippets:
                snippets.append(feature_code)
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

        result = tool_create_tfidf_features(df, text_col, max_features)
        if result["success"]:
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
                importance_section += "Lowest importance features (consider dropping):\n"
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
- Create 3-8 focused features per iteration
- Call create_feature SEPARATELY for each feature (one snippet per call)
- DO NOT drop existing columns — use the tool to ADD features
- Keep feature names descriptive and unique
- STRICT: DO NOT use stateful transformations (StandardScaler, Target Encoding, global means via groupby). These leak validation targets during CV.
- Use the 'state' dictionary to store and retrieve data-dependent parameters (like means, counts, or scalers) between 'fit' and 'transform' phases.
- The code runs inside a framework that provides 'is_fit' (bool) and 'state' (dict) in the local scope.
- DO NOT reference columns with ZERO VARIANCE (only 1 unique value) — they are dropped by the pipeline.

## Provided Framework Tools
- `from bluecast.preprocessing.feature_creation import add_polynomial_features`
  Signature: add_polynomial_features(df, cols=['...'], degree=2)
- `from bluecast.preprocessing.feature_creation import add_interaction_features`
  Signature: add_interaction_features(df, cols_a=['...'], cols_b=['...'], operations=['mul', 'div', 'add', 'sub'])
- `from bluecast.preprocessing.feature_creation import add_groupby_agg_feats`
  Signature: add_groupby_agg_feats(df, groupby_cols=['...'], to_group_cols=['...'], num_col_prefix='agg', target_col='target', aggregations=['min', 'max', 'mean'])
- `from bluecast.preprocessing.feature_creation import add_binned_features`
  Signature: add_binned_features(df, cols=['...'], num_bins=5)
- `from bluecast.preprocessing.feature_creation import StateAwareGroupbyAggregator`
  Usage:
  if 'my_agg' not in state:
      state['my_agg'] = StateAwareGroupbyAggregator(groupby_cols=['cat_col'], agg_cols=['num_col'], aggregations=['mean'])
  if is_fit:
      df = state['my_agg'].fit_transform(df)
  else:
      df = state['my_agg'].transform(df)
{importance_section}

Dataset overview:
{data_summary}

Analysis findings:
{profile}
{hints}"""

    def get_tools(self) -> List[ToolDefinition]:
        return [
            TOOL_DEFINITIONS["create_feature"],
            TOOL_DEFINITIONS["create_tfidf_features"],
        ]
