"""Feature engineer agent: creates new features based on data analysis."""

from typing import List

from bluecast.ai.agents.base import BaseAgent
from bluecast.ai.providers.base import ToolDefinition
from bluecast.ai.tools import (
    TOOL_DEFINITIONS,
    tool_create_feature,
    tool_create_tfidf_features,
)


class FeatureEngineerAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_tool_impl(
            "create_feature",
            self._create_feature_wrapper,
        )
        self.register_tool_impl(
            "create_tfidf_features",
            self._create_tfidf_wrapper,
        )

    def _create_feature_wrapper(self, feature_code: str, description: str = "", **kw):
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
            if feature_code not in self.context.feature_code_snippets:
                self.context.feature_code_snippets.append(feature_code)

            existing_code = self.context.feature_engineering_code or ""
            if feature_code not in existing_code:
                self.context.feature_engineering_code = (
                    existing_code + f"\n# {description}\n{feature_code}\n"
                )
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
            # Store the state-aware TFIDF code snippet for replay at inference
            tfidf_code = (
                f"from bluecast.preprocessing.feature_creation import TfIdfTextEncoder\n"
                f"if 'tfidf_{text_col}' not in state:\n"
                f"    state['tfidf_{text_col}'] = TfIdfTextEncoder(max_features={max_features})\n"
                f"if is_fit:\n"
                f"    df = state['tfidf_{text_col}'].fit_transform(df, '{text_col}')\n"
                f"else:\n"
                f"    df = state['tfidf_{text_col}'].transform(df, '{text_col}')\n"
            )
            self.context.feature_code_snippets.append(tfidf_code)
            existing_code = self.context.feature_engineering_code or ""
            self.context.feature_engineering_code = (
                existing_code + f"\n# TF-IDF on '{text_col}'\n{tfidf_code}\n"
            )
        return result

    @property
    def name(self) -> str:
        return "FeatureEngineer"

    def system_prompt(self) -> str:
        data_summary = self.context.get_data_summary()
        profile = self.context.data_profile or "Not yet profiled."
        hints = ""
        if self.context.data_warnings:
            hints = "\nData warnings:\n" + "\n".join(
                f"- {w}" for w in self.context.data_warnings
            )

        return f"""You are a feature engineer for the BlueCast AutoML framework.

Your job is to create useful features that will improve model performance.
Use the create_feature tool to add features to the DataFrame.

The code you write runs with 'df' (the DataFrame), 'np' (numpy), and 'pd' (pandas) available.

Guidelines:
- Create ratio features from related numerical columns
- Create interaction features (products, differences)
- Bin continuous features into categories
- Create frequency-encoded features from categoricals (Warning: stateless only, e.g. mapping dictionaries, not groupbys)
- Handle missing values if needed (fill with constants or create indicator columns)
- Do NOT drop existing columns (the model may need them)
- Keep feature names descriptive and unique
- If a feature creation fails, try a different approach
- STRICT: DO NOT use stateful transformations manually (e.g. StandardScaler, Target Encoding, global means via groupby). These leak validation targets during CV or crash on single-row test sets during inference. Stick to stateless row-by-row math.

CRITICAL PIPELINE EXECUTION CONSTRAINTS:
1. Moreso than ever, YOU MUST SEPARATE YOUR FEATURE ENGINEERING INTO MULTIPLE INDEPENDENT SNIPPETS! Call `create_feature` separately for each new feature you invent. DO NOT combine them into one massive code block. This prevents a single failed line of code from dropping all your other valid features.
2. DO NOT use or attempt to encode columns that have ZERO VARIANCE (only 1 unique value) or are entirely missing. The BlueCast pipeline automatically DROPS these columns before your features run. If you reference them, your snippet will crash with a ColumnNotFoundError!

CRITICAL ARCHITECTURE CONSTRAINTS:
If the overarching plan involves XGBoost, HistGB, or Linear models (e.g., in 'ultimate' mode):
1. You MUST LEAVE CATEGORICAL COLUMNS UNENCODED (do NOT use `.cat.codes` or `factorize()`). The BlueCast pipeline has powerful native Target Encoding that will automatically handle `object` and `category` text data if you leave them alone. If you convert them to integers, BlueCast will treat them as continuous variables and performance will be ruined!
2. You MUST impute ALL missing values with simple constants (e.g., `fillna(0)`).
3. If Linear models are used, you SHOULD scale numerical features statelessly (e.g., `df['col'] = df['col'] / df['col'].max()` where max is a hardcoded constant, NOT a dynamic `.max()`).
(CatBoost is the only model that handles unencoded categories natively. But BlueCast Auto natively handles them for the rest!).

Text Data:
- If a column contains free text (long strings, descriptions, names), use
  create_tfidf_features to extract word-level features automatically
- This is much better than ignoring text columns

Preprocessing Logic:
- If a categorical column has too many rare categories, group them:
  df['col'] = df['col'].where(df['col'].map(df['col'].value_counts()) > 10, 'other')
- Create hierarchical groupings when categories have semantic structure
- Map ordinal categories to numbers when appropriate

Provided Framework Tools:
You can import and use these pre-built BlueCast stateless functions to save time and reduce errors:
- `from bluecast.preprocessing.feature_creation import add_groupby_agg_feats`
  Signature: add_groupby_agg_feats(df, groupby_cols=['...'], to_group_cols=['...'], num_col_prefix='agg', target_col='target', aggregations=['min', 'max', 'mean'])
- `from bluecast.preprocessing.feature_creation import add_polynomial_features`
  Signature: add_polynomial_features(df, cols=['...'], degree=2)
- `from bluecast.preprocessing.feature_creation import add_interaction_features`
  Signature: add_interaction_features(df, cols_a=['...'], cols_b=['...'], operations=['mul', 'div', 'add', 'sub'])
- `from bluecast.preprocessing.feature_creation import add_binned_features`
  Signature: add_binned_features(df, cols=['...'], num_bins=5)
- `from bluecast.preprocessing.feature_creation import add_datetime_features`
  Signature: add_datetime_features(df, date_cols=['...'])

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
