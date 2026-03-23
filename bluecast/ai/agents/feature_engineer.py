"""Feature engineer agent: creates new features based on data analysis."""

from typing import List

from bluecast.ai.agents.base import BaseAgent
from bluecast.ai.providers.base import ToolDefinition
from bluecast.ai.tools import TOOL_DEFINITIONS, tool_create_feature


class FeatureEngineerAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_tool_impl(
            "create_feature",
            self._create_feature_wrapper,
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
            existing_code = self.context.feature_engineering_code or ""
            self.context.feature_engineering_code = (
                existing_code + f"\n# {description}\n{feature_code}\n"
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
- Create frequency-encoded features from categoricals
- Handle missing values if needed (fill or create indicator columns)
- Do NOT drop existing columns (the model may need them)
- Keep feature names descriptive and unique
- If a feature creation fails, try a different approach

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
        return [TOOL_DEFINITIONS["create_feature"]]
