"""Data analyst agent: profiles data, checks quality, detects issues."""

from typing import List

from bluecast.ai.agents.base import BaseAgent
from bluecast.ai.providers.base import ToolDefinition
from bluecast.ai.tools import (
    TOOL_DEFINITIONS,
    tool_check_correlations,
    tool_check_leakage,
    tool_describe_data,
)


class DataAnalystAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_tool_impl(
            "describe_data",
            lambda **kw: tool_describe_data(
                self.context.df_train, self.context.target_col
            ),
        )
        self.register_tool_impl(
            "check_correlations",
            lambda threshold=0.8, **kw: tool_check_correlations(
                self.context.df_train, self.context.target_col, threshold
            ),
        )
        self.register_tool_impl(
            "check_leakage",
            lambda **kw: tool_check_leakage(
                self.context.df_train, self.context.target_col
            ),
        )

    @property
    def name(self) -> str:
        return "DataAnalyst"

    def system_prompt(self) -> str:
        data_summary = self.context.get_data_summary()
        return f"""You are a data analyst for the BlueCast AutoML framework.

Your job is to thoroughly profile the dataset, identify data quality issues,
and provide actionable insights for feature engineering and modelling.

Use your tools to:
1. Describe the data (types, distributions, nulls)
2. Check for high correlations
3. Check for potential target leakage

After analysis, provide a structured summary with:
- Problem type (binary/multiclass/regression)
- Key data quality issues
- Features that are most/least important
- Recommendations for feature engineering
- Warnings about leakage, class imbalance, or other issues

Current dataset overview:
{data_summary}"""

    def get_tools(self) -> List[ToolDefinition]:
        return [
            TOOL_DEFINITIONS["describe_data"],
            TOOL_DEFINITIONS["check_correlations"],
            TOOL_DEFINITIONS["check_leakage"],
        ]
