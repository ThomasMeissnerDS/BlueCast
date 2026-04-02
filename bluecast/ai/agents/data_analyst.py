"""Data analyst agent: profiles data, checks quality, detects issues."""

from typing import List

from bluecast.ai.agents.base import BaseAgent
from bluecast.ai.providers.base import ToolDefinition
from bluecast.ai.tools import (
    TOOL_DEFINITIONS,
    tool_check_correlations,
    tool_check_group_statistics,
    tool_check_leakage,
    tool_check_outliers,
    tool_check_temporal_patterns,
    tool_check_uniqueness,
    tool_describe_data,
    tool_inspect_rows,
    tool_run_sql_query,
    tool_web_search,
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
        self.register_tool_impl(
            "check_uniqueness",
            lambda **kw: tool_check_uniqueness(self.context.df_train),
        )
        self.register_tool_impl(
            "check_outliers",
            lambda n_show=5, contamination=0.05, **kw: tool_check_outliers(
                self.context.df_train, n_show, contamination
            ),
        )
        self.register_tool_impl(
            "inspect_rows",
            lambda indices="", condition="", **kw: tool_inspect_rows(
                self.context.df_train, indices, condition
            ),
        )
        self.register_tool_impl(
            "run_sql_query",
            lambda query, **kw: tool_run_sql_query(self.context.df_train, query),
        )
        self.register_tool_impl(
            "check_temporal_patterns",
            lambda **kw: tool_check_temporal_patterns(
                self.context.df_train, self.context.target_col
            ),
        )
        self.register_tool_impl(
            "check_group_statistics",
            lambda group_col, agg_col, **kw: tool_check_group_statistics(
                self.context.df_train, group_col, agg_col
            ),
        )
        self.register_tool_impl(
            "web_search",
            lambda query, **kw: tool_web_search(query),
        )

    @property
    def name(self) -> str:
        return "DataAnalyst"

    def system_prompt(self) -> str:
        data_summary = self.context.get_data_summary()
        domain_ctx = ""
        if self.context.context_file_contents:
            domain_ctx = "\n\nDomain knowledge provided:\n" + "\n".join(
                self.context.context_file_contents[:3]
            )

        return f"""You are a senior data analyst for the BlueCast AutoML framework.

Your job is to thoroughly profile the dataset, identify data quality issues,
detect oddities, and provide actionable insights for feature engineering and modelling.

Use your tools systematically:
1. **describe_data** — Get dtypes, distributions, nulls, target overview
2. **check_uniqueness** — Find ID columns, understand cardinality
3. **check_correlations** — Find highly correlated features and target relationships
4. **check_leakage** — Detect potential target leakage
5. **check_outliers** — Use IsolationForest to find anomalous rows
6. **inspect_rows** — Drill into suspicious rows from outlier detection
7. **check_temporal_patterns** — Detect datetime columns, check for drift
8. **check_group_statistics** — Analyze distributions within groups
9. **run_sql_query** — Write custom SQL for deep-dive analysis
10. **web_search** — Look up domain knowledge about the dataset or problem

After analysis, provide a comprehensive structured summary with:
- Problem type (binary/multiclass/regression)
- Key data quality issues (nulls, duplicates, constant columns)
- Outlier analysis: how many, which features drive them, should they be removed?
- Feature importance signals (correlations with target)
- Cardinality analysis: which columns are IDs, which are useful categoricals
- Temporal patterns: is there data drift over time?
- Group-level insights: do certain groups behave differently?
- Specific recommendations for tree-based models (which handle missing values and unencoded categories naturally).
- Specific recommendations for linear models (which strictly require missing value imputation, scaling, and categorical encoding).
- Warnings about leakage, class imbalance, or other issues

Be thorough. Look for oddities the model might struggle with.

Current dataset overview:
{data_summary}{domain_ctx}"""

    def get_tools(self) -> List[ToolDefinition]:
        return [
            TOOL_DEFINITIONS["describe_data"],
            TOOL_DEFINITIONS["check_correlations"],
            TOOL_DEFINITIONS["check_leakage"],
            TOOL_DEFINITIONS["check_uniqueness"],
            TOOL_DEFINITIONS["check_outliers"],
            TOOL_DEFINITIONS["inspect_rows"],
            TOOL_DEFINITIONS["run_sql_query"],
            TOOL_DEFINITIONS["check_temporal_patterns"],
            TOOL_DEFINITIONS["check_group_statistics"],
            TOOL_DEFINITIONS["web_search"],
        ]
