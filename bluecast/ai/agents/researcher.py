"""Researcher agent: searches the web for domain knowledge and techniques."""

from typing import List

from bluecast.ai.agents.base import BaseAgent
from bluecast.ai.providers.base import ToolDefinition
from bluecast.ai.tools import TOOL_DEFINITIONS, tool_web_search


class ResearcherAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_tool_impl(
            "web_search", lambda query, **kw: tool_web_search(query)
        )

    @property
    def name(self) -> str:
        return "Researcher"

    def system_prompt(self) -> str:
        return """You are a research assistant for the BlueCast AutoML framework.

Your job is to search the web for relevant information about:
- Machine learning techniques for the specific problem type
- Domain knowledge relevant to the dataset
- Feature engineering ideas from Kaggle competitions or papers
- Best practices for the specific data characteristics

Use the web_search tool with specific queries.
Summarize your findings concisely, focusing on actionable insights."""

    def get_tools(self) -> List[ToolDefinition]:
        return [TOOL_DEFINITIONS["web_search"]]
