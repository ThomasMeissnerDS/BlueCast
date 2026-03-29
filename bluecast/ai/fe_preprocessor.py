"""Dynamic CustomPreprocessing that replays LLM-generated feature code."""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from bluecast.preprocessing.custom import CustomPreprocessing

logger = logging.getLogger(__name__)


class AIFeaturePreprocessor(CustomPreprocessing):
    """Replay feature-engineering code snippets from the AI agent pipeline.

    Each snippet is a string of Python code that was successfully executed
    by the FeatureEngineer agent during training.  The same snippets are
    re-executed at inference time so that the test DataFrame has the same
    schema as the training DataFrame.

    The code runs with ``df``, ``np``, and ``pd`` in scope — identical to
    the ``tool_create_feature`` execution environment.
    """

    def __init__(self, code_snippets: Optional[List[str]] = None):
        super().__init__()
        self.code_snippets: List[str] = code_snippets or []
        self.state: dict = {}

    def _apply_snippets(self, df: pd.DataFrame, is_fit: bool) -> pd.DataFrame:
        """Execute every stored snippet against *df* in order.

        Each snippet is wrapped in try/except so that a single broken
        snippet does not crash the entire pipeline.  The snippet is
        skipped and a warning is logged.
        """
        for i, code in enumerate(self.code_snippets):
            try:
                local_vars = {
                    "df": df,
                    "np": np,
                    "pd": pd,
                    "state": self.state,
                    "is_fit": is_fit,
                }
                exec(code, {}, local_vars)  # noqa: S102
                df = local_vars.get("df", df)
            except Exception as e:
                logger.warning(
                    f"FE snippet {i + 1}/{len(self.code_snippets)} "
                    f"failed ({type(e).__name__}: {e}), skipping."
                )
        return df

    def fit_transform(
        self, df: pd.DataFrame, target: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        df = self._apply_snippets(df, is_fit=True)
        return df, target

    def transform(
        self,
        df: pd.DataFrame,
        target: Optional[pd.Series] = None,
        prediction_mode: bool = False,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        df = self._apply_snippets(df, is_fit=False)
        return df, target
