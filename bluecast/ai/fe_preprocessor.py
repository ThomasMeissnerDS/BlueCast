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
        self.snippet_fallbacks: dict = {}
        self.robust_snippets: List[str] = []

    def _simulate_unseen_data(self, code: str, df: pd.DataFrame) -> bool:
        import copy

        try:
            dummy = df.sample(min(50, len(df)), replace=True).copy()
            for col in dummy.columns:
                mask = np.random.rand(len(dummy)) < 0.05
                if pd.api.types.is_numeric_dtype(dummy[col]):
                    dummy.loc[mask, col] = np.nan
                else:
                    dummy.loc[mask, col] = "UNSEEN_VALUE"

            # Use deepcopy to prevent state corruption during simulation
            test_state = copy.deepcopy(self.state)
            local_vars = {
                "df": dummy,
                "np": np,
                "pd": pd,
                "state": test_state,
                "is_fit": False,
            }
            exec(code, local_vars)  # noqa: S102
            return True
        except Exception as e:
            logger.warning(f"Snippet rejected during simulation: {e}")
            return False

    def _apply_snippets(self, df: pd.DataFrame, is_fit: bool) -> pd.DataFrame:
        snippets_to_run = self.code_snippets if is_fit else self.robust_snippets

        for i, code in enumerate(snippets_to_run):
            pre_cols = set(df.columns)
            try:
                if is_fit:
                    # Test snippet robustness before applying
                    if not self._simulate_unseen_data(code, df):
                        continue

                local_vars = {
                    "df": df,
                    "np": np,
                    "pd": pd,
                    "state": self.state,
                    "is_fit": is_fit,
                }
                exec(code, local_vars)  # noqa: S102
                df = local_vars.get("df", df)

                if is_fit:
                    self.robust_snippets.append(code)
                    new_cols = list(set(df.columns) - pre_cols)
                    if new_cols:
                        fallbacks = {}
                        for col in new_cols:
                            if pd.api.types.is_numeric_dtype(df[col]):
                                fallbacks[col] = df[col].median()
                            else:
                                fallbacks[col] = (
                                    df[col].mode()[0]
                                    if not df[col].mode().empty
                                    else "MISSING"
                                )
                        self.snippet_fallbacks[i] = fallbacks

            except Exception as e:
                logger.warning(
                    f"FE snippet failed during {'fit' if is_fit else 'transform'} "
                    f"({type(e).__name__}: {e})."
                )
                if not is_fit and i in self.snippet_fallbacks:
                    logger.warning("Applying graceful degradation fallbacks.")
                    for col, val in self.snippet_fallbacks[i].items():
                        if col not in df.columns:
                            df[col] = val
        return df

    def fit_transform(
        self, df: pd.DataFrame, target: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        self.state.clear()
        self.snippet_fallbacks.clear()
        self.robust_snippets.clear()
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
