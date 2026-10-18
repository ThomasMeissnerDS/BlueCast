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
        import collections

        self.states: dict = {}
        self.state: collections.ChainMap = collections.ChainMap()
        self.snippet_fallbacks: dict = {}
        self.robust_snippets: List[str] = []

    def _simulate_unseen_data(
        self,
        code: str,
        df: pd.DataFrame,
        snippet_state: dict,
        expected_new_cols: Optional[List[str]] = None,
    ) -> bool:
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
            test_state = copy.deepcopy(snippet_state)
            local_vars = {
                "df": dummy,
                "np": np,
                "pd": pd,
                "state": test_state,
                "is_fit": False,
            }
            exec(code, local_vars)  # noqa: S102

            if expected_new_cols:
                dummy_after = local_vars.get("df", dummy)
                missing = [c for c in expected_new_cols if c not in dummy_after.columns]
                if missing:
                    logger.warning(
                        f"Snippet rejected: missing expected columns {missing} during inference simulation."
                    )
                    return False

            return True
        except Exception as e:
            logger.warning(f"Snippet rejected during simulation: {e}")
            return False

    def _apply_snippets(self, df: pd.DataFrame, is_fit: bool) -> pd.DataFrame:
        snippets_to_run = self.code_snippets if is_fit else self.robust_snippets

        for i, code in enumerate(snippets_to_run):
            pre_cols = set(df.columns)

            # Use isolated state per snippet
            if is_fit:
                if i not in self.states:
                    self.states[i] = {}

            snippet_state = self.states.get(i, {})

            try:
                # Use a copy during is_fit to allow rollback if the snippet is invalid
                work_df = df.copy() if is_fit else df

                local_vars = {
                    "df": work_df,
                    "np": np,
                    "pd": pd,
                    "state": snippet_state,
                    "is_fit": is_fit,
                }
                exec(code, local_vars)  # noqa: S102
                df_after = local_vars.get("df", work_df)

                if is_fit:
                    new_cols = list(set(df_after.columns) - pre_cols)

                    # Test if the snippet successfully creates the SAME columns on unseen inference data
                    if not self._simulate_unseen_data(
                        code, df, snippet_state, expected_new_cols=new_cols
                    ):
                        continue

                    # If simulation passed, commit the changes
                    df = df_after
                    self.robust_snippets.append(code)
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
                else:
                    df = df_after

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

        # Update global ChainMap state for backward compatibility
        if is_fit:
            import collections

            self.state = collections.ChainMap(
                *[self.states[i] for i in range(len(self.states) - 1, -1, -1)]
            )

        return df

    def fit_transform(
        self, df: pd.DataFrame, target: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        import collections

        self.states.clear()
        self.state = collections.ChainMap()
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
