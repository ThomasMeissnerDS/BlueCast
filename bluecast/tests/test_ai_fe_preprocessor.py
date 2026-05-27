"""Tests for AIFeaturePreprocessor."""

import numpy as np
import pandas as pd

from bluecast.ai.fe_preprocessor import AIFeaturePreprocessor


class TestAIFeaturePreprocessor:
    def test_fit_transform_basic(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        target = pd.Series([0, 1, 0])
        
        snippets = [
            "df['c'] = df['a'] + df['b']",
            "df['d'] = df['a'] * 2"
        ]
        
        prep = AIFeaturePreprocessor(snippets)
        df_out, target_out = prep.fit_transform(df.copy(), target)
        
        assert "c" in df_out.columns
        assert "d" in df_out.columns
        assert df_out["c"].tolist() == [5, 7, 9]
        assert len(prep.robust_snippets) == 2

    def test_transform_reuses_state(self):
        df_train = pd.DataFrame({"a": [1, 2, 3]})
        df_test = pd.DataFrame({"a": [4, 5, 6]})
        
        # Snippet uses state to store the mean during fit
        snippets = [
            "if 'mean' not in state:\n"
            "    state['mean'] = df['a'].mean()\n"
            "df['centered'] = df['a'] - state['mean']"
        ]
        
        prep = AIFeaturePreprocessor(snippets)
        
        # Fit phase: mean of [1, 2, 3] is 2.0
        prep.fit_transform(df_train, pd.Series([0, 1, 0]))
        assert prep.state["mean"] == 2.0
        
        # Transform phase: should use 2.0, not mean of test data
        df_out, _ = prep.transform(df_test)
        assert df_out["centered"].tolist() == [2.0, 3.0, 4.0]

    def test_simulate_unseen_data(self):
        prep = AIFeaturePreprocessor([])
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "x", "y", "z"]})
        
        # This code should work on unseen data (adds a column safely)
        good_code = "df['c'] = df['a'] * 10"
        assert prep._simulate_unseen_data(good_code, df) is True
        
        # This code will crash on unseen data (assumes a missing column)
        bad_code = "df['c'] = df['nonexistent'] * 10"
        assert prep._simulate_unseen_data(bad_code, df) is False

    def test_snippet_rejection(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        target = pd.Series([0, 1, 0])
        
        snippets = [
            "df['good'] = df['a'] + 1",
            "df['bad'] = df['nonexistent'] + 1",
            "df['good2'] = df['a'] + 2"
        ]
        
        prep = AIFeaturePreprocessor(snippets)
        df_out, _ = prep.fit_transform(df.copy(), target)
        
        # Only robust snippets should be applied
        assert "good" in df_out.columns
        assert "good2" in df_out.columns
        assert "bad" not in df_out.columns
        assert len(prep.robust_snippets) == 2

    def test_snippet_failure_graceful_fallback(self):
        # A snippet works on train but fails on test (e.g. unseen category)
        df_train = pd.DataFrame({"a": [1, 2, 3]})
        df_test = pd.DataFrame({"a": [4, 5, 6]})
        
        snippets = [
            "if is_fit:\n"
            "    df['new_col'] = df['a'] * 2\n"
            "else:\n"
            "    raise ValueError('Simulated failure during transform')"
        ]
        
        prep = AIFeaturePreprocessor(snippets)
        df_train_out, _ = prep.fit_transform(df_train, pd.Series([0, 1, 0]))
        
        # new_col is created during fit
        assert "new_col" in df_train_out.columns
        assert prep.snippet_fallbacks[0]["new_col"] == 4.0  # median of 2, 4, 6
        
        # Transform fails, should use fallback median
        df_test_out, _ = prep.transform(df_test)
        assert "new_col" in df_test_out.columns
        assert df_test_out["new_col"].tolist() == [4.0, 4.0, 4.0]

    def test_fit_transform_clears_state(self):
        df = pd.DataFrame({"a": [1, 2]})
        snippets = ["state['key'] = 1"]
        prep = AIFeaturePreprocessor(snippets)
        
        prep.fit_transform(df, pd.Series([0, 1]))
        assert "key" in prep.state
        
        # Second fit_transform should clear the state
        snippets = ["state['key2'] = 2"]
        prep.code_snippets = snippets
        prep.fit_transform(df, pd.Series([0, 1]))
        
        assert "key2" in prep.state
        assert "key" not in prep.state
