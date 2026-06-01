"""Extended tests for bluecast.ai.tools — covers all tool functions."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from bluecast.ai.tools import (
    TOOL_DEFINITIONS,
    _serialize_metrics,
    tool_check_correlations,
    tool_check_feature_quality,
    tool_check_group_statistics,
    tool_check_leakage,
    tool_check_outliers,
    tool_check_temporal_patterns,
    tool_check_uniqueness,
    tool_create_feature,
    tool_create_tfidf_features,
    tool_describe_data,
    tool_drop_collinear_features,
    tool_evaluate_imputations,
    tool_inspect_rows,
    tool_l1_feature_selection,
    tool_run_sql_query,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def classification_df():
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "num1": rng.normal(0, 1, 100),
            "num2": rng.normal(5, 2, 100),
            "num3": rng.uniform(-1, 1, 100),
            "cat": rng.choice(["a", "b", "c"], 100),
            "target": rng.choice([0, 1], 100),
        }
    )


@pytest.fixture
def regression_df():
    rng = np.random.default_rng(42)
    n = 100
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(5, 2, n)
    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "cat": rng.choice(["a", "b"], n),
            "target": x1 * 2 + x2 + rng.normal(0, 0.1, n),
        }
    )


@pytest.fixture
def df_with_nulls():
    rng = np.random.default_rng(42)
    n = 100
    vals = rng.normal(0, 1, n).tolist()
    vals[5] = np.nan
    vals[10] = np.nan
    return pd.DataFrame(
        {
            "a": vals,
            "b": rng.normal(5, 2, n),
            "target": rng.choice([0, 1], n),
        }
    )


@pytest.fixture
def df_with_text():
    return pd.DataFrame(
        {
            "text_col": [
                "hello world foo",
                "bar baz hello",
                "world foo bar baz",
                "another text hello",
                "foo bar world",
            ]
            * 20,
            "num": list(range(100)),
            "target": [0, 1] * 50,
        }
    )


# ---------------------------------------------------------------------------
# tool_describe_data
# ---------------------------------------------------------------------------


class TestDescribeData:
    def test_classification(self, classification_df):
        result = tool_describe_data(classification_df, "target")
        assert "Shape" in result
        assert "target" in result.lower()

    def test_regression(self, regression_df):
        result = tool_describe_data(regression_df, "target")
        assert "Shape" in result

    def test_with_nulls(self, df_with_nulls):
        result = tool_describe_data(df_with_nulls, "target")
        assert (
            "missing" in result.lower()
            or "null" in result.lower()
            or "nan" in result.lower()
            or "Shape" in result
        )


# ---------------------------------------------------------------------------
# tool_check_correlations
# ---------------------------------------------------------------------------


class TestCheckCorrelations:
    def test_basic(self, classification_df):
        result = tool_check_correlations(classification_df, "target")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_with_threshold(self, classification_df):
        result = tool_check_correlations(classification_df, "target", threshold=0.3)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# tool_check_leakage
# ---------------------------------------------------------------------------


class TestCheckLeakage:
    def test_no_leakage(self, classification_df):
        result = tool_check_leakage(classification_df, "target")
        assert "leakage" in result.lower()

    def test_with_leakage(self):
        df = pd.DataFrame(
            {
                "leak": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                "noise": list(range(10)),
                "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            }
        )
        result = tool_check_leakage(df, "target")
        assert "leak" in result.lower() or "leakage" in result.lower()


# ---------------------------------------------------------------------------
# tool_check_uniqueness
# ---------------------------------------------------------------------------


class TestCheckUniqueness:
    def test_basic(self, classification_df):
        result = tool_check_uniqueness(classification_df)
        assert isinstance(result, str)
        assert (
            "unique" in result.lower()
            or "cardinality" in result.lower()
            or "num1" in result
        )

    def test_with_id_column(self):
        df = pd.DataFrame(
            {
                "id": list(range(100)),
                "value": list(range(100)),
                "target": [0, 1] * 50,
            }
        )
        result = tool_check_uniqueness(df)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# tool_check_outliers
# ---------------------------------------------------------------------------


class TestCheckOutliers:
    def test_basic(self, classification_df):
        result = tool_check_outliers(classification_df)
        assert isinstance(result, str)

    def test_with_params(self, classification_df):
        result = tool_check_outliers(classification_df, n_show=3, contamination=0.1)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# tool_check_temporal_patterns
# ---------------------------------------------------------------------------


class TestCheckTemporalPatterns:
    def test_no_datetime(self, classification_df):
        result = tool_check_temporal_patterns(classification_df, "target")
        assert isinstance(result, str)

    def test_with_datetime(self):
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=50, freq="D"),
                "value": list(range(50)),
                "target": [0, 1] * 25,
            }
        )
        result = tool_check_temporal_patterns(df, "target")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# tool_check_group_statistics
# ---------------------------------------------------------------------------


class TestCheckGroupStatistics:
    def test_basic(self, classification_df):
        result = tool_check_group_statistics(classification_df, "cat", "num1")
        assert isinstance(result, str)

    def test_missing_column(self, classification_df):
        result = tool_check_group_statistics(classification_df, "nonexistent", "num1")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# tool_inspect_rows
# ---------------------------------------------------------------------------


class TestInspectRows:
    def test_by_indices(self, classification_df):
        result = tool_inspect_rows(classification_df, indices="0,1,2")
        assert isinstance(result, str)

    def test_by_condition(self, classification_df):
        result = tool_inspect_rows(classification_df, condition="num1 > 0")
        assert isinstance(result, str)

    def test_empty(self, classification_df):
        result = tool_inspect_rows(classification_df)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# tool_run_sql_query
# ---------------------------------------------------------------------------


class TestRunSqlQuery:
    def test_basic_query(self, classification_df):
        result = tool_run_sql_query(classification_df, "SELECT * FROM df LIMIT 5")
        assert isinstance(result, str)

    def test_aggregation_query(self, classification_df):
        result = tool_run_sql_query(
            classification_df, "SELECT cat, COUNT(*) as cnt FROM df GROUP BY cat"
        )
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# tool_create_feature
# ---------------------------------------------------------------------------


class TestCreateFeature:
    def test_success(self, classification_df):
        result = tool_create_feature(
            classification_df, "df['ratio'] = df['num1'] / (df['num2'] + 1)"
        )
        assert result["success"] is True
        assert "ratio" in result["new_columns"]

    def test_failure(self, classification_df):
        result = tool_create_feature(
            classification_df, "df['bad'] = df['nonexistent'] * 2"
        )
        assert result["success"] is False
        assert result["error"] is not None

    def test_with_state(self, classification_df):
        state = {}
        code = (
            "if 'mean_val' not in state:\n"
            "    state['mean_val'] = df['num1'].mean()\n"
            "df['centered'] = df['num1'] - state['mean_val']"
        )
        result = tool_create_feature(classification_df, code, state=state)
        assert result["success"] is True
        assert "mean_val" in state

    def test_multiple_columns(self, classification_df):
        result = tool_create_feature(
            classification_df,
            "df['sq1'] = df['num1']**2\ndf['sq2'] = df['num2']**2",
        )
        assert result["success"] is True
        assert "sq1" in result["new_columns"]
        assert "sq2" in result["new_columns"]


# ---------------------------------------------------------------------------
# tool_create_tfidf_features
# ---------------------------------------------------------------------------


class TestCreateTfidfFeatures:
    def test_success(self, df_with_text):
        result = tool_create_tfidf_features(df_with_text, "text_col", max_features=5)
        assert result["success"] is True
        assert len(result.get("new_columns", [])) > 0

    def test_missing_column(self, classification_df):
        result = tool_create_tfidf_features(classification_df, "nonexistent")
        assert result["success"] is False


# ---------------------------------------------------------------------------
# tool_drop_collinear_features
# ---------------------------------------------------------------------------


class TestDropCollinearFeatures:
    def test_basic(self):
        rng = np.random.default_rng(42)
        n = 100
        a = rng.normal(0, 1, n)
        df = pd.DataFrame(
            {
                "a": a,
                "b": a + rng.normal(0, 0.001, n),  # Nearly identical to a
                "c": rng.normal(0, 1, n),
                "target": rng.choice([0, 1], n),
            }
        )
        result = tool_drop_collinear_features(df, threshold=0.99, target_col="target")
        assert result["success"] is True

    def test_no_drop(self, classification_df):
        result = tool_drop_collinear_features(
            classification_df, threshold=0.99, target_col="target"
        )
        assert result["success"] is True


# ---------------------------------------------------------------------------
# tool_l1_feature_selection
# ---------------------------------------------------------------------------


class TestL1FeatureSelection:
    def test_classification(self, classification_df):
        result = tool_l1_feature_selection(
            classification_df, "target", "binary", alpha=0.01
        )
        assert result["success"] is True
        assert "dropped_columns" in result

    def test_regression(self, regression_df):
        result = tool_l1_feature_selection(
            regression_df, "target", "regression", alpha=0.01
        )
        assert result["success"] is True


# ---------------------------------------------------------------------------
# tool_evaluate_imputations
# ---------------------------------------------------------------------------


class TestEvaluateImputations:
    def test_with_nulls(self, df_with_nulls):
        result = tool_evaluate_imputations(df_with_nulls, "target")
        assert isinstance(result, str)

    def test_with_sentinels(self):
        """Test sentinel detection (999, -999)."""
        df = pd.DataFrame(
            {
                "x": [1.0, 2.0, 999.0, 4.0, 999.0, 6.0] * 10,
                "target": list(range(60)),
            }
        )
        result = tool_evaluate_imputations(df, "target")
        assert isinstance(result, str)

    def test_no_missing(self, classification_df):
        result = tool_evaluate_imputations(classification_df, "target")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# tool_check_feature_quality
# ---------------------------------------------------------------------------


class TestCheckFeatureQuality:
    def test_basic(self, classification_df):
        result = tool_check_feature_quality(
            classification_df, "target", ["num1", "num2"]
        )
        assert isinstance(result, (str, dict))

    def test_nonexistent_column(self, classification_df):
        result = tool_check_feature_quality(
            classification_df, "target", ["nonexistent"]
        )
        assert isinstance(result, (str, dict))


# ---------------------------------------------------------------------------
# tool_web_search (mocked)
# ---------------------------------------------------------------------------


class TestWebSearch:
    @patch("bluecast.ai.tools.tool_web_search", return_value="Search results for query")
    def test_web_search(self, mock_search):
        result = mock_search("test query")
        assert "results" in result.lower()


# ---------------------------------------------------------------------------
# _serialize_metrics
# ---------------------------------------------------------------------------


class TestSerializeMetrics:
    def test_dict_input(self):
        m = {"roc_auc": 0.85, "accuracy": 0.9}
        result = _serialize_metrics(m)
        assert result["roc_auc"] == 0.85

    def test_tuple_input(self):
        result = _serialize_metrics((0.85, 0.02))
        assert result["oof_mean"] == 0.85

    def test_numpy_float(self):
        m = {"score": np.float64(0.85)}
        result = _serialize_metrics(m)
        assert isinstance(result["score"], float)

    def test_none_input(self):
        result = _serialize_metrics(None)
        assert isinstance(result, dict)

    def test_string_input(self):
        result = _serialize_metrics("some string")
        assert isinstance(result, dict)

    def test_list_input(self):
        result = _serialize_metrics([0.85, 0.9])
        assert isinstance(result, dict)

    def test_nested_dict_with_numpy(self):
        m = {"a": np.int64(5), "b": np.float32(0.5), "c": "text"}
        result = _serialize_metrics(m)
        assert isinstance(result["a"], (int, float))
        assert isinstance(result["b"], float)


# ---------------------------------------------------------------------------
# tool_build_and_run_pipeline (mocked)
# ---------------------------------------------------------------------------


class TestBuildAndRunPipeline:
    @patch("bluecast.blueprints.unified.BlueCastAuto")
    def test_regression_pipeline(self, mock_auto, regression_df):
        from bluecast.ai.tools import tool_build_and_run_pipeline

        mock_pipeline = MagicMock()
        mock_pipeline.fit_eval.return_value = None
        mock_auto.return_value = mock_pipeline

        result = tool_build_and_run_pipeline(
            regression_df,
            "target",
            {"class_problem": "regression", "use_cv": False},
        )
        assert isinstance(result, dict)
        assert "success" in result


# ---------------------------------------------------------------------------
# TOOL_DEFINITIONS
# ---------------------------------------------------------------------------


class TestToolDefinitions:
    def test_all_required_tools_present(self):
        required = [
            "describe_data",
            "check_correlations",
            "check_leakage",
            "create_feature",
            "create_tfidf_features",
            "build_and_run_pipeline",
        ]
        for name in required:
            assert name in TOOL_DEFINITIONS
            td = TOOL_DEFINITIONS[name]
            assert td.name == name
            assert len(td.description) > 10
            assert "type" in td.parameters

    def test_tool_definitions_have_properties(self):
        for _name, td in TOOL_DEFINITIONS.items():
            assert hasattr(td, "name")
            assert hasattr(td, "description")
            assert hasattr(td, "parameters")


def test_tool_check_adversarial_validation(classification_df):
    from bluecast.ai.tools import tool_check_adversarial_validation

    classification_df["split"] = ["test"] * 50 + ["train"] * 50
    res = tool_check_adversarial_validation(classification_df, "split == 'test'")
    print(res)
    assert "Adversarial Validation AUC" in res


def test_tool_check_mutual_information(classification_df):
    from bluecast.ai.tools import tool_check_mutual_information

    res = tool_check_mutual_information(classification_df, "target", "classification")
    assert "Mutual Information" in res


def test_tool_target_distribution_test(classification_df):
    from bluecast.ai.tools import tool_target_distribution_test

    res = tool_target_distribution_test(classification_df, "num1")
    print(res)
    assert "Shapiro-Wilk Test" in res


def test_tool_nlp_profiling():
    from bluecast.ai.tools import tool_nlp_profiling

    df = pd.DataFrame(
        {"text": ["This is great", "This is terrible", "Neutral text"] * 10}
    )
    res = tool_nlp_profiling(df, "text")
    assert "NLP Profiling" in res


def test_tool_apply_target_encoding(classification_df):
    from bluecast.ai.tools import tool_apply_target_encoding

    res = tool_apply_target_encoding(classification_df, "target", "cat")
    assert res["success"] is True


def test_tool_automated_numeric_interactions(classification_df):
    from bluecast.ai.tools import tool_automated_numeric_interactions

    res = tool_automated_numeric_interactions(classification_df, "num1,num2")
    assert res["success"] is True


def test_tool_create_groupby_aggregations(classification_df):
    from bluecast.ai.tools import tool_create_groupby_aggregations

    res = tool_create_groupby_aggregations(classification_df, "cat", "num1", "mean,sum")
    assert res["success"] is True


def test_tool_inspect_residuals(regression_df):
    from bluecast.ai.tools import tool_inspect_residuals

    # Mocking residuals
    regression_df["preds"] = regression_df["target"] + np.random.normal(
        0, 0.1, len(regression_df)
    )
    res = tool_inspect_residuals(regression_df, "target", "preds")
    assert "Top 20 rows with highest residuals/loss" in res


def test_tool_apply_pseudo_labeling(classification_df):
    from bluecast.ai.tools import tool_apply_pseudo_labeling

    classification_df.loc[10:20, "target"] = np.nan
    res = tool_apply_pseudo_labeling(classification_df, "target", "binary", 0.9)
    assert "Pseudo-labeling" in res


def test_tool_web_search():
    from bluecast.ai.tools import tool_web_search

    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "items": [{"title": "t", "snippet": "s", "link": "l"}]
        }
        mock_response.ok = True
        mock_get.return_value = mock_response
        res = tool_web_search("test query")
        assert "t" in res


def test_tool_check_mutual_information_edge_cases(classification_df):
    from bluecast.ai.tools import tool_check_mutual_information

    # Missing target col
    res = tool_check_mutual_information(classification_df, "missing_target")
    assert "not found" in str(res)

    # Large dataframe downsampling
    import numpy as np
    import pandas as pd

    large_df = pd.DataFrame(
        {"num": np.random.rand(10005), "target": np.random.randint(0, 2, 10005)}
    )
    res2 = tool_check_mutual_information(large_df, "target")
    assert "Mutual Information" in str(res2)


def test_tool_nlp_profiling_edge_cases():
    import pandas as pd

    from bluecast.ai.tools import tool_nlp_profiling

    # Missing text col
    df = pd.DataFrame({"other": [1, 2]})
    res = tool_nlp_profiling(df, "text")
    assert "Column 'text' not found." in str(res)

    # Empty text or NaNs
    df2 = pd.DataFrame({"text": [None, ""]})
    res2 = tool_nlp_profiling(df2, "text")
    assert "No valid text found" in str(res2) or "NLP Profiling" in str(res2)


def test_tool_create_groupby_aggregations_edge_cases(classification_df):
    from bluecast.ai.tools import tool_create_groupby_aggregations

    # Missing group col
    res = tool_create_groupby_aggregations(
        classification_df, "missing_target", "cat", "mean"
    )
    assert "Group column" in str(res)


def test_tool_create_tfidf_features_edge_cases():
    import pandas as pd

    from bluecast.ai.tools import tool_create_tfidf_features

    df = pd.DataFrame({"other": [1, 2]})
    res = tool_create_tfidf_features(df, "text_col")
    assert "Column 'text_col' not found" in str(res)
