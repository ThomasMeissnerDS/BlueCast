"""Tests for bluecast.eda.dashboard — covers CSS, layout, summary, query filter."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "num1": rng.normal(0, 1, 50),
            "num2": rng.normal(5, 2, 50),
            "cat": rng.choice(["a", "b", "c"], 50),
            "target": rng.choice([0, 1], 50),
        }
    )


@pytest.fixture
def regression_df():
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "x1": rng.normal(0, 1, 50),
            "x2": rng.normal(5, 2, 50),
            "target": rng.normal(10, 3, 50),
        }
    )


class TestBuildDashboardCSS:
    def test_css_contains_accent(self):
        from bluecast.eda.dashboard import build_dashboard_css

        css = build_dashboard_css(
            "#667eea", "linear-gradient(135deg, #667eea, #764ba2)"
        )
        assert "#667eea" in css
        assert "linear-gradient" in css
        assert "body" in css
        assert "main-container" in css

    def test_css_classification(self):
        from bluecast.eda.dashboard import (
            CLASSIFICATION_ACCENT,
            CLASSIFICATION_GRADIENT,
            build_dashboard_css,
        )

        css = build_dashboard_css(CLASSIFICATION_ACCENT, CLASSIFICATION_GRADIENT)
        assert CLASSIFICATION_ACCENT in css


class TestBuildDashboardLayout:
    def test_layout_structure(self):
        try:
            import dash  # noqa: F401
        except ImportError:
            pytest.skip("dash not installed")

        from bluecast.eda.dashboard import build_dashboard_layout

        layout = build_dashboard_layout(
            title="Test Dashboard",
            data_info_lines=["50 rows", "4 columns"],
            plot_options=[{"label": "Correlation", "value": "correlation"}],
            feature_x_options=[{"label": "num1", "value": "num1"}],
            feature_y_options=[{"label": "num2", "value": "num2"}],
            default_feature_x="num1",
            default_feature_y="num2",
        )
        assert layout is not None

    def test_layout_import_error(self):
        import sys

        with patch.dict(
            sys.modules, {"dash": None, "dash.dcc": None, "dash.html": None}
        ):
            # Force reimport to trigger ImportError
            try:
                import dash  # noqa: F401

                pytest.skip("dash is actually available")
            except (ImportError, TypeError):
                pass  # Expected


class TestApplyQueryFilter:
    def test_no_filter(self, sample_df):
        from bluecast.eda.dashboard import apply_query_filter

        result_df, msg = apply_query_filter(sample_df, "")
        assert len(result_df) == len(sample_df)
        assert "no filter" in msg.lower()

    def test_whitespace_filter(self, sample_df):
        from bluecast.eda.dashboard import apply_query_filter

        result_df, msg = apply_query_filter(sample_df, "   ")
        assert len(result_df) == len(sample_df)

    def test_valid_filter(self, sample_df):
        from bluecast.eda.dashboard import apply_query_filter

        result_df, msg = apply_query_filter(sample_df, "num1 > 0")
        assert len(result_df) <= len(sample_df)
        assert "✅" in msg

    def test_invalid_filter(self, sample_df):
        from bluecast.eda.dashboard import apply_query_filter

        result_df, msg = apply_query_filter(sample_df, "invalid_col > 999")
        assert "❌" in msg


class TestBuildSummaryHtml:
    def test_basic_summary(self, sample_df):
        try:
            import dash  # noqa: F401
        except ImportError:
            pytest.skip("dash not installed")

        from bluecast.eda.dashboard import build_summary_html

        children = build_summary_html(sample_df, "target")
        assert children is not None
        assert len(children) > 0

    def test_regression_summary(self, regression_df):
        try:
            import dash  # noqa: F401
        except ImportError:
            pytest.skip("dash not installed")

        from bluecast.eda.dashboard import build_summary_html

        children = build_summary_html(regression_df, "target")
        assert children is not None

    def test_summary_with_nulls(self):
        try:
            import dash  # noqa: F401
        except ImportError:
            pytest.skip("dash not installed")

        from bluecast.eda.dashboard import build_summary_html

        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "target": [0, 1, 0]})
        children = build_summary_html(df, "target")
        assert children is not None


class TestCreateEdaDashboard:
    def test_missing_target_raises(self, sample_df):
        from bluecast.eda.dashboard import create_eda_dashboard

        with pytest.raises(ValueError, match="not found"):
            create_eda_dashboard(sample_df, "nonexistent", run_server=False)

    def test_classification_detection(self, sample_df):
        from bluecast.eda.dashboard import create_eda_dashboard

        with patch(
            "bluecast.eda.analyse.create_eda_dashboard_classification"
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            create_eda_dashboard(sample_df, "target", run_server=False)
            mock_cls.assert_called_once()

    def test_regression_detection(self, regression_df):
        from bluecast.eda.dashboard import create_eda_dashboard

        with patch("bluecast.eda.analyse.create_eda_dashboard_regression") as mock_reg:
            mock_reg.return_value = MagicMock()
            create_eda_dashboard(regression_df, "target", run_server=False)
            mock_reg.assert_called_once()


class TestRunDashboardApp:
    def test_standalone_server(self):
        from bluecast.eda.dashboard import run_dashboard_app

        mock_app = MagicMock()
        run_dashboard_app(mock_app, "test", 8050, True, None)
        mock_app.run.assert_called_once()

    def test_no_run_no_jupyter(self):
        from bluecast.eda.dashboard import run_dashboard_app

        mock_app = MagicMock()
        result = run_dashboard_app(mock_app, "test", 8050, False, None)
        mock_app.run.assert_not_called()
        assert result is mock_app

    def test_jupyter_mode_fallback(self):
        from bluecast.eda.dashboard import run_dashboard_app

        mock_app = MagicMock()

        with patch.dict("sys.modules", {"jupyter_dash": None}):
            run_dashboard_app(mock_app, "test", 8050, True, "inline")
            mock_app.run.assert_called_once_with(debug=False, port=8050)


class TestConstants:
    def test_plot_options_exist(self):
        from bluecast.eda.dashboard import (
            CLASSIFICATION_EXTRA_PLOT_OPTIONS,
            COMMON_PLOT_OPTIONS,
            REGRESSION_EXTRA_PLOT_OPTIONS,
        )

        assert len(COMMON_PLOT_OPTIONS) >= 5
        assert len(REGRESSION_EXTRA_PLOT_OPTIONS) >= 1
        assert len(CLASSIFICATION_EXTRA_PLOT_OPTIONS) >= 1

    def test_dark_theme_layout(self):
        from bluecast.eda.dashboard import DARK_THEME_LAYOUT

        assert "paper_bgcolor" in DARK_THEME_LAYOUT
        assert "font" in DARK_THEME_LAYOUT
