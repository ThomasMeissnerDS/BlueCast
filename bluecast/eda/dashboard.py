"""Shared dashboard infrastructure for BlueCast EDA dashboards.

Provides CSS generation, layout building, callback wiring, and a unified
entry point that auto-detects regression vs classification.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DARK_THEME_LAYOUT: Dict[str, Any] = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#ffffff", "family": "Segoe UI"},
    "xaxis": {
        "gridcolor": "#404040",
        "zerolinecolor": "#404040",
        "tickfont": {"color": "#ffffff"},
    },
    "yaxis": {
        "gridcolor": "#404040",
        "zerolinecolor": "#404040",
        "tickfont": {"color": "#ffffff"},
    },
    "margin": {"t": 60, "b": 60, "l": 60, "r": 60},
}

REGRESSION_ACCENT = "#667eea"
REGRESSION_GRADIENT = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"

CLASSIFICATION_ACCENT = "#f093fb"
CLASSIFICATION_GRADIENT = "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"

DEFAULT_N_TARGET_BINS = 4
DEFAULT_CONTAMINATION = 0.1
DEFAULT_TEST_SIZE = 0.3

# Shared plot types available in both dashboards
COMMON_PLOT_OPTIONS = [
    {"label": "🔗 Correlation Heatmap", "value": "correlation"},
    {"label": "📈 Distribution Plot", "value": "distribution"},
    {"label": "🎯 PCA Analysis", "value": "pca"},
    {"label": "📦 Box Plot", "value": "boxplot"},
    {"label": "🔍 Benford's Law Analysis", "value": "benfords_law"},
    {"label": "❌ Missing Values Matrix", "value": "missing_values"},
    {"label": "📊 Category Frequency", "value": "category_frequency"},
    {"label": "🔗 Theil U Heatmap", "value": "theil_u"},
    {"label": "📈 ECDF Analysis", "value": "ecdf"},
    {"label": "🚨 Outlier Detection (IsolationForest)", "value": "outlier_detection"},
]

REGRESSION_EXTRA_PLOT_OPTIONS = [
    {"label": "📊 Scatter with Regression", "value": "scatter_with_regression"},
    {"label": "⚖️ Feature Coefficients", "value": "coefficients"},
    {"label": "🎯 Distribution by Target Bins", "value": "distribution_by_target"},
    {"label": "🎻 Violin Plot by Target Bins", "value": "violin_by_target"},
]

CLASSIFICATION_EXTRA_PLOT_OPTIONS = [
    {"label": "📊 Target Distribution", "value": "target_distribution"},
    {"label": "📊 Feature by Target", "value": "feature_by_target"},
]


# ---------------------------------------------------------------------------
# CSS Builder
# ---------------------------------------------------------------------------


def build_dashboard_css(accent_color: str, gradient: str) -> str:
    """Generate the full CSS string for a dashboard, parameterized by accent color."""
    return f"""
    <!DOCTYPE html>
    <html>
        <head>
            {{%metas%}}
            <title>{{%title%}}</title>
            {{%favicon%}}
            {{%css%}}
            <style>
                body {{
                    background-color: #1e1e1e;
                    color: #ffffff;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 0;
                }}
                .main-container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background: {gradient};
                    padding: 30px;
                    border-radius: 15px;
                    margin-bottom: 30px;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                }}
                .header h1 {{
                    margin: 0;
                    color: white;
                    font-size: 2.5rem;
                    font-weight: 700;
                    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
                }}
                .header .data-info {{
                    color: rgba(255, 255, 255, 0.9);
                    border-top: 1px solid rgba(255, 255, 255, 0.3);
                    padding-top: 15px;
                    margin-top: 15px;
                }}
                .controls-container {{
                    background: #2d2d2d;
                    padding: 25px;
                    border-radius: 12px;
                    margin-bottom: 25px;
                    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
                }}
                .control-group label {{
                    display: block;
                    margin-bottom: 8px;
                    font-weight: 500;
                    color: #e0e0e0;
                    font-size: 0.95rem;
                }}
                .graph-container {{
                    background: #2d2d2d;
                    padding: 20px;
                    border-radius: 12px;
                    margin-bottom: 25px;
                    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
                    min-height: 500px;
                }}
                .summary-container {{
                    background: #2d2d2d;
                    padding: 20px;
                    border-radius: 12px;
                    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
                }}
                .summary-container h3 {{
                    color: {accent_color};
                    margin-top: 0;
                }}
                .summary-table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                .summary-table th, .summary-table td {{
                    padding: 8px 12px;
                    text-align: left;
                    border-bottom: 1px solid #404040;
                }}
                .summary-table th {{
                    color: {accent_color};
                }}
                .dash-dropdown .Select-control {{
                    background-color: #3a3a3a !important;
                    border-color: #555 !important;
                    color: white !important;
                }}
                .dash-dropdown .Select-value-label {{
                    color: white !important;
                }}
                .dash-dropdown .Select-menu-outer {{
                    background-color: #3a3a3a !important;
                    border-color: #555 !important;
                }}
                .dash-dropdown .Select-option {{
                    background-color: #3a3a3a !important;
                    color: white !important;
                }}
                .dash-dropdown .Select-option:hover {{
                    background-color: {accent_color} !important;
                }}
                .dash-dropdown .Select-option.is-focused {{
                    background-color: {accent_color} !important;
                }}
                .dash-dropdown .Select-placeholder {{
                    color: #999 !important;
                }}
                .dash-dropdown .Select-input input {{
                    color: white !important;
                }}
            </style>
        </head>
        <body>
            {{%app_entry%}}
            <footer>
                {{%config%}}
                {{%scripts%}}
                {{%renderer%}}
            </footer>
        </body>
    </html>
    """


# ---------------------------------------------------------------------------
# Layout Builder
# ---------------------------------------------------------------------------


def build_dashboard_layout(
    title: str,
    data_info_lines: List[str],
    plot_options: List[Dict[str, str]],
    feature_x_options: List[Dict[str, str]],
    feature_y_options: List[Dict[str, str]],
    default_feature_x: Optional[str],
    default_feature_y: Optional[str],
) -> Any:
    """Build the standard dashboard layout with controls, graph, and summary."""
    try:
        from dash import dcc, html
    except ImportError:
        raise ImportError("Dash is required. Install with: pip install dash")

    return html.Div(
        className="main-container",
        children=[
            # Header
            html.Div(
                className="header",
                children=[
                    html.H1(title),
                    html.Div(
                        className="data-info",
                        children=[html.P(line) for line in data_info_lines],
                        style={"margin": "20px 0 0 0", "fontSize": "1rem"},
                    ),
                ],
            ),
            # Controls
            html.Div(
                className="controls-container",
                children=[
                    html.Div(
                        style={"display": "flex", "gap": "20px", "flexWrap": "wrap"},
                        children=[
                            html.Div(
                                className="control-group",
                                style={"flex": "1", "minWidth": "300px"},
                                children=[
                                    html.Label("📊 Select Plot Type:"),
                                    dcc.Dropdown(
                                        id="plot-type",
                                        options=plot_options,
                                        value="correlation",
                                        className="dash-dropdown",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="control-group",
                                style={"flex": "1", "minWidth": "250px"},
                                children=[
                                    html.Label("📐 Select Feature X:"),
                                    dcc.Dropdown(
                                        id="feature-x-dropdown",
                                        options=feature_x_options,
                                        value=default_feature_x,
                                        className="dash-dropdown",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="control-group",
                                style={"flex": "1", "minWidth": "250px"},
                                children=[
                                    html.Label("📏 Select Feature Y:"),
                                    dcc.Dropdown(
                                        id="feature-y-dropdown",
                                        options=feature_y_options,
                                        value=default_feature_y,
                                        className="dash-dropdown",
                                    ),
                                ],
                            ),
                        ],
                    )
                ],
            ),
            # Query filter
            html.Div(
                className="controls-container",
                children=[
                    html.Div(
                        className="control-group",
                        children=[
                            html.Label("🔍 Data Filter (Pandas Query Syntax):"),
                            html.P(
                                "Filter data using pandas query syntax. "
                                "Examples: 'column > 100', 'category == \"A\"'. "
                                "Press Enter or click outside to apply.",
                                style={
                                    "fontSize": "0.9rem",
                                    "color": "#999",
                                    "margin": "5px 0 10px 0",
                                },
                            ),
                            dcc.Input(
                                id="pandas-query-input",
                                type="text",
                                placeholder="column_name > 100 & category == 'value'",
                                debounce=True,
                                style={
                                    "width": "100%",
                                    "backgroundColor": "#4a4a4a",
                                    "color": "#ffffff",
                                    "border": "1px solid #666",
                                    "borderRadius": "8px",
                                    "padding": "10px",
                                    "fontFamily": "monospace",
                                    "fontSize": "0.9rem",
                                },
                                value="",
                            ),
                            html.Div(
                                id="query-status",
                                style={"marginTop": "10px", "fontSize": "0.9rem"},
                            ),
                        ],
                    )
                ],
            ),
            # Graph with loading indicator
            html.Div(
                className="graph-container",
                children=[
                    dcc.Loading(
                        id="loading-graph",
                        type="circle",
                        color="#667eea",
                        children=[
                            dcc.Graph(
                                id="main-plot",
                                config={"displayModeBar": True, "scrollZoom": True},
                                style={"height": "600px"},
                            ),
                        ],
                    ),
                ],
            ),
            # Summary
            html.Div(
                className="summary-container",
                id="data-summary",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Callback Builder
# ---------------------------------------------------------------------------


def apply_query_filter(df: pd.DataFrame, query_text: str):
    """Apply a query filter, returning (filtered_df, status_message).

    Returns the original df (not a copy) when no filter is applied.
    """
    from bluecast.eda.analyse import _apply_pandas_query_filter

    if not query_text or not query_text.strip():
        return df, f"📊 Showing all {len(df):,} rows (no filter applied)."

    try:
        filtered = _apply_pandas_query_filter(df, query_text)
        if len(filtered) == 0:
            return df, "❌ Query returned no results. Showing original data."
        return filtered, (
            f"✅ Query applied. Showing {len(filtered):,} of {len(df):,} rows."
        )
    except Exception as e:
        return df, f"❌ Query Error: {str(e)}"


def build_summary_html(df: pd.DataFrame, target_col: str):
    """Build the summary section HTML."""
    try:
        from dash import html
    except ImportError:
        return None

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    null_count = df.isnull().sum().sum()
    null_pct = (df.isnull().mean() * 100).round(1)
    cols_with_nulls = null_pct[null_pct > 0]

    children = [
        html.H3("📊 Dataset Summary"),
        html.P(
            f"Rows: {len(df):,} | "
            f"Numeric: {len(num_cols)} | "
            f"Categorical: {len(cat_cols)} | "
            f"Total nulls: {null_count:,}"
        ),
    ]

    if len(cols_with_nulls) > 0:
        children.append(html.P("Columns with missing values:"))
        for col_name, pct in cols_with_nulls.items():
            children.append(
                html.P(f"  {col_name}: {pct}%", style={"marginLeft": "20px"})
            )

    if target_col in df.columns:
        target = df[target_col]
        if target.nunique() <= 20:
            dist = target.value_counts().to_dict()
            children.append(html.P(f"Target distribution: {dist}"))
        else:
            children.append(
                html.P(
                    f"Target: mean={target.mean():.4f}, "
                    f"std={target.std():.4f}, "
                    f"range=[{target.min():.4f}, {target.max():.4f}]"
                )
            )

    return children


def run_dashboard_app(
    app: Any,
    dashboard_name: str,
    port: int,
    run_server: bool,
    jupyter_mode: Optional[str],
) -> Any:
    """Start the Dash app with proper Jupyter/standalone handling."""
    if jupyter_mode is not None:
        try:
            from jupyter_dash import JupyterDash  # noqa: F401

            app.run(
                mode=jupyter_mode,
                port=port,
                debug=False,
            )
        except ImportError:
            logger.info(
                f"jupyter-dash not available. Starting {dashboard_name} "
                f"as standalone server on port {port}."
            )
            if run_server:
                app.run(debug=False, port=port)
    elif run_server:
        logger.info(f"Starting {dashboard_name} on http://localhost:{port}")
        app.run(debug=False, port=port)

    return app


# ---------------------------------------------------------------------------
# Auto-detect entry point
# ---------------------------------------------------------------------------


def create_eda_dashboard(
    df: pd.DataFrame,
    target_col: str,
    port: int = 8050,
    run_server: bool = True,
    jupyter_mode: Optional[str] = None,
    n_target_bins: int = DEFAULT_N_TARGET_BINS,
    contamination: float = DEFAULT_CONTAMINATION,
) -> Any:
    """Create an EDA dashboard, auto-detecting regression vs classification.

    :param df: DataFrame to analyze.
    :param target_col: Target column name.
    :param port: Port for the dashboard server.
    :param run_server: Whether to start the server.
    :param jupyter_mode: Jupyter mode ('inline', 'external', 'tab', 'jupyterlab').
    :param n_target_bins: Number of bins for binning the target in regression plots.
    :param contamination: IsolationForest contamination parameter for outlier detection.
    :returns: The Dash app instance.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    n_unique = df[target_col].nunique()
    is_numeric = pd.api.types.is_numeric_dtype(df[target_col])

    if is_numeric and n_unique > 20:
        problem_type = "regression"
    else:
        problem_type = "classification"

    logger.info(
        f"Auto-detected problem type: {problem_type} (target has {n_unique} unique values)"
    )

    if problem_type == "regression":
        from bluecast.eda.analyse import create_eda_dashboard_regression

        return create_eda_dashboard_regression(
            df, target_col, port=port, run_server=run_server, jupyter_mode=jupyter_mode
        )
    else:
        from bluecast.eda.analyse import create_eda_dashboard_classification

        return create_eda_dashboard_classification(
            df, target_col, port=port, run_server=run_server, jupyter_mode=jupyter_mode
        )
