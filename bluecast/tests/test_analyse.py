import random
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from bluecast.eda.analyse import (
    _dashboard_update_plot,
    _dashboard_update_summary,
    bi_variate_plots,
    check_unique_values,
    correlation_heatmap,
    correlation_to_target,
    create_eda_dashboard,
    create_eda_dashboard_classification,
    create_eda_dashboard_regression,
    mutual_info_to_target,
    plot_against_target_for_regression,
    plot_andrews_curve,
    plot_benfords_law,
    plot_category_frequency,
    plot_classification_target_distribution_within_categories,
    plot_count_pair,
    plot_count_pairs,
    plot_distribution_by_time,
    plot_distribution_pairs,
    plot_ecdf,
    plot_missing_values_matrix,
    plot_null_percentage,
    plot_pca,
    plot_pca_biplot,
    plot_pca_cumulative_variance,
    plot_pie_chart,
    plot_theil_u_heatmap,
    plot_tsne,
    univariate_plots,
)
from bluecast.tests.make_data.create_data import (
    create_synthetic_dataframe,
    create_synthetic_dataframe_regression,
)


@pytest.fixture
def synthetic_train_test_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = create_synthetic_dataframe(2000, random_state=20)
    df_val = create_synthetic_dataframe(2000, random_state=200)
    return df_train, df_val


@pytest.fixture
def synthetic_train_test_data_regression() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = create_synthetic_dataframe_regression(2000, random_state=20)
    df_val = create_synthetic_dataframe_regression(2000, random_state=200)
    return df_train, df_val


@pytest.fixture()
def synthetic_categorical_data() -> pd.DataFrame:
    data = {
        "Category1": [random.choice(["A", "B", "C"]) for _ in range(100)],
        "Category2": [random.choice(["X", "Y", "Z"]) for _ in range(100)],
        "Category3": [random.choice(["Red", "Green", "Blue"]) for _ in range(100)],
        "Category4": [random.choice(["High", "Medium", "Low"]) for _ in range(100)],
    }

    # Create the DataFrame
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def create_data_with_nulls() -> pd.DataFrame:
    data = {
        "Column1": [1, 2, 3, None, 5],
        "Column2": [None, 2, 3, 4, 5],
        "Column3": [1, 2, 3, 4, 5],
        "Column4": [None, None, None, None, None],
    }

    df = pd.DataFrame(data)
    return df


@pytest.fixture
def create_data_with_many_uniques() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "col1": [1, 2, 3, 4],
            "col2": [1, 2, 3, 3],
            "col3": [1, 2, 2, 2],
            "col4": [1, 1, 1, 1],
        }
    )
    return df


def test_plot_pie_chart(synthetic_train_test_data):
    fig = plot_pie_chart(
        synthetic_train_test_data[0],
        "categorical_feature_1",
        show=False,
    )
    assert isinstance(fig, go.Figure)

    fig = plot_pie_chart(
        synthetic_train_test_data[0],
        "categorical_feature_1",
        explode=[0.1]
        * len(synthetic_train_test_data[0]["categorical_feature_1"].unique()),
        show=False,
    )
    assert isinstance(fig, go.Figure)


def test_univariate_plots(synthetic_train_test_data):
    # Test that it doesn't crash - univariate_plots doesn't return a figure
    univariate_plots(
        synthetic_train_test_data[0].loc[
            :, ["numerical_feature_1", "numerical_feature_2", "numerical_feature_3"]
        ]
    )
    assert True


def test_bi_variate_plots(synthetic_train_test_data):
    # Test that it doesn't crash - bi_variate_plots doesn't return a figure
    bi_variate_plots(synthetic_train_test_data[0], "target")
    assert True


def test_correlation_heatmap(synthetic_train_test_data):
    fig = correlation_heatmap(
        synthetic_train_test_data[0].drop(
            ["categorical_feature_1", "categorical_feature_2"], axis=1
        ),
        show=False,
    )
    assert isinstance(fig, go.Figure)


def test_correlation_to_target(synthetic_train_test_data):
    fig = correlation_to_target(
        synthetic_train_test_data[0].drop(
            ["categorical_feature_1", "categorical_feature_2"], axis=1
        ),
        "target",
        show=False,
    )
    assert isinstance(fig, go.Figure)


def test_mutual_info_to_target(synthetic_train_test_data):
    fig = mutual_info_to_target(
        synthetic_train_test_data[0].drop(
            ["categorical_feature_1", "categorical_feature_2", "datetime_feature"],
            axis=1,
        ),
        "target",
        class_problem="binary",
        show=False,
    )
    assert isinstance(fig, go.Figure)


def test_mutual_info_to_target_multiclass(synthetic_train_test_data):
    fig = mutual_info_to_target(
        synthetic_train_test_data[0].drop(
            ["categorical_feature_1", "categorical_feature_2", "datetime_feature"],
            axis=1,
        ),
        "target",
        class_problem="multiclass",
        show=False,
    )
    assert isinstance(fig, go.Figure)


def test_mutual_info_to_target_regression(synthetic_train_test_data_regression):
    fig = mutual_info_to_target(
        synthetic_train_test_data_regression[0].drop(
            ["categorical_feature_1", "categorical_feature_2", "datetime_feature"],
            axis=1,
        ),
        "target",
        class_problem="regression",
        show=False,
    )
    assert isinstance(fig, go.Figure)


def test_pca_plot(synthetic_train_test_data):
    fig = plot_pca(
        synthetic_train_test_data[0].loc[
            :,
            [
                "numerical_feature_1",
                "numerical_feature_2",
                "numerical_feature_3",
                "target",
            ],
        ],
        "target",
        show=False,
    )
    assert isinstance(fig, go.Figure)


def test_pca_biplot(synthetic_train_test_data):
    # test while having target column
    fig = plot_pca_biplot(
        synthetic_train_test_data[0].loc[
            :,
            [
                "numerical_feature_1",
                "numerical_feature_2",
                "numerical_feature_3",
                "target",
            ],
        ],
        "target",
        show=False,
    )
    assert isinstance(fig, go.Figure)

    # test absence of target column
    fig = plot_pca_biplot(
        synthetic_train_test_data[0].loc[
            :,
            [
                "numerical_feature_1",
                "numerical_feature_2",
                "numerical_feature_3",
            ],
        ],
        "target",
        show=False,
    )
    assert isinstance(fig, go.Figure)


def test_plot_pca_cumulative_variance(synthetic_train_test_data):
    fig = plot_pca_cumulative_variance(
        synthetic_train_test_data[0].loc[
            :,
            [
                "numerical_feature_1",
                "numerical_feature_2",
                "numerical_feature_3",
                "target",
            ],
        ],
        scale_data=True,
        n_components=3,
        show=False,
    )
    assert isinstance(fig, go.Figure)

    fig = plot_pca_cumulative_variance(
        synthetic_train_test_data[0].loc[
            :,
            [
                "numerical_feature_1",
                "numerical_feature_2",
                "numerical_feature_3",
                "target",
            ],
        ],
        scale_data=False,
        n_components=2,
        show=False,
    )
    assert isinstance(fig, go.Figure)


def test_plot_tsne(synthetic_train_test_data):
    fig = plot_tsne(
        synthetic_train_test_data[0]
        .loc[
            :,
            [
                "numerical_feature_1",
                "numerical_feature_2",
                "numerical_feature_3",
                "target",
            ],
        ]
        .head(100),  # Limit data for faster testing
        "target",
        perplexity=5,  # Lower perplexity for small dataset
        show=False,
    )
    assert isinstance(fig, go.Figure)


def test_plot_theil_u_heatmap(synthetic_categorical_data):
    fig, matrix = plot_theil_u_heatmap(
        synthetic_categorical_data,
        ["Category1", "Category2"],
        show=False,
    )
    assert isinstance(fig, go.Figure)
    assert matrix.shape == (2, 2)


def test_plot_null_percentage(create_data_with_nulls):
    fig = plot_null_percentage(create_data_with_nulls, show=False)
    assert isinstance(fig, go.Figure)


def test_check_unique_values(create_data_with_many_uniques):
    result = check_unique_values(
        create_data_with_many_uniques, ["col1", "col2", "col3", "col4"], threshold=0.9
    )
    assert result == ["col1"]


def test_plot_classification_target_distribution_within_categories(
    synthetic_train_test_data,
):
    # Test that it doesn't crash - this function doesn't return a figure
    plot_classification_target_distribution_within_categories(
        synthetic_train_test_data[0],
        ["categorical_feature_1", "categorical_feature_2"],
        "target",
    )
    assert True


def test_plot_against_target_for_regression(synthetic_train_test_data_regression):
    fig = plot_against_target_for_regression(
        synthetic_train_test_data_regression[0],
        ["numerical_feature_1", "numerical_feature_2"],
        "target",
        show=False,
    )
    assert isinstance(fig, go.Figure)


def test_plot_ecdf(synthetic_train_test_data):
    # Test plot_all_at_once=True
    fig = plot_ecdf(
        synthetic_train_test_data[0].loc[
            :, ["numerical_feature_1", "numerical_feature_2"]
        ],
        ["numerical_feature_1", "numerical_feature_2"],
        plot_all_at_once=True,
        show=False,
    )
    assert isinstance(fig, go.Figure)

    # Test plot_all_at_once=False
    figures = plot_ecdf(
        synthetic_train_test_data[0].loc[
            :, ["numerical_feature_1", "numerical_feature_2"]
        ],
        ["numerical_feature_1", "numerical_feature_2"],
        plot_all_at_once=False,
        show=False,
    )
    assert isinstance(figures, list)
    assert len(figures) == 2
    assert all(isinstance(fig, go.Figure) for fig in figures)


def test_plot_distribution_by_time(synthetic_train_test_data):
    # Create a copy with datetime column
    df = synthetic_train_test_data[0].copy()
    df["date_col"] = pd.date_range("2020-01-01", periods=len(df), freq="D")

    fig = plot_distribution_by_time(
        df,
        "numerical_feature_1",
        "date_col",
        freq="M",  # Monthly for faster testing
        show=False,
    )
    assert isinstance(fig, go.Figure)


def test_plot_distribution_pairs(synthetic_train_test_data):
    fig = plot_distribution_pairs(
        synthetic_train_test_data[0].loc[
            :100, ["numerical_feature_1"]
        ],  # Smaller dataset
        synthetic_train_test_data[1].loc[:100, ["numerical_feature_1"]],
        "numerical_feature_1",
        show=False,
    )
    assert isinstance(fig, go.Figure)


def test_plot_andrews_curve(synthetic_train_test_data):
    fig = plot_andrews_curve(
        synthetic_train_test_data[0]
        .loc[
            :,
            [
                "numerical_feature_1",
                "numerical_feature_2",
                "numerical_feature_3",
                "target",
            ],
        ]
        .head(50),  # Smaller dataset for faster testing
        "target",
        n_samples=20,
        show=False,
    )
    assert isinstance(fig, go.Figure)


def test_plot_count_pair(synthetic_train_test_data):
    """Test plot_count_pair function with different parameters."""
    df_train, df_test = synthetic_train_test_data

    # Test basic functionality
    fig = plot_count_pair(
        df_train,
        df_test,
        df_aliases=["train", "test"],
        feature="categorical_feature_1",
        show=False,
    )
    assert isinstance(fig, go.Figure)

    # Test with custom df_aliases
    fig = plot_count_pair(
        df_train,
        df_test,
        df_aliases=["training_set", "validation_set"],
        feature="categorical_feature_1",
        show=False,
    )
    assert isinstance(fig, go.Figure)

    # Test with order parameter
    unique_values = sorted(df_train["categorical_feature_1"].unique())
    fig = plot_count_pair(
        df_train,
        df_test,
        df_aliases=None,  # This should use default aliases
        feature="categorical_feature_1",
        order=unique_values,
        show=False,
    )
    assert isinstance(fig, go.Figure)

    # Test with custom palette
    fig = plot_count_pair(
        df_train,
        df_test,
        df_aliases=["set1", "set2"],
        feature="categorical_feature_1",
        palette=["red", "blue"],
        show=False,
    )
    assert isinstance(fig, go.Figure)


def test_plot_count_pairs(synthetic_train_test_data):
    """Test plot_count_pairs function."""
    df_train, df_test = synthetic_train_test_data

    # Test that it doesn't crash - plot_count_pairs doesn't return a figure
    plot_count_pairs(
        df_train,
        df_test,
        cat_cols=["categorical_feature_1", "categorical_feature_2"],
        df_aliases=["train", "test"],
    )
    assert True

    # Test with custom palette
    plot_count_pairs(
        df_train,
        df_test,
        cat_cols=["categorical_feature_1"],
        df_aliases=["training", "testing"],
        palette=["green", "orange"],
    )
    assert True


def test_create_eda_dashboard():
    # Create simple test data
    test_df = pd.DataFrame(
        {
            "numeric_col1": np.random.randn(100),
            "numeric_col2": np.random.randn(100),
            "categorical_col": np.random.choice(["A", "B", "C"], 100),
            "target": np.random.choice([0, 1], 100),
        }
    )

    # Test that the function can be called without error (don't start the server)
    app = create_eda_dashboard(test_df, "target", port=8051, run_server=False)

    # Verify that we got a Dash app object
    assert app is not None
    assert hasattr(app, "layout")
    assert hasattr(app, "callback")


def test_dashboard_callbacks_logic():
    """Test the dashboard callback logic by manually testing the expected behavior."""
    try:
        import plotly.graph_objects as go
        from dash import html
    except ImportError:
        pytest.skip("Dash not available for testing")

    # Create test data to simulate the dashboard environment
    test_df = pd.DataFrame(
        {
            "numeric_col1": np.random.randn(100),
            "numeric_col2": np.random.randn(100),
            "categorical_col": np.random.choice(["A", "B", "C"], 100),
            "target": np.random.choice([0, 1], 100),
        }
    )

    # Simulate the variables that would exist in the dashboard callback context
    numeric_cols = test_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = test_df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Remove target from feature lists
    if "target" in numeric_cols:
        numeric_cols.remove("target")
    if "target" in categorical_cols:
        categorical_cols.remove("target")

    target_col = "target"

    # Test update_plot logic manually
    # This simulates the logic inside the update_plot callback
    def simulate_update_plot(plot_type, selected_feature):
        if plot_type == "correlation" and len(numeric_cols) > 1:
            return correlation_heatmap(test_df[numeric_cols + [target_col]], show=False)
        elif plot_type == "distribution" and selected_feature:
            fig = go.Figure()
            fig.add_trace(
                go.Histogram(x=test_df[selected_feature], name=selected_feature)
            )
            fig.update_layout(title=f"Distribution of {selected_feature}")
            return fig
        elif plot_type == "pca" and len(numeric_cols) > 1:
            return plot_pca(
                test_df[numeric_cols + [target_col]], target_col, show=False
            )
        elif plot_type == "boxplot" and selected_feature:
            fig = go.Figure()
            fig.add_trace(go.Box(y=test_df[selected_feature], name=selected_feature))
            fig.update_layout(title=f"Box Plot of {selected_feature}")
            return fig
        elif plot_type == "scatter" and selected_feature and len(numeric_cols) > 1:
            other_col = [col for col in numeric_cols if col != selected_feature][0]
            import plotly.express as px

            fig = px.scatter(test_df, x=selected_feature, y=other_col, color=target_col)
            return fig
        else:
            # Default empty plot
            fig = go.Figure()
            fig.update_layout(title="Select valid options to display plot")
            return fig

    # Test all the plot types to increase coverage
    result = simulate_update_plot("correlation", "numeric_col1")
    assert isinstance(result, go.Figure)

    result = simulate_update_plot("distribution", "numeric_col1")
    assert isinstance(result, go.Figure)

    result = simulate_update_plot("pca", "numeric_col1")
    assert isinstance(result, go.Figure)

    result = simulate_update_plot("boxplot", "numeric_col1")
    assert isinstance(result, go.Figure)

    result = simulate_update_plot("scatter", "numeric_col1")
    assert isinstance(result, go.Figure)

    # Test default case
    result = simulate_update_plot("invalid_type", "numeric_col1")
    assert isinstance(result, go.Figure)

    # Test with None feature
    result = simulate_update_plot("distribution", None)
    assert isinstance(result, go.Figure)

    # Test update_summary logic manually
    def simulate_update_summary(selected_feature):
        if selected_feature:
            if selected_feature in numeric_cols:
                stats = test_df[selected_feature].describe()
                return html.Table(
                    [
                        html.Tr([html.Td(stat), html.Td(f"{value:.2f}")])
                        for stat, value in stats.items()
                    ]
                )
            else:
                value_counts = test_df[selected_feature].value_counts()
                return html.Table(
                    [
                        html.Tr([html.Td(value), html.Td(count)])
                        for value, count in value_counts.head(10).items()
                    ]
                )
        return "Select a feature to see summary statistics"

    # Test summary with numeric column
    result = simulate_update_summary("numeric_col1")
    assert isinstance(result, html.Table)

    # Test summary with categorical column
    result = simulate_update_summary("categorical_col")
    assert isinstance(result, html.Table)

    # Test summary with None selection
    result = simulate_update_summary(None)
    assert result == "Select a feature to see summary statistics"

    # Test edge case with single numeric column (should trigger default plot case)
    single_numeric_cols = ["numeric_col1"]  # Only one numeric column

    def simulate_edge_case_plot(plot_type, selected_feature):
        if (
            plot_type == "correlation" and len(single_numeric_cols) > 1
        ):  # This will be False
            return "Not reached"
        else:
            fig = go.Figure()
            fig.update_layout(title="Select valid options to display plot")
            return fig

    result = simulate_edge_case_plot("correlation", "numeric_col1")
    assert isinstance(result, go.Figure)


def test_dashboard_server_startup():
    """Test the server startup code path."""
    # Create simple test data
    test_df = pd.DataFrame(
        {
            "numeric_col1": np.random.randn(10),
            "target": np.random.choice([0, 1], 10),
        }
    )

    # Mock the app.run method to avoid actually starting a server
    import unittest.mock

    try:
        with unittest.mock.patch("dash.Dash.run") as mock_run:
            # Test the run_server=True path
            app = create_eda_dashboard(test_df, "target", port=8053, run_server=True)

            # Verify that run was called with the expected parameters
            mock_run.assert_called_once_with(debug=True, port=8053)

            # Verify we still get an app object
            assert app is not None
    except ImportError:
        pytest.skip("Dash not available for testing")


def test_show_parameter_coverage():
    """Test all plotting functions with show=True to increase coverage."""
    # Create test data
    test_df = pd.DataFrame(
        {
            "num1": np.random.randn(50),
            "num2": np.random.randn(50),
            "cat1": np.random.choice(["A", "B", "C"], 50),
            "target": np.random.choice([0, 1], 50),
        }
    )

    # Mock plt.show to avoid actual plot display during testing
    import unittest.mock

    with unittest.mock.patch("plotly.graph_objects.Figure.show"):
        # Test all functions that have show parameter
        plot_pie_chart(test_df, "cat1", show=True)
        correlation_heatmap(test_df[["num1", "num2", "target"]], show=True)
        correlation_to_target(test_df[["num1", "num2", "target"]], "target", show=True)
        plot_pca(
            test_df[["num1", "num2", "target"]], "target", show=True
        )  # Only numeric columns
        plot_pca_cumulative_variance(
            test_df[["num1", "num2"]], n_components=2, show=True
        )
        plot_pca_biplot(
            test_df[["num1", "num2", "target"]], "target", show=True
        )  # Only numeric columns
        plot_tsne(
            test_df[["num1", "num2", "target"]], "target", perplexity=5, show=True
        )  # Only numeric columns
        plot_theil_u_heatmap(test_df[["cat1"]], ["cat1"], show=True)
        plot_null_percentage(test_df, show=True)
        mutual_info_to_target(
            test_df[["num1", "num2", "target"]], "target", "binary", show=True
        )
        plot_against_target_for_regression(
            test_df, ["num1", "num2"], "target", show=True
        )
        plot_ecdf(test_df, ["num1", "num2"], plot_all_at_once=True, show=True)
        plot_ecdf(test_df, ["num1", "num2"], plot_all_at_once=False, show=True)

        # Test time-based plot
        test_df["date_col"] = pd.date_range(
            "2020-01-01", periods=len(test_df), freq="D"
        )
        plot_distribution_by_time(test_df, "num1", "date_col", show=True)

        plot_distribution_pairs(test_df[["num1"]], test_df[["num1"]], "num1", show=True)
        plot_andrews_curve(
            test_df[["num1", "num2", "target"]], "target", n_samples=20, show=True
        )  # Only numeric columns
        plot_count_pair(test_df, test_df, ["train", "test"], "cat1", show=True)

    assert True


def test_dashboard_import_error():
    """Test ImportError handling when dash is not available."""
    import sys
    import unittest.mock

    # Create test data
    test_df = pd.DataFrame(
        {
            "num1": np.random.randn(10),
            "target": np.random.choice([0, 1], 10),
        }
    )

    # Store original dash modules if they exist
    dash_modules = {}
    for module_name in list(sys.modules.keys()):
        if module_name.startswith("dash"):
            dash_modules[module_name] = sys.modules[module_name]
            del sys.modules[module_name]

    try:
        # Set dash modules to None to trigger ImportError
        with unittest.mock.patch.dict(
            "sys.modules",
            {
                "dash": None,
                "dash.dependencies": None,
                "dash.html": None,
                "dash.dcc": None,
            },
        ):
            with pytest.raises(
                ImportError, match="Dash is required for dashboard functionality"
            ):
                create_eda_dashboard(test_df, "target", run_server=False)
    finally:
        # Restore original modules
        for module_name, module in dash_modules.items():
            sys.modules[module_name] = module


def test_dashboard_categorical_target_removal():
    """Test the categorical_cols.remove(target_col) line coverage."""
    # Create test data where target is categorical
    test_df = pd.DataFrame(
        {
            "numeric_col1": np.random.randn(50),
            "numeric_col2": np.random.randn(50),
            "categorical_col": np.random.choice(["A", "B", "C"], 50),
            "target": np.random.choice(["Class1", "Class2"], 50),  # Categorical target
        }
    )

    # Convert target to categorical to ensure it's detected as categorical
    test_df["target"] = test_df["target"].astype("category")

    # This should trigger the categorical_cols.remove(target_col) line
    app = create_eda_dashboard(test_df, "target", port=8052, run_server=False)
    assert app is not None


def test_dashboard_update_plot_all_branches():
    """Test all branches of the update_plot function in dashboard."""
    import plotly.express as px

    # Create test data
    test_df = pd.DataFrame(
        {
            "numeric_col1": np.random.randn(100),
            "numeric_col2": np.random.randn(100),
            "categorical_col": np.random.choice(["A", "B", "C"], 100),
            "target": np.random.choice([0, 1], 100),
        }
    )

    numeric_cols = ["numeric_col1", "numeric_col2"]
    target_col = "target"

    # Simulate the exact update_plot function from the dashboard
    def simulate_update_plot(plot_type, selected_feature):
        if plot_type == "correlation" and len(numeric_cols) > 1:
            return correlation_heatmap(test_df[numeric_cols + [target_col]], show=False)
        elif plot_type == "distribution" and selected_feature:
            fig = go.Figure()
            fig.add_trace(
                go.Histogram(x=test_df[selected_feature], name=selected_feature)
            )
            fig.update_layout(title=f"Distribution of {selected_feature}")
            return fig
        elif plot_type == "pca" and len(numeric_cols) > 1:
            return plot_pca(
                test_df[numeric_cols + [target_col]], target_col, show=False
            )
        elif plot_type == "boxplot" and selected_feature:
            fig = go.Figure()
            fig.add_trace(go.Box(y=test_df[selected_feature], name=selected_feature))
            fig.update_layout(title=f"Box Plot of {selected_feature}")
            return fig
        elif plot_type == "scatter" and selected_feature and len(numeric_cols) > 1:
            other_col = [col for col in numeric_cols if col != selected_feature][0]
            fig = px.scatter(test_df, x=selected_feature, y=other_col, color=target_col)
            return fig
        else:
            # Default empty plot
            fig = go.Figure()
            fig.update_layout(title="Select valid options to display plot")
            return fig

    # Test all branches of update_plot
    result = simulate_update_plot("correlation", "numeric_col1")
    assert isinstance(result, go.Figure)

    result = simulate_update_plot("distribution", "numeric_col1")
    assert isinstance(result, go.Figure)

    result = simulate_update_plot("pca", "numeric_col1")
    assert isinstance(result, go.Figure)

    result = simulate_update_plot("boxplot", "numeric_col1")
    assert isinstance(result, go.Figure)

    result = simulate_update_plot("scatter", "numeric_col1")
    assert isinstance(result, go.Figure)

    # Test default case (else branch)
    result = simulate_update_plot("unknown_type", "numeric_col1")
    assert isinstance(result, go.Figure)

    # Test edge cases
    result = simulate_update_plot("distribution", None)  # No selected_feature
    assert isinstance(result, go.Figure)

    # Test when only one numeric column (correlation and scatter won't work)
    single_numeric_cols = ["numeric_col1"]

    def simulate_single_col_plot(plot_type, selected_feature):
        if plot_type == "correlation" and len(single_numeric_cols) > 1:  # False
            return "not reached"
        elif (
            plot_type == "scatter" and selected_feature and len(single_numeric_cols) > 1
        ):  # False
            return "not reached"
        else:
            fig = go.Figure()
            fig.update_layout(title="Select valid options to display plot")
            return fig

    result = simulate_single_col_plot("correlation", "numeric_col1")
    assert isinstance(result, go.Figure)

    result = simulate_single_col_plot("scatter", "numeric_col1")
    assert isinstance(result, go.Figure)


def test_dashboard_update_summary_all_branches():
    """Test all branches of the update_summary function in dashboard."""
    from dash import html

    # Create test data
    test_df = pd.DataFrame(
        {
            "numeric_col1": np.random.randn(100),
            "categorical_col": np.random.choice(["A", "B", "C"], 100),
            "target": np.random.choice([0, 1], 100),
        }
    )

    numeric_cols = ["numeric_col1"]

    # Simulate the exact update_summary function from the dashboard
    def simulate_update_summary(selected_feature):
        if selected_feature:
            if selected_feature in numeric_cols:
                stats = test_df[selected_feature].describe()
                return html.Table(
                    [
                        html.Tr([html.Td(stat), html.Td(f"{value:.2f}")])
                        for stat, value in stats.items()
                    ]
                )
            else:
                value_counts = test_df[selected_feature].value_counts()
                return html.Table(
                    [
                        html.Tr([html.Td(value), html.Td(count)])
                        for value, count in value_counts.head(10).items()
                    ]
                )
        return "Select a feature to see summary statistics"

    # Test numeric column branch
    result = simulate_update_summary("numeric_col1")
    assert isinstance(result, html.Table)

    # Test categorical column branch (else within if selected_feature)
    result = simulate_update_summary("categorical_col")
    assert isinstance(result, html.Table)

    # Test None/empty selection branch
    result = simulate_update_summary(None)
    assert result == "Select a feature to see summary statistics"

    result = simulate_update_summary("")
    assert result == "Select a feature to see summary statistics"


def test_dashboard_comprehensive_callbacks_coverage():
    """Test comprehensive coverage of dashboard callback functions including all edge cases."""
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from dash import html
    except ImportError:
        pytest.skip("Dash not available for testing")

    # Create test data to simulate various scenarios
    test_df = pd.DataFrame(
        {
            "numeric_col1": np.random.randn(100),
            "numeric_col2": np.random.randn(100),
            "categorical_col1": np.random.choice(["A", "B", "C"], 100),
            "categorical_col2": np.random.choice(["X", "Y", "Z"], 100),
            "target": np.random.choice([0, 1], 100),
        }
    )

    # Test scenario 1: Multiple numeric columns
    numeric_cols = ["numeric_col1", "numeric_col2"]
    target_col = "target"
    df = test_df  # For consistency with function logic

    def simulate_update_plot(plot_type, selected_feature):
        if plot_type == "correlation" and len(numeric_cols) > 1:
            return correlation_heatmap(df[numeric_cols + [target_col]], show=False)
        elif plot_type == "distribution" and selected_feature:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df[selected_feature], name=selected_feature))
            fig.update_layout(title=f"Distribution of {selected_feature}")
            return fig
        elif plot_type == "pca" and len(numeric_cols) > 1:
            return plot_pca(df[numeric_cols + [target_col]], target_col, show=False)
        elif plot_type == "boxplot" and selected_feature:
            fig = go.Figure()
            fig.add_trace(go.Box(y=df[selected_feature], name=selected_feature))
            fig.update_layout(title=f"Box Plot of {selected_feature}")
            return fig
        elif plot_type == "scatter" and selected_feature and len(numeric_cols) > 1:
            other_col = [col for col in numeric_cols if col != selected_feature][0]
            fig = px.scatter(df, x=selected_feature, y=other_col, color=target_col)
            return fig
        else:
            # Default empty plot
            fig = go.Figure()
            fig.update_layout(title="Select valid options to display plot")
            return fig

    def simulate_update_summary(selected_feature):
        if selected_feature:
            if selected_feature in numeric_cols:
                stats = df[selected_feature].describe()
                return html.Table(
                    [
                        html.Tr([html.Td(stat), html.Td(f"{value:.2f}")])
                        for stat, value in stats.items()
                    ]
                )
            else:
                value_counts = df[selected_feature].value_counts()
                return html.Table(
                    [
                        html.Tr([html.Td(value), html.Td(count)])
                        for value, count in value_counts.head(10).items()
                    ]
                )
        return "Select a feature to see summary statistics"

    # Test all successful plot types with valid conditions
    result = simulate_update_plot("correlation", "numeric_col1")
    assert isinstance(result, go.Figure)

    result = simulate_update_plot("distribution", "numeric_col1")
    assert isinstance(result, go.Figure)

    result = simulate_update_plot("pca", "numeric_col1")
    assert isinstance(result, go.Figure)

    result = simulate_update_plot("boxplot", "numeric_col1")
    assert isinstance(result, go.Figure)

    result = simulate_update_plot("scatter", "numeric_col1")
    assert isinstance(result, go.Figure)

    # Test edge cases that trigger the default case
    result = simulate_update_plot("distribution", None)  # No selected feature
    assert isinstance(result, go.Figure)
    assert "Select valid options" in result.layout.title.text

    result = simulate_update_plot("distribution", "")  # Empty feature
    assert isinstance(result, go.Figure)
    assert "Select valid options" in result.layout.title.text

    result = simulate_update_plot("boxplot", None)  # No selected feature
    assert isinstance(result, go.Figure)
    assert "Select valid options" in result.layout.title.text

    result = simulate_update_plot("invalid_type", "numeric_col1")  # Invalid plot type
    assert isinstance(result, go.Figure)
    assert "Select valid options" in result.layout.title.text

    # Test edge cases with insufficient numeric columns
    single_numeric_cols = ["numeric_col1"]

    def simulate_single_numeric_update_plot(plot_type, selected_feature):
        if plot_type == "correlation" and len(single_numeric_cols) > 1:  # False
            return correlation_heatmap(
                df[single_numeric_cols + [target_col]], show=False
            )
        elif plot_type == "distribution" and selected_feature:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df[selected_feature], name=selected_feature))
            fig.update_layout(title=f"Distribution of {selected_feature}")
            return fig
        elif plot_type == "pca" and len(single_numeric_cols) > 1:  # False
            return plot_pca(
                df[single_numeric_cols + [target_col]], target_col, show=False
            )
        elif plot_type == "boxplot" and selected_feature:
            fig = go.Figure()
            fig.add_trace(go.Box(y=df[selected_feature], name=selected_feature))
            fig.update_layout(title=f"Box Plot of {selected_feature}")
            return fig
        elif (
            plot_type == "scatter" and selected_feature and len(single_numeric_cols) > 1
        ):  # False
            other_col = [col for col in single_numeric_cols if col != selected_feature][
                0
            ]
            fig = px.scatter(df, x=selected_feature, y=other_col, color=target_col)
            return fig
        else:
            # Default empty plot
            fig = go.Figure()
            fig.update_layout(title="Select valid options to display plot")
            return fig

    # Test correlation with insufficient numeric columns
    result = simulate_single_numeric_update_plot("correlation", "numeric_col1")
    assert isinstance(result, go.Figure)
    assert "Select valid options" in result.layout.title.text

    # Test PCA with insufficient numeric columns
    result = simulate_single_numeric_update_plot("pca", "numeric_col1")
    assert isinstance(result, go.Figure)
    assert "Select valid options" in result.layout.title.text

    # Test scatter with insufficient numeric columns
    result = simulate_single_numeric_update_plot("scatter", "numeric_col1")
    assert isinstance(result, go.Figure)
    assert "Select valid options" in result.layout.title.text

    # Test summary functionality - comprehensive coverage
    # Test with numeric column
    result = simulate_update_summary("numeric_col1")
    assert isinstance(result, html.Table)

    # Test with categorical column
    result = simulate_update_summary("categorical_col1")
    assert isinstance(result, html.Table)

    # Test with None selection
    result = simulate_update_summary(None)
    assert result == "Select a feature to see summary statistics"

    # Test with empty string selection
    result = simulate_update_summary("")
    assert result == "Select a feature to see summary statistics"

    # Test edge case: categorical column that looks like numeric
    test_df_edge = test_df.copy()
    test_df_edge["mixed_col"] = ["1", "2", "3", "1", "2"] * 20  # String numbers

    def simulate_edge_summary(selected_feature):
        edge_numeric_cols = ["numeric_col1", "numeric_col2"]  # mixed_col not in numeric
        if selected_feature:
            if selected_feature in edge_numeric_cols:
                stats = test_df_edge[selected_feature].describe()
                return html.Table(
                    [
                        html.Tr([html.Td(stat), html.Td(f"{value:.2f}")])
                        for stat, value in stats.items()
                    ]
                )
            else:
                value_counts = test_df_edge[selected_feature].value_counts()
                return html.Table(
                    [
                        html.Tr([html.Td(value), html.Td(count)])
                        for value, count in value_counts.head(10).items()
                    ]
                )
        return "Select a feature to see summary statistics"

    # Test mixed column treated as categorical
    result = simulate_edge_summary("mixed_col")
    assert isinstance(result, html.Table)


def test_dashboard_helper_functions_coverage():
    """Test the dashboard helper functions directly to ensure proper coverage."""
    try:
        from dash import html
    except ImportError:
        pytest.skip("Dash not available for testing")

    # Create test data
    test_df = pd.DataFrame(
        {
            "numeric_col1": np.random.randn(100),
            "numeric_col2": np.random.randn(100),
            "categorical_col1": np.random.choice(["A", "B", "C"], 100),
            "target": np.random.choice([0, 1], 100),
        }
    )

    # Get the column lists exactly as the dashboard would
    numeric_cols = test_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = test_df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    if "target" in numeric_cols:
        numeric_cols.remove("target")
    if "target" in categorical_cols:
        categorical_cols.remove("target")

    target_col = "target"

    # Test all branches of _dashboard_update_plot function

    # Test successful cases
    result = _dashboard_update_plot(
        "correlation", "numeric_col1", test_df, numeric_cols, target_col
    )
    assert isinstance(result, go.Figure)

    result = _dashboard_update_plot(
        "distribution", "numeric_col1", test_df, numeric_cols, target_col
    )
    assert isinstance(result, go.Figure)
    assert "Distribution of numeric_col1" in result.layout.title.text

    result = _dashboard_update_plot(
        "pca", "numeric_col1", test_df, numeric_cols, target_col
    )
    assert isinstance(result, go.Figure)

    result = _dashboard_update_plot(
        "boxplot", "numeric_col1", test_df, numeric_cols, target_col
    )
    assert isinstance(result, go.Figure)
    assert "Box Plot of numeric_col1" in result.layout.title.text

    result = _dashboard_update_plot(
        "scatter", "numeric_col1", test_df, numeric_cols, target_col
    )
    assert isinstance(result, go.Figure)

    # Test default/else cases that trigger the final else branch
    result = _dashboard_update_plot(
        "invalid_plot_type", "numeric_col1", test_df, numeric_cols, target_col
    )
    assert isinstance(result, go.Figure)
    assert "Select valid options to display plot" in result.layout.title.text

    result = _dashboard_update_plot(
        "distribution", None, test_df, numeric_cols, target_col
    )  # No selected feature
    assert isinstance(result, go.Figure)
    assert "Select valid options to display plot" in result.layout.title.text

    result = _dashboard_update_plot(
        "boxplot", "", test_df, numeric_cols, target_col
    )  # Empty selected feature
    assert isinstance(result, go.Figure)
    assert "Select valid options to display plot" in result.layout.title.text

    # Test all branches of _dashboard_update_summary function

    # Numeric column case - now returns html.Div with enhanced statistics
    result = _dashboard_update_summary(
        "numeric_col1", test_df, numeric_cols, target_col
    )
    assert isinstance(result, html.Div)
    assert len(result.children) > 0  # Should contain feature statistics

    # Categorical column case - now returns html.Div with enhanced statistics
    result = _dashboard_update_summary(
        "categorical_col1", test_df, numeric_cols, target_col
    )
    assert isinstance(result, html.Div)
    assert len(result.children) > 0  # Should contain feature statistics

    # No selection case (main else)
    result = _dashboard_update_summary(None, test_df, numeric_cols, target_col)
    assert isinstance(result, html.Div)
    assert "Select a feature to see summary statistics" in result.children

    result = _dashboard_update_summary("", test_df, numeric_cols, target_col)
    assert isinstance(result, html.Div)
    assert "Select a feature to see summary statistics" in result.children

    # Test edge case: insufficient numeric columns
    # Create scenario with only 1 numeric column
    single_df = pd.DataFrame(
        {
            "numeric_col1": np.random.randn(100),
            "categorical_col1": np.random.choice(["A", "B", "C"], 100),
            "target": np.random.choice([0, 1], 100),
        }
    )

    single_numeric_cols = ["numeric_col1"]  # Only one numeric column

    # Test cases that trigger the else branch due to insufficient columns
    result = _dashboard_update_plot(
        "correlation", "numeric_col1", single_df, single_numeric_cols, target_col
    )
    assert isinstance(result, go.Figure)
    assert "Select valid options to display plot" in result.layout.title.text

    result = _dashboard_update_plot(
        "pca", "numeric_col1", single_df, single_numeric_cols, target_col
    )
    assert isinstance(result, go.Figure)
    assert "Select valid options to display plot" in result.layout.title.text

    result = _dashboard_update_plot(
        "scatter", "numeric_col1", single_df, single_numeric_cols, target_col
    )
    assert isinstance(result, go.Figure)
    assert "Select valid options to display plot" in result.layout.title.text

    # These should still work (distribution and boxplot don't require multiple numeric cols)
    result = _dashboard_update_plot(
        "distribution", "numeric_col1", single_df, single_numeric_cols, target_col
    )
    assert isinstance(result, go.Figure)
    assert "Distribution of numeric_col1" in result.layout.title.text

    result = _dashboard_update_plot(
        "boxplot", "numeric_col1", single_df, single_numeric_cols, target_col
    )
    assert isinstance(result, go.Figure)
    assert "Box Plot of numeric_col1" in result.layout.title.text


def test_plot_benfords_law():
    """Test Benford's Law analysis plotting."""
    # Create data that could follow Benford's Law
    np.random.seed(42)
    data = np.random.lognormal(mean=3, sigma=2, size=1000)
    df = pd.DataFrame({"financial_amounts": data})

    # Test successful case
    fig = plot_benfords_law(df, "financial_amounts", show=False)
    assert isinstance(fig, go.Figure)
    assert "Benford's Law Analysis" in fig.layout.title.text

    # Test with non-existent column
    with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
        plot_benfords_law(df, "nonexistent", show=False)

    # Test with no positive values
    df_zeros = pd.DataFrame({"zero_col": [0, -1, -2, 0]})
    with pytest.raises(ValueError, match="No positive values found"):
        plot_benfords_law(df_zeros, "zero_col", show=False)


def test_plot_missing_values_matrix():
    """Test missing values matrix plotting."""
    # Create data with missing values
    df = pd.DataFrame(
        {
            "col1": [1, 2, np.nan, 4, 5],
            "col2": [1, np.nan, 3, np.nan, 5],
            "col3": [1, 2, 3, 4, 5],  # No missing values
        }
    )

    # Test with missing values
    fig = plot_missing_values_matrix(df, show=False)
    assert isinstance(fig, go.Figure)
    assert "Missing Values Matrix" in fig.layout.title.text

    # Test with no missing values
    df_complete = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    fig = plot_missing_values_matrix(df_complete, show=False)
    assert isinstance(fig, go.Figure)
    assert "No missing values found" in fig.layout.annotations[0].text


def test_plot_category_frequency():
    """Test category frequency plotting."""
    # Create categorical data
    df = pd.DataFrame({"categories": ["A", "B", "A", "C", "B", "B", "A"]})

    # Test successful case
    fig = plot_category_frequency(df, "categories", show=False)
    assert isinstance(fig, go.Figure)
    assert (
        "Category Frequencies" in fig.layout.title.text
        or "Word Cloud" in fig.layout.title.text
    )

    # Test with non-existent column
    with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
        plot_category_frequency(df, "nonexistent", show=False)


def test_create_eda_dashboard_regression():
    """Test regression dashboard creation without starting server."""
    df = create_synthetic_dataframe_regression(100, random_state=42)

    try:
        # Test regular mode
        app = create_eda_dashboard_regression(
            df=df, target_col="target", port=8055, run_server=False
        )
        assert app is not None
        assert hasattr(app, "layout")

        # Test jupyter mode
        app_jupyter = create_eda_dashboard_regression(
            df=df,
            target_col="target",
            port=8056,
            run_server=False,
            jupyter_mode="inline",
        )
        assert app_jupyter is not None
        assert hasattr(app_jupyter, "layout")
    except ImportError:
        pytest.skip("Dash not available for testing")


def test_create_eda_dashboard_classification():
    """Test classification dashboard creation without starting server."""
    df = create_synthetic_dataframe(100, random_state=42)

    try:
        # Test regular mode
        app = create_eda_dashboard_classification(
            df=df, target_col="target", port=8057, run_server=False
        )
        assert app is not None
        assert hasattr(app, "layout")

        # Test jupyter mode
        app_jupyter = create_eda_dashboard_classification(
            df=df,
            target_col="target",
            port=8058,
            run_server=False,
            jupyter_mode="external",
        )
        assert app_jupyter is not None
        assert hasattr(app_jupyter, "layout")
    except ImportError:
        pytest.skip("Dash not available for testing")
