import random
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from bluecast.eda.analyse import (
    bi_variate_plots,
    check_unique_values,
    correlation_heatmap,
    correlation_to_target,
    create_eda_dashboard,
    mutual_info_to_target,
    plot_against_target_for_regression,
    plot_andrews_curve,
    plot_classification_target_distribution_within_categories,
    plot_distribution_by_time,
    plot_distribution_pairs,
    plot_ecdf,
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
