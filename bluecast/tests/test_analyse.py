import random
from typing import Tuple

import pandas as pd
import pytest

from bluecast.eda.analyse import (
    bi_variate_plots,
    check_unique_values,
    correlation_heatmap,
    correlation_to_target,
    mutual_info_to_target,
    plot_count_pairs,
    plot_ecdf,
    plot_null_percentage,
    plot_pca,
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
    plot_pie_chart(
        synthetic_train_test_data[0],
        "categorical_feature_1",
    )
    assert True

    plot_pie_chart(
        synthetic_train_test_data[0],
        "categorical_feature_1",
        explode=[0.1]
        * len(synthetic_train_test_data[0]["categorical_feature_1"].unique()),
    )


def test_univariate_plots(synthetic_train_test_data):
    univariate_plots(
        synthetic_train_test_data[0].loc[
            :, ["numerical_feature_1", "numerical_feature_2", "numerical_feature_3"]
        ]
    )
    assert True


def test_bi_variate_plots(synthetic_train_test_data):
    bi_variate_plots(synthetic_train_test_data[0], "target")
    assert True


def test_correlation_heatmap(synthetic_train_test_data):
    correlation_heatmap(
        synthetic_train_test_data[0].drop(
            ["categorical_feature_1", "categorical_feature_2"], axis=1
        )
    )
    assert True


def test_correlation_to_target(synthetic_train_test_data):
    correlation_to_target(
        synthetic_train_test_data[0].drop(
            ["categorical_feature_1", "categorical_feature_2"], axis=1
        ),
        "target",
    )
    assert True


def test_mutual_info_to_target(synthetic_train_test_data):
    mutual_info_to_target(
        synthetic_train_test_data[0].drop(
            ["categorical_feature_1", "categorical_feature_2", "datetime_feature"],
            axis=1,
        ),
        "target",
        class_problem="binary",
    )


def test_mutual_info_to_target_multiclass(synthetic_train_test_data):
    mutual_info_to_target(
        synthetic_train_test_data[0].drop(
            ["categorical_feature_1", "categorical_feature_2", "datetime_feature"],
            axis=1,
        ),
        "target",
        class_problem="multiclass",
    )


def test_mutual_info_to_target_regression(synthetic_train_test_data_regression):
    mutual_info_to_target(
        synthetic_train_test_data_regression[0].drop(
            ["categorical_feature_1", "categorical_feature_2", "datetime_feature"],
            axis=1,
        ),
        "target",
        class_problem="regression",
    )


def test_pca_plot(synthetic_train_test_data):
    plot_pca(
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
    )
    assert True


def test_plot_pca_cumulative_variance(synthetic_train_test_data):
    plot_pca_cumulative_variance(
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
    )
    assert True
    plot_pca_cumulative_variance(
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
    )
    assert True


def test_plot_tsne(synthetic_train_test_data):
    plot_tsne(
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
        perplexity=30,
        random_state=0,
    )
    assert True


def test_plot_theil_u_heatmap(synthetic_categorical_data):
    columns_of_interest = synthetic_categorical_data.columns.to_list()
    theil_matrix = plot_theil_u_heatmap(synthetic_categorical_data, columns_of_interest)
    assert True
    assert theil_matrix[0, 0] == 1.0


def test_plot_count_pairs(synthetic_categorical_data):
    plot_count_pairs(
        synthetic_categorical_data,
        synthetic_categorical_data.head(2),
        cat_cols=synthetic_categorical_data.columns.to_list(),
    )
    assert True


def test_plot_null_percentage(create_data_with_nulls):
    plot_null_percentage(create_data_with_nulls)
    assert True


def test_check_unique_values(create_data_with_many_uniques):
    # Test with threshold of 0.9
    assert check_unique_values(
        create_data_with_many_uniques, ["col1", "col2", "col3", "col4"], 0.9
    ) == ["col1"]

    # Test with threshold of 0.8
    assert check_unique_values(
        create_data_with_many_uniques, ["col1", "col2", "col3"], 0.70
    ) == ["col1", "col2"]

    # Test with threshold of 0.5
    assert check_unique_values(
        create_data_with_many_uniques, ["col1", "col2", "col3"], 0.5
    ) == ["col1", "col2", "col3"]


def test_plot_ecdf(synthetic_train_test_data):
    num_data = synthetic_train_test_data[0]
    num_cols = [
        "numerical_feature_1",
        "numerical_feature_2",
        "numerical_feature_3",
        "target",
    ]
    plot_ecdf(num_data, num_cols, plot_all_at_once=False)
    plot_ecdf(num_data, num_cols, plot_all_at_once=True)
    assert True
