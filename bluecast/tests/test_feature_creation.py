from collections import namedtuple

import pandas as pd
import polars as pl
import pytest

from bluecast.preprocessing.feature_creation import (
    AddRowLevelAggFeatures,
    FeatureClusteringScorer,
    GroupLevelAggFeatures,
)


@pytest.fixture
def sample_dataframe():
    data = {
        "A": [1, 2, 3, 4],
        "B": [5, 6, 7, 8],
        "C": [9, 10, 11, 12],
        "target": [0, 1, 0, 1],
    }
    return pd.DataFrame(data)


def test_initialization():
    aggregator = AddRowLevelAggFeatures()
    assert aggregator.original_features == []


def test_get_original_features(sample_dataframe):
    aggregator = AddRowLevelAggFeatures()
    aggregator.get_original_features(sample_dataframe, "target")
    assert aggregator.original_features == ["A", "B", "C"]


def test_add_row_level_mean(sample_dataframe):
    aggregator = AddRowLevelAggFeatures()
    df = aggregator.add_row_level_mean(sample_dataframe, ["A", "B", "C"], "row_mean")
    expected_mean = sample_dataframe[["A", "B", "C"]].mean(axis=1)
    pd.testing.assert_series_equal(
        df["row_mean"], expected_mean, check_names=False, check_like=True
    )


def test_add_row_level_std(sample_dataframe):
    aggregator = AddRowLevelAggFeatures()
    df = aggregator.add_row_level_std(sample_dataframe, ["A", "B", "C"], "row_std")
    expected_std = sample_dataframe[["A", "B", "C"]].std(axis=1)
    pd.testing.assert_series_equal(df["row_std"], expected_std, check_names=False)


def test_add_row_level_min(sample_dataframe):
    aggregator = AddRowLevelAggFeatures()
    df = aggregator.add_row_level_min(sample_dataframe, ["A", "B", "C"], "row_min")
    expected_min = sample_dataframe[["A", "B", "C"]].min(axis=1)
    pd.testing.assert_series_equal(df["row_min"], expected_min, check_names=False)


def test_add_row_level_max(sample_dataframe):
    aggregator = AddRowLevelAggFeatures()
    df = aggregator.add_row_level_max(sample_dataframe, ["A", "B", "C"], "row_max")
    expected_max = sample_dataframe[["A", "B", "C"]].max(axis=1)
    pd.testing.assert_series_equal(df["row_max"], expected_max, check_names=False)


def test_add_row_level_sum(sample_dataframe):
    aggregator = AddRowLevelAggFeatures()
    df = aggregator.add_row_level_sum(sample_dataframe, ["A", "B", "C"], "row_sum")
    expected_sum = sample_dataframe[["A", "B", "C"]].sum(axis=1)
    pd.testing.assert_series_equal(df["row_sum"], expected_sum, check_names=False)


def test_add_row_level_agg_features(sample_dataframe):
    aggregator = AddRowLevelAggFeatures()
    df = aggregator.add_row_level_agg_features(sample_dataframe, "target")

    # Verify if the correct columns are added
    expected_mean = sample_dataframe[["A", "B", "C"]].mean(axis=1)
    expected_std = sample_dataframe[["A", "B", "C"]].std(axis=1)
    expected_min = sample_dataframe[["A", "B", "C"]].min(axis=1)
    expected_max = sample_dataframe[["A", "B", "C"]].max(axis=1)
    expected_sum = sample_dataframe[["A", "B", "C"]].sum(axis=1)

    pd.testing.assert_series_equal(df["row_mean"], expected_mean, check_names=False)
    pd.testing.assert_series_equal(df["row_std"], expected_std, check_names=False)
    pd.testing.assert_series_equal(df["row_min"], expected_min, check_names=False)
    pd.testing.assert_series_equal(df["row_max"], expected_max, check_names=False)
    pd.testing.assert_series_equal(df["row_sum"], expected_sum, check_names=False)


@pytest.fixture
def synthetic_data():
    test_df = pd.DataFrame(
        {
            "customer_id": [f"M{i}" for i in range(100)],
            "recency": [i for i in range(100)][::-1],  # inverse order
            "frequency": [i for i in range(100)],
            "monetary": [i for i in range(100)],
        }
    )
    return test_df


def test_higher_is_better(synthetic_data):
    """Test if higher is better than lower param works"""
    test_df = synthetic_data

    cluster_setting = namedtuple(
        "cluster_settings", ("nb_clusters", "higher_is_better")
    )

    cluster_settings = {
        "recency": cluster_setting(2, False),
        "frequency": cluster_setting(3, True),
        "monetary": cluster_setting(2, True),
    }

    rfm_c = FeatureClusteringScorer(cluster_settings)
    cluster_results = rfm_c.fit_predict_cluster(test_df, keep_original_features=False)
    assert cluster_results.head(1)["recency"].values[0] == 1
    assert cluster_results.tail(1)["recency"].values[0] == 2

    assert cluster_results.head(1)["frequency"].values[0] == 1
    assert cluster_results.tail(1)["frequency"].values[0] == 3

    assert cluster_results.head(1)["monetary"].values[0] == 1
    assert cluster_results.tail(1)["monetary"].values[0] == 2

    cluster_settings = {
        "recency": cluster_setting(2, True),
        "frequency": cluster_setting(3, False),
        "monetary": cluster_setting(2, False),
    }

    rfm_c = FeatureClusteringScorer(cluster_settings)
    cluster_results = rfm_c.fit_predict_cluster(test_df, keep_original_features=False)
    assert cluster_results.head(1)["recency"].values[0] == 2
    assert cluster_results.tail(1)["recency"].values[0] == 1

    assert cluster_results.head(1)["frequency"].values[0] == 3
    assert cluster_results.tail(1)["frequency"].values[0] == 1

    assert cluster_results.head(1)["monetary"].values[0] == 2
    assert cluster_results.tail(1)["monetary"].values[0] == 1


def test_keep_original_features(synthetic_data):
    """Test if keep original features works"""
    test_df = synthetic_data

    cluster_setting = namedtuple(
        "cluster_settings", ("nb_clusters", "higher_is_better")
    )

    cluster_settings = {
        "recency": cluster_setting(2, False),
        "frequency": cluster_setting(3, True),
        "monetary": cluster_setting(2, True),
    }

    rfm_c = FeatureClusteringScorer(cluster_settings)
    cluster_results = rfm_c.fit_predict_cluster(test_df, keep_original_features=False)
    assert (
        len(cluster_results.columns) == len(cluster_settings) + 1
    )  # +1 for total score

    cluster_results = rfm_c.predict_cluster(test_df, keep_original_features=False)
    assert (
        len(cluster_results.columns) == len(cluster_settings) + 1
    )  # +1 for total score

    rfm_c = FeatureClusteringScorer(cluster_settings)
    cluster_results = rfm_c.fit_predict_cluster(test_df, keep_original_features=True)
    assert (
        len(cluster_results.columns) == len(cluster_settings) + len(test_df.columns) + 1
    )  # +1 for total score

    assert "recency" in cluster_results.columns
    assert "frequency" in cluster_results.columns
    assert "monetary" in cluster_results.columns
    assert "total_score" in cluster_results.columns
    assert "customer_id_original" in cluster_results.columns
    assert "recency_original" in cluster_results.columns
    assert "frequency_original" in cluster_results.columns
    assert "monetary_original" in cluster_results.columns

    cluster_results = rfm_c.predict_cluster(test_df, keep_original_features=True)
    assert (
        len(cluster_results.columns) == len(cluster_settings) + len(test_df.columns) + 1
    )  # +1 for total score

    assert "recency" in cluster_results.columns
    assert "frequency" in cluster_results.columns
    assert "monetary" in cluster_results.columns
    assert "total_score" in cluster_results.columns
    assert "customer_id_original" in cluster_results.columns
    assert "recency_original" in cluster_results.columns
    assert "frequency_original" in cluster_results.columns
    assert "monetary_original" in cluster_results.columns


def test_changing_features(synthetic_data):
    """Test if can use custom features"""
    test_df = synthetic_data
    test_df["loyalty"] = [i for i in range(100)]

    cluster_setting = namedtuple(
        "cluster_settings", ("nb_clusters", "higher_is_better")
    )

    cluster_settings = {
        "recency": cluster_setting(2, False),
        "frequency": cluster_setting(3, True),
        "monetary": cluster_setting(2, True),
        "loyalty": cluster_setting(4, True),
    }

    rfm_c = FeatureClusteringScorer(cluster_settings)
    cluster_results = rfm_c.fit_predict_cluster(test_df, keep_original_features=True)

    assert "loyalty" in cluster_results.columns
    assert "loyalty_original" in cluster_results.columns
    assert cluster_results.head(1)["loyalty"].values[0] == 1
    assert cluster_results.tail(1)["loyalty"].values[0] == 4
    assert cluster_results["total_score"].max() == 11

    cluster_results = rfm_c.predict_cluster(test_df, keep_original_features=True)

    assert "loyalty" in cluster_results.columns
    assert "loyalty_original" in cluster_results.columns
    assert cluster_results.head(1)["loyalty"].values[0] == 1
    assert cluster_results.tail(1)["loyalty"].values[0] == 4
    assert cluster_results["total_score"].max() == 11


# test grouplevelaggfeature creator
@pytest.fixture
def polars_dataframe(sample_dataframe):
    return pl.from_pandas(sample_dataframe)


def test_initialization_grouplevelaggfeatures():
    aggregator = GroupLevelAggFeatures()
    assert aggregator.original_features == []
    assert aggregator.agg_features_created == []


def test_create_groupby_agg_features_pandas(sample_dataframe):
    aggregator = GroupLevelAggFeatures()
    groupby_columns = ["target"]
    columns_to_agg = ["A", "B"]
    target_col = "C"
    aggregations = ["mean", "sum"]

    result_df = aggregator.create_groupby_agg_features(
        sample_dataframe, groupby_columns, columns_to_agg, target_col, aggregations
    )

    expected_columns = ["target", "A_mean", "A_sum", "B_mean", "B_sum"]
    assert list(result_df.columns) == expected_columns
    assert aggregator.original_features == ["A", "B", "C", "target"]
    assert aggregator.agg_features_created == ["A_mean", "A_sum", "B_mean", "B_sum"]


def test_create_groupby_agg_features_polars(polars_dataframe):
    aggregator = GroupLevelAggFeatures()
    groupby_columns = ["target"]
    columns_to_agg = ["A", "B"]
    target_col = "C"
    aggregations = ["min", "max"]

    result_df = aggregator.create_groupby_agg_features(
        polars_dataframe, groupby_columns, columns_to_agg, target_col, aggregations
    )

    expected_columns = ["target", "A_min", "A_max", "B_min", "B_max"]
    assert list(result_df.columns) == expected_columns
    assert aggregator.original_features == ["A", "B", "C", "target"]
    assert aggregator.agg_features_created == ["A_min", "A_max", "B_min", "B_max"]


def test_create_groupby_agg_features_no_columns_to_agg(sample_dataframe):
    aggregator = GroupLevelAggFeatures()
    groupby_columns = ["target"]
    columns_to_agg = None
    target_col = "C"

    result_df = aggregator.create_groupby_agg_features(
        sample_dataframe, groupby_columns, columns_to_agg, target_col
    )

    expected_columns = [
        "target",
        "A_min",
        "A_max",
        "A_mean",
        "A_sum",
        "B_min",
        "B_max",
        "B_mean",
        "B_sum",
        "target_min",
        "target_max",
        "target_mean",
        "target_sum",
    ]
    assert list(result_df.columns) == expected_columns
    assert aggregator.original_features == ["A", "B", "C", "target"]
    assert aggregator.agg_features_created == [
        "A_min",
        "A_max",
        "A_mean",
        "A_sum",
        "B_min",
        "B_max",
        "B_mean",
        "B_sum",
        "target_min",
        "target_max",
        "target_mean",
        "target_sum",
    ]


def test_create_groupby_agg_features_no_target_column(sample_dataframe):
    aggregator = GroupLevelAggFeatures()
    groupby_columns = ["target"]
    columns_to_agg = ["A", "B"]
    target_col = None

    result_df = aggregator.create_groupby_agg_features(
        sample_dataframe, groupby_columns, columns_to_agg, target_col
    )

    expected_columns = [
        "target",
        "A_min",
        "A_max",
        "A_mean",
        "A_sum",
        "B_min",
        "B_max",
        "B_mean",
        "B_sum",
    ]
    assert list(result_df.columns) == expected_columns
    assert aggregator.original_features == ["A", "B", "C", "target"]
    assert aggregator.agg_features_created == [
        "A_min",
        "A_max",
        "A_mean",
        "A_sum",
        "B_min",
        "B_max",
        "B_mean",
        "B_sum",
    ]


def test_create_groupby_agg_features_default_aggregations(sample_dataframe):
    aggregator = GroupLevelAggFeatures()
    groupby_columns = ["target"]
    columns_to_agg = ["A"]
    target_col = None

    result_df = aggregator.create_groupby_agg_features(
        sample_dataframe, groupby_columns, columns_to_agg, target_col
    )

    expected_columns = ["target", "A_min", "A_max", "A_mean", "A_sum"]
    assert list(result_df.columns) == expected_columns
    assert aggregator.original_features == ["A", "B", "C", "target"]
    assert aggregator.agg_features_created == ["A_min", "A_max", "A_mean", "A_sum"]
