from collections import namedtuple

import pandas as pd
import pytest

from bluecast.preprocessing.feature_creation import FeatureClusteringScorer


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
    cluster_results = rfm_c.fit_predict_cluster_rfm(
        test_df, keep_original_features=False
    )
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
    cluster_results = rfm_c.fit_predict_cluster_rfm(
        test_df, keep_original_features=False
    )
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
    cluster_results = rfm_c.fit_predict_cluster_rfm(
        test_df, keep_original_features=False
    )
    assert (
        len(cluster_results.columns) == len(cluster_settings) + 1
    )  # +1 for total score

    rfm_c = FeatureClusteringScorer(cluster_settings)
    cluster_results = rfm_c.fit_predict_cluster_rfm(
        test_df, keep_original_features=True
    )
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
    cluster_results = rfm_c.fit_predict_cluster_rfm(
        test_df, keep_original_features=True
    )

    assert "loyalty" in cluster_results.columns
    assert "loyalty_original" in cluster_results.columns
    assert cluster_results.head(1)["loyalty"].values[0] == 1
    assert cluster_results.tail(1)["loyalty"].values[0] == 4
    assert cluster_results["total_score"].max() == 11
