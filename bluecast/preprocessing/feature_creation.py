from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


class FeatureClusteringScorer:
    def __init__(
        self,
        cluster_settings: Dict[str, Any],
        random_state: int = 25,
    ):
        self.random_state = random_state  # control randomness
        self.cluster_settings = cluster_settings  # settings for each feature
        self.scalers: Dict[str, MinMaxScaler] = {}  # storing scalers per feature
        self.cluster_classes: Dict[str, KMeans] = {}  # storing Kmeans class per feature
        self.cluster_mappings: Dict[str, Dict[int, int]] = (
            {}
        )  # storing reindex mapping for cluster ids

    def _fit_reindex_clusters_by_mean(
        self, temp_df: pd.DataFrame, feature_name: str, higher_is_better: bool = True
    ) -> np.ndarray:
        """
        Fix cluster indices.

        Cluster indices do not follow the order of the original feature (i.e. highest value might nbe cluster 0).
        This function reindexes the cluster idx, so the total value make sense.

        :param temp_df: DataFrame containing two columns: the 'cluster' and the original feature
        :param feature_name: String indicating the name of the original feature.
        :param higher_is_better: Boolean indicating if the cluster index should raise with increasing values of the
            original feature.
        :return: Nmpy array with corrected cluster indices
        """
        # Calculate the mean of the feature values for each cluster
        cluster_means = temp_df.groupby("cluster")[feature_name].mean()

        # Sort the clusters by their mean values
        if higher_is_better:
            sorted_clusters = cluster_means.sort_values(ascending=True).index
        else:
            sorted_clusters = cluster_means.sort_values(ascending=False).index

        # Create a mapping from old cluster IDs to new cluster IDs
        self.cluster_mappings[feature_name] = {
            old: new for new, old in enumerate(sorted_clusters)
        }

        # Apply the mapping to reindex the cluster IDs in the DataFrame
        temp_df["cluster"] = temp_df["cluster"].map(self.cluster_mappings[feature_name])

        return temp_df["cluster"].values

    def _predict_reindex_clusters_by_mean(
        self, temp_df: pd.DataFrame, feature_name: str
    ) -> np.ndarray:
        """
        Fix cluster indices.

        Cluster indices do not follow the order of the original feature (i.e. highest value might nbe cluster 0).
        This function reindexes the cluster idx, so the total value make sense.

        :param temp_df: DataFrame containing two columns: the 'cluster' and the original feature
        :param feature_name: String indicating the name of the original feature.
        :return: Nmpy array with corrected cluster indices
        """
        # Apply the mapping to reindex the cluster IDs in the DataFrame
        temp_df["cluster"] = temp_df["cluster"].map(self.cluster_mappings[feature_name])

        return temp_df["cluster"].values

    def _fit_cluster_feature(
        self,
        df: pd.DataFrame,
        feature_name: str,
        nb_clusters: int,
        higher_is_better: bool,
    ) -> np.ndarray:
        """
        Cluster individual feature.

        :param df: DataFrame with original features.
        :param feature_name: String indicating the feature name.
        :param nb_clusters: Integer indicating how many clusters shall be found.
        :return: Numpy array with cluster ids
        """
        self.scalers[feature_name] = MinMaxScaler()
        feat_scaled = self.scalers[feature_name].fit_transform(df[[feature_name]])

        self.cluster_classes[feature_name] = KMeans(
            random_state=self.random_state, n_clusters=nb_clusters
        )
        clusters_found = self.cluster_classes[feature_name].fit_predict(feat_scaled)

        # sort and reindex cluster such as the highest value rewards most points
        temp_df = pd.DataFrame(
            {feature_name: df[feature_name], "cluster": clusters_found}
        )
        reindexed_clusters = self._fit_reindex_clusters_by_mean(
            temp_df, feature_name, higher_is_better
        )

        # move away from 0 index for scoring
        return reindexed_clusters + 1

    def _predict_cluster_feature(
        self, df: pd.DataFrame, feature_name: str
    ) -> np.ndarray:
        """
        Cluster individual feature.

        :param df: DataFrame with original features.
        :param feature_name: String indicating the feature name.
        :return: Numpy array with cluster ids
        """
        feat_scaled = self.scalers[feature_name].transform(df[[feature_name]])

        clusters_found = self.cluster_classes[feature_name].predict(feat_scaled)

        # sort and reindex cluster such as the highest value rewards most points
        temp_df = pd.DataFrame(
            {feature_name: df[feature_name], "cluster": clusters_found}
        )
        reindexed_clusters = self._predict_reindex_clusters_by_mean(
            temp_df, feature_name
        )

        # move away from 0 index for scoring
        return reindexed_clusters + 1

    def fit_predict_cluster_rfm(
        self, df: pd.DataFrame, keep_original_features: bool = True
    ):
        """
        Calculate cluster (i.e. RFM) scores based on input features.

        :param df: Pandas DataFrame including the original features. Additional feature will be ignored.
        :param keep_original_features: If true, return clusters and original dataframe. Otherwise return RFM results
            only.
        :return: Pandas DataFrame with RFM scores
        """
        if keep_original_features:
            cluster_results_df = df.copy()
            cluster_results_df.columns = [
                f"{col}_original" for col in cluster_results_df.columns.to_list()
            ]
        else:
            cluster_results_df = pd.DataFrame()

        for feature, cluster_setting in self.cluster_settings.items():
            clusters_found = self._fit_cluster_feature(
                df,
                feature,
                cluster_setting.nb_clusters,
                cluster_setting.higher_is_better,
            )
            cluster_results_df[feature] = clusters_found

        cluster_results_df["total_score"] = cluster_results_df[
            self.cluster_settings.keys()
        ].sum(axis=1)
        return cluster_results_df

    def predict_cluster_rfm(
        self, df: pd.DataFrame, keep_original_features: bool = True
    ):
        """
        Calculate cluster (i.e. RFM) scores based on input features.

        :param df: Pandas DataFrame including the original features. Additional feature will be ignored.
        :param keep_original_features: If true, return clusters and original dataframe. Otherwise return RFM results
            only.
        :return: Pandas DataFrame with RFM scores
        """
        if keep_original_features:
            cluster_results_df = df.copy()
            cluster_results_df.columns = [
                f"{col}_original" for col in cluster_results_df.columns.to_list()
            ]
        else:
            cluster_results_df = pd.DataFrame()

        for feature, _cluster_setting in self.cluster_settings.items():
            clusters_found = self._predict_cluster_feature(df, feature)
            cluster_results_df[feature] = clusters_found

        cluster_results_df["total_score"] = cluster_results_df[
            self.cluster_settings.keys()
        ].sum(axis=1)
        return cluster_results_df