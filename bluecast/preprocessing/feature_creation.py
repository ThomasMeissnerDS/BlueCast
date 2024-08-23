from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import polars as pl
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


class AddRowLevelAggFeatures:
    def __init__(self):
        self.original_features: List[Union[str, int, float]] = []

    def get_original_features(
        self, df: pd.DataFrame, target_col: Optional[str]
    ) -> None:
        if isinstance(target_col, str):
            self.original_features = df.drop(target_col, axis=1).columns.to_list()
        else:
            self.original_features = df.columns.to_list()

    def add_row_level_mean(
        self,
        df: pd.DataFrame,
        feature_to_agg: List[Union[str, int, float]],
        agg_col_name: str = "row_mean",
    ) -> pd.DataFrame:
        """
        Add row level mean of features to a dataframe.

        :param df: Pandas DataFrame holding all features.
        :param feature_to_agg: List of column names indicating which features to aggregate.
        :param agg_col_name: Name of the new column.
        :return: Original Pandas DataFrame with added row level means.
        """
        df[agg_col_name] = df[feature_to_agg].mean(axis=1)
        return df

    def add_row_level_std(
        self,
        df: pd.DataFrame,
        feature_to_agg: List[Union[str, int, float]],
        agg_col_name: str = "row_std",
    ) -> pd.DataFrame:
        """
        Add row level standard deviation of features to a dataframe.

        :param df: Pandas DataFrame holding all features.
        :param feature_to_agg: List of column names indicating which features to aggregate.
        :param agg_col_name: Name of the new column.
        :return: Original Pandas DataFrame with added row level means.
        """
        df[agg_col_name] = df[feature_to_agg].std(axis=1)
        return df

    def add_row_level_min(
        self,
        df: pd.DataFrame,
        feature_to_agg: List[Union[str, int, float]],
        agg_col_name: str = "row_min",
    ) -> pd.DataFrame:
        """
        Add row level min of features to a dataframe.

        :param df: Pandas DataFrame holding all features.
        :param feature_to_agg: List of column names indicating which features to aggregate.
        :param agg_col_name: Name of the new column.
        :return: Original Pandas DataFrame with added row level means.
        """
        df[agg_col_name] = df[feature_to_agg].min(axis=1)
        return df

    def add_row_level_max(
        self,
        df: pd.DataFrame,
        feature_to_agg: List[Union[str, int, float]],
        agg_col_name: str = "row_max",
    ) -> pd.DataFrame:
        """
        Add row level max of features to a dataframe.

        :param df: Pandas DataFrame holding all features.
        :param feature_to_agg: List of column names indicating which features to aggregate.
        :param agg_col_name: Name of the new column.
        :return: Original Pandas DataFrame with added row level means.
        """
        df[agg_col_name] = df[feature_to_agg].max(axis=1)
        return df

    def add_row_level_sum(
        self,
        df: pd.DataFrame,
        feature_to_agg: List[Union[str, int, float]],
        agg_col_name: str = "row_sum",
    ) -> pd.DataFrame:
        """
        Add row level sum of features to a dataframe.

        :param df: Pandas DataFrame holding all features.
        :param feature_to_agg: List of column names indicating which features to aggregate.
        :param agg_col_name: Name of the new column.
        :return: Original Pandas DataFrame with added row level means.
        """
        df[agg_col_name] = df[feature_to_agg].sum(axis=1)
        return df

    def add_row_level_agg_features(
        self, df: pd.DataFrame, target_col: Optional[str] = None
    ) -> pd.DataFrame:
        self.get_original_features(df, target_col)
        df = self.add_row_level_mean(df, self.original_features)
        df = self.add_row_level_std(df, self.original_features)
        df = self.add_row_level_min(df, self.original_features)
        df = self.add_row_level_max(df, self.original_features)
        df = self.add_row_level_sum(df, self.original_features)
        return df


class GroupLevelAggFeatures:
    def __init__(self):
        self.original_features: List[Union[str, int, float]] = []
        self.agg_features_created: List[Union[str, int, float]] = []

    def create_groupby_agg_features(
        self,
        df: Union[pd.DataFrame, pl.DataFrame],
        groupby_columns: List[str],
        columns_to_agg: Optional[List[str]],
        target_col: Optional[str],
        aggregations: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Create aggregations based on groups for a given DataFrame.

        :param df: Either Pandas or Polars DataFrame.
        :param groupby_columns: List of column names to use for the groupby.
        :param columns_to_agg: List of columns to aggregate. If empty all columns except
            target column (target_col) will be chosen.
        :param target_col: Target column name. Will be ignored during aggregation.
        :param aggregations: Aggregations to perform. If not provided, ["min", "max", "mean", "sum"] will be used.
        :return: Aggregated  Pandas DataFrame
        """
        if not isinstance(aggregations, list):
            aggregations = ["min", "max", "mean", "sum"]

        if isinstance(df, pd.DataFrame):
            df = pl.from_dataframe(df)

        self.original_features = df.columns

        # Determine which columns to aggregate
        if not columns_to_agg:
            columns_to_agg = df.columns

        # Remove the target column from the aggregation list if specified
        if isinstance(columns_to_agg, list):
            if target_col in columns_to_agg:
                columns_to_agg.remove(target_col)

        # Define the aggregation operations
        agg_ops = []
        if isinstance(columns_to_agg, list):
            for col in columns_to_agg:
                for agg in aggregations:
                    agg_ops.append(getattr(pl.col(col), agg)().alias(f"{col}_{agg}"))
                    self.agg_features_created.append(f"{col}_{agg}")

        df_grouped = df.group_by(groupby_columns).agg(agg_ops)

        return df_grouped.to_pandas()


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

    def fit_predict_cluster(
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

    def predict_cluster(self, df: pd.DataFrame, keep_original_features: bool = True):
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
