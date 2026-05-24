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
        :return: Original Pandas DataFrame with added row level standard deviations.
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
        :return: Original Pandas DataFrame with added row level minimums.
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
        :return: Original Pandas DataFrame with added row level maximums.
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
        :return: Original Pandas DataFrame with added row level sums.
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
            df = pl.from_pandas(df)

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

        return df_grouped.to_pandas(use_pyarrow_extension_array=True)


def add_groupby_agg_feats(
    df: pd.DataFrame,
    groupby_cols: List[str],
    to_group_cols: List[str],
    num_col_prefix: str,
    target_col: str,
    aggregations: List[str],
) -> pd.DataFrame:
    """
    Add groupby aggregation features to a DataFrame.

    :param df: Pandas DataFrame containing all relevant columns.
    :param groupby_cols: List of columns to use as groups.
    :param to_group_cols: List of columns to aggregate
    :param num_col_prefix: Prefix to add to the new columns
    :param target_col: String indicating the target column
    :param aggregations: List of aggregations to perform. If not provided, ["min", "max", "mean", "sum"] will be used.
    :return: Returns enriched DataFrame
    """
    group_agg_creator = GroupLevelAggFeatures()

    if not isinstance(aggregations, list):
        aggregations = ["min", "max", "mean", "sum"]

    num_aggs = group_agg_creator.create_groupby_agg_features(
        df=df,
        groupby_columns=groupby_cols,
        columns_to_agg=to_group_cols,
        target_col=target_col,
        aggregations=aggregations,  # falls back to some aggs
    )

    # update column names to avoid conflicts
    rename_dict = {}
    for col in to_group_cols:
        for agg in aggregations:
            rename_dict[col + "_" + agg] = num_col_prefix + "_" + str(col) + "_" + agg

    num_aggs = num_aggs.rename(columns=rename_dict)

    # joining the train information everywhere
    df = df.merge(num_aggs, on=groupby_cols, how="left")

    return df


class FeatureClusteringScorer:
    def __init__(
        self,
        cluster_settings: Dict[str, Any],
        random_state: int = 25,
    ):
        self.random_state = random_state  # control randomness
        self.cluster_settings = cluster_settings  # settings for each feature
        self.scalers: dict[str, MinMaxScaler] = {}  # storing scalers per feature
        self.cluster_classes: dict[str, KMeans] = {}  # storing Kmeans class per feature
        self.cluster_mappings: dict[str, dict[int, int]] = (
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


# ---------------------------------------------------------------------------
# Stateless Feature Engineering Utilities (LLM-Friendly)
# ---------------------------------------------------------------------------


def add_polynomial_features(
    df: pd.DataFrame, cols: List[str], degree: int = 2
) -> pd.DataFrame:
    """
    Statelessly adds polynomial features (power) for selected columns.
    Great for non-linear regression relationships.
    """
    df_out = df.copy()
    for col in cols:
        if col in df_out.columns:
            for d in range(2, degree + 1):
                df_out[f"{col}_pow_{d}"] = df_out[col] ** d
    return df_out


def add_interaction_features(
    df: pd.DataFrame,
    cols_a: List[str],
    cols_b: List[str],
    operations: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Creates interaction features among two lists of numeric columns.
    Supported ops: 'mul', 'div', 'add', 'sub'.
    """
    if operations is None:
        operations = ["mul", "div"]
    df_out = df.copy()
    for col1 in cols_a:
        for col2 in cols_b:
            if col1 == col2:
                continue
            if "mul" in operations:
                df_out[f"{col1}_mul_{col2}"] = df_out[col1] * df_out[col2]
            if "add" in operations:
                df_out[f"{col1}_add_{col2}"] = df_out[col1] + df_out[col2]
            if "sub" in operations:
                df_out[f"{col1}_sub_{col2}"] = df_out[col1] - df_out[col2]
            if "div" in operations:
                # Add a small epsilon to avoid division by zero
                df_out[f"{col1}_div_{col2}"] = df_out[col1] / (df_out[col2] + 1e-6)
    return df_out


def add_binned_features(
    df: pd.DataFrame,
    cols: List[str],
    num_bins: int = 5,
    state: Optional[Dict[str, Any]] = None,
    is_fit: bool = True,
) -> pd.DataFrame:
    """Bin continuous features into quantile bins.

    When ``state`` is provided, bin edges are stored during fit and reused
    during transform so that train and test data use identical bin boundaries.
    Without ``state`` the function is stateless (bin edges computed from the
    input data each time), which can cause train/test inconsistency.

    :param df: Input DataFrame.
    :param cols: Columns to bin.
    :param num_bins: Number of quantile bins.
    :param state: Optional dict to store/retrieve bin edges across fit/transform.
    :param is_fit: If True, compute bin edges; if False, reuse stored edges.
    :returns: DataFrame with ``{col}_binned`` columns added.
    """
    df_out = df.copy()
    for col in cols:
        if col not in df_out.columns:
            continue

        state_key = f"_binned_edges_{col}_{num_bins}"

        # Transform mode: reuse stored bin edges
        if state is not None and not is_fit and state_key in state:
            bins = state[state_key]
            df_out[f"{col}_binned"] = (
                pd.cut(
                    df_out[col],
                    bins=bins,
                    labels=False,
                    include_lowest=True,
                )
                .fillna(-1)
                .astype(int)
            )
        else:
            # Fit mode (or stateless fallback): compute bin edges
            try:
                result, bins = pd.qcut(
                    df_out[col],
                    q=num_bins,
                    labels=False,
                    retbins=True,
                    duplicates="drop",
                )
                df_out[f"{col}_binned"] = result
                if state is not None:
                    # Extend outer bins to -inf/+inf for unseen test values
                    bins = bins.copy()
                    bins[0] = -np.inf
                    bins[-1] = np.inf
                    state[state_key] = bins
            except Exception:
                df_out[f"{col}_binned"] = pd.cut(
                    df_out[col],
                    bins=num_bins,
                    labels=False,
                )
    return df_out


def add_datetime_features(df: pd.DataFrame, date_cols: List[str]) -> pd.DataFrame:
    """
    A lightweight stateless datetime extractor for pandas.
    Extracts year, month, day, dayofweek, and hour.
    """
    df_out = df.copy()
    for col in date_cols:
        if col in df_out.columns:
            # Attempt to convert to datetime if it's not already
            try:
                dt_series = pd.to_datetime(df_out[col], errors="coerce")
                df_out[f"{col}_year"] = dt_series.dt.year
                df_out[f"{col}_month"] = dt_series.dt.month
                df_out[f"{col}_day"] = dt_series.dt.day
                df_out[f"{col}_dayofweek"] = dt_series.dt.dayofweek
                df_out[f"{col}_hour"] = dt_series.dt.hour
                df_out = df_out.drop(columns=[col])
            except Exception:
                pass
    return df_out


class TfIdfTextEncoder:
    """
    Stateful TF-IDF encoder for seamless integration into AIFeaturePreprocessor.
    Safely separates fit_transform (training data) from transform (inference data).
    """

    def __init__(self, max_features: int = 50, stop_words: str = "english"):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            max_features=max_features, stop_words=stop_words
        )
        self.is_fitted = False
        self.feature_names: list[str] = []

    def fit_transform(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        if text_col not in df.columns:
            return df

        text_series = df[text_col].fillna("").astype(str)
        tfidf_matrix = self.vectorizer.fit_transform(text_series)
        self.is_fitted = True

        try:
            words = self.vectorizer.get_feature_names_out()
        except AttributeError:
            words = self.vectorizer.get_feature_names()

        self.feature_names = [f"tfidf_{text_col}_{w}" for w in words]
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(), columns=self.feature_names, index=df.index
        )

        for col in self.feature_names:
            df[col] = tfidf_df[col]

        return df

    def transform(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        if text_col not in df.columns or not self.is_fitted:
            return df

        text_series = df[text_col].fillna("").astype(str)
        tfidf_matrix = self.vectorizer.transform(text_series)

        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(), columns=self.feature_names, index=df.index
        )

        for col in self.feature_names:
            df[col] = tfidf_df[col]

        return df


class StateAwareGroupbyAggregator:
    """
    Stateless-acting Groupby Aggregator for AIFeaturePreprocessor.
    Calculates aggregations during 'fit_transform' and stores them in state.
    Mappings are applied during 'transform' to ensure no leakage from test set.
    """

    def __init__(
        self,
        groupby_cols: List[str],
        agg_cols: List[str],
        aggregations: Optional[List[str]] = None,
        prefix: str = "state_agg",
    ):
        self.groupby_cols = groupby_cols
        self.agg_cols = agg_cols
        self.aggregations = aggregations if aggregations is not None else ["mean"]
        self.prefix = prefix
        self.mappings: Dict[str, pd.DataFrame] = {}
        self.is_fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        # Perform aggregations
        agg_results = (
            df_out.groupby(self.groupby_cols)[self.agg_cols]
            .agg(self.aggregations)
            .reset_index()
        )

        # Flatten column names
        new_cols = []
        for col in agg_results.columns.values:
            if isinstance(col, tuple):
                # MultiIndex: (feature, agg)
                if col[0] in self.groupby_cols:
                    new_cols.append(col[0])
                else:
                    new_cols.append(f"{self.prefix}_{col[0]}_{col[1]}")
            else:
                # Single index
                new_cols.append(col)

        agg_results.columns = new_cols
        self.mappings["aggs"] = agg_results
        self.is_fitted = True

        # Drop columns from df_out that will be added by the merge to prevent duplicates
        cols_to_drop = [
            c
            for c in agg_results.columns
            if c in df_out.columns and c not in self.groupby_cols
        ]
        if cols_to_drop:
            df_out = df_out.drop(columns=cols_to_drop)

        # Merge back to original df
        df_out = df_out.merge(agg_results, on=self.groupby_cols, how="left")
        return df_out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            return df

        df_out = df.copy()

        cols_to_drop = [
            c
            for c in self.mappings["aggs"].columns
            if c in df_out.columns and c not in self.groupby_cols
        ]
        if cols_to_drop:
            df_out = df_out.drop(columns=cols_to_drop)

        df_out = df_out.merge(self.mappings["aggs"], on=self.groupby_cols, how="left")
        return df_out


def add_pca_features(
    df: pd.DataFrame,
    cols: List[str],
    n_components: int = 3,
    state: Optional[Dict[str, Any]] = None,
    is_fit: bool = True,
    prefix: str = "pca",
) -> pd.DataFrame:
    """Add PCA components as new features. Stateful: stores the fitted PCA in state.

    During fit (is_fit=True), fits a PCA on the specified columns and stores
    the fitted PCA scaler in `state` under key ``f'{prefix}_pca'``.
    During transform (is_fit=False), uses the previously fitted PCA from state.

    Missing values are filled with 0 before PCA. Columns are scaled to zero
    mean and unit variance before PCA.

    :param df: DataFrame with the columns to transform.
    :param cols: List of numeric column names to include in PCA.
    :param n_components: Number of PCA components to create.
    :param state: Mutable dict for storing fitted transformers between
        fit and transform phases (required for CV consistency).
    :param is_fit: True during training, False during inference/validation.
    :param prefix: Prefix for the new PCA column names.
    :return: DataFrame with added PCA columns.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if state is None:
        state = {}

    # Filter to columns that actually exist
    valid_cols = [c for c in cols if c in df.columns]
    if len(valid_cols) < 2:
        return df

    cols_hash = "_".join(valid_cols)
    state_key = f"{prefix}_pca_{cols_hash}"

    # Clamp n_components to the number of available columns
    n_components = min(n_components, len(valid_cols))

    X = df[valid_cols].fillna(0).values

    if is_fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=n_components, random_state=42)
        components = pca.fit_transform(X_scaled)

        state[state_key] = {"scaler": scaler, "pca": pca, "cols": valid_cols}
    else:
        if state_key not in state:
            # Not fitted yet — skip gracefully
            return df
        fitted = state[state_key]
        X_scaled = fitted["scaler"].transform(X)
        components = fitted["pca"].transform(X_scaled)

    for i in range(components.shape[1]):
        df[f"{prefix}_{i + 1}"] = components[:, i]

    return df
