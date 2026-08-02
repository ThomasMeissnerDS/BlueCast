from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

from bluecast.preprocessing.custom import CustomPreprocessing
from bluecast.preprocessing.remove_collinearity import remove_correlated_columns


class LinearModelPreprocessingConfig:
    """Configuration for linear model preprocessing.

    :param scaler: Scaler type. Options: 'standard', 'power', 'robust', 'minmax'.
    :param imputation_strategy: Strategy for imputing missing values. Options: 'mean', 'median', 'constant'.
    :param collinearity_threshold: Correlation threshold for removing collinear features.
    :param add_polynomial_features: Whether to add polynomial interaction features.
    :param polynomial_degree: Degree of polynomial features.
    :param polynomial_interaction_only: If True, only interaction features are generated (no x^2).
    :param max_polynomial_features: Cap on the number of polynomial features to prevent explosion.
    """

    def __init__(
        self,
        scaler: Literal["standard", "power", "robust", "minmax"] = "standard",
        imputation_strategy: Literal["mean", "median", "constant"] = "median",
        collinearity_threshold: float = 0.9,
        add_polynomial_features: bool = False,
        polynomial_degree: int = 2,
        polynomial_interaction_only: bool = True,
        max_polynomial_features: int = 50,
    ):
        self.scaler = scaler
        self.imputation_strategy = imputation_strategy
        self.collinearity_threshold = collinearity_threshold
        self.add_polynomial_features = add_polynomial_features
        self.polynomial_degree = polynomial_degree
        self.polynomial_interaction_only = polynomial_interaction_only
        self.max_polynomial_features = max_polynomial_features


class PreprocessingForLinearModels(CustomPreprocessing):
    """Preprocessing pipeline tailored for linear models.

    Handles imputation, scaling, collinearity removal, and optional polynomial features.

    :param num_columns: List of numerical column names. If None, will auto-detect at fit time.
    :param config: LinearModelPreprocessingConfig instance. If None, uses defaults.
    """

    def __init__(
        self,
        num_columns: Optional[List] = None,
        config: Optional[LinearModelPreprocessingConfig] = None,
    ):
        super().__init__()
        self.config = config or LinearModelPreprocessingConfig()

        self.missing_val_imputer = SimpleImputer(
            missing_values=np.nan, strategy=self.config.imputation_strategy
        )
        self.scaler_instance = self._create_scaler()
        self.poly_transformer = None

        if isinstance(num_columns, list):
            self.num_columns = num_columns
        else:
            self.num_columns = []
        self.non_correlated_columns: List[Union[str, float, int]] = []
        self._auto_detected = num_columns is None
        self.poly_feature_names: List[str] = []

    def _create_scaler(self):
        scaler_type = self.config.scaler
        if scaler_type == "standard":
            return StandardScaler()
        elif scaler_type == "power":
            return PowerTransformer(method="yeo-johnson")
        elif scaler_type == "robust":
            return RobustScaler()
        elif scaler_type == "minmax":
            return MinMaxScaler()
        else:
            return StandardScaler()

    def _auto_detect_num_columns(self, df: pd.DataFrame) -> List[str]:
        """Auto-detect numerical columns from the DataFrame."""
        return df.select_dtypes(include=[np.number]).columns.tolist()

    def _add_polynomial_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Add polynomial interaction features to numeric columns."""
        if not self.config.add_polynomial_features:
            return df

        from sklearn.preprocessing import PolynomialFeatures

        num_cols_for_poly = self.non_correlated_columns
        if len(num_cols_for_poly) < 2:
            return df

        n_features_to_use = min(
            len(num_cols_for_poly), self.config.max_polynomial_features
        )
        cols_for_poly = num_cols_for_poly[:n_features_to_use]

        if fit:
            poly = PolynomialFeatures(
                degree=self.config.polynomial_degree,
                interaction_only=self.config.polynomial_interaction_only,
                include_bias=False,
            )
            poly_data = poly.fit_transform(df[cols_for_poly])
            self.poly_feature_names = poly.get_feature_names_out(cols_for_poly).tolist()
            self.poly_transformer = poly
        else:
            if self.poly_transformer is None:
                return df
            poly_data = self.poly_transformer.transform(df[cols_for_poly])

        new_feature_names = [
            name for name in self.poly_feature_names if name not in cols_for_poly
        ]
        new_feature_indices = [
            self.poly_feature_names.index(name) for name in new_feature_names
        ]
        if new_feature_indices:
            poly_df = pd.DataFrame(
                poly_data[:, new_feature_indices],
                columns=new_feature_names,
                index=df.index,
            )
            df = pd.concat([df, poly_df], axis=1)

        return df

    def fit_transform(
        self, df: pd.DataFrame, target: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        if self._auto_detected or not self.num_columns:
            self.num_columns = self._auto_detect_num_columns(df)

        active_num_cols = [c for c in self.num_columns if c in df.columns]

        df.loc[:, active_num_cols] = df.loc[:, active_num_cols].replace(
            [np.inf, -np.inf], np.nan
        )

        if len(active_num_cols) > 0:
            df.loc[:, active_num_cols] = self.missing_val_imputer.fit_transform(
                df.loc[:, active_num_cols]
            )
            df.loc[:, active_num_cols] = self.scaler_instance.fit_transform(
                df.loc[:, active_num_cols]
            )

        df_non_numerical = df.loc[
            :, [col for col in df.columns.tolist() if col not in active_num_cols]
        ]

        self.non_correlated_columns = remove_correlated_columns(
            df.loc[:, active_num_cols], self.config.collinearity_threshold
        ).columns.tolist()
        df_numerical = df.loc[:, self.non_correlated_columns]

        df = pd.concat([df_numerical, df_non_numerical], axis=1)
        df = self._add_polynomial_features(df, fit=True)

        return df, target

    def transform(
        self,
        df: pd.DataFrame,
        target: Optional[pd.Series] = None,
        prediction_mode: bool = False,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        active_num_cols = [c for c in self.num_columns if c in df.columns]

        df.loc[:, active_num_cols] = df.loc[:, active_num_cols].replace(
            [np.inf, -np.inf], np.nan
        )

        if len(active_num_cols) > 0:
            df.loc[:, active_num_cols] = self.missing_val_imputer.transform(
                df.loc[:, active_num_cols]
            )
            df.loc[:, active_num_cols] = self.scaler_instance.transform(
                df.loc[:, active_num_cols]
            )

        df_non_numerical = df.loc[
            :, [col for col in df.columns.tolist() if col not in active_num_cols]
        ]
        df_numerical = df.loc[:, self.non_correlated_columns]
        df = pd.concat([df_numerical, df_non_numerical], axis=1)
        df = self._add_polynomial_features(df, fit=False)

        return df, target
