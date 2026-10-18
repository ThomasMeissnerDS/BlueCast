import logging
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from category_encoders import OneHotEncoder, TargetEncoder
from category_encoders.wrapper import NestedCVWrapper
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", "is_categorical_dtype")


class BinaryClassTargetEncoder:
    """Target encode categorical features in the context of binary classification using NestedCVWrapper."""

    def __init__(
        self, cat_columns: List[Union[str, float, int]], random_state: int = 200
    ):
        self.encoders: Dict[str, NestedCVWrapper] = {}
        self.prediction_mode: bool = False
        self.cat_columns = cat_columns
        self.random_state = random_state

    def fit_target_encode_binary_class(
        self, x: pd.DataFrame, y: pd.Series
    ) -> pd.DataFrame:
        """Fit target encoder using NestedCVWrapper and transform columns."""
        logging.info("Start fitting binary target encoder with NestedCVWrapper.")

        # Check and encode target if it is not numeric (i.e. string labels)
        if y.dtype == "object" or y.dtype.name == "category":
            logging.info(
                "Encoding string targets temporarily for binary classification."
            )
            label_encoder = LabelEncoder()
            y = pd.Series(label_encoder.fit_transform(y))
            y.index = x.index

        smoothing = np.max([np.log10(len(x.index)) * 5, 10])

        # Wrap TargetEncoder with NestedCVWrapper
        enc = NestedCVWrapper(
            TargetEncoder(
                cols=self.cat_columns,
                smoothing=smoothing,
            ),
            cv=5,  # Specify number of folds for cross-validation
            random_state=self.random_state,
        )

        x.loc[:, self.cat_columns] = enc.fit_transform(x[self.cat_columns], y)
        x[self.cat_columns] = x[self.cat_columns].astype(float)
        self.encoders["target_encoder_all_cols"] = enc
        return x

    def transform_target_encode_binary_class(self, x: pd.DataFrame) -> pd.DataFrame:
        """Transform categories based on already trained encoder."""
        logging.info("Start transforming categories with binary target encoder.")
        enc = self.encoders["target_encoder_all_cols"]
        x.loc[:, self.cat_columns] = enc.transform(x[self.cat_columns])
        x[self.cat_columns] = x[self.cat_columns].astype(float)
        return x


class MultiClassTargetEncoder:
    """Target encode categorical features in the context of multiclass classification using NestedCVWrapper."""

    def __init__(
        self,
        cat_columns: List[Union[str, float, int]],
        target_col: Union[str, float, int],
        random_state: int = 200,
    ):
        self.encoders: Dict[str, Union[NestedCVWrapper, OneHotEncoder]] = {}
        self.prediction_mode: bool = False
        self.cat_columns = cat_columns
        self.class_names: List[Optional[Union[str, float, int]]] = []
        self.target_col = target_col
        self.random_state = random_state

    def fit_target_encode_multiclass(
        self, x: pd.DataFrame, y: pd.Series
    ) -> pd.DataFrame:
        """Fit target encoder using NestedCVWrapper and transform columns."""
        logging.info("Start fitting multiclass target encoder with NestedCVWrapper.")

        # Check and encode target if it is not numeric (i.e. string labels)
        if y.dtype == "object" or y.dtype.name == "category":
            logging.info(
                "Encoding string targets temporarily for multiclass classification."
            )
            label_encoder = LabelEncoder()
            y = pd.Series(label_encoder.fit_transform(y))
            y.index = x.index

        algorithm = "multiclass_target_encoding_onehotter"
        enc = OneHotEncoder(
            # drop_invariant=True, handle_unknown="ignore"
        )
        enc.fit(y)
        y_onehot = enc.transform(y)
        self.class_names = y_onehot.columns.to_list()
        self.cat_columns = [c for c in self.cat_columns if c != self.target_col]

        x_obj = x.loc[:, self.cat_columns].copy()
        x = x.loc[:, ~x.columns.isin(self.cat_columns)]
        for class_ in self.class_names:
            smoothing = np.max([np.log10(len(x.index)) * 5, 10])

            # Wrap TargetEncoder with NestedCVWrapper for each class
            target_enc = NestedCVWrapper(
                TargetEncoder(
                    cols=self.cat_columns,
                    smoothing=smoothing,
                ),
                cv=5,  # Specify number of folds for cross-validation
                random_state=self.random_state,
            )
            target_enc.fit(x_obj, y_onehot[class_])
            self.encoders[f"multiclass_target_encoder_all_cols_{class_}"] = target_enc
            temp = target_enc.transform(x_obj)
            temp.columns = [str(col) + "_" + str(class_) for col in temp.columns]
            x = pd.concat([x, temp], axis=1)
        self.encoders[f"{algorithm}_all_cols"] = enc
        return x

    def transform_target_encode_multiclass(self, x: pd.DataFrame) -> pd.DataFrame:
        """Transform categories based on already trained encoder."""
        logging.info("Start transforming categories with multiclass target encoder.")
        algorithm = "multiclass_target_encoding_onehotter"
        enc = self.encoders[f"{algorithm}_all_cols"]

        x_obj = x.loc[:, self.cat_columns].copy()
        x = x.loc[:, ~x.columns.isin(self.cat_columns)]
        for class_ in self.class_names:
            target_enc = self.encoders[f"multiclass_target_encoder_all_cols_{class_}"]
            temp = target_enc.transform(x_obj)
            temp.columns = [str(col) + "_" + str(class_) for col in temp.columns]
            x = pd.concat([x, temp], axis=1)
        self.encoders[f"{algorithm}_all_cols"] = enc
        return x


class RegressionTargetEncoder:
    """Target encode categorical features in the context of regression using NestedCVWrapper."""

    def __init__(
        self, cat_columns: List[Union[str, float, int]], random_state: int = 200
    ):
        self.encoders: Dict[str, NestedCVWrapper] = {}
        self.prediction_mode: bool = False
        self.cat_columns = cat_columns
        self.random_state = random_state

    def fit_target_encode_regression(
        self, x: pd.DataFrame, y: pd.Series
    ) -> pd.DataFrame:
        """Fit target encoder using NestedCVWrapper and transform columns."""
        logging.info("Start fitting regression target encoder with NestedCVWrapper.")

        y = pd.qcut(y, q=10, labels=False, duplicates="drop")

        smoothing = np.max([np.log10(len(x.index)) * 5, 10])

        # Use KFold for regression
        enc = NestedCVWrapper(
            TargetEncoder(
                cols=self.cat_columns,
                smoothing=smoothing,
            ),
            cv=StratifiedKFold(
                n_splits=5, shuffle=True, random_state=self.random_state
            ),
            random_state=self.random_state,
        )

        x.loc[:, self.cat_columns] = enc.fit_transform(x[self.cat_columns], y)
        x[self.cat_columns] = x[self.cat_columns].astype(float)
        self.encoders["target_encoder_all_cols"] = enc
        return x

    def transform_target_encode_regression(self, x: pd.DataFrame) -> pd.DataFrame:
        """Transform categories based on already trained encoder."""
        logging.info("Start transforming categories with regression target encoder.")
        enc = self.encoders["target_encoder_all_cols"]
        x.loc[:, self.cat_columns] = enc.transform(x[self.cat_columns])
        x[self.cat_columns] = x[self.cat_columns].astype(float)
        return x


class OOFTargetEncoder:
    """Out-of-Fold Target Encoder with Bayesian Smoothing.

    Designed to strictly prevent target leakage while keeping robust encodings for rare categories.
    """

    def __init__(
        self,
        cat_columns: List[Union[str, float, int]],
        random_state: int = 42,
        cv: int = 5,
        smoothing: float = 10.0,
    ):
        self.cat_columns = cat_columns
        self.random_state = random_state
        self.cv = cv
        self.smoothing = smoothing
        self.global_mean: float = 0.0
        self.global_mappings: Dict[str, pd.DataFrame] = {}
        self.is_fitted: bool = False

    def _bayesian_smoothing(
        self, df_cat: pd.Series, target: pd.Series, global_mean: float
    ) -> pd.Series:
        """Calculate Bayesian smoothed target encoding for a single categorical column."""
        stats = target.groupby(df_cat).agg(["count", "mean"])

        # Bayesian smoothing formula: (count * mean + smoothing * global_mean) / (count + smoothing)
        smoothed = (stats["count"] * stats["mean"] + self.smoothing * global_mean) / (
            stats["count"] + self.smoothing
        )
        return smoothed

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform using Out-of-Fold methodology."""
        logging.info(
            f"Start OOF Target Encoding for {len(self.cat_columns)} columns..."
        )

        # Encode targets temporarily if they are strings
        y_encoded = y
        if y.dtype == "object" or y.dtype.name == "category":
            y_encoded = pd.Series(LabelEncoder().fit_transform(y), index=y.index)

        self.global_mean = y_encoded.mean()
        X_encoded = X.copy()

        kf = StratifiedKFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )

        # Determine if regression or classification for StratifiedKFold
        # If too many unique values, fallback to normal KFold logic by binning
        if y_encoded.nunique() > 20:
            y_stratify = pd.qcut(y_encoded, q=10, labels=False, duplicates="drop")
        else:
            y_stratify = y_encoded

        # OOF Loop
        for cat_col in self.cat_columns:
            oof_encoded = pd.Series(np.nan, index=X.index)

            for train_idx, val_idx in kf.split(X, y_stratify):
                X_tr, y_tr = X.iloc[train_idx][cat_col], y_encoded.iloc[train_idx]
                X_va = X.iloc[val_idx][cat_col]

                smoothed_mapping = self._bayesian_smoothing(
                    X_tr, y_tr, self.global_mean
                )
                oof_encoded.iloc[val_idx] = X_va.map(smoothed_mapping)

            # Fill unseen categories in validation folds with global mean
            oof_encoded = oof_encoded.fillna(self.global_mean)
            X_encoded[cat_col] = oof_encoded.astype(float)

            # Calculate global mapping for transform (test set)
            self.global_mappings[str(cat_col)] = self._bayesian_smoothing(
                X[cat_col], y_encoded, self.global_mean
            )

        self.is_fitted = True
        return X_encoded

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform unseen data using global mappings."""
        if not self.is_fitted:
            raise RuntimeError(
                "OOFTargetEncoder must be fitted before calling transform."
            )

        logging.info("Start transforming data with OOF Target Encoder...")
        X_encoded = X.copy()

        for cat_col in self.cat_columns:
            if cat_col in X_encoded.columns:
                mapping = self.global_mappings.get(str(cat_col))
                if mapping is not None:
                    X_encoded[cat_col] = (
                        X_encoded[cat_col]
                        .map(mapping)
                        .fillna(self.global_mean)
                        .astype(float)
                    )

        return X_encoded
