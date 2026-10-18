"""
AutoFE Module for BlueCast.

This module implements a highly performant feature generator inspired by the OpenFE "Expand-and-Reduce" paradigm.
It creates new interactions (arithmetic, GroupByThenMean) and rigorously prunes them using a fast XGBoost heuristic
(training a shallow tree against the base margin's pseudo-residuals).
"""

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder


class BlueCastFeatureGenerator:
    """
    BlueCastFeatureGenerator implements an Expand-and-Reduce auto feature engineering pipeline.

    - Expand: Generates new features using basic arithmetic interactions and group-by-then-mean logic.
    - Reduce (Pruning): Evaluates generated features individually against the pseudo-residuals of a base model
      and prunes features with an XGBoost gain below a given threshold.

    :param autofe_pruning_threshold: The minimum gain required for a generated feature to be retained.
    :param cat_columns: List of categorical columns to be used for GroupBy operations.
    :param class_problem: The type of classification problem ("binary" or "multiclass").
    :param random_state: Global random state for reproducibility.
    """

    def __init__(
        self,
        autofe_pruning_threshold: float = 0.01,
        cat_columns: Optional[List[Union[str, float, int]]] = None,
        class_problem: str = "binary",
        random_state: int = 42,
        target_column: Optional[str] = None,
    ):
        self.autofe_pruning_threshold = autofe_pruning_threshold
        self.cat_columns = cat_columns if cat_columns is not None else []
        self.class_problem = class_problem
        self.random_state = random_state
        self.target_column = target_column
        self.retained_features: List[str] = []
        self._fitted: bool = False
        self._group_by_means: dict = {}

    def _expand_space(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Generate new interactive features.
        """
        generated_cols = []
        new_features = {}

        # Exclude the target column from feature generation to prevent
        # target leakage and inference-time schema mismatches
        exclude_cols = set(self.cat_columns)
        if self.target_column:
            exclude_cols.add(self.target_column)
        num_cols = [c for c in df.columns if c not in exclude_cols]

        # Arithmetic interactions
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                col1, col2 = num_cols[i], num_cols[j]
                
                # Addition
                new_col_add = f"{col1}_add_{col2}"
                new_features[new_col_add] = (df[col1] + df[col2]).astype(np.float32)
                generated_cols.append(new_col_add)
                
                # Subtraction
                new_col_sub = f"{col1}_sub_{col2}"
                new_features[new_col_sub] = (df[col1] - df[col2]).astype(np.float32)
                generated_cols.append(new_col_sub)
                
                # Multiplication
                new_col_mul = f"{col1}_mul_{col2}"
                new_features[new_col_mul] = (df[col1] * df[col2]).astype(np.float32)
                generated_cols.append(new_col_mul)
                
                # Division (with protection against division by zero)
                new_col_div = f"{col1}_div_{col2}"
                new_features[new_col_div] = (df[col1] / (df[col2].replace(0, 1e-6))).astype(np.float32)
                generated_cols.append(new_col_div)
                
        # GroupByThenMean interactions
        if not self._fitted:
            self._group_by_means = {}

        for cat_col in self.cat_columns:
            for num_col in num_cols:
                new_col_grp = f"{num_col}_groupby_mean_{cat_col}"
                
                if not self._fitted:
                    means = df.groupby(cat_col)[num_col].mean().astype(np.float32).to_dict()
                    self._group_by_means[new_col_grp] = means
                
                grp_feature = df[cat_col].map(self._group_by_means[new_col_grp]).astype(np.float32)
                
                # Fill NaNs from unseen categories in test set with global mean
                if grp_feature.isna().any():
                    grp_feature = grp_feature.fillna(df[num_col].mean())
                    
                new_features[new_col_grp] = grp_feature
                generated_cols.append(new_col_grp)

        if new_features:
            df_new = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
        else:
            df_new = df.copy()

        return df_new, generated_cols

    def _feature_boost_pruning(
        self, df: pd.DataFrame, y: pd.Series, generated_cols: List[str]
    ) -> List[str]:
        """
        Prune features by training a shallow XGBoost tree for each generated feature
        against the residuals of a base model trained on original features.
        """
        logging.info("Starting AutoFE Feature Pruning...")
        
        # 1. Train base model on original features to get base margin
        original_cols = [c for c in df.columns if c not in generated_cols and c != self.target_column]
        X_base = df[original_cols].copy()

        # Handle categoricals for base model
        for col in self.cat_columns:
            if col in X_base.columns:
                X_base[col] = X_base[col].astype("category")

        d_base = xgb.DMatrix(X_base, label=y, enable_categorical=True)

        params_base = {
            "objective": "multi:softprob" if self.class_problem == "multiclass" else "binary:logistic",
            "eval_metric": "mlogloss" if self.class_problem == "multiclass" else "logloss",
            "max_depth": 6,
            "random_state": self.random_state,
        }
        if self.class_problem == "multiclass":
            params_base["num_class"] = y.nunique()

        logging.info("Training AutoFE base model...")
        base_model = xgb.train(params_base, d_base, num_boost_round=10)
        
        # Get base margin
        base_margin = base_model.predict(d_base, output_margin=True)
        
        if self.class_problem == "multiclass" and base_margin.ndim > 1:
            # Flatten or reshape if multiclass, or set as list
            # DMatrix handles 2D base_margin correctly in newer XGBoost versions
            pass

        retained = []
        
        # 2. Train a weak learner per feature against base margin
        params_prune = {
            "objective": "multi:softprob" if self.class_problem == "multiclass" else "binary:logistic",
            "max_depth": 2,
            "tree_method": "hist", 
            "random_state": self.random_state,
        }
        if self.class_problem == "multiclass":
            params_prune["num_class"] = y.nunique()

        logging.info(f"Evaluating {len(generated_cols)} generated features...")
        for col in generated_cols:
            d_feat = xgb.DMatrix(df[[col]], label=y)
            d_feat.set_base_margin(base_margin)
            
            try:
                feat_model = xgb.train(params_prune, d_feat, num_boost_round=1)
                scores = feat_model.get_score(importance_type="gain")
                gain = scores.get(col, 0.0)
                
                if gain >= self.autofe_pruning_threshold:
                    retained.append(col)
            except Exception as e:
                # If feature is constant or causes error, we discard it
                continue
                
        logging.info(f"AutoFE retained {len(retained)} out of {len(generated_cols)} features.")
        return retained

    def fit_transform(self, df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Coordinate the expand and reduce phases during training.
        """
        logging.info("Starting AutoFE expansion...")
        df_expanded, generated_cols = self._expand_space(df)
        
        self.retained_features = self._feature_boost_pruning(df_expanded, y, generated_cols)
        self._fitted = True
        
        drop_cols = [c for c in generated_cols if c not in self.retained_features]
        return df_expanded.drop(columns=drop_cols)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply retained features to new data.
        """
        if not self._fitted:
            raise RuntimeError("BlueCastFeatureGenerator is not fitted yet.")
            
        df_expanded, generated_cols = self._expand_space(df)
        drop_cols = [c for c in generated_cols if c not in self.retained_features]
        return df_expanded.drop(columns=drop_cols)
