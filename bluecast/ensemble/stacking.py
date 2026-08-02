"""Stacking ensemble: train a meta-learner on out-of-fold predictions."""

import logging
from typing import Any, Optional

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import Ridge


class StackingEnsemble:
    """Train a meta-learner on rank-transformed OOF predictions.

    The stacking approach trains a second-stage model (default: Ridge regression)
    on the out-of-fold predictions from each base model. At prediction time, each
    base model's predictions are rank-transformed and fed through the meta-learner.

    :param meta_learner: A scikit-learn compatible estimator. If None, uses Ridge(alpha=10.0).
    :param use_ranks: Whether to rank-transform predictions before stacking.
    """

    def __init__(
        self,
        meta_learner: Optional[Any] = None,
        use_ranks: bool = True,
        clip_predictions: bool = True,
    ):
        if meta_learner is None:
            self.meta_learner = Ridge(alpha=10.0, fit_intercept=True, random_state=42)
        else:
            self.meta_learner = meta_learner
        self.use_ranks = use_ranks
        self.clip_predictions = clip_predictions
        self.is_fitted = False

    @staticmethod
    def _convert_to_ranks(predictions: np.ndarray) -> np.ndarray:
        """Convert predictions to rank percentiles in [0, 1]."""
        n = len(predictions)
        if n <= 1:
            return predictions
        ranks = rankdata(predictions, method="ordinal").astype(np.float64)
        return (ranks - 1) / (n - 1)

    def _rank_transform_matrix(self, X: np.ndarray) -> np.ndarray:
        """Apply rank transformation to each column of prediction matrix."""
        if not self.use_ranks:
            return X
        X_ranked = np.zeros_like(X, dtype=np.float64)
        for col in range(X.shape[1]):
            X_ranked[:, col] = self._convert_to_ranks(X[:, col])
        return X_ranked

    def fit(self, oof_predictions: np.ndarray, y_true: np.ndarray) -> None:
        """Fit the meta-learner on out-of-fold predictions.

        :param oof_predictions: Array of shape (n_samples, n_models) with OOF predictions.
        :param y_true: Array of shape (n_samples,) with true targets.
        """
        X = self._rank_transform_matrix(oof_predictions)
        self.meta_learner.fit(X, y_true)
        self.is_fitted = True
        logging.info(
            f"Stacking meta-learner fitted on {X.shape[1]} base models, "
            f"{X.shape[0]} samples."
        )

    def predict(self, predictions: np.ndarray) -> np.ndarray:
        """Generate stacked predictions.

        :param predictions: Array of shape (n_samples, n_models) with base model predictions.
        :returns: Array of shape (n_samples,) with stacked predictions.
        """
        if not self.is_fitted:
            raise RuntimeError("StackingEnsemble has not been fitted yet.")
        X = self._rank_transform_matrix(predictions)
        preds = self.meta_learner.predict(X)
        if self.clip_predictions:
            return np.clip(preds, 0, 1)
        return preds
