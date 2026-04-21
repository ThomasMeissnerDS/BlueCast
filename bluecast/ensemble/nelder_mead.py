"""Nelder-Mead optimization ensemble: optimize blending weights."""

import logging
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def _default_classification_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Default metric for classification Nelder-Mead (ROC AUC)."""
    return roc_auc_score(y_true, y_pred)


def _default_regression_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Default metric for regression Nelder-Mead (negative RMSE, higher is better)."""
    return -np.sqrt(np.mean((y_true - y_pred) ** 2))


def _mae_regression_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE metric for regression Nelder-Mead (negative MAE, higher is better)."""
    return -np.mean(np.abs(y_true - y_pred))


def _convert_to_ranks(predictions: np.ndarray) -> np.ndarray:
    """Convert predictions to rank percentiles in [0, 1]."""
    n = len(predictions)
    if n <= 1:
        return predictions.astype(np.float64)
    ranks = rankdata(predictions, method="ordinal").astype(np.float64)
    return (ranks - 1) / (n - 1)


class NelderMeadEnsemble:
    """Optimization-based blending using Nelder-Mead.
    
    Finds optimal weights for the given models via scipy.optimize.minimize.

    :param blending_method: 'rank' normalizes model outputs via rank transform, 
        'probability' uses raw predictions.
    :param eval_metric: Callable(y_true, y_pred) -> float, higher is better.
    """

    def __init__(
        self,
        blending_method: str = "probability",
        eval_metric: Optional[Callable] = None,
        is_classification: bool = True,
    ):
        self.blending_method = blending_method
        self.is_classification = is_classification

        if eval_metric:
            self.eval_metric = eval_metric
        elif is_classification:
            self.eval_metric = _default_classification_metric
        else:
            self.eval_metric = _default_regression_metric

        self.weights: np.ndarray = np.array([])
        self.is_fitted: bool = False
        self._y_min: Optional[float] = None
        self._y_max: Optional[float] = None

    def _prepare_predictions(self, preds_list: List[np.ndarray]) -> List[np.ndarray]:
        """Optionally rank-transform predictions."""
        if self.blending_method == "rank":
            return [_convert_to_ranks(p) for p in preds_list]
        return [p.astype(np.float64) for p in preds_list]

    def fit(
        self,
        oof_predictions: List[np.ndarray],
        y_true: np.ndarray,
        model_names: Optional[List[str]] = None,
    ) -> None:
        """Run Nelder-Mead optimization to select weights.

        :param oof_predictions: List of arrays, each of shape (n_samples,) with OOF predictions.
        :param y_true: Array of shape (n_samples,) with true targets.
        :param model_names: Optional list of model names for logging.
        """
        n_models = len(oof_predictions)
        preds = self._prepare_predictions(oof_predictions)
        y = y_true.astype(np.float64)
        preds_matrix = np.column_stack(preds)

        # Store target range for prediction clipping (regression only)
        if not self.is_classification:
            self._y_min = float(np.min(y))
            self._y_max = float(np.max(y))

        def objective(weights):
            # Normalize weights to sum to 1.0 INSIDE the objective.
            # This ensures the optimizer explores the actual prediction landscape
            # that will be used during inference (where weights are also normalized).
            w_sum = np.sum(weights)
            if abs(w_sum) < 1e-10:
                return 1e9  # Degenerate case: all weights near zero
            w_norm = weights / w_sum
            
            # Penalty for any individual weight being extreme
            if np.any(w_norm < -1.0) or np.any(w_norm > 2.0):
                return 1e9
                
            blended = np.dot(preds_matrix, w_norm)
            if self.is_classification:
                blended = np.clip(blended, 0.0, 1.0)
            score = self.eval_metric(y, blended)
            
            # Light L2 regularization toward uniform weights
            l2_reg = 1e-4 * np.sum((w_norm - 1.0/n_models)**2)
            
            return -score + l2_reg  # Minimize negative score

        initial_weights = np.ones(n_models) / n_models
        
        res = minimize(
            objective, 
            initial_weights, 
            method='Nelder-Mead', 
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        raw_weights = res.x
        
        # Normalize weights to sum to 1.0 (matches what objective() used)
        weight_sum = np.sum(raw_weights)
        if abs(weight_sum) > 1e-10:
            self.weights = raw_weights / weight_sum
        else:
            self.weights = np.ones(n_models) / n_models
            logging.warning("Nelder-Mead: Degenerate weights, falling back to uniform")
        
        self.is_fitted = True
        
        # Compute final score with normalized weights
        blended_final = np.dot(preds_matrix, self.weights)
        final_score = self.eval_metric(y, blended_final)
        logging.info(f"Nelder-Mead optimization complete. Final Score: {-final_score:.6f}")
        logging.info(f"Nelder-Mead optimal weights: {self.weights} (sum={np.sum(self.weights):.4f})")

    def predict(self, predictions: List[np.ndarray]) -> np.ndarray:
        """Blend predictions using fitted weights.

        :param predictions: List of arrays, each of shape (n_samples,) with predictions.
        :returns: Blended predictions array of shape (n_samples,).
        """
        if not self.is_fitted:
            raise RuntimeError("NelderMeadEnsemble has not been fitted yet.")

        preds = self._prepare_predictions(predictions)
        preds_matrix = np.column_stack(preds)
        blended = np.dot(preds_matrix, self.weights)

        if self.is_classification:
            return np.clip(blended, 0.0, 1.0)
        
        # Safety clip for regression: training range + 50% margin
        # Wide enough to allow legitimate extrapolation, tight enough 
        # to catch catastrophic errors
        if self._y_min is not None and self._y_max is not None:
            y_range = self._y_max - self._y_min
            margin = 0.5 * y_range
            blended = np.clip(blended, self._y_min - margin, self._y_max + margin)
        
        return blended
