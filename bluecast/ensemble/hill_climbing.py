"""Hill climbing ensemble: greedy forward selection with weighted blending."""

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def _default_classification_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Default metric for classification hill climbing (ROC AUC)."""
    return roc_auc_score(y_true, y_pred)


def _default_regression_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Default metric for regression hill climbing (negative RMSE, higher is better)."""
    return -np.sqrt(np.mean((y_true - y_pred) ** 2))


def _convert_to_ranks(predictions: np.ndarray) -> np.ndarray:
    """Convert predictions to rank percentiles in [0, 1]."""
    n = len(predictions)
    if n <= 1:
        return predictions.astype(np.float64)
    ranks = rankdata(predictions, method="ordinal").astype(np.float64)
    return (ranks - 1) / (n - 1)


class HillClimbingEnsemble:
    """Greedy forward selection to find optimal model combinations and weights.

    Starting from the best individual model, iteratively adds models that improve
    the ensemble score. Supports rank-based or probability-based blending and
    optional negative weights for diverse model combinations.

    :param weight_min: Minimum weight for candidate models.
    :param weight_max: Maximum weight for candidate models.
    :param weight_step: Step size for weight search.
    :param tolerance: Minimum improvement required to add a model.
    :param blending_method: 'rank' normalizes model outputs via rank transform,
        'probability' uses raw predictions.
    :param eval_metric: Callable(y_true, y_pred) -> float, higher is better.
    """

    def __init__(
        self,
        weight_min: float = -0.3,
        weight_max: float = 0.5,
        weight_step: float = 0.01,
        tolerance: float = 1e-7,
        blending_method: str = "rank",
        eval_metric: Optional[Callable] = None,
        is_classification: bool = True,
    ):
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.weight_step = weight_step
        self.tolerance = tolerance
        self.blending_method = blending_method
        self.eval_metric = eval_metric or _default_classification_metric
        self.is_classification = is_classification

        self.selected_indices: List[int] = []
        self.weights_map: Dict[int, float] = {}
        self.history: List[Dict] = []
        self.is_fitted: bool = False

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
        """Run greedy hill climbing to select models and weights.

        :param oof_predictions: List of arrays, each of shape (n_samples,) with OOF
            predictions from one model.
        :param y_true: Array of shape (n_samples,) with true targets.
        :param model_names: Optional list of model names for logging.
        """
        n_models = len(oof_predictions)
        if model_names is None:
            model_names = [f"model_{i}" for i in range(n_models)]

        preds = self._prepare_predictions(oof_predictions)
        y = y_true.astype(np.float64)

        individual_scores = [self.eval_metric(y, p) for p in preds]
        start_idx = int(np.argmax(individual_scores))

        ensemble = preds[start_idx].copy()
        ensemble_score = individual_scores[start_idx]

        self.selected_indices = [start_idx]
        remaining = [i for i in range(n_models) if i != start_idx]
        self.weights_map = {start_idx: 1.0}

        self.history = [
            {
                "iteration": 0,
                "model": model_names[start_idx],
                "model_idx": start_idx,
                "weight": 1.0,
                "score": ensemble_score,
            }
        ]

        logging.info(
            f"HC Start: {model_names[start_idx]} | Score: {ensemble_score:.6f}"
        )

        weight_candidates = np.arange(
            self.weight_min, self.weight_max + 1e-12, self.weight_step
        )

        iteration = 0
        while remaining:
            iteration += 1
            best_improvement = 0.0
            best_info: Optional[Tuple[int, float, str]] = None
            best_combined: Optional[np.ndarray] = None
            best_score: float = ensemble_score
            best_new_weights: Dict[int, float] = dict(self.weights_map)

            for idx in remaining:
                candidate = preds[idx]

                local_best_imp = -np.inf
                local_best_w = 0.0
                local_best_score = 0.0

                for w in weight_candidates:
                    if self.is_classification:
                        combined = np.clip(ensemble * (1.0 - w) + candidate * w, 0.0, 1.0)
                    else:
                        combined = ensemble * (1.0 - w) + candidate * w
                    score = self.eval_metric(y, combined)
                    improvement = score - ensemble_score

                    if improvement > local_best_imp:
                        local_best_imp = improvement
                        local_best_w = float(w)
                        local_best_score = score

                if local_best_imp > best_improvement:
                    best_improvement = local_best_imp
                    best_info = (idx, local_best_w, model_names[idx])
                    best_score = local_best_score
                    if self.is_classification:
                        best_combined = np.clip(
                            ensemble * (1.0 - local_best_w) + candidate * local_best_w,
                            0.0,
                            1.0,
                        )
                    else:
                        best_combined = ensemble * (1.0 - local_best_w) + candidate * local_best_w
                    new_w = {
                        k: v * (1.0 - local_best_w) for k, v in self.weights_map.items()
                    }
                    new_w[idx] = local_best_w
                    best_new_weights = new_w

            if best_improvement < self.tolerance or best_info is None:
                break

            add_idx, w, add_name = best_info
            ensemble = best_combined
            ensemble_score = best_score
            self.weights_map = best_new_weights
            self.selected_indices.append(add_idx)
            remaining.remove(add_idx)

            self.history.append(
                {
                    "iteration": iteration,
                    "model": add_name,
                    "model_idx": add_idx,
                    "weight": w,
                    "score": ensemble_score,
                }
            )

            logging.info(
                f"HC [{iteration:2d}] {add_name:<50s} w={w:+.4f} | "
                f"Score: {ensemble_score:.6f} (+{best_improvement:.6f})"
            )

        self.is_fitted = True
        logging.info(
            f"Hill Climbing complete: {len(self.selected_indices)} models selected."
        )

    def predict(self, predictions: List[np.ndarray]) -> np.ndarray:
        """Blend predictions using fitted weights.

        :param predictions: List of arrays, each of shape (n_samples,) with predictions.
        :returns: Blended predictions array of shape (n_samples,).
        """
        if not self.is_fitted:
            raise RuntimeError("HillClimbingEnsemble has not been fitted yet.")

        preds = self._prepare_predictions(predictions)

        ensemble = np.zeros_like(preds[0], dtype=np.float64)
        for idx, weight in self.weights_map.items():
            ensemble += weight * preds[idx]

        if self.is_classification:
            return np.clip(ensemble, 0.0, 1.0)
        return ensemble

    def get_selected_model_info(self) -> List[Dict]:
        """Return information about selected models and their weights."""
        return [
            {"model_idx": idx, "weight": self.weights_map[idx]}
            for idx in self.selected_indices
        ]
