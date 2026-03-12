"""Configuration for ensemble strategies."""

from typing import Any, Callable, Literal, Optional


class EnsembleConfig:
    """Configuration for how multiple model predictions are combined.

    :param ensemble_strategy: Strategy for combining predictions. 'mean' uses simple averaging,
        'stacking' trains a meta-learner on out-of-fold predictions, 'hill_climbing' uses greedy
        forward selection to find optimal model weights.
    :param mean_type: Type of averaging when ensemble_strategy='mean'. Options: 'arithmetic',
        'median', 'geometric', 'harmonic'.
    :param stacking_meta_learner: Scikit-learn compatible estimator to use as meta-learner for
        stacking. Defaults to Ridge(alpha=10.0) if None.
    :param stacking_use_ranks: Whether to rank-transform predictions before stacking.
    :param hc_weight_min: Minimum weight for hill climbing. Negative values allow anti-correlated
        models.
    :param hc_weight_max: Maximum weight for hill climbing.
    :param hc_weight_step: Step size for weight search in hill climbing.
    :param hc_tolerance: Minimum improvement required to add a model in hill climbing.
    :param hc_blending_method: Whether to use rank-transformed or raw predictions in hill climbing.
    :param hc_allow_negative_weights: Whether negative weights are allowed in hill climbing.
    :param hc_eval_metric: Custom evaluation metric for hill climbing. Should accept (y_true, y_pred)
        and return a score where higher is better. If None, uses ROC AUC for classification and
        negative RMSE for regression.
    """

    def __init__(
        self,
        ensemble_strategy: Literal["mean", "stacking", "hill_climbing"] = "mean",
        mean_type: Literal[
            "arithmetic", "median", "geometric", "harmonic"
        ] = "arithmetic",
        stacking_meta_learner: Optional[Any] = None,
        stacking_use_ranks: bool = True,
        hc_weight_min: float = -0.3,
        hc_weight_max: float = 0.5,
        hc_weight_step: float = 0.01,
        hc_tolerance: float = 1e-7,
        hc_blending_method: Literal["rank", "probability"] = "rank",
        hc_allow_negative_weights: bool = True,
        hc_eval_metric: Optional[Callable] = None,
    ):
        self.ensemble_strategy = ensemble_strategy
        self.mean_type = mean_type
        self.stacking_meta_learner = stacking_meta_learner
        self.stacking_use_ranks = stacking_use_ranks
        self.hc_weight_min = hc_weight_min
        self.hc_weight_max = hc_weight_max
        self.hc_weight_step = hc_weight_step
        self.hc_tolerance = hc_tolerance
        self.hc_blending_method = hc_blending_method
        self.hc_allow_negative_weights = hc_allow_negative_weights
        self.hc_eval_metric = hc_eval_metric

        if not hc_allow_negative_weights:
            self.hc_weight_min = 0.0

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{self.__class__.__name__}({params})"
