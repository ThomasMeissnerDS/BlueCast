import logging
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from bluecast.conformal_prediction.effectiveness_nonconformity_measures import (
    convert_expected_effectiveness_nonconformity_input_types,
)


def prediction_set_coverage(
    y_true: Union[np.ndarray, pd.Series],
    prediction_sets: Union[pd.Series, pd.DataFrame],
) -> float:
    """
    Calculate the percentage of prediction sets that include the true label.

    :param y_true: Ground truth labels.
    :param prediction_sets: Predicted probabilities of shape (n_samples, 1) where each row is a set of classes.
    """
    y_hat = convert_expected_effectiveness_nonconformity_input_types(prediction_sets)
    set_with_corr_label = [str(label) in str(ps) for label, ps in zip(y_true, y_hat)]
    return np.mean(np.asarray(set_with_corr_label))


def prediction_interval_coverage(
    y_true: Union[np.ndarray, pd.Series],
    prediction_intervals: pd.DataFrame,
    alphas: List[float],
) -> Dict[float, float]:
    """
    Calculate the percentage of prediction intervals that cover the true value.

    :param y_true: Ground truth labels.
    :param prediction_intervals: DataFrame with predicted bands according to provided confidence levels. This must
        contain columns of format f"{alpha}_low" and f"{1-alpha}_high" for each format.
    :param alphas: List of alphas indicating which confidence levels to check
    """
    if isinstance(y_true, pd.Series):
        y_true = y_true.values

    coverages = {}
    for alpha in alphas:
        coverages[alpha] = np.mean(
            np.where(
                (
                    (prediction_intervals[f"{alpha}_low"] <= y_true)
                    & (prediction_intervals[f"{1 - alpha}_high"] >= y_true)
                ),
                1,
                0,
            )
        )

    return coverages


def prediction_interval_coverage_by_group(
    y_true: Union[np.ndarray, pd.Series],
    prediction_intervals: pd.DataFrame,
    alphas: List[float],
    groups: Union[np.ndarray, pd.Series],
) -> Dict[Any, Dict[float, float]]:
    """Calculate per-group coverage of prediction intervals.

    :param y_true: Ground truth values.
    :param prediction_intervals: DataFrame with predicted bands.
    :param alphas: List of significance levels.
    :param groups: Group labels for each sample.
    :returns: Nested dict mapping group -> alpha -> coverage.
    """
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(groups, pd.Series):
        groups = groups.values

    unique_groups = np.unique(groups)
    group_coverages = {}

    for group in unique_groups:
        mask = groups == group
        group_y = y_true[mask]
        group_intervals = prediction_intervals.loc[mask]

        coverages = {}
        for alpha in alphas:
            coverages[alpha] = np.mean(
                np.where(
                    (
                        (group_intervals[f"{alpha}_low"].values <= group_y)
                        & (group_intervals[f"{1 - alpha}_high"].values >= group_y)
                    ),
                    1,
                    0,
                )
            )
        group_coverages[group] = coverages

    return group_coverages


def prediction_interval_spans_by_group(
    prediction_intervals: pd.DataFrame,
    alphas: List[float],
    groups: Union[np.ndarray, pd.Series],
) -> Dict[Any, Dict[float, float]]:
    """Calculate per-group mean interval width.

    :param prediction_intervals: DataFrame with predicted bands.
    :param alphas: List of significance levels.
    :param groups: Group labels for each sample.
    :returns: Nested dict mapping group -> alpha -> mean_span.
    """
    if isinstance(groups, pd.Series):
        groups = groups.values

    unique_groups = np.unique(groups)
    group_spans = {}

    for group in unique_groups:
        mask = groups == group
        group_intervals = prediction_intervals.loc[mask]

        spans = {}
        for alpha in alphas:
            spans[alpha] = np.mean(
                group_intervals[f"{1 - alpha}_high"].values
                - group_intervals[f"{alpha}_low"].values
            )
        group_spans[group] = spans

    return group_spans


def conformal_fairness_check(
    coverages_by_group: Dict[Any, Dict[float, float]],
    target_coverage: float,
    tolerance: float = 0.05,
) -> Dict[str, Any]:
    """Check whether conformal prediction coverage is fair across groups.

    A model's uncertainty quantification is considered fair if each group's
    empirical coverage is within `tolerance` of the `target_coverage`.
    Groups with coverage significantly below target are under-covered
    (the model is overconfident for them); groups above are over-covered
    (the model is conservative for them).

    :param coverages_by_group: Output from `prediction_interval_coverage_by_group`,
        mapping group -> alpha -> coverage.
    :param target_coverage: The expected coverage level (e.g. 0.9 for alpha=0.1).
    :param tolerance: Maximum allowed deviation from target_coverage. Default 0.05.
    :returns: Dict with 'is_fair' (bool), 'group_results' (per-group pass/fail),
        and 'coverage_range' (min/max across groups).

    Usage::

        coverages = prediction_interval_coverage_by_group(
            y_true, intervals, [0.1], groups
        )
        result = conformal_fairness_check(coverages, target_coverage=0.9)
        if not result['is_fair']:
            print("Coverage is uneven across groups!")
    """
    logger = logging.getLogger(__name__)

    group_results: Dict[Any, Dict[str, Any]] = {}
    all_coverages: List[float] = []
    all_fair = True

    for group, alpha_coverages in coverages_by_group.items():
        group_info: Dict[str, Any] = {}
        for alpha, coverage in alpha_coverages.items():
            deviation = abs(coverage - target_coverage)
            is_within = deviation <= tolerance
            group_info[f"alpha_{alpha}_coverage"] = round(coverage, 4)
            group_info[f"alpha_{alpha}_deviation"] = round(deviation, 4)
            group_info[f"alpha_{alpha}_fair"] = is_within
            all_coverages.append(coverage)
            if not is_within:
                all_fair = False
                logger.warning(
                    f"Conformal fairness: group '{group}' coverage={coverage:.3f} "
                    f"deviates by {deviation:.3f} from target {target_coverage:.3f}"
                )
        group_results[group] = group_info

    coverage_range: Tuple[float, float] = (
        (round(min(all_coverages), 4), round(max(all_coverages), 4))
        if all_coverages
        else (0.0, 0.0)
    )

    return {
        "is_fair": all_fair,
        "target_coverage": target_coverage,
        "tolerance": tolerance,
        "coverage_range": coverage_range,
        "group_results": group_results,
    }
