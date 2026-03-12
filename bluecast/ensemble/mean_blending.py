"""Simple mean-based blending strategies."""

import logging
from typing import Literal

import numpy as np
import pandas as pd


def blend_predictions_mean(
    result_df: pd.DataFrame,
    pred_cols: list,
    mean_type: Literal["arithmetic", "median", "geometric", "harmonic"] = "arithmetic",
) -> pd.Series:
    """Blend predictions from multiple models using a mean-based strategy.

    :param result_df: DataFrame with predictions from each model in separate columns.
    :param pred_cols: List of column names containing predictions.
    :param mean_type: Type of averaging to use.
    :returns: Series with blended predictions.
    """
    df = result_df.loc[:, pred_cols]

    if mean_type == "arithmetic":
        return df.mean(axis=1)
    elif mean_type == "median":
        return df.median(axis=1)
    elif mean_type == "geometric":
        log_preds = np.log(df.clip(lower=1e-15))
        return np.exp(log_preds.mean(axis=1))
    elif mean_type == "harmonic":
        n = len(pred_cols)
        return n / np.sum(1.0 / df.clip(lower=1e-15), axis=1)
    else:
        logging.warning(
            f"Unknown mean_type '{mean_type}', falling back to arithmetic mean."
        )
        return df.mean(axis=1)
