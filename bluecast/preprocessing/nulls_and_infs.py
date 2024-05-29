"""
This module contains functions to handle nulls and infinite values.

Only the handling of infinite values is part of the preprocessing pipeline as Xgboost can handle missing values out of
the box.
"""

import logging
from typing import Union

import numpy as np
import pandas as pd


def fill_infinite_values(df: pd.DataFrame, fill_with: Union[int, float] = 0):
    """Replace infinite values with NaN or given value."""
    logging.info("Start filling infinite values.")
    df = df.replace([np.inf, -np.inf], fill_with)
    return df


def fill_nulls(df: pd.DataFrame, fill_with: Union[int, float] = 0):
    """Replace null values with NaN or given value."""
    logging.info("Start filling infinite nulls.")
    df = df.fillna(fill_with)
    return df
