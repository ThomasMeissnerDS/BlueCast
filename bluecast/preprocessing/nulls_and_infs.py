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
    """Replace infinite values with the given value (default 0)."""
    logging.info("Start filling infinite values.")
    df = df.replace([np.inf, -np.inf], fill_with)
    return df


def fill_nulls(df: pd.DataFrame, fill_with: Union[int, float] = 0):
    """Replace null values with the given value (default 0)."""
    logging.info("Start filling null values.")
    df = df.fillna(fill_with)
    return df
