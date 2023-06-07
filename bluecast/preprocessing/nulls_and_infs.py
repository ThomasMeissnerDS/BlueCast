from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd

from bluecast.preprocessing.general_utils import logger


def fill_infinite_values(df: pd.DataFrame, fill_with: Union[int, float] = 0):
    """Replace infinite values with NaN or given value."""
    logger(f"{datetime.utcnow()}: Start filling infinite values.")
    df = df.replace([np.inf, -np.inf], fill_with)
    return df


def fill_nulls(df: pd.DataFrame, fill_with: Union[int, float] = 0):
    """Replace null values with NaN or given value."""
    logger(f"{datetime.utcnow()}: Start filling infinite nulls.")
    df = df.fillna(fill_with)
    return df
