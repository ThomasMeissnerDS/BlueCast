import pandas as pd
import numpy as np
from typing import Union


def fill_infinite_values(df: pd.DataFrame, fill_with: Union[int, float] = 0):
    """Replace infinite values with NaN or given value."""
    df = df.replace([np.inf, -np.inf], fill_with)
    return df


def fill_nulls(df: pd.DataFrame, fill_with: Union[int, float] = 0):
    """Replace null values with NaN or given value."""
    df = df.fillna(fill_with)
    return df


