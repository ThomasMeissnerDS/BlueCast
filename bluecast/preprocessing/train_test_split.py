"""
This module contains functions to split data into train and test sets.

The train-test split can be done in two ways:
    - Randomly
    - Based on a provided order (i.e. time)
"""

from datetime import datetime
from typing import Optional

import pandas as pd
from sklearn import model_selection

from bluecast.general_utils.general_utils import logger


def train_test_split_cross(
    df: pd.DataFrame,
    target_col: str,
    train_size=0.80,
    random_state: int = 100,
    stratify: bool = False,
):
    """Split data into train and test. Stratification is possible."""
    logger(
        f"{datetime.utcnow()}: Start executing train-test split with train size of {train_size}."
    )
    target = df[target_col].copy()
    df = df.drop(target_col, axis=1)

    if stratify:
        stratify_data = target
    else:
        stratify_data = None

    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        df,
        target,
        train_size=train_size,
        random_state=random_state,
        stratify=stratify_data,
    )
    if target_col in x_train.columns:
        x_train = x_train.drop(target_col, axis=1)
        x_test = x_test.drop(target_col, axis=1)
    return x_train, x_test, y_train, y_test


def train_test_split_time(
    df: pd.DataFrame, target_col: str, split_by_col: str, train_size: float = 0.80
):
    """Split data into train and test based on a provided order (i.e. time)."""
    logger(
        f"{datetime.utcnow()}: Start executing ordered train-test split with train size of {train_size}."
    )
    length = len(df.index)
    train_length = int(length * train_size)
    test_length = length - train_length
    if not split_by_col:
        df = df.sort_index()
    elif split_by_col:
        df = df.sort_values(by=[split_by_col])
    else:
        pass
    x_train = df.head(train_length)
    x_test = df.tail(test_length)
    y_train = x_train[target_col]
    y_test = x_test[target_col]
    # remove target column from x_train and x_test
    if target_col in x_train.columns:
        x_train = x_train.drop(target_col, axis=1)
        x_test = x_test.drop(target_col, axis=1)
    return x_train, x_test, y_train, y_test


def train_test_split(
    df: pd.DataFrame,
    target_col: str,
    split_by_col: Optional[str] = None,
    train_size: float = 0.80,
    random_state: int = 0,
    stratify: bool = False,
):
    if split_by_col is not None:
        x_train, x_test, y_train, y_test = train_test_split_time(
            df,
            target_col,
            split_by_col,
            train_size=train_size,
        )
    else:
        x_train, x_test, y_train, y_test = train_test_split_cross(
            df,
            target_col,
            train_size=train_size,
            random_state=random_state,
            stratify=stratify,
        )
    return x_train, x_test, y_train, y_test
