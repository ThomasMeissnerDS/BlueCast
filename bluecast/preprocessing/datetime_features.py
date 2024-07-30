"""
Module for extracting date parts from datetime columns.

Cyclic transformations are not implemented as they aren't a good match for tree models.
"""

import logging
from typing import List, Optional, Union

import pandas as pd


def date_converter(
    df: pd.DataFrame,
    date_columns: Optional[List[Union[str, int, float]]],
    date_parts: List[str],
) -> pd.DataFrame:
    """
    Takes in a df and loops through datetime columns to and extracts the date parts month, day, dayofweek
    and hour and adds them as additional columns.
    :param date_columns: List of datetime columns.
    :param df: Dataframe to be processed.
    :param date_parts: List of date parts to be extracted.
    :return: Returns modified df.
    """
    logging.info("Start date column conversion.")
    if not date_columns:
        return df

    if not date_parts:
        date_parts = ["year", "_week_of_year", "month", "day", "dayofweek", "hour"]

    for c in date_columns:
        if "year" in date_parts:
            df[str(c) + "_year"] = df[c].dt.year.astype(int)
        if "month" in date_parts:
            df[str(c) + "_month"] = df[c].dt.month.astype(int)
        if "week_of_year" in date_parts:
            df[str(c) + "_week_of_year"] = df[c].dt.isocalendar().week.astype(int)
        if "day" in date_parts:
            df[str(c) + "_day"] = df[c].dt.day.astype(int)
        if "dayofweek" in date_parts:
            df[str(c) + "_dayofweek"] = df[c].dt.dayofweek.astype(int)
        if "hour" in date_parts:
            df[str(c) + "_hour"] = df[c].dt.hour.astype(int)
        df = df.drop(c, axis=1)
    return df
