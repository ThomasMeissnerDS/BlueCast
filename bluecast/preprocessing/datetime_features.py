"""
Module for extracting date parts from datetime columns.
"""

import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd


def date_converter(
    df: pd.DataFrame,
    date_columns: Optional[List[Union[str, int, float]]],
    date_parts: Optional[List[str]],
) -> pd.DataFrame:
    """
    Takes in a df and loops through datetime columns to extract date parts and adds them as additional columns.
    Additionally, creates cyclic features for time components if there is more than one unique value for the time unit.
    :param date_columns: List of datetime columns.
    :param df: Dataframe to be processed.
    :param date_parts: List of date parts to be extracted.
    :return: Returns modified df.
    """
    logging.info("Start date column conversion.")
    if not date_columns:
        return df

    if not date_parts:
        date_parts = ["year", "week_of_year", "month", "day", "dayofweek", "hour"]

    date_part_periods = {
        "month": 12,
        "week_of_year": 52,
        "day": 31,
        "dayofweek": 7,
        "hour": 24,
    }

    for c in date_columns:
        # create dummy variable such that the next part does not fail
        date_part_values = pd.Series([0] * len(df))
        for date_part in date_parts:
            if date_part == "year" and df[c].dt.year.astype(float).nunique() > 1:
                df[str(c) + "_year"] = df[c].dt.year.astype(float)
            elif date_part == "month" and df[c].dt.month.astype(float).nunique() > 1:
                date_part_values = df[c].dt.month.astype(float)
                df[str(c) + "_month"] = date_part_values
            elif (
                date_part == "week_of_year"
                and df[c].dt.isocalendar().week.astype(float).nunique() > 1
            ):
                date_part_values = df[c].dt.isocalendar().week.astype(float)
                df[str(c) + "_week_of_year"] = date_part_values
            elif date_part == "day" and df[c].dt.day.astype(float).nunique() > 1:
                date_part_values = df[c].dt.day.astype(float)
                df[str(c) + "_day"] = date_part_values
            elif (
                date_part == "dayofweek"
                and df[c].dt.dayofweek.astype(float).nunique() > 1
            ):
                date_part_values = df[c].dt.dayofweek.astype(float)
                df[str(c) + "_dayofweek"] = date_part_values
            elif date_part == "hour" and df[c].dt.hour.astype(float).nunique() > 1:
                date_part_values = df[c].dt.hour.astype(float)
                df[str(c) + "_hour"] = date_part_values
            else:
                pass

            # For date parts with a defined period, create cyclic features if there is more than one unique value
            if date_part in date_part_periods and date_part_values.nunique() > 1:
                period = date_part_periods[date_part]
                df[str(c) + "_" + date_part + "_sin"] = np.sin(
                    2 * np.pi * date_part_values / period
                )
                df[str(c) + "_" + date_part + "_cos"] = np.cos(
                    2 * np.pi * date_part_values / period
                )
        # Drop the original date column
        df = df.drop(c, axis=1)
    return df
