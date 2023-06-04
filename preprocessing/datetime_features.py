import pandas as pd
from typing import List, Union


def date_converter(df: pd.DataFrame, date_columns: List[Union[str, int, float]],
                   date_parts: List[str] = None) -> pd.DataFrame:
    """
    Takes in a df and loops through datetime columns to and extracts the date parts month, day, dayofweek
    and hour and adds them as additional columns.
    :param date_columns: List of datetime columns.
    :param df: Dataframe to be processed.
    :param date_parts: List of date parts to be extracted.
    :return: Returns modified df.
    """
    if not date_parts:
        date_parts = ["month", "day", "dayofweek", "hour"]

    for c in date_columns:
        if "month" in date_parts:
            df[c + "_month"] = df[c].dt.month
        if "day" in date_parts:
            df[c + "_day"] = df[c].dt.day
        if "dayofweek" in date_parts:
            df[c + "_dayofweek"] = df[c].dt.dayofweek
        if "hour" in date_parts:
            df[c + "_hour"] = df[c].dt.hour
        df = df.drop(c, axis=1)
    return df
