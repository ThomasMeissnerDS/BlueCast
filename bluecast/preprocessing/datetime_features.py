"""
Module for extracting date parts from datetime columns.
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


class DatePartExtractor:
    """
    Class for extracting date parts from datetime columns and creating cyclic features.
    """

    def __init__(
        self,
        date_columns: Optional[List[Union[str, int, float]]] = None,
        date_parts: Optional[List[str]] = None,
    ):
        """
        Initializes the DatePartExtractor.
        :param date_columns: List of datetime columns.
        :param date_parts: List of date parts to be extracted.
        """
        self.date_columns = date_columns
        if date_parts is None:
            self.date_parts = [
                "year",
                "dayofyear",
                "week_of_year",
                "month",
                "day",
                "dayofweek",
                "hour",
            ]
        else:
            self.date_parts = date_parts
        self.date_part_periods = {
            "year": 1,  # Year cyclicity is not meaningful, but kept for consistency
            "dayofyear": 365,
            "month": 12,
            "week_of_year": 52,
            "day": 31,
            "dayofweek": 7,
            "hour": 24,
        }
        self.included_date_parts: Dict[Union[str, int, float], List[str]] = {}
        self.included_cyclic_features: Dict[Union[str, int, float], List[str]] = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fits to the data and transforms it, extracting date parts and creating cyclic features.
        :param df: Dataframe to be processed.
        :return: Returns modified dataframe.
        """
        logging.info("Start date column conversion.")
        if not self.date_columns:
            return df

        df = df.copy()  # To avoid modifying the original dataframe

        for c in self.date_columns:
            if c not in df.columns:
                logging.warning(f"Date column {c} not found in dataframe.")
                continue
            self.included_date_parts[c] = []
            self.included_cyclic_features[c] = []
            for date_part in self.date_parts:
                date_part_values = None
                if date_part == "year":
                    date_part_values = df[c].dt.year.astype(float)
                elif date_part == "dayofyear":
                    date_part_values = df[c].dt.dayofyear.astype(float)
                elif date_part == "month":
                    date_part_values = df[c].dt.month.astype(float)
                elif date_part == "week_of_year":
                    date_part_values = df[c].dt.isocalendar().week.astype(float)
                elif date_part == "day":
                    date_part_values = df[c].dt.day.astype(float)
                elif date_part == "dayofweek":
                    date_part_values = df[c].dt.dayofweek.astype(float)
                elif date_part == "hour":
                    date_part_values = df[c].dt.hour.astype(float)
                else:
                    logging.warning(f"Unknown date part '{date_part}' specified.")
                    continue  # Skip unknown date_parts

                if date_part_values.nunique() > 1:
                    df[f"{c}_{date_part}"] = date_part_values
                    self.included_date_parts[c].append(date_part)

                    if (
                        date_part in self.date_part_periods
                        and date_part_values.nunique() > 1
                    ):
                        period = self.date_part_periods[date_part]
                        df[f"{c}_{date_part}_sin"] = np.sin(
                            2 * np.pi * date_part_values / period
                        )
                        df[f"{c}_{date_part}_cos"] = np.cos(
                            2 * np.pi * date_part_values / period
                        )
                        self.included_cyclic_features[c].append(date_part)
            # Drop the original date column
            df = df.drop(c, axis=1)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the data using the same transformations as fitted during fit_transform.
        Always drops the original date columns to maintain consistency.
        :param df: Dataframe to be transformed.
        :return: Returns modified dataframe.
        """
        logging.info("Start date column conversion (transform).")
        if not self.date_columns:
            return df

        df = df.copy()  # To avoid modifying the original dataframe

        for c in self.date_columns:
            if c not in df.columns:
                logging.warning(f"Date column {c} not found in dataframe.")
                # Even if the column is missing, attempt to drop it to maintain consistency
                continue

            if c not in self.included_date_parts:
                logging.warning(
                    f"Date column {c} was not processed during fit_transform."
                )
                # Drop the date column regardless of whether it was processed
                df = df.drop(c, axis=1)
                continue

            for date_part in self.included_date_parts[c]:
                date_part_values = None
                if date_part == "year":
                    date_part_values = df[c].dt.year.astype(float)
                elif date_part == "dayofyear":
                    date_part_values = df[c].dt.dayofyear.astype(float)
                elif date_part == "month":
                    date_part_values = df[c].dt.month.astype(float)
                elif date_part == "week_of_year":
                    date_part_values = df[c].dt.isocalendar().week.astype(float)
                elif date_part == "day":
                    date_part_values = df[c].dt.day.astype(float)
                elif date_part == "dayofweek":
                    date_part_values = df[c].dt.dayofweek.astype(float)
                elif date_part == "hour":
                    date_part_values = df[c].dt.hour.astype(float)
                else:
                    logging.warning(f"Unknown date part '{date_part}' specified.")
                    continue  # Skip unknown date_parts

                df[f"{c}_{date_part}"] = date_part_values

                if date_part in self.included_cyclic_features.get(c, []):
                    period = self.date_part_periods[date_part]
                    df[f"{c}_{date_part}_sin"] = np.sin(
                        2 * np.pi * date_part_values / period
                    )
                    df[f"{c}_{date_part}_cos"] = np.cos(
                        2 * np.pi * date_part_values / period
                    )
            # Drop the original date column
            df = df.drop(c, axis=1)
        return df
