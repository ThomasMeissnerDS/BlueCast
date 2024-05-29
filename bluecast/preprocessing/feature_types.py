"""
Feature type detection and casting.

This is a convenience class to detect and cast feature types in a DataFrame. It can be used to detect numerical,
categorical and datetime columns. It also casts columns to a specific type.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd


class FeatureTypeDetector:
    """Detect and cast feature types in DataFrame.

    Column names for individual feature types can be provided. Otherwise types will be inferred and casted accordingly.
    """

    def __init__(
        self,
        num_columns: Optional[List[Union[str, int, float]]] = None,
        cat_columns: Optional[List[Union[str, int, float]]] = None,
        date_columns: Optional[List[Union[str, int, float]]] = None,
        all_null_cols: Optional[List[Union[str, int, float]]] = None,
        zero_var_cols: Optional[List[Union[str, int, float]]] = None,
    ):
        if not num_columns:
            num_columns = []
        self.num_columns = num_columns

        if not cat_columns:
            cat_columns = []
        self.cat_columns = cat_columns

        if not date_columns:
            date_columns = []
        self.date_columns = date_columns

        if not all_null_cols:
            all_null_cols = []
        self.all_null_cols = all_null_cols

        if not zero_var_cols:
            zero_var_cols = []
        self.zero_var_cols = zero_var_cols

        self.detected_col_types: Dict[str, str] = {}
        self.num_dtypes = [
            "int8",
            "int16",
            "int32",
            "int64",
            "float16",
            "float32",
            "float64",
        ]

    def drop_all_null_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop all columns with only null values."""
        if not self.all_null_cols:
            self.all_null_cols = df.columns[df.isnull().all()].tolist()

        if self.all_null_cols:
            df = df.drop(self.all_null_cols, axis=1)

        logging.info(
            f"Dropped the following columns as being Nulls only: {self.all_null_cols}."
        )
        return df

    def drop_zero_variance_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop all columns with only one unique value."""
        if not self.zero_var_cols:
            for col in df.columns:
                if df[col].nunique() == 1:
                    self.zero_var_cols.append(col)

        if self.zero_var_cols:
            df = df.drop(self.zero_var_cols, axis=1)
        logging.info(
            f"Dropped the following columns as constants only: {self.zero_var_cols}."
        )
        return df

    def identify_num_columns(self, df: pd.DataFrame):
        """Identify numerical columns based on already existing data type."""
        # detect numeric columns by type
        if not self.num_columns:
            num_col_list = []
            for vartype in self.num_dtypes:
                num_cols = df.select_dtypes(include=[vartype]).columns
                for col in num_cols:
                    num_col_list.append(col)
            self.num_columns = num_col_list
        return self.num_columns

    def identify_bool_columns(
        self, df: pd.DataFrame
    ) -> Tuple[List[Union[str, float, int]], List[Union[str, float, int]]]:
        """Identify boolean columns based on data type"""
        bool_cols = list(df.select_dtypes(["bool"]))
        for col in bool_cols:
            # df[col] = df[col].astype(bool)
            self.detected_col_types[col] = "bool"

        try:
            no_bool_df = df.loc[:, ~df.columns.isin(bool_cols)]
            no_bool_cols = no_bool_df.columns.to_list()
        except Exception:
            no_bool_cols = df.columns.to_list()
        return bool_cols, no_bool_cols

    def identify_date_time_columns(
        self, df: pd.DataFrame, no_bool_cols: List[Union[str, float, int]]
    ):
        """Try casting to datetime. Expected is a datetime format of YYYY-MM-DD"""
        if self.date_columns and self.num_columns:
            date_columns = []
            # convert date columns from object to datetime type
            for col in self.date_columns:
                if col not in self.num_columns:
                    try:
                        df[col] = pd.to_datetime(df[col], yearfirst=True)
                        date_columns.append(col)
                        self.detected_col_types[str(col)] = "datetime[ns]"
                    except Exception:
                        pass
            self.date_columns = date_columns
        if not self.date_columns and self.num_columns:
            date_columns = []
            # convert date columns from object to datetime type
            for col in no_bool_cols:
                # check if column is not numerical and has a length of 10 (YYYY-MM-DD)
                if (
                    col not in self.num_columns
                    and df[col].astype(str).str.len().max() >= 10
                ):
                    try:
                        df[col] = pd.to_datetime(df[col], yearfirst=True)
                        date_columns.append(col)
                        self.detected_col_types[str(col)] = "datetime[ns]"
                    except Exception:
                        pass
            self.date_columns = date_columns

    def cast_rest_columns_to_object(
        self, df: pd.DataFrame, bool_cols: List[Union[str, float, int]]
    ) -> pd.DataFrame:
        """Treat remaining columns.

        Takes remaining columns and tries to cast them as numerical. If not successful, then columns are assumed to be
        categorical.
        """
        if bool_cols and self.date_columns:
            no_bool_dt_cols = bool_cols + self.date_columns
        elif bool_cols and not self.date_columns:
            no_bool_dt_cols = bool_cols
        elif not bool_cols and self.date_columns:
            no_bool_dt_cols = self.date_columns
        else:
            no_bool_dt_cols = []
        no_bool_datetime_df = df.loc[:, ~df.columns.isin(no_bool_dt_cols)]
        no_bool_datetime_cols = no_bool_datetime_df.columns.to_list()
        cat_columns = []
        for col in no_bool_datetime_cols:
            if col in self.cat_columns:
                df[col] = df[col].astype(str)
                self.detected_col_types[col] = "object"
                cat_columns.append(col)
            else:
                try:
                    if (df[col] - df[col].astype(int)).sum() == 0:
                        df[col] = df[col].astype(int)
                        self.detected_col_types[col] = "int"
                    else:
                        df[col] = df[col].astype(float)
                        self.detected_col_types[col] = "float"
                except Exception:
                    df[col] = df[col].astype(str)
                    self.detected_col_types[col] = "object"
                    cat_columns.append(col)
        self.cat_columns = cat_columns
        return df

    def fit_transform_feature_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify and transform feature types.

        Wrapper function to orchester different detection methods.
        """
        logging.info("Start detecting and casting feature types.")
        df = self.drop_all_null_columns(df)
        df = self.drop_zero_variance_columns(df)
        self.identify_num_columns(df)
        bool_cols, no_bool_cols = self.identify_bool_columns(df)
        self.identify_date_time_columns(df, no_bool_cols)
        df = self.cast_rest_columns_to_object(df, bool_cols)
        return df

    def transform_feature_types(
        self, df: pd.DataFrame, ignore_cols: List[Union[str, float, int, None]]
    ) -> pd.DataFrame:
        """Transform feature types based on already mapped types."""
        """
        Loops through the dataframe and detects column types and type casts them accordingly.
        :return: Returns casted dataframe
        """
        logging.info("Start casting feature types.")
        df = self.drop_all_null_columns(df)
        df = self.drop_zero_variance_columns(df)
        for key in self.detected_col_types:
            if ignore_cols and key not in ignore_cols and key in df.columns:
                if self.detected_col_types[key] == "datetime[ns]":
                    df[key] = pd.to_datetime(df[key], yearfirst=True)
                else:
                    df[key] = df[key].astype(self.detected_col_types[key])
        return df
