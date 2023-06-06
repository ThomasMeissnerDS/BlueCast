import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import dill as pickle
import numpy as np
import pandas as pd
import xgboost as xgb


def check_gpu_support() -> str:
    data = np.random.rand(50, 2)
    label = np.random.randint(2, size=50)
    d_train = xgb.DMatrix(data, label=label)
    params = {"tree_method": "gpu_hist", "steps": 2}
    try:
        xgb.train(params, d_train)
        print("Xgboost uses GPU.")
        return "gpu_hist"
    except Exception:
        print("Xgboost uses CPU.")
        return "exact"


def logger(message: str) -> None:
    logging.info(message)
    print(message)


def save_to_production(
    class_instance: Any,
    file_path: Optional[str] = None,
    file_name: str = "automl_instance",
    file_type: str = ".dat",
) -> None:
    """
    Takes a pretrained model and saves it via dill.
    :param class_instance: Takes instance of a BlueCast class.
    :param file_path: Takes a string containing the full absolute path.
    :param file_name: Takes a string containing the whole file name.
    :param file_type: Takes the expected type of file to export.
    :return:
    """
    if file_path:
        full_path = file_path + file_name + file_type
    else:
        full_path = file_name + file_type
    filehandler = open(full_path, "wb")
    pickle.dump(class_instance, filehandler)
    filehandler.close()


def load_for_production(
    file_path: Optional[str] = None,
    file_name: str = "automl_instance",
    file_type: str = ".dat",
) -> Any:
    """
    Load in a pretrained auto ml model. This function will try to load the model as provided.
    It has a fallback logic to impute .dat as file_type in case the import fails initially.
    :param file_path: Takes a string containing the full absolute path.
    :param file_name: Takes a string containing the whole file name.
    :param file_type: Takes the expected type of file to import.
    :return:
    """
    if file_path:
        full_path = file_path + file_name
    else:
        full_path = file_name
    try:
        filehandler = open(full_path, "rb")
    except Exception:
        filehandler = open(full_path + file_type, "rb")
    automl_model = pickle.load(filehandler)
    filehandler.close()
    return automl_model


class FeatureTypeDetector:
    """Detect and cast feature types in DataFrame.

    Column names for individual feature types can be provided. Otherwise types will be inferred and casted accordingly.
    """

    def __init__(
        self,
        num_columns: Optional[List[Union[str, int, float]]] = None,
        cat_columns: Optional[List[Union[str, int, float]]] = None,
        date_columns: Optional[List[Union[str, int, float]]] = None,
    ):
        self.num_columns = num_columns
        self.cat_columns = cat_columns
        self.date_columns = date_columns
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
            df[col] = df[col].astype(bool)
            self.detected_col_types[col] = "bool"

        # detect and cast datetime columns
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
                if col not in self.num_columns:
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
            try:
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
        self.identify_num_columns(df)
        bool_cols, no_bool_cols = self.identify_bool_columns(df)
        self.identify_date_time_columns(df, no_bool_cols)
        df = self.cast_rest_columns_to_object(df, bool_cols)
        return df

    def transform_feature_types(
        self, df: pd.DataFrame, ignore_cols: List[Union[str, float, int, None]] = None
    ) -> pd.DataFrame:
        """Transform feature types based on already mapped types."""
        """
        Loops through the dataframe and detects column types and type casts them accordingly.
        :return: Returns casted dataframe
        """
        for key in self.detected_col_types:
            if ignore_cols and key not in ignore_cols:
                if self.detected_col_types[key] == "datetime[ns]":
                    df[key] = pd.to_datetime(df[key], yearfirst=True)
                else:
                    df[key] = df[key].astype(self.detected_col_types[key])
        return df
