import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
import xgboost as xgb


def check_gpu_support() -> str:
    data = np.random.rand(50, 2)
    label = np.random.randint(2, size=50)
    D_train = xgb.DMatrix(data, label=label)
    params = {"tree_method": "gpu_hist", "steps": 2}
    try:
        xgb.train(params, D_train)
        print("Xgboost uses GPU.")
        return "gpu_hist"
    except Exception:
        print("Xgboost uses CPU.")
        return "exact"


class FeatureTypeDetector:
    def __init__(self,
                 num_columns: Optional[List[Union[str, int, float]]] = None,
                 cat_columns: Optional[List[Union[str, int, float]]] = None,
                 date_columns: Optional[List[Union[str, int, float]]] = None):
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

    def fit_transform_feature_types(self, df: pd.DataFrame) -> pd.DataFrame:
        # detect numeric columns by type
        if not self.num_columns:
            num_col_list = []
            for vartype in self.num_dtypes:
                num_cols = df.select_dtypes(include=[vartype]).columns
                for col in num_cols:
                    num_col_list.append(col)
            self.num_columns = num_col_list

        # detect and cast boolean columns
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

        if self.date_columns:
            date_columns = []
            # convert date columns from object to datetime type
            for col in self.date_columns:
                if col not in self.num_columns:
                    try:
                        df[col] = pd.to_datetime(df[col], yearfirst=True)
                        date_columns.append(col)
                        self.detected_col_types[col] = "datetime[ns]"
                    except Exception:
                        pass
            self.date_columns = date_columns
        if not self.date_columns:
            date_columns = []
            # convert date columns from object to datetime type
            for col in no_bool_cols:
                if col not in self.num_columns:
                    try:
                        df[col] = pd.to_datetime(df[col], yearfirst=True)
                        date_columns.append(col)
                        self.detected_col_types[col] = "datetime[ns]"
                    except Exception:
                        pass
            self.date_columns = date_columns

        # detect and cast floats
        no_bool_dt_cols = bool_cols + self.date_columns
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

    def transform_feature_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Loops through the dataframe and detects column types and type casts them accordingly.
        :return: Returns casted dataframe
        """
        for key in self.detected_col_types:
            if self.detected_col_types[key] == "datetime[ns]":
                df[key] = pd.to_datetime(
                    df[key], yearfirst=True
                )
            else:
                df[key] = df[key].astype(
                    self.detected_col_types[key]
                )
        return df
