from category_encoders import (
    OneHotEncoder,
    TargetEncoder,
)

import pandas as pd
from typing import Dict, List, Union


class BinaryClassTargetEncoder:
    def __init__(self, cat_columns: List[str]):
        self.encoders: Dict[str, Union[List[str], TargetEncoder]] = {}
        self.prediction_mode: bool = False
        self.cat_columns = cat_columns

    def fit_target_encode_binary_class(self, x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        enc = TargetEncoder(cols=self.cat_columns)
        x[self.cat_columns] = enc.fit_transform(
            x[self.cat_columns], y
        )
        self.encoders[
            "target_encoder_all_cols"
        ] = enc
        return x

    def transform_target_encode_binary_class(self, x: pd.DataFrame) -> pd.DataFrame:
        enc = self.encoders["target_encoder_all_cols"]
        x[self.cat_columns] = enc.transform(
            x[self.cat_columns]
        )
        return x


class MultiClassTargetEncoder:
    def __init__(self, cat_columns: List[str]):
        self.encoders: Dict[str, Union[List[str], TargetEncoder, OneHotEncoder]] = {}
        self.prediction_mode: bool = False
        self.cat_columns = cat_columns

    def fit_target_encode_multiclass(self, x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        algorithm = "multiclass_target_encoding_onehotter"
        enc = OneHotEncoder()
        enc.fit(y)
        y_onehot = enc.transform(y)
        class_names = y_onehot.columns
        self.encoders["seen_targets"] = class_names
        x_obj = x.loc[:, self.cat_columns].copy()
        x = x.loc[:, ~x.columns.isin(self.cat_columns)]
        for class_ in class_names:
            target_enc = TargetEncoder()
            target_enc.fit(x_obj, y_onehot[class_])
            self.encoders[
                f"multiclass_target_encoder_all_cols_{class_}"
            ] = target_enc
            temp = target_enc.transform(x_obj)
            temp.columns = [str(x) + "_" + str(class_) for x in temp.columns]
            x = pd.concat([x, temp], axis=1)
        self.encoders[f"{algorithm}_all_cols"] = enc
        return x

    def transform_target_encode_multiclass(self, x: pd.DataFrame) -> pd.DataFrame:
        algorithm = "multiclass_target_encoding_onehotter"
        enc = self.encoders[
            f"{algorithm}_all_cols"
        ]
        class_names = self.encoders["seen_targets"]
        x_obj = x.loc[:, self.cat_columns].copy()
        x = x.loc[:, ~x.columns.isin(self.cat_columns)]
        for class_ in class_names:
            target_enc = self.encoders[
                f"multiclass_target_encoder_all_cols_{class_}"
            ]
            temp = target_enc.transform(x_obj)
            temp.columns = [str(x) + "_" + str(class_) for x in temp.columns]
            x = pd.concat([x, temp], axis=1)
        self.encoders[f"{algorithm}_all_cols"] = enc
        return x
