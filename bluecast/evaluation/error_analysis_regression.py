from typing import List, Union

import numpy as np
import pandas as pd
import polars as pl

from bluecast.blueprints.cast import BlueCast
from bluecast.blueprints.cast_cv import BlueCastCV
from bluecast.blueprints.cast_cv_regression import BlueCastCVRegression
from bluecast.blueprints.cast_regression import BlueCastRegression
from bluecast.evaluation.base_classes import (
    DataReader,
    ErrorAnalyser,
    ErrorPreprocessor,
)


class OutOfFoldDataReaderRegression(DataReader):
    def __init__(self, bluecast_instance: Union[BlueCast, BlueCastRegression]):
        self.bluecast_instance: Union[BlueCast, BlueCastRegression] = bluecast_instance
        self.class_problem = bluecast_instance.class_problem
        self.target_column = bluecast_instance.target_column
        self.prediction_columns: List[str] = []

    def read_data_from_bluecast_instance(self) -> pl.DataFrame:
        if isinstance(
            self.bluecast_instance.conf_training.out_of_fold_dataset_store_path, str
        ):
            oof_dataset = pl.read_parquet(
                self.bluecast_instance.conf_training.out_of_fold_dataset_store_path
                + f"oof_data_{self.bluecast_instance.conf_training.global_random_state}.parquet"
            )
        else:
            raise ValueError(
                "out_of_fold_dataset_store_path has not been configured in Training config"
            )

        self.prediction_columns = ["predictions"]
        return oof_dataset

    def read_data_from_bluecast_cv_instance(self) -> pl.DataFrame:
        raise ValueError("Please use OutOfFoldDataReaderRegressionCV class instead.")


class OutOfFoldDataReaderRegressionCV(DataReader):
    def __init__(self, bluecast_instance: Union[BlueCastCV, BlueCastCVRegression]):
        self.bluecast_instance: Union[BlueCastCV, BlueCastCVRegression] = (
            bluecast_instance
        )
        self.class_problem = bluecast_instance.bluecast_models[0].class_problem
        self.target_column = bluecast_instance.bluecast_models[0].target_column
        self.prediction_columns: List[str] = []

    def read_data_from_bluecast_instance(self) -> pl.DataFrame:
        raise ValueError("Please use OutOfFoldDataReaderRegression class instead.")

    def read_data_from_bluecast_cv_instance(self) -> pl.DataFrame:
        oof_datasets = []

        if isinstance(
            self.bluecast_instance.bluecast_models[
                0
            ].conf_training.out_of_fold_dataset_store_path,
            str,
        ):
            path: str = self.bluecast_instance.bluecast_models[
                0
            ].conf_training.out_of_fold_dataset_store_path
        else:
            raise ValueError(
                "out_of_fold_dataset_store_path has not been configured in Training config"
            )

        for idx in range(len(self.bluecast_instance.bluecast_models)):
            temp_df = pl.read_parquet(
                path
                + f"oof_data_{self.bluecast_instance.bluecast_models[idx].conf_training.global_random_state}.parquet"
            )
            oof_datasets.append(temp_df)

        oof_dataset = pl.concat(oof_datasets)
        self.prediction_columns = ["predictions"]
        return oof_dataset


class ErrorAnalyserRegressionMixin(ErrorAnalyser):
    def analyse_errors(
        self, df: Union[pd.DataFrame, pl.DataFrame], descending: bool = True
    ) -> pl.DataFrame:
        groupby_cols = [
            col
            for col in df.columns
            if col not in ["prediction_error", "predictions", "target_quantiles"]
        ]
        quantiles = [round(i, 2) for i in np.linspace(0, 0.95, 10)]
        numeric_columns = [
            col
            for col in df.select(pl.col(pl.NUMERIC_DTYPES)).columns
            if "target_quantiles" not in col
        ]

        error_dfs = []

        for col in groupby_cols:
            if col in numeric_columns:
                error_df = (
                    df.select(
                        pl.col("target_quantiles"),
                        pl.col(col).rank("ordinal").qcut(quantiles),
                        pl.col("prediction_error"),
                    )
                    .group_by([col, "target_quantiles"])
                    .agg(pl.mean("prediction_error"))
                )
                error_df = error_df.with_columns(pl.col(col).cast(pl.String))
                error_df = error_df.rename({col: "column_subset"})
                error_df = error_df.with_columns(pl.lit(col).alias("column_name"))
            else:
                error_df = (
                    df.select(
                        pl.col("target_quantiles"),
                        pl.col(col),
                        pl.col("prediction_error"),
                    )
                    .group_by([col, "target_quantiles"])
                    .agg(pl.mean("prediction_error"))
                )
                error_df = error_df.with_columns(pl.col(col).cast(pl.String))
                error_df = error_df.rename({col: "column_subset"})
                error_df = error_df.with_columns(pl.lit(col).alias("column_name"))
            error_dfs.append(error_df)

        return pl.concat(error_dfs).sort("prediction_error", descending=descending)


class ErrorAnalyserRegression(
    OutOfFoldDataReaderRegression, ErrorPreprocessor, ErrorAnalyserRegressionMixin
):
    def stack_predictions_by_class(self, df: pl.DataFrame) -> pl.DataFrame:
        quantiles = [round(i, 2) for i in np.linspace(0, 0.95, 10)]
        df = df.with_columns(
            pl.col(self.target_column)
            .rank("ordinal")
            .qcut(quantiles)
            .alias("target_quantiles")
        )
        return df

    def calculate_errors(self, df: Union[pd.DataFrame, pl.DataFrame]):
        """
        Analyse errors of predictions on out of fold data.

        :param df: DataFrame holding out of fold data and predictions.
        :param loss_func: Function that takes (y_true, y_pred) and returns a score. Will be used to evaluate
            prediction errors.
        :return: None
        """
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        df["prediction_error"] = np.abs(
            df["target"].astype(float) - df["predictions"].astype(float)
        )

        if isinstance(df, pd.DataFrame):
            df = pl.from_dataframe(df)

        return df

    def analyse_segment_errors(self) -> pl.DataFrame:
        oof_data = self.read_data_from_bluecast_instance()
        stacked_oof_data = self.stack_predictions_by_class(oof_data)
        errors = self.calculate_errors(stacked_oof_data)
        errors_analysed = self.analyse_errors(errors.drop(self.target_column))
        return errors_analysed


class ErrorAnalyserRegressionCV(
    OutOfFoldDataReaderRegressionCV, ErrorPreprocessor, ErrorAnalyserRegressionMixin
):
    def stack_predictions_by_class(self, df: pl.DataFrame) -> pl.DataFrame:
        quantiles = [round(i, 2) for i in np.linspace(0, 0.95, 10)]
        df = df.with_columns(
            pl.col(self.target_column)
            .rank("ordinal")
            .qcut(quantiles)
            .alias("target_quantiles")
        )
        return df

    def calculate_errors(self, df: Union[pd.DataFrame, pl.DataFrame]):
        """
        Analyse errors of predictions on out of fold data.

        :param df: DataFrame holding out of fold data and predictions.
        :param loss_func: Function that takes (y_true, y_pred) and returns a score. Will be used to evaluate
            prediction errors.
        :return: None
        """
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        df["prediction_error"] = np.abs(
            df["target"].astype(float) - df["predictions"].astype(float)
        )

        if isinstance(df, pd.DataFrame):
            df = pl.from_dataframe(df)

        return df

    def analyse_segment_errors(self) -> pl.DataFrame:
        oof_data = self.read_data_from_bluecast_cv_instance()
        stacked_oof_data = self.stack_predictions_by_class(oof_data)
        errors = self.calculate_errors(stacked_oof_data)
        errors_analysed = self.analyse_errors(errors.drop(self.target_column))
        return errors_analysed
