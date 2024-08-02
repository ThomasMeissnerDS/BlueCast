"""
Module for error analysis.

This step follows the training step. Ideally
it uses stored out of fold datasets from using the 'fit_eval' methods.
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd
import polars as pl

from bluecast.blueprints.cast import BlueCast
from bluecast.blueprints.cast_cv import BlueCastCV
from bluecast.eda.analyse import plot_error_distributions
from bluecast.evaluation.base_classes import (
    DataReader,
    ErrorAnalyser,
    ErrorDistributionPlotter,
    ErrorPreprocessor,
)
from bluecast.preprocessing.encode_target_labels import TargetLabelEncoder


class OutOfFoldDataReader(DataReader):
    def __init__(self, bluecast_instance: BlueCast):
        self.bluecast_instance: BlueCast = bluecast_instance
        self.class_problem = bluecast_instance.class_problem
        self.target_column = bluecast_instance.target_column
        self.target_classes: List[Union[str, int, float]] = []
        self.prediction_columns: List[str] = []
        self.target_label_encoder: Optional[TargetLabelEncoder] = None

    def read_data_from_bluecast_instance(self) -> pl.DataFrame:
        """
        Read out of fold datasetsfrom defined storage location.

        :return: Out of fold dataset.
        """
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

        self.target_classes = sorted(
            oof_dataset.unique(subset=[self.target_column])
            .select(self.target_column)
            .to_series()
            .to_list()
        )
        self.prediction_columns = [
            f"predictions_class_{target_class}" for target_class in self.target_classes
        ]
        self.target_label_encoder = self.target_label_encoder
        return oof_dataset

    def read_data_from_bluecast_cv_instance(self) -> pl.DataFrame:
        """
        Function to fail when called.

        Please use read_data_from_bluecast_instance instead.
        :return: Will raise an error.
        """
        raise ValueError("Please use OutOfFoldDataReaderCV class instead.")


class OutOfFoldDataReaderCV(DataReader):
    def __init__(self, bluecast_instance: BlueCastCV):
        self.bluecast_instance: BlueCastCV = bluecast_instance
        self.class_problem = bluecast_instance.bluecast_models[0].class_problem
        self.target_column = bluecast_instance.bluecast_models[0].target_column
        self.target_classes: List[Union[str, int, float]] = []
        self.prediction_columns: List[str] = []
        self.target_label_encoder: Optional[TargetLabelEncoder] = None

    def read_data_from_bluecast_instance(self) -> pl.DataFrame:
        """
        Function to fail when called.

        Please use read_data_from_bluecast_cv_instance instead.
        :return: Will raise an error.
        """
        raise ValueError("Please use OutOfFoldDataReader class instead.")

    def read_data_from_bluecast_cv_instance(self) -> pl.DataFrame:
        """
        Read out of fold datasets from defined storage location.

        :return: Concatenated out of fold dataset.
        """
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
        self.target_classes = sorted(
            oof_dataset.unique(subset=[self.target_column])
            .select(self.target_column)
            .to_series()
            .to_list()
        )
        self.prediction_columns = [
            f"predictions_class_{target_class}" for target_class in self.target_classes
        ]
        self.target_label_encoder = self.bluecast_instance.bluecast_models[
            0
        ].target_label_encoder
        return oof_dataset


class ErrorAnalyserClassificationMixin(ErrorAnalyser):
    def analyse_errors(
        self, df: Union[pd.DataFrame, pl.DataFrame], descending: bool = True
    ) -> pl.DataFrame:
        """
        Find mean absolute errors for all subsegments
        :param df: Preprocessed out of fold DataFrame.
        :param descending: Bool indicating if errors shall be ordered descending in final DataFrame.
        :return: Polars DataFrame with all subsegments and mean absolute error in each of them.
        """
        groupby_cols = [
            col for col in df.columns if col not in ["prediction_error", "target_class"]
        ]
        quantiles = [round(i, 2) for i in np.linspace(0, 0.95, 10)]
        numeric_columns = df.select(pl.col(pl.NUMERIC_DTYPES)).columns

        error_dfs = []

        for col in groupby_cols:
            if col in numeric_columns:
                error_df = (
                    df.select(
                        pl.col("target_class"),
                        pl.col(col).rank("ordinal").qcut(quantiles),
                        pl.col("prediction_error"),
                    )
                    .group_by([col, "target_class"])
                    .agg(pl.mean("prediction_error"))
                )
                error_df = error_df.with_columns(pl.col(col).cast(pl.String))
                error_df = error_df.rename({col: "column_subset"})
                error_df = error_df.with_columns(pl.lit(col).alias("column_name"))
            else:
                error_df = (
                    df.select(
                        pl.col("target_class"), pl.col(col), pl.col("prediction_error")
                    )
                    .group_by([col, "target_class"])
                    .agg(pl.mean("prediction_error"))
                )
                error_df = error_df.with_columns(pl.col(col).cast(pl.String))
                error_df = error_df.rename({col: "column_subset"})
                error_df = error_df.with_columns(pl.lit(col).alias("column_name"))
            error_dfs.append(error_df)

        return pl.concat(error_dfs).sort("prediction_error", descending=descending)


class ErrorDistributionPlotterMixin(ErrorDistributionPlotter):
    def __init__(self, ignore_columns_during_visualization: Optional[List[str]] = None):

        if not isinstance(ignore_columns_during_visualization, list):
            ignore_columns_during_visualization = []
        self.ignore_columns_during_visualization = ignore_columns_during_visualization

    def plot_error_distributions(
        self,
        df: pl.DataFrame,
        target_column: str = "target_class",
    ):
        res_df = df.to_pandas().drop(self.ignore_columns_during_visualization, axis=1)

        plot_error_distributions(
            res_df,
            target=target_column,
            prediction_error="prediction_error",
            num_cols_grid=2,
        )


class ErrorAnalyserClassification(
    OutOfFoldDataReader,
    ErrorPreprocessor,
    ErrorAnalyserClassificationMixin,
    ErrorDistributionPlotterMixin,
):
    def __init__(
        self,
        bluecast_instance: BlueCast,
        ignore_columns_during_visualization=None,
    ):
        OutOfFoldDataReader.__init__(self, bluecast_instance)
        ErrorDistributionPlotterMixin.__init__(
            self, ignore_columns_during_visualization
        )

    def stack_predictions_by_class(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Stack class predictions into a long format.

        BlueCast returns predictions for each class as separate columns. This function returns a DataFrame where
        all predictions are stacked as a single 'prediction' column.
        :param df: Polars DataFrame with wide predictions  format.
        :return: Polars DataFrame with stacked predictions.
        """
        stacked_df = []
        for cls in self.target_classes:
            cls_pred_col = [
                col for col in self.prediction_columns if str(cls) in str(col)
            ]
            other_cls_pred_col = [
                col for col in self.prediction_columns if str(cls) not in str(col)
            ]  # TODO: Check if similar names cause trouble
            temp_df = df.filter(pl.col(self.target_column) == cls).drop(
                other_cls_pred_col
            )
            temp_df = temp_df.rename({cls_pred_col[0]: "prediction"})
            temp_df = temp_df.with_columns(pl.lit(cls).alias("target_class"))
            stacked_df.append(temp_df)

        return pl.concat(stacked_df)

    def calculate_errors(self, df: Union[pd.DataFrame, pl.DataFrame]) -> pl.DataFrame:
        """
        Analyse errors of predictions on out of fold data.

        :param df: DataFrame holding out of fold data and predictions.
        :return: Polars DataFrame with additional 'prediction_error' column.
        """
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        df["prediction_error"] = np.abs(
            1 - df["target_class_predicted_probas"].astype(float)
        )

        if isinstance(df, pd.DataFrame):
            df = pl.from_dataframe(df)

        return df

    def analyse_segment_errors(self) -> pl.DataFrame:
        """
        Pipeline for error analysis.

        Reads the out of fold datasets from the output location defined in the training config inside the provided
        BlueCast instance, preprocess the data and calculate errors for all subsegments of the data.
        Numerical columns will be split into quantiles to get subsegments.
        :return: Polars DataFrame with subsegments and errors.
        """
        oof_data = self.read_data_from_bluecast_instance()
        stacked_oof_data = self.stack_predictions_by_class(oof_data)
        errors = self.calculate_errors(stacked_oof_data)
        self.plot_error_distributions(errors, "target_class")
        errors_analysed = self.analyse_errors(errors.drop(self.target_column))
        return errors_analysed


class ErrorAnalyserClassificationCV(
    OutOfFoldDataReaderCV,
    ErrorPreprocessor,
    ErrorAnalyserClassificationMixin,
    ErrorDistributionPlotterMixin,
):
    def __init__(
        self,
        bluecast_instance: BlueCastCV,
        ignore_columns_during_visualization=None,
    ):
        OutOfFoldDataReaderCV.__init__(self, bluecast_instance)
        ErrorDistributionPlotterMixin.__init__(
            self, ignore_columns_during_visualization
        )

    def stack_predictions_by_class(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Stack class predictions into a long format.

        BlueCast returns predictions for each class as separate columns. This function returns a DataFrame where
        all predictions are stacked as a single 'prediction' column.
        :param df: Polars DataFrame with wide predictions  format.
        :return: Polars DataFrame with stacked predictions.
        """
        stacked_df = []
        for cls in self.target_classes:
            cls_pred_col = [
                col for col in self.prediction_columns if str(cls) in str(col)
            ]
            other_cls_pred_col = [
                col for col in self.prediction_columns if str(cls) not in str(col)
            ]  # TODO: Check if similar names cause trouble
            temp_df = df.filter(pl.col(self.target_column) == cls).drop(
                other_cls_pred_col
            )
            temp_df = temp_df.rename({cls_pred_col[0]: "prediction"})
            temp_df = temp_df.with_columns(pl.lit(cls).alias("target_class"))
            stacked_df.append(temp_df)

        return pl.concat(stacked_df)

    def calculate_errors(self, df: Union[pd.DataFrame, pl.DataFrame]) -> pl.DataFrame:
        """
        Analyse errors of predictions on out of fold data.

        :param df: DataFrame holding out of fold data and predictions.
        :return: Polars DataFrame with additional 'prediction_error' column.
        """
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        df["prediction_error"] = np.abs(
            1 - df["target_class_predicted_probas"].astype(float)
        )

        if isinstance(df, pd.DataFrame):
            df = pl.from_dataframe(df)

        return df

    def analyse_segment_errors(self) -> pl.DataFrame:
        """
        Pipeline for error analysis.

        Reads the out of fold datasets from the output location defined in the training config inside the provided
        BlueCast instance, preprocess the data and calculate errors for all subsegments of the data.
        Numerical columns will be split into quantiles to get subsegments.
        :return: Polars DataFrame with subsegments and errors.
        """
        oof_data = self.read_data_from_bluecast_cv_instance()
        stacked_oof_data = self.stack_predictions_by_class(oof_data)
        errors = self.calculate_errors(stacked_oof_data)
        self.plot_error_distributions(errors, "target_class")
        errors_analysed = self.analyse_errors(errors.drop(self.target_column))
        return errors_analysed
