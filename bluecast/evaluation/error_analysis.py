"""
Module for error analysis.

This step follows the training step. Ideally
it uses stored out of fold datasets from using the 'fit_eval' methods.
"""

from typing import Callable, List, Optional, Union

import pandas as pd
import polars as pl

from bluecast.blueprints.cast import BlueCast
from bluecast.blueprints.cast_cv import BlueCastCV
from bluecast.blueprints.cast_cv_regression import BlueCastCVRegression
from bluecast.blueprints.cast_regression import BlueCastRegression
from bluecast.evaluation.base_classes import DataReader, ErrorAnalyser
from bluecast.preprocessing.encode_target_labels import TargetLabelEncoder


class OutOfFoldDataReader(DataReader):
    def __init__(self, bluecast_instance: Union[BlueCast, BlueCastRegression]):

        self.class_problem = bluecast_instance.class_problem
        self.target_column = bluecast_instance.target_column
        self.target_classes: List[Union[str, int, float]] = []
        self.prediction_columns: List[str] = []
        self.target_label_encoder: Optional[TargetLabelEncoder] = None

    def read_data_from_bluecast_instance(
        self, bluecast_instance: Union[BlueCast, BlueCastRegression]
    ) -> pl.DataFrame:
        if isinstance(
            bluecast_instance.conf_training.out_of_fold_dataset_store_path, str
        ):
            oof_dataset = pl.read_parquet(
                bluecast_instance.conf_training.out_of_fold_dataset_store_path
                + f"oof_data_{bluecast_instance.conf_training.global_random_state}.parquet"
            )
        else:
            raise ValueError(
                "out_of_fold_dataset_store_path has not been configured in Training config"
            )

        self.target_classes = (
            oof_dataset.unique(subset=[self.target_column])
            .select(self.target_column)
            .to_series()
            .to_list()
        )
        self.prediction_columns = [
            f"predictions_class_{target_class}" for target_class in self.target_classes
        ]
        self.target_label_encoder = bluecast_instance.target_label_encoder
        return oof_dataset

    def read_data_from_bluecast_cv_instance(
        self, bluecast_instance: Union[BlueCastCV, BlueCastCVRegression]
    ) -> pl.DataFrame:
        raise ValueError("Please use OutOfFoldDataReaderCV class instead.")

    def read_data_from_path(
        self,
        bluecast_instance: Union[BlueCast, BlueCastRegression],
    ) -> pl.DataFrame:
        """
        Read and create DataFrame for analyse_errors function

        :param bluecast_instance: Instance of BlueCast that created the data.
        """
        oof_df = self.read_data_from_bluecast_instance(bluecast_instance)
        return oof_df


class OutOfFoldDataReaderCV(DataReader):
    def __init__(self, bluecast_instance: Union[BlueCastCV, BlueCastCVRegression]):
        self.class_problem = bluecast_instance.bluecast_models[0].class_problem
        self.target_column = bluecast_instance.bluecast_models[0].target_column
        self.target_classes: List[Union[str, int, float]] = []
        self.prediction_columns: List[str] = []
        self.target_label_encoder: Optional[TargetLabelEncoder] = None

    def read_data_from_bluecast_instance(
        self, bluecast_instance: Union[BlueCast, BlueCastRegression]
    ) -> pl.DataFrame:
        raise ValueError("Please use OutOfFoldDataReader class instead.")

    def read_data_from_bluecast_cv_instance(
        self, bluecast_instance: Union[BlueCastCV, BlueCastCVRegression]
    ) -> pl.DataFrame:
        oof_datasets = []

        if isinstance(
            bluecast_instance.bluecast_models[
                0
            ].conf_training.out_of_fold_dataset_store_path,
            str,
        ):
            path: str = bluecast_instance.bluecast_models[
                0
            ].conf_training.out_of_fold_dataset_store_path
        else:
            raise ValueError(
                "out_of_fold_dataset_store_path has not been configured in Training config"
            )

        for idx in range(len(bluecast_instance.bluecast_models)):
            temp_df = pl.read_parquet(
                path
                + f"oof_data_{bluecast_instance.bluecast_models[idx].conf_training.global_random_state}.parquet"
            )
            oof_datasets.append(temp_df)

        oof_dataset = pl.concat(oof_datasets)
        self.target_classes = (
            oof_dataset.unique(subset=[self.target_column])
            .select(self.target_column)
            .to_series()
            .to_list()
        )
        self.prediction_columns = [
            f"predictions_class_{target_class}" for target_class in self.target_classes
        ]
        self.target_label_encoder = bluecast_instance.bluecast_models[
            0
        ].target_label_encoder
        return oof_dataset

    def read_data_from_path(
        self,
        bluecast_instance: Union[BlueCastCV, BlueCastCVRegression],
    ) -> pl.DataFrame:
        """
        Read and create DataFrame for analyse_errors function

        :param bluecast_instance: Instance of BlueCast that created the data.
        """
        oof_df = self.read_data_from_bluecast_cv_instance(bluecast_instance)
        return oof_df


class ErrorAnalyserClassification(ErrorAnalyser, OutOfFoldDataReader):
    def stack_predictions_by_class(self, df: pl.DataFrame) -> pl.DataFrame:
        stacked_df = []
        for cls in self.target_classes:
            temp_df = df.filter(self.target_column == cls)
            stacked_df.append(temp_df)

        return pl.concat(stacked_df)

    def analyse_errors(
        self, df: Union[pd.DataFrame, pl.DataFrame], loss_func: Callable
    ):
        """
        Analyse errors of predictions on out of fold data.

        :param df: DataFrame holding out of fold data and predictions.
        :param loss_func: Function that takes (y_true, y_pred) and returns a score. Will be used to evaluate
            prediction errors.
        :return: None
        """
        if isinstance(df, pd.DataFrame):
            df = pl.from_dataframe(df)

    def show_leaderboard(self) -> pd.DataFrame:
        pass


class ErrorAnalyserClassificationCV(ErrorAnalyser, OutOfFoldDataReaderCV):
    def stack_predictions_by_class(self, df: pl.DataFrame) -> pl.DataFrame:
        stacked_df = []
        for cls in self.target_classes:
            temp_df = df.filter(self.target_column == cls)
            stacked_df.append(temp_df)

        return pl.concat(stacked_df)

    def analyse_errors(
        self, df: Union[pd.DataFrame, pl.DataFrame], loss_func: Callable
    ):
        """
        Analyse errors of predictions on out of fold data.

        :param df: DataFrame holding out of fold data and predictions.
        :param loss_func: Function that takes (y_true, y_pred) and returns a score. Will be used to evaluate
            prediction errors.
        :return: None
        """
        if isinstance(df, pd.DataFrame):
            df = pl.from_dataframe(df)

    def show_leaderboard(self) -> pd.DataFrame:
        pass
