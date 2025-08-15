"""
Module for error analysis with DuckDB backend.

This step follows the training step. Ideally
it uses stored out of fold datasets from using the 'fit_eval' methods.
Enhanced with DuckDB for better performance and analytics capabilities.
"""

import os
import tempfile
from typing import Dict, List, Optional, Union

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl

from bluecast.blueprints.cast import BlueCast
from bluecast.blueprints.cast_cv import BlueCastCV
from bluecast.evaluation.base_classes import (
    DataReader,
    ErrorAnalyser,
    ErrorDistributionPlotter,
    ErrorPreprocessor,
)
from bluecast.preprocessing.encode_target_labels import TargetLabelEncoder


class DuckDBErrorAnalysisEngine:
    """
    DuckDB-based engine for error analysis providing enhanced analytics capabilities.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize DuckDB error analysis engine.

        :param db_path: Path to DuckDB database file. If None, creates temporary database.
        """
        if db_path is None:
            self.temp_dir: Optional[str] = tempfile.mkdtemp()
            self.db_path = os.path.join(self.temp_dir, "error_analysis.duckdb")
        else:
            self.db_path = db_path
            self.temp_dir = None

        self._init_database()

    def _init_database(self) -> None:
        """Initialize database schema for error analysis."""
        with duckdb.connect(self.db_path) as conn:
            # Create sequence for error analysis
            conn.execute("CREATE SEQUENCE IF NOT EXISTS error_analysis_seq START 1")

            # Create main error analysis table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS error_analysis (
                    id INTEGER DEFAULT nextval('error_analysis_seq'),
                    experiment_id VARCHAR NOT NULL,
                    target_column VARCHAR NOT NULL,
                    prediction_column VARCHAR,
                    prediction_error DOUBLE,
                    absolute_error DOUBLE,
                    squared_error DOUBLE,
                    target_class VARCHAR,
                    predicted_class VARCHAR,
                    prediction_confidence DOUBLE,
                    is_correct_prediction BOOLEAN,
                    -- Feature values stored as JSON for flexibility
                    feature_values JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create sequence for error statistics
            conn.execute("CREATE SEQUENCE IF NOT EXISTS error_statistics_seq START 1")

            # Create aggregated error statistics table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS error_statistics (
                    id INTEGER DEFAULT nextval('error_statistics_seq'),
                    experiment_id VARCHAR NOT NULL,
                    feature_name VARCHAR NOT NULL,
                    feature_value VARCHAR,
                    target_class VARCHAR,
                    sample_count INTEGER,
                    mean_error DOUBLE,
                    median_error DOUBLE,
                    std_error DOUBLE,
                    min_error DOUBLE,
                    max_error DOUBLE,
                    percentile_25 DOUBLE,
                    percentile_75 DOUBLE,
                    error_variance DOUBLE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

    def load_data(
        self, df: pd.DataFrame, experiment_id: str, target_column: str
    ) -> None:
        """
        Load error analysis data into DuckDB.

        :param df: DataFrame with predictions and features
        :param experiment_id: Unique identifier for this experiment
        :param target_column: Name of the target column
        """
        with duckdb.connect(self.db_path) as conn:
            # Insert data into error_analysis table
            df_copy = df.copy()
            df_copy["experiment_id"] = experiment_id
            df_copy["target_column"] = target_column

            # Convert DataFrame to DuckDB for efficient processing
            conn.register("temp_df", df_copy)

            conn.execute(
                """
                INSERT INTO error_analysis (
                    experiment_id, target_column, prediction_column, prediction_error,
                    absolute_error, squared_error, target_class, predicted_class,
                    prediction_confidence, is_correct_prediction, feature_values, created_at
                )
                SELECT
                    experiment_id,
                    target_column,
                    'prediction' as prediction_column,
                    prediction_error,
                    ABS(prediction_error) as absolute_error,
                    POWER(prediction_error, 2) as squared_error,
                    target_class::VARCHAR as target_class,
                    NULL as predicted_class,
                    NULL as prediction_confidence,
                    NULL as is_correct_prediction,
                    NULL as feature_values,
                    CURRENT_TIMESTAMP as created_at
                FROM temp_df
                WHERE prediction_error IS NOT NULL
            """
            )

    def compute_error_statistics(self, experiment_id: str) -> Dict[str, pd.DataFrame]:
        """
        Compute comprehensive error statistics.

        :param experiment_id: Experiment identifier
        :return: Dictionary of statistical DataFrames
        """
        with duckdb.connect(self.db_path) as conn:
            # Overall statistics
            overall_stats = conn.execute(
                """
                SELECT
                    COUNT(*) as total_samples,
                    AVG(absolute_error) as mean_absolute_error,
                    MEDIAN(absolute_error) as median_absolute_error,
                    STDDEV(absolute_error) as std_absolute_error,
                    MIN(absolute_error) as min_error,
                    MAX(absolute_error) as max_error,
                    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY absolute_error) as q25_error,
                    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY absolute_error) as q75_error,
                    VAR_POP(absolute_error) as error_variance
                FROM error_analysis
                WHERE experiment_id = ?
            """,
                [experiment_id],
            ).df()

            # Error distribution by target class
            class_stats = conn.execute(
                """
                SELECT
                    target_class,
                    COUNT(*) as sample_count,
                    AVG(absolute_error) as mean_error,
                    MEDIAN(absolute_error) as median_error,
                    STDDEV(absolute_error) as std_error,
                    MIN(absolute_error) as min_error,
                    MAX(absolute_error) as max_error
                FROM error_analysis
                WHERE experiment_id = ? AND target_class IS NOT NULL
                GROUP BY target_class
                ORDER BY mean_error DESC
            """,
                [experiment_id],
            ).df()

            # Top error samples
            top_errors = conn.execute(
                """
                SELECT *
                FROM error_analysis
                WHERE experiment_id = ?
                ORDER BY absolute_error DESC
                LIMIT 20
            """,
                [experiment_id],
            ).df()

            return {
                "overall_statistics": overall_stats,
                "class_statistics": class_stats,
                "top_errors": top_errors,
            }

    def create_error_visualizations(
        self, experiment_id: str, target_column: str = "target_class"
    ) -> Dict[str, go.Figure]:
        """
        Create comprehensive error visualizations using Plotly.

        :param experiment_id: Experiment identifier
        :param target_column: Target column name
        :return: Dictionary of Plotly figures
        """
        with duckdb.connect(self.db_path) as conn:
            # Get data for visualizations
            data = conn.execute(
                """
                SELECT * FROM error_analysis
                WHERE experiment_id = ?
            """,
                [experiment_id],
            ).df()

        if data.empty:
            return {}

        figures = {}

        # 1. Error distribution histogram
        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Histogram(
                x=data["absolute_error"],
                nbinsx=50,
                name="Error Distribution",
                marker_color="lightblue",
                opacity=0.7,
            )
        )
        fig_hist.update_layout(
            title="Distribution of Prediction Errors",
            xaxis_title="Absolute Error",
            yaxis_title="Frequency",
            template="plotly_white",
        )
        figures["error_distribution"] = fig_hist

        # 2. Error by target class (if available)
        if "target_class" in data.columns and data["target_class"].notna().any():
            fig_box = px.box(
                data,
                x="target_class",
                y="absolute_error",
                title="Error Distribution by Target Class",
                template="plotly_white",
            )
            fig_box.update_xaxes(title="Target Class")
            fig_box.update_yaxes(title="Absolute Error")
            figures["error_by_class"] = fig_box

        # 3. Error over time/order
        if len(data) > 1:
            data_with_index = data.copy()
            data_with_index["sample_index"] = range(len(data))

            fig_time = px.line(
                data_with_index,
                x="sample_index",
                y="absolute_error",
                title="Prediction Errors Over Sample Order",
                template="plotly_white",
            )
            fig_time.update_xaxes(title="Sample Index")
            fig_time.update_yaxes(title="Absolute Error")
            figures["error_over_time"] = fig_time

        # 4. Error heatmap (if multiple classes)
        if "target_class" in data.columns and data["target_class"].nunique() > 1:
            # Create error matrix
            error_matrix = (
                data.groupby("target_class")["absolute_error"]
                .agg(["mean", "std", "count"])
                .reset_index()
            )

            fig_heatmap = go.Figure(
                data=go.Heatmap(
                    z=error_matrix["mean"],
                    x=["Mean Error"],
                    y=error_matrix["target_class"],
                    colorscale="Reds",
                    showscale=True,
                    text=error_matrix["mean"].round(4),
                    texttemplate="%{text}",
                    textfont={"size": 12},
                )
            )
            fig_heatmap.update_layout(
                title="Mean Error by Target Class", template="plotly_white"
            )
            figures["error_heatmap"] = fig_heatmap

        # 5. Q-Q plot for error distribution normality
        from scipy import stats

        if len(data) > 10:
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(data)))
            sample_quantiles = np.sort(data["absolute_error"])

            fig_qq = go.Figure()
            fig_qq.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=sample_quantiles,
                    mode="markers",
                    name="Q-Q Plot",
                    marker=dict(color="blue", size=6),
                )
            )

            # Add reference line
            min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
            max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
            fig_qq.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    name="Reference Line",
                    line=dict(color="red", dash="dash"),
                )
            )

            fig_qq.update_layout(
                title="Q-Q Plot: Error Distribution vs Normal Distribution",
                xaxis_title="Theoretical Quantiles (Normal)",
                yaxis_title="Sample Quantiles (Errors)",
                template="plotly_white",
            )
            figures["qq_plot"] = fig_qq

        return figures

    def close(self) -> None:
        """Close database connection and cleanup temporary files if created."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil

            shutil.rmtree(self.temp_dir)


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
        Read out of fold datasets from defined storage location.

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

        :return: Out of fold dataset.
        """
        all_oof_data = []

        for bluecast_model in self.bluecast_instance.bluecast_models:
            if isinstance(
                bluecast_model.conf_training.out_of_fold_dataset_store_path, str
            ):
                oof_dataset = pl.read_parquet(
                    bluecast_model.conf_training.out_of_fold_dataset_store_path
                    + f"oof_data_{bluecast_model.conf_training.global_random_state}.parquet"
                )
                all_oof_data.append(oof_dataset)
            else:
                raise ValueError(
                    "out_of_fold_dataset_store_path has not been configured in Training config"
                )

        combined_oof_data = pl.concat(all_oof_data)

        self.target_classes = sorted(
            combined_oof_data.unique(subset=[self.target_column])
            .select(self.target_column)
            .to_series()
            .to_list()
        )
        self.prediction_columns = [
            f"predictions_class_{target_class}" for target_class in self.target_classes
        ]
        return combined_oof_data


class ErrorAnalyserClassificationMixin(ErrorAnalyser):
    def __init__(self):
        self.duckdb_engine = DuckDBErrorAnalysisEngine()

    def analyse_errors(
        self, df: Union[pd.DataFrame, pl.DataFrame], descending: bool = True
    ) -> pl.DataFrame:
        """
        Find mean absolute errors for all subsegments using DuckDB for enhanced analysis.

        :param df: Preprocessed out of fold DataFrame.
        :param descending: Bool indicating if errors shall be ordered descending in final DataFrame.
        :return: Polars DataFrame with all subsegments and mean absolute error in each of them.
        """
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        # Load data into DuckDB
        experiment_id = f"exp_{np.random.randint(1000000)}"
        self.duckdb_engine.load_data(df, experiment_id, "target_class")

        # Compute enhanced statistics
        self.duckdb_engine.compute_error_statistics(experiment_id)

        # Create comprehensive visualizations
        figures = self.duckdb_engine.create_error_visualizations(experiment_id)

        # Show visualizations
        for name, fig in figures.items():
            print(f"\n=== {name.replace('_', ' ').title()} ===")
            fig.show()

        # Return enhanced error analysis
        with duckdb.connect(self.duckdb_engine.db_path) as conn:
            enhanced_analysis = conn.execute(
                """
                SELECT
                    target_class,
                    COUNT(*) as sample_count,
                    AVG(absolute_error) as mean_error,
                    MEDIAN(absolute_error) as median_error,
                    STDDEV(absolute_error) as std_error,
                    MIN(absolute_error) as min_error,
                    MAX(absolute_error) as max_error,
                    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY absolute_error) as q25_error,
                    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY absolute_error) as q75_error
                FROM error_analysis
                WHERE experiment_id = ?
                GROUP BY target_class
                ORDER BY mean_error {}
            """.format(
                    "DESC" if descending else "ASC"
                ),
                [experiment_id],
            ).df()

        return pl.from_pandas(enhanced_analysis)


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
        """
        Enhanced error distribution plotting using Plotly with better visualizations.
        """
        res_df = df.to_pandas()
        # Only drop columns that actually exist
        cols_to_drop = [
            col
            for col in self.ignore_columns_during_visualization
            if col in res_df.columns
        ]
        if cols_to_drop:
            res_df = res_df.drop(cols_to_drop, axis=1)

        if (
            target_column not in res_df.columns
            or "prediction_error" not in res_df.columns
        ):
            raise ValueError("Required columns missing for error distribution plotting")

        # Get feature columns (excluding target and prediction columns)
        feature_columns = [
            col
            for col in res_df.columns
            if col
            not in [target_column, "prediction_error", "prediction", "predictions"]
        ]

        # Create enhanced violin plots for each feature
        for feature in feature_columns[:5]:  # Limit to first 5 features for performance
            unique_values = sorted(res_df[feature].unique())

            if len(unique_values) <= 10:  # Only plot if reasonable number of categories
                fig = px.violin(
                    res_df,
                    x=feature,
                    y="prediction_error",
                    color=target_column,
                    title=f"Enhanced Error Analysis: {feature} vs Prediction Error",
                    template="plotly_white",
                )

                # Add box plot overlay
                fig.update_traces(meanline_visible=True, showlegend=True)

                # Add statistical annotations
                fig.add_annotation(
                    text=f"Total samples: {len(res_df)}<br>Mean error: {res_df['prediction_error'].mean():.4f}",
                    xref="paper",
                    yref="paper",
                    x=0.02,
                    y=0.98,
                    showarrow=False,
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1,
                )

                fig.update_xaxes(tickangle=45)
                fig.show()


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
        ErrorAnalyserClassificationMixin.__init__(self)

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
            df = pl.from_pandas(df)

        return df

    def analyse_segment_errors(self) -> pl.DataFrame:
        """
        Enhanced pipeline for error analysis with DuckDB backend.

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
        ErrorAnalyserClassificationMixin.__init__(self)

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
            df = pl.from_pandas(df)

        return df

    def analyse_segment_errors(self) -> pl.DataFrame:
        """
        Enhanced pipeline for error analysis with DuckDB backend.

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
