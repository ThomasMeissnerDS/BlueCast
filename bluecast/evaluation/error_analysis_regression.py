"""
Module for regression error analysis with DuckDB backend.

Enhanced error analysis for regression tasks with DuckDB for better
performance and analytics capabilities.
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
from plotly.subplots import make_subplots

from bluecast.blueprints.cast_cv_regression import BlueCastCVRegression
from bluecast.blueprints.cast_regression import BlueCastRegression
from bluecast.evaluation.base_classes import (
    DataReader,
    ErrorAnalyser,
    ErrorDistributionPlotter,
    ErrorPreprocessor,
)


class DuckDBRegressionErrorAnalysisEngine:
    """
    DuckDB-based engine for regression error analysis with enhanced analytics.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize DuckDB regression error analysis engine.

        :param db_path: Path to DuckDB database file. If None, creates temporary database.
        """
        if db_path is None:
            self.temp_dir: Optional[str] = tempfile.mkdtemp()
            self.db_path: str = os.path.join(
                self.temp_dir, "regression_error_analysis.duckdb"
            )
        else:
            self.db_path = db_path
            self.temp_dir = None

        self._init_database()

    def _init_database(self) -> None:
        """Initialize database schema for regression error analysis."""
        with duckdb.connect(self.db_path) as conn:
            # Create sequence for regression error analysis
            conn.execute(
                "CREATE SEQUENCE IF NOT EXISTS regression_error_analysis_seq START 1"
            )

            # Create main regression error analysis table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS regression_error_analysis (
                    id INTEGER DEFAULT nextval('regression_error_analysis_seq'),
                    experiment_id VARCHAR NOT NULL,
                    target_column VARCHAR NOT NULL,
                    target_value DOUBLE,
                    predicted_value DOUBLE,
                    prediction_error DOUBLE,
                    absolute_error DOUBLE,
                    squared_error DOUBLE,
                    percentage_error DOUBLE,
                    target_quantile VARCHAR,
                    residual DOUBLE,
                    -- Feature values stored as JSON for flexibility
                    feature_values JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create sequence for regression error statistics
            conn.execute(
                "CREATE SEQUENCE IF NOT EXISTS regression_error_statistics_seq START 1"
            )

            # Create regression statistics table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS regression_error_statistics (
                    id INTEGER DEFAULT nextval('regression_error_statistics_seq'),
                    experiment_id VARCHAR NOT NULL,
                    feature_name VARCHAR NOT NULL,
                    feature_value VARCHAR,
                    target_quantile VARCHAR,
                    sample_count INTEGER,
                    mean_absolute_error DOUBLE,
                    median_absolute_error DOUBLE,
                    std_error DOUBLE,
                    min_error DOUBLE,
                    max_error DOUBLE,
                    rmse DOUBLE,
                    mape DOUBLE,
                    r_squared DOUBLE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

    def load_regression_data(
        self, df: pd.DataFrame, experiment_id: str, target_column: str
    ) -> None:
        """
        Load regression error analysis data into DuckDB.

        :param df: DataFrame with predictions and features
        :param experiment_id: Unique identifier for this experiment
        :param target_column: Name of the target column
        """
        with duckdb.connect(self.db_path) as conn:
            df_copy = df.copy()
            df_copy["experiment_id"] = experiment_id
            df_copy["target_column"] = target_column

            # Calculate additional error metrics
            if "predictions" in df_copy.columns and target_column in df_copy.columns:
                df_copy["residual"] = df_copy[target_column] - df_copy["predictions"]
                df_copy["percentage_error"] = np.where(
                    df_copy[target_column] != 0,
                    100 * np.abs(df_copy["residual"]) / np.abs(df_copy[target_column]),
                    np.nan,
                )

            # Ensure target_quantiles column exists for compatibility
            if "target_quantiles" not in df_copy.columns:
                df_copy["target_quantiles"] = None

            conn.register("temp_df", df_copy)

            conn.execute(
                f"""
                INSERT INTO regression_error_analysis (
                    experiment_id, target_column, target_value, predicted_value,
                    prediction_error, absolute_error, squared_error, percentage_error,
                    target_quantile, residual, feature_values, created_at
                )
                SELECT
                    experiment_id,
                    target_column,
                    CASE
                        WHEN TRY_CAST({target_column} AS DOUBLE) IS NOT NULL
                        THEN CAST({target_column} AS DOUBLE)
                        ELSE NULL
                    END as target_value,
                    CASE
                        WHEN TRY_CAST(predictions AS DOUBLE) IS NOT NULL
                        THEN CAST(predictions AS DOUBLE)
                        ELSE NULL
                    END as predicted_value,
                    prediction_error,
                    ABS(prediction_error) as absolute_error,
                    POWER(prediction_error, 2) as squared_error,
                    percentage_error,
                    target_quantiles as target_quantile,
                    residual,
                    NULL as feature_values,
                    CURRENT_TIMESTAMP as created_at
                FROM temp_df
                WHERE prediction_error IS NOT NULL
            """
            )

    def compute_regression_statistics(
        self, experiment_id: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute comprehensive regression error statistics.

        :param experiment_id: Experiment identifier
        :return: Dictionary of statistical DataFrames
        """
        with duckdb.connect(self.db_path) as conn:
            # Overall regression statistics
            overall_stats = conn.execute(
                """
                SELECT
                    COUNT(*) as total_samples,
                    AVG(absolute_error) as mean_absolute_error,
                    MEDIAN(absolute_error) as median_absolute_error,
                    STDDEV(absolute_error) as std_absolute_error,
                    MIN(absolute_error) as min_error,
                    MAX(absolute_error) as max_error,
                    SQRT(AVG(squared_error)) as rmse,
                    AVG(percentage_error) as mean_absolute_percentage_error,
                    -- R-squared calculation
                    1 - (SUM(squared_error) / (VAR_POP(target_value) * COUNT(*))) as r_squared,
                    CORR(target_value, predicted_value) as correlation
                FROM regression_error_analysis
                WHERE experiment_id = ?
            """,
                [experiment_id],
            ).df()

            # Error distribution by target quantiles
            quantile_stats = conn.execute(
                """
                SELECT
                    target_quantile,
                    COUNT(*) as sample_count,
                    AVG(absolute_error) as mean_error,
                    MEDIAN(absolute_error) as median_error,
                    STDDEV(absolute_error) as std_error,
                    MIN(absolute_error) as min_error,
                    MAX(absolute_error) as max_error,
                    SQRT(AVG(squared_error)) as rmse
                FROM regression_error_analysis
                WHERE experiment_id = ? AND target_quantile IS NOT NULL
                GROUP BY target_quantile
                ORDER BY mean_error DESC
            """,
                [experiment_id],
            ).df()

            # Residual analysis
            residual_stats = conn.execute(
                """
                SELECT
                    COUNT(*) as total_samples,
                    AVG(residual) as mean_residual,
                    MEDIAN(residual) as median_residual,
                    STDDEV(residual) as std_residual,
                    MIN(residual) as min_residual,
                    MAX(residual) as max_residual,
                    -- Test for heteroscedasticity
                    CORR(ABS(residual), predicted_value) as heteroscedasticity_corr
                FROM regression_error_analysis
                WHERE experiment_id = ?
            """,
                [experiment_id],
            ).df()

            # Top error samples
            top_errors = conn.execute(
                """
                SELECT *
                FROM regression_error_analysis
                WHERE experiment_id = ?
                ORDER BY absolute_error DESC
                LIMIT 20
            """,
                [experiment_id],
            ).df()

            return {
                "overall_statistics": overall_stats,
                "quantile_statistics": quantile_stats,
                "residual_statistics": residual_stats,
                "top_errors": top_errors,
            }

    def create_regression_visualizations(
        self, experiment_id: str
    ) -> Dict[str, go.Figure]:
        """
        Create comprehensive regression error visualizations using Plotly.

        :param experiment_id: Experiment identifier
        :return: Dictionary of Plotly figures
        """
        with duckdb.connect(self.db_path) as conn:
            data = conn.execute(
                """
                SELECT * FROM regression_error_analysis
                WHERE experiment_id = ?
            """,
                [experiment_id],
            ).df()

        if data.empty:
            return {}

        figures = {}

        # 1. Residual plot
        fig_residual = go.Figure()
        fig_residual.add_trace(
            go.Scatter(
                x=data["predicted_value"],
                y=data["residual"],
                mode="markers",
                name="Residuals",
                marker=dict(color="blue", size=6, opacity=0.6),
            )
        )

        # Add zero line
        fig_residual.add_hline(y=0, line_dash="dash", line_color="red")

        fig_residual.update_layout(
            title="Residual Plot: Residuals vs Predicted Values",
            xaxis_title="Predicted Values",
            yaxis_title="Residuals",
            template="plotly_white",
        )
        figures["residual_plot"] = fig_residual

        # 2. Predicted vs Actual scatter plot
        fig_scatter = go.Figure()
        fig_scatter.add_trace(
            go.Scatter(
                x=data["target_value"],
                y=data["predicted_value"],
                mode="markers",
                name="Predictions",
                marker=dict(color="blue", size=6, opacity=0.6),
            )
        )

        # Add perfect prediction line
        min_val = min(data["target_value"].min(), data["predicted_value"].min())
        max_val = max(data["target_value"].max(), data["predicted_value"].max())
        fig_scatter.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Perfect Prediction",
                line=dict(color="red", dash="dash"),
            )
        )

        fig_scatter.update_layout(
            title="Predicted vs Actual Values",
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values",
            template="plotly_white",
        )
        figures["predicted_vs_actual"] = fig_scatter

        # 3. Error distribution histogram
        fig_error_dist = go.Figure()
        fig_error_dist.add_trace(
            go.Histogram(
                x=data["absolute_error"],
                nbinsx=50,
                name="Error Distribution",
                marker_color="lightcoral",
                opacity=0.7,
            )
        )
        fig_error_dist.update_layout(
            title="Distribution of Absolute Errors",
            xaxis_title="Absolute Error",
            yaxis_title="Frequency",
            template="plotly_white",
        )
        figures["error_distribution"] = fig_error_dist

        # 4. Error by target quantiles (if available)
        if "target_quantile" in data.columns and data["target_quantile"].notna().any():
            fig_quantile = px.box(
                data,
                x="target_quantile",
                y="absolute_error",
                title="Error Distribution by Target Quantiles",
                template="plotly_white",
            )
            fig_quantile.update_xaxes(title="Target Quantiles")
            fig_quantile.update_yaxes(title="Absolute Error")
            figures["error_by_quantiles"] = fig_quantile

        # 5. Residual Q-Q plot for normality check
        from scipy import stats

        if len(data) > 10:
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(data)))
            sample_quantiles = np.sort(data["residual"])

            fig_qq = go.Figure()
            fig_qq.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=sample_quantiles,
                    mode="markers",
                    name="Q-Q Plot",
                    marker=dict(color="green", size=6),
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
                title="Q-Q Plot: Residual Distribution vs Normal Distribution",
                xaxis_title="Theoretical Quantiles (Normal)",
                yaxis_title="Sample Quantiles (Residuals)",
                template="plotly_white",
            )
            figures["residual_qq_plot"] = fig_qq

        # 6. Error vs prediction confidence (if available)
        if len(data) > 1:
            data_with_index = data.copy()
            data_with_index["sample_index"] = range(len(data))

            fig_error_trend = px.line(
                data_with_index,
                x="sample_index",
                y="absolute_error",
                title="Error Trend Over Sample Order",
                template="plotly_white",
            )
            fig_error_trend.update_xaxes(title="Sample Index")
            fig_error_trend.update_yaxes(title="Absolute Error")
            figures["error_trend"] = fig_error_trend

        return figures

    def close(self) -> None:
        """Close database connection and cleanup temporary files if created."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil

            shutil.rmtree(self.temp_dir)


class OutOfFoldDataReaderRegression(DataReader):
    def __init__(self, bluecast_instance: BlueCastRegression):
        self.bluecast_instance: BlueCastRegression = bluecast_instance
        self.class_problem = bluecast_instance.class_problem
        self.target_column = bluecast_instance.target_column
        self.prediction_columns: List[str] = []

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

        self.prediction_columns = ["predictions"]
        return oof_dataset

    def read_data_from_bluecast_cv_instance(self) -> pl.DataFrame:
        """
        Function to fail when called.

        Please use read_data_from_bluecast_instance instead.
        :return: Will raise an error.
        """
        raise ValueError("Please use OutOfFoldDataReaderRegressionCV class instead.")


class OutOfFoldDataReaderRegressionCV(DataReader):
    def __init__(self, bluecast_instance: BlueCastCVRegression):
        self.bluecast_instance: BlueCastCVRegression = bluecast_instance
        self.class_problem = bluecast_instance.bluecast_models[0].class_problem
        self.target_column = bluecast_instance.bluecast_models[0].target_column
        self.prediction_columns: List[str] = []

    def read_data_from_bluecast_instance(self) -> pl.DataFrame:
        """
        Function to fail when called.

        Please use read_data_from_bluecast_cv_instance instead.
        :return: Will raise an error.
        """
        raise ValueError("Please use OutOfFoldDataReaderRegression class instead.")

    def read_data_from_bluecast_cv_instance(self) -> pl.DataFrame:
        """
        Read out of fold datasets from defined storage location for CV regression.

        :return: Combined out of fold dataset.
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
        self.prediction_columns = ["predictions"]
        return combined_oof_data


class ErrorAnalyserRegressionMixin(ErrorAnalyser):
    def __init__(self):
        self.duckdb_engine = DuckDBRegressionErrorAnalysisEngine()

    def analyse_errors(
        self,
        df: Union[pd.DataFrame, pl.DataFrame],
        descending: bool = True,
        target_column: str = "target_quantiles",
    ) -> pl.DataFrame:
        """
        Enhanced regression error analysis using DuckDB for better insights.

        :param df: Preprocessed out of fold DataFrame.
        :param descending: Bool indicating if errors shall be ordered descending in final DataFrame.
        :return: Polars DataFrame with enhanced error analysis results.
        """
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        # Load data into DuckDB
        experiment_id = f"reg_exp_{np.random.randint(1000000)}"
        self.duckdb_engine.load_regression_data(df, experiment_id, target_column)

        # Compute enhanced statistics
        stats = self.duckdb_engine.compute_regression_statistics(experiment_id)

        # Print comprehensive statistics
        print("\n=== Regression Error Analysis Summary ===")
        if not stats["overall_statistics"].empty:
            overall = stats["overall_statistics"].iloc[0]
            print(f"Mean Absolute Error: {overall['mean_absolute_error']:.4f}")
            print(f"RMSE: {overall['rmse']:.4f}")
            print(f"R-squared: {overall['r_squared']:.4f}")
            print(f"Correlation: {overall['correlation']:.4f}")

        # Create comprehensive visualizations
        figures = self.duckdb_engine.create_regression_visualizations(experiment_id)

        # Show visualizations
        for name, fig in figures.items():
            print(f"\n=== {name.replace('_', ' ').title()} ===")
            fig.show()

        # Return enhanced error analysis by quantiles
        if not stats["quantile_statistics"].empty:
            return pl.from_pandas(stats["quantile_statistics"])
        else:
            # Fallback to overall statistics
            return pl.from_pandas(stats["overall_statistics"])


class ErrorDistributionRegressionPlotterMixin(ErrorDistributionPlotter):
    def __init__(self, ignore_columns_during_visualization: Optional[List[str]] = None):
        if not isinstance(ignore_columns_during_visualization, list):
            ignore_columns_during_visualization = []
        self.ignore_columns_during_visualization = ignore_columns_during_visualization

    def plot_error_distributions(
        self,
        df: pl.DataFrame,
        target_column: str = "target_quantiles",
    ):
        """
        Enhanced error distribution plotting for regression using Plotly.
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

        # Get feature columns
        feature_columns = [
            col
            for col in res_df.columns
            if col
            not in [target_column, "prediction_error", "prediction", "predictions"]
        ]

        # Create enhanced visualizations for key features
        for feature in feature_columns[:3]:  # Limit for performance
            unique_values = sorted(res_df[feature].unique())

            if len(unique_values) <= 15:  # Only plot if reasonable number of categories
                # Create subplot with violin and box plots
                fig = make_subplots(
                    rows=1,
                    cols=2,
                    subplot_titles=("Error Distribution", "Error Statistics"),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}]],
                )

                # Violin plot
                for target_val in res_df[target_column].unique():
                    subset = res_df[res_df[target_column] == target_val]
                    fig.add_trace(
                        go.Violin(
                            x=subset[feature],
                            y=subset["prediction_error"],
                            name=f"Target: {target_val}",
                            showlegend=True,
                        ),
                        row=1,
                        col=1,
                    )

                # Box plot for summary statistics
                fig.add_trace(
                    go.Box(
                        x=res_df[feature],
                        y=res_df["prediction_error"],
                        name="Overall Distribution",
                        showlegend=False,
                    ),
                    row=1,
                    col=2,
                )

                fig.update_layout(
                    title=f"Regression Error Analysis: {feature}",
                    template="plotly_white",
                    height=500,
                )

                fig.update_xaxes(title_text=feature, row=1, col=1, tickangle=45)
                fig.update_xaxes(title_text=feature, row=1, col=2, tickangle=45)
                fig.update_yaxes(title_text="Prediction Error", row=1, col=1)
                fig.update_yaxes(title_text="Prediction Error", row=1, col=2)

                fig.show()


class ErrorAnalyserRegression(
    OutOfFoldDataReaderRegression,
    ErrorPreprocessor,
    ErrorAnalyserRegressionMixin,
    ErrorDistributionRegressionPlotterMixin,
):
    def __init__(
        self,
        bluecast_instance: BlueCastRegression,
        ignore_columns_during_visualization=None,
    ):
        OutOfFoldDataReaderRegression.__init__(self, bluecast_instance)
        ErrorDistributionRegressionPlotterMixin.__init__(
            self, ignore_columns_during_visualization
        )
        ErrorAnalyserRegressionMixin.__init__(self)

    def stack_predictions_by_class(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add additional column with binned target.

        :param df: Polars DataFrame with original targets.
        :return: Polars DataFrame with additional binned targets column.
        """
        quantiles = [round(i, 2) for i in np.linspace(0, 0.95, 10)]
        df = df.with_columns(
            pl.col(self.target_column)
            .rank("ordinal")
            .qcut(quantiles)
            .alias("target_quantiles")
        )
        return df

    def calculate_errors(self, df: Union[pd.DataFrame, pl.DataFrame]) -> pl.DataFrame:
        """
        Analyse errors of predictions on out of fold data.

        :param df: DataFrame holding out of fold data and predictions.
        :return: Polars DataFrame with additional 'prediction_error' column.
        """
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        df["prediction_error"] = np.abs(
            df[self.target_column].astype(float) - df["predictions"].astype(float)
        )

        if isinstance(df, pd.DataFrame):
            df = pl.from_pandas(df)

        return df

    def analyse_segment_errors(self) -> pl.DataFrame:
        """
        Enhanced pipeline for regression error analysis with DuckDB backend.

        Reads the out of fold datasets from the output location defined in the training config inside the provided
        BlueCast instance, preprocess the data and calculate errors for all subsegments of the data.
        Numerical columns will be split into quantiles to get subsegments.
        :return: Polars DataFrame with subsegments and errors.
        """
        oof_data = self.read_data_from_bluecast_instance()
        stacked_oof_data = self.stack_predictions_by_class(oof_data)
        errors = self.calculate_errors(stacked_oof_data)
        self.plot_error_distributions(errors, "target_quantiles")
        errors_analysed = self.analyse_errors(errors.drop(self.target_column))
        return errors_analysed


class ErrorAnalyserRegressionCV(
    OutOfFoldDataReaderRegressionCV,
    ErrorPreprocessor,
    ErrorAnalyserRegressionMixin,
    ErrorDistributionRegressionPlotterMixin,
):
    def __init__(
        self,
        bluecast_instance: BlueCastCVRegression,
        ignore_columns_during_visualization=None,
    ):
        OutOfFoldDataReaderRegressionCV.__init__(self, bluecast_instance)
        ErrorDistributionRegressionPlotterMixin.__init__(
            self, ignore_columns_during_visualization
        )
        ErrorAnalyserRegressionMixin.__init__(self)

    def stack_predictions_by_class(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add additional column with binned target.

        :param df: Polars DataFrame with original targets.
        :return: Polars DataFrame with additional binned targets column.
        """
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
        :return: Polars DataFrame with additional 'prediction_error' column.
        """
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        df["prediction_error"] = np.abs(
            df[self.target_column].astype(float) - df["predictions"].astype(float)
        )

        if isinstance(df, pd.DataFrame):
            df = pl.from_pandas(df)

        return df

    def analyse_segment_errors(self) -> pl.DataFrame:
        """
        Enhanced pipeline for regression error analysis with DuckDB backend.

        Reads the out of fold datasets from the output location defined in the training config inside the provided
        BlueCast instance, preprocess the data and calculate errors for all subsegments of the data.
        Numerical columns will be split into quantiles to get subsegments.
        :return: Polars DataFrame with subsegments and errors.
        """
        oof_data = self.read_data_from_bluecast_cv_instance()
        stacked_oof_data = self.stack_predictions_by_class(oof_data)
        errors = self.calculate_errors(stacked_oof_data)
        self.plot_error_distributions(errors, "target_quantiles")
        errors_analysed = self.analyse_errors(errors.drop(self.target_column))
        return errors_analysed
