import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

import duckdb
import pandas as pd

from bluecast.config.base_classes import BaseClassExperimentTracker
from bluecast.config.training_config import TrainingConfig


class ExperimentTracker(BaseClassExperimentTracker):
    """
    DuckDB-based implementation of ExperimentTracker used in BlueCast
    and BlueCastCV pipelines. This triggers automatically as long
    as the default Xgboost model is used. For custom ml models
    users need to create an own Tracker. The base class from
    bluecast.config.base_classes can be used as an inspiration.

    Uses separate tables for hyperparameter tuning and evaluation data
    for better organization and query performance.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize DuckDB-based experiment tracker.

        :param db_path: Path to DuckDB database file. If None, creates temporary database.
        """
        if db_path is None:
            # Create temporary database file
            self.temp_dir: Optional[str] = tempfile.mkdtemp()
            self.db_path = os.path.join(self.temp_dir, "experiment_tracker.duckdb")
        else:
            self.db_path = db_path
            self.temp_dir = None

        self._init_database()

    def _init_database(self) -> None:
        """Initialize database schema with separate tables for different data types."""
        with duckdb.connect(self.db_path) as conn:
            # Create sequence for hyperparameter experiments
            conn.execute("CREATE SEQUENCE IF NOT EXISTS hyperparameter_seq START 1")

            # Create hyperparameter tuning table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS hyperparameter_experiments (
                    id INTEGER DEFAULT nextval('hyperparameter_seq'),
                    experiment_id INTEGER NOT NULL,
                    score_category VARCHAR NOT NULL,
                    eval_scores DOUBLE,
                    metric_used VARCHAR NOT NULL,
                    metric_higher_is_better BOOLEAN NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    -- Training config fields
                    global_random_state INTEGER,
                    autotune_model BOOLEAN,
                    enable_feature_selection BOOLEAN,
                    hyperparameter_tuning_rounds INTEGER,
                    hyperparameter_tuning_max_runtime_secs INTEGER,
                    train_split_stratify BOOLEAN,
                    -- Model parameters (JSON)
                    model_parameters JSON
                )
            """
            )

            # Create sequence for evaluation experiments
            conn.execute("CREATE SEQUENCE IF NOT EXISTS evaluation_seq START 1")

            # Create evaluation table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS evaluation_experiments (
                    id INTEGER DEFAULT nextval('evaluation_seq'),
                    experiment_id INTEGER NOT NULL,
                    score_category VARCHAR NOT NULL,
                    eval_scores DOUBLE,
                    metric_used VARCHAR NOT NULL,
                    metric_higher_is_better BOOLEAN NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    -- Training config fields
                    global_random_state INTEGER,
                    autotune_model BOOLEAN,
                    enable_feature_selection BOOLEAN,
                    hyperparameter_tuning_rounds INTEGER,
                    hyperparameter_tuning_max_runtime_secs INTEGER,
                    train_split_stratify BOOLEAN,
                    -- Model parameters (JSON)
                    model_parameters JSON
                )
            """
            )

    def add_results(
        self,
        experiment_id: int,
        score_category: Literal["simple_train_test_score", "cv_score", "oof_score"],
        training_config: TrainingConfig,
        model_parameters: Dict[Any, Any],
        eval_scores: Union[float, int, None],
        metric_used: str,
        metric_higher_is_better: bool,
    ) -> None:
        """
        Add an individual experiment result into the tracker.

        :param experiment_id: Sequential id. Make sure add an increment.
        :param score_category: Chose one of ["simple_train_test_score", "cv_score", "oof_score"].
            "simple_train_test_score" is the default where a simple train-test split is done. "cv_score" is called
            when cross validation has been enabled in the TrainingConfig.
        :param training_config: TrainingConfig instance from bluecast.config.training_config.
        :param model_parameters: Dictionary with parameters of ml model (i.e. learning rate)
        :param eval_scores: The actual score of the experiment.
        :param metric_used: The name of the eval metric.
        :param metric_higher_is_better: True or False.
        """
        # Determine which table to use based on score category
        # Hyperparameter tuning results go to hyperparameter_experiments
        # Final evaluation results go to evaluation_experiments
        table_name = (
            "hyperparameter_experiments"
            if score_category in ["simple_train_test_score", "cv_score"]
            else "evaluation_experiments"
        )

        with duckdb.connect(self.db_path) as conn:
            conn.execute(
                f"""
                INSERT INTO {table_name} (
                    experiment_id, score_category, eval_scores, metric_used,
                    metric_higher_is_better, created_at, global_random_state,
                    autotune_model, enable_feature_selection, hyperparameter_tuning_rounds,
                    hyperparameter_tuning_max_runtime_secs, train_split_stratify, model_parameters
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    experiment_id,
                    score_category,
                    eval_scores,
                    metric_used,
                    metric_higher_is_better,
                    datetime.utcnow(),
                    training_config.global_random_state,
                    training_config.autotune_model,
                    training_config.enable_feature_selection,
                    training_config.hyperparameter_tuning_rounds,
                    training_config.hyperparameter_tuning_max_runtime_secs,
                    training_config.train_split_stratify,
                    model_parameters,
                ],
            )

    def retrieve_results_as_df(self, table: Optional[str] = None) -> pd.DataFrame:
        """
        Convert ExperimentTracker information into a Pandas DataFrame.

        :param table: Which table to query ("hyperparameter_experiments", "evaluation_experiments", or None for both)
        :return: Pandas DataFrame with experiment results
        """
        with duckdb.connect(self.db_path) as conn:
            if table == "hyperparameter_experiments":
                query = "SELECT * FROM hyperparameter_experiments ORDER BY created_at"
            elif table == "evaluation_experiments":
                query = "SELECT * FROM evaluation_experiments ORDER BY created_at"
            else:
                # Union both tables
                query = """
                    SELECT *, 'hyperparameter' as table_source FROM hyperparameter_experiments
                    UNION ALL
                    SELECT *, 'evaluation' as table_source FROM evaluation_experiments
                    ORDER BY created_at
                """

            return conn.execute(query).df()

    def get_best_score(
        self, target_metric: str, table: Optional[str] = None
    ) -> Union[int, float]:
        """
        Get the best score for a target metric.

        :param target_metric: The metric to find the best score for
        :param table: Which table to query ("hyperparameter_experiments", "evaluation_experiments", or None for both)
        :return: Best score value
        """
        with duckdb.connect(self.db_path) as conn:
            if table == "hyperparameter_experiments":
                base_query = "SELECT * FROM hyperparameter_experiments"
            elif table == "evaluation_experiments":
                base_query = "SELECT * FROM evaluation_experiments"
            else:
                base_query = """
                    SELECT * FROM hyperparameter_experiments
                    UNION ALL
                    SELECT * FROM evaluation_experiments
                """

            query = f"{base_query} WHERE metric_used = ?"
            results_df = conn.execute(query, [target_metric]).df()

            if results_df.empty:
                raise ValueError("No results have been found in experiment tracker")

            # Get the last row to check metric_higher_is_better
            if results_df.iloc[-1]["metric_higher_is_better"]:
                return results_df["eval_scores"].max()
            else:
                return results_df["eval_scores"].min()

    def get_hyperparameter_results(self) -> pd.DataFrame:
        """Get only hyperparameter tuning results."""
        return self.retrieve_results_as_df(table="hyperparameter_experiments")

    def get_evaluation_results(self) -> pd.DataFrame:
        """Get only evaluation results."""
        return self.retrieve_results_as_df(table="evaluation_experiments")

    def get_experiment_summary(self) -> Dict[str, Any]:
        """
        Get a summary of experiments stored in the tracker.

        :return: Dictionary with experiment statistics
        """
        with duckdb.connect(self.db_path) as conn:
            hyperparameter_count = conn.execute(
                "SELECT COUNT(*) FROM hyperparameter_experiments"
            ).fetchone()[0]
            evaluation_count = conn.execute(
                "SELECT COUNT(*) FROM evaluation_experiments"
            ).fetchone()[0]

            unique_experiments = conn.execute(
                """
                SELECT COUNT(DISTINCT experiment_id) FROM (
                    SELECT experiment_id FROM hyperparameter_experiments
                    UNION
                    SELECT experiment_id FROM evaluation_experiments
                )
            """
            ).fetchone()[0]

            metrics_used = conn.execute(
                """
                SELECT DISTINCT metric_used FROM (
                    SELECT metric_used FROM hyperparameter_experiments
                    UNION
                    SELECT metric_used FROM evaluation_experiments
                )
            """
            ).fetchall()

            return {
                "total_hyperparameter_experiments": hyperparameter_count,
                "total_evaluation_experiments": evaluation_count,
                "unique_experiments": unique_experiments,
                "metrics_used": [m[0] for m in metrics_used],
            }

    def close(self) -> None:
        """Close database connection and cleanup temporary files if created."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil

            shutil.rmtree(self.temp_dir)

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:  # Catch specific exceptions instead of bare except
            pass

    # Legacy compatibility methods for existing code
    @property
    def experiment_id(self) -> List[int]:
        """Legacy compatibility property."""
        with duckdb.connect(self.db_path) as conn:
            results = conn.execute(
                """
                SELECT experiment_id FROM (
                    SELECT experiment_id FROM hyperparameter_experiments
                    UNION ALL
                    SELECT experiment_id FROM evaluation_experiments
                ) ORDER BY experiment_id
            """
            ).fetchall()
            return [r[0] for r in results]
