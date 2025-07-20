"""Tests for base_classes module functionality."""

import os
import pickle
import tempfile
from unittest.mock import Mock, patch

import optuna
import pytest

from bluecast.config.training_config import (
    CatboostFinalParamConfig,
    CatboostTuneParamsConfig,
    TrainingConfig,
    XgboostFinalParamConfig,
    XgboostTuneParamsConfig,
)
from bluecast.experimentation.tracking import ExperimentTracker
from bluecast.ml_modelling.base_classes import CatboostBaseModel, XgboostBaseModel


class SimpleSampler:
    """A simple class that can be pickled for testing."""

    def __init__(self, seed):
        self.seed = seed


class TestXgboostBaseModel:
    """Test the XgboostBaseModel base class functionality."""

    @pytest.fixture
    def xgboost_base_model(self):
        """Create a basic XgboostBaseModel instance for testing."""
        return XgboostBaseModel(
            class_problem="binary",
            conf_training=TrainingConfig(),
            conf_xgboost=XgboostTuneParamsConfig(),
            conf_params_xgboost=XgboostFinalParamConfig(),
            experiment_tracker=ExperimentTracker(),
        )

    def test_create_optuna_study_basic(self, xgboost_base_model):
        """Test basic optuna study creation without database backend."""
        study = xgboost_base_model._create_optuna_study(
            direction="minimize", study_name="test_study"
        )

        assert isinstance(study, optuna.Study)
        assert study.study_name == "test_study"
        assert study.direction == optuna.study.StudyDirection.MINIMIZE

    def test_create_optuna_study_with_sampler_and_pruner(self, xgboost_base_model):
        """Test optuna study creation with sampler and pruner."""
        sampler = optuna.samplers.TPESampler(seed=42)
        pruner = optuna.pruners.MedianPruner()

        study = xgboost_base_model._create_optuna_study(
            direction="maximize",
            sampler=sampler,
            study_name="test_study_with_sampler",
            pruner=pruner,
        )

        assert isinstance(study, optuna.Study)
        assert study.study_name == "test_study_with_sampler"
        assert study.direction == optuna.study.StudyDirection.MAXIMIZE

    def test_create_optuna_study_with_database_backend(self, xgboost_base_model):
        """Test optuna study creation with database backend configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_study.db")
            xgboost_base_model.conf_training.optuna_db_backend_path = db_path

            study = xgboost_base_model._create_optuna_study(
                direction="minimize", study_name="test_db_study"
            )

            assert isinstance(study, optuna.Study)
            assert study.study_name == "test_db_study"
            # Verify that the database file is created
            assert os.path.exists(db_path)

    def test_create_optuna_study_with_database_backend_and_tpe_sampler(
        self, xgboost_base_model
    ):
        """Test optuna study creation with database backend and TPESampler (which doesn't save state due to no seed attr)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_study_sampler.db")
            sampler_path = os.path.join(tmpdir, "test_study_sampler_sampler.pkl")

            xgboost_base_model.conf_training.optuna_db_backend_path = db_path
            sampler = optuna.samplers.TPESampler(seed=42)

            study = xgboost_base_model._create_optuna_study(
                direction="minimize",
                sampler=sampler,
                study_name="test_db_sampler_study",
            )

            assert isinstance(study, optuna.Study)
            assert os.path.exists(db_path)
            # TPESampler doesn't have accessible seed attribute, so sampler file should NOT be created
            assert not os.path.exists(sampler_path)

    def test_create_optuna_study_with_database_backend_and_simple_sampler_with_seed(
        self, xgboost_base_model
    ):
        """Test optuna study creation with database backend and simple sampler that has seed attribute."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_study_simple_sampler.db")
            sampler_path = os.path.join(tmpdir, "test_study_simple_sampler_sampler.pkl")

            xgboost_base_model.conf_training.optuna_db_backend_path = db_path

            # Create a simple sampler with seed attribute
            sampler = SimpleSampler(seed=42)

            with patch("logging.info") as mock_log:
                study = xgboost_base_model._create_optuna_study(
                    direction="minimize",
                    sampler=sampler,
                    study_name="test_db_simple_sampler_study",
                )

                assert isinstance(study, optuna.Study)
                assert os.path.exists(db_path)
                assert os.path.exists(sampler_path)

                # Verify that the sampler was saved correctly
                with open(sampler_path, "rb") as f:
                    saved_sampler = pickle.load(f)
                assert saved_sampler.seed == 42

                # Verify logging was called
                mock_log.assert_called_with(f"Saved sampler state to {sampler_path}")

    def test_create_optuna_study_with_database_backend_and_sampler_without_seed(
        self, xgboost_base_model
    ):
        """Test optuna study creation with database backend and sampler without seed attribute."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_study_no_seed.db")
            sampler_path = os.path.join(tmpdir, "test_study_no_seed_sampler.pkl")

            xgboost_base_model.conf_training.optuna_db_backend_path = db_path

            # Create a mock sampler without seed attribute
            sampler = Mock()
            delattr(sampler, "seed") if hasattr(sampler, "seed") else None

            study = xgboost_base_model._create_optuna_study(
                direction="minimize",
                sampler=sampler,
                study_name="test_db_no_seed_study",
            )

            assert isinstance(study, optuna.Study)
            assert os.path.exists(db_path)
            # Sampler file should not be created if sampler doesn't have seed
            assert not os.path.exists(sampler_path)

    def test_create_optuna_study_sampler_save_error_handling(self, xgboost_base_model):
        """Test error handling when sampler state saving fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_study_error.db")
            xgboost_base_model.conf_training.optuna_db_backend_path = db_path

            # Create a simple sampler with seed attribute
            sampler = SimpleSampler(seed=42)

            # Mock pickle.dump to raise an exception
            with patch("pickle.dump", side_effect=Exception("Mocked pickle error")):
                with patch("logging.warning") as mock_warning:
                    study = xgboost_base_model._create_optuna_study(
                        direction="minimize",
                        sampler=sampler,
                        study_name="test_error_study",
                    )

                    assert isinstance(study, optuna.Study)
                    assert os.path.exists(db_path)

                    # Verify that warning was logged
                    mock_warning.assert_called_with(
                        "Could not save sampler state: Mocked pickle error"
                    )

    def test_create_optuna_study_with_database_backend_none(self, xgboost_base_model):
        """Test optuna study creation when database backend path is None."""
        xgboost_base_model.conf_training.optuna_db_backend_path = None
        sampler = optuna.samplers.TPESampler(seed=42)

        study = xgboost_base_model._create_optuna_study(
            direction="minimize", sampler=sampler, study_name="test_no_db_study"
        )

        assert isinstance(study, optuna.Study)
        assert study.study_name == "test_no_db_study"

    def test_create_optuna_study_with_database_backend_non_string(
        self, xgboost_base_model
    ):
        """Test optuna study creation when database backend path is not a string."""
        xgboost_base_model.conf_training.optuna_db_backend_path = 123  # Not a string

        study = xgboost_base_model._create_optuna_study(
            direction="minimize", study_name="test_non_string_db_study"
        )

        assert isinstance(study, optuna.Study)
        assert study.study_name == "test_non_string_db_study"

    def test_create_optuna_study_load_if_exists_with_database(self, xgboost_base_model):
        """Test that load_if_exists is set to True when using database backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_load_exists.db")
            xgboost_base_model.conf_training.optuna_db_backend_path = db_path

            # Mock optuna.create_study to capture kwargs
            with patch("optuna.create_study") as mock_create:
                mock_study = Mock()
                mock_create.return_value = mock_study

                _ = xgboost_base_model._create_optuna_study(
                    direction="minimize", study_name="test_load_study"
                )

                # Verify that create_study was called with load_if_exists=True
                mock_create.assert_called_once()
                call_kwargs = mock_create.call_args.kwargs
                assert call_kwargs["load_if_exists"] is True
                assert call_kwargs["storage"] == f"sqlite:///{db_path}"


class TestCatboostBaseModel:
    """Test the CatboostBaseModel base class functionality."""

    @pytest.fixture
    def catboost_base_model(self):
        """Create a basic CatboostBaseModel instance for testing."""
        return CatboostBaseModel(
            class_problem="binary",
            conf_training=TrainingConfig(),
            conf_catboost=CatboostTuneParamsConfig(),
            conf_params_catboost=CatboostFinalParamConfig(),
            experiment_tracker=ExperimentTracker(),
        )

    def test_catboost_create_optuna_study_basic(self, catboost_base_model):
        """Test basic optuna study creation for CatBoost without database backend."""
        study = catboost_base_model._create_optuna_study(
            direction="minimize", study_name="test_catboost_study"
        )

        assert isinstance(study, optuna.Study)
        assert study.study_name == "test_catboost_study"
        assert study.direction == optuna.study.StudyDirection.MINIMIZE

    def test_catboost_create_optuna_study_with_database_backend(
        self, catboost_base_model
    ):
        """Test CatBoost optuna study creation with database backend configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_catboost_study.db")
            catboost_base_model.conf_training.optuna_db_backend_path = db_path

            study = catboost_base_model._create_optuna_study(
                direction="minimize", study_name="test_catboost_db_study"
            )

            assert isinstance(study, optuna.Study)
            assert study.study_name == "test_catboost_db_study"
            assert os.path.exists(db_path)

    def test_catboost_create_optuna_study_with_database_backend_and_simple_sampler(
        self, catboost_base_model
    ):
        """Test CatBoost optuna study creation with database backend and simple sampler with seed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_catboost_sampler.db")
            sampler_path = os.path.join(tmpdir, "test_catboost_sampler_sampler.pkl")

            catboost_base_model.conf_training.optuna_db_backend_path = db_path

            # Create a simple sampler with seed attribute
            sampler = SimpleSampler(seed=123)

            with patch("logging.info") as mock_log:
                study = catboost_base_model._create_optuna_study(
                    direction="maximize",
                    sampler=sampler,
                    study_name="test_catboost_sampler_study",
                )

                assert isinstance(study, optuna.Study)
                assert os.path.exists(db_path)
                assert os.path.exists(sampler_path)

                # Verify that the sampler was saved correctly
                with open(sampler_path, "rb") as f:
                    saved_sampler = pickle.load(f)
                assert saved_sampler.seed == 123

                # Verify logging was called
                mock_log.assert_called_with(f"Saved sampler state to {sampler_path}")

    def test_catboost_create_optuna_study_sampler_save_error_handling(
        self, catboost_base_model
    ):
        """Test CatBoost error handling when sampler state saving fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_catboost_error.db")
            catboost_base_model.conf_training.optuna_db_backend_path = db_path

            # Create a simple sampler with seed attribute
            sampler = SimpleSampler(seed=456)

            with patch(
                "builtins.open", side_effect=PermissionError("Permission denied")
            ):
                with patch("logging.warning") as mock_warning:
                    study = catboost_base_model._create_optuna_study(
                        direction="minimize",
                        sampler=sampler,
                        study_name="test_catboost_error_study",
                    )

                    assert isinstance(study, optuna.Study)

                    # Verify that warning was logged
                    mock_warning.assert_called_with(
                        "Could not save sampler state: Permission denied"
                    )


class TestBaseClassesIntegration:
    """Integration tests for base classes functionality."""

    def test_xgboost_database_backend_file_creation_with_error_handling(self):
        """Test database backend creation with error handling for permission issues."""
        model = XgboostBaseModel(
            class_problem="binary",
            conf_training=TrainingConfig(),
            conf_xgboost=XgboostTuneParamsConfig(),
        )

        # Try to create database in a non-existent directory
        model.conf_training.optuna_db_backend_path = "/non/existent/path/test.db"

        # This should raise an OperationalError from SQLite
        with pytest.raises(
            Exception, match=".*"
        ):  # Could be OperationalError or similar
            _ = model._create_optuna_study(
                direction="minimize", study_name="test_permission_study"
            )

    def test_sampler_replacement_strategy_with_simple_sampler(self):
        """Test that .db is correctly replaced with _sampler.pkl in path."""
        model = XgboostBaseModel(
            class_problem="binary",
            conf_training=TrainingConfig(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test various .db path formats
            test_cases = [
                "study.db",
                "deep/path/study.db",
                "study_name_with_underscores.db",
                "study.db.db",  # Edge case with multiple .db
            ]

            for db_filename in test_cases:
                db_path = os.path.join(tmpdir, db_filename)
                expected_sampler_path = db_path.replace(".db", "_sampler.pkl")

                # Ensure directory exists for nested paths
                os.makedirs(os.path.dirname(db_path), exist_ok=True)

                model.conf_training.optuna_db_backend_path = db_path

                # Use simple sampler with seed attribute
                sampler = SimpleSampler(seed=42)

                study = model._create_optuna_study(
                    direction="minimize",
                    sampler=sampler,
                    study_name=f"test_path_{db_filename.replace('/', '_').replace('.', '_')}",
                )

                assert isinstance(study, optuna.Study)
                assert os.path.exists(expected_sampler_path)

                # Verify sampler can be loaded
                with open(expected_sampler_path, "rb") as f:
                    loaded_sampler = pickle.load(f)
                assert loaded_sampler.seed == 42

    def test_study_kwargs_construction(self):
        """Test that study kwargs are constructed correctly."""
        model = XgboostBaseModel(
            class_problem="binary",
            conf_training=TrainingConfig(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "kwargs_test.db")
            model.conf_training.optuna_db_backend_path = db_path

            sampler = optuna.samplers.TPESampler(seed=42)
            pruner = optuna.pruners.MedianPruner()

            with patch("optuna.create_study") as mock_create:
                mock_study = Mock()
                mock_create.return_value = mock_study

                _ = model._create_optuna_study(
                    direction="maximize",
                    sampler=sampler,
                    study_name="kwargs_test_study",
                    pruner=pruner,
                )

                # Verify all expected kwargs are passed
                mock_create.assert_called_once()
                call_kwargs = mock_create.call_args.kwargs

                assert call_kwargs["direction"] == "maximize"
                assert call_kwargs["study_name"] == "kwargs_test_study"
                assert call_kwargs["sampler"] == sampler
                assert call_kwargs["pruner"] == pruner
                assert call_kwargs["storage"] == f"sqlite:///{db_path}"
                assert call_kwargs["load_if_exists"] is True

    def test_tpe_sampler_behavior_documentation(self):
        """Test to document the current TPESampler behavior (no accessible seed attribute)."""
        sampler = optuna.samplers.TPESampler(seed=42)

        # Document that TPESampler doesn't have an accessible seed attribute
        assert not hasattr(sampler, "seed")

        # This means the current sampler saving logic in base_classes.py
        # will never trigger for TPESampler, which might be a bug in the implementation

        model = XgboostBaseModel(
            class_problem="binary",
            conf_training=TrainingConfig(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "tpe_test.db")
            sampler_path = os.path.join(tmpdir, "tpe_test_sampler.pkl")

            model.conf_training.optuna_db_backend_path = db_path

            study = model._create_optuna_study(
                direction="minimize", sampler=sampler, study_name="tpe_test_study"
            )

            assert isinstance(study, optuna.Study)
            assert os.path.exists(db_path)
            # Sampler file will NOT be created due to missing seed attribute
            assert not os.path.exists(sampler_path)
