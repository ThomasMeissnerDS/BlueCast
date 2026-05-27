from unittest.mock import patch

import xgboost as xgb

from bluecast.general_utils.general_utils import check_gpu_support


def test_check_gpu_support_cpu_fallback():
    """Test that the function falls back to CPU when no GPU is detected."""

    # Mock xgb.train to always raise an XGBoostError, simulating no GPU support
    with patch("xgboost.train", side_effect=xgb.core.XGBoostError("No GPU support")):
        params = check_gpu_support()
        assert params == {"tree_method": "hist"}


def test_check_gpu_support_gpu_available():
    """Test that the function detects GPU support correctly."""

    # Mock xgb.train to work correctly, simulating GPU support
    with patch("xgboost.train") as mock_train:
        params = check_gpu_support()
        # Verify that GPU parameters are returned
        assert params in [
            {"tree_method": "hist", "device": "cuda"},
            {"tree_method": "hist", "device": "gpu"},
        ]

        # Ensure that xgb.train was called at least once
        mock_train.assert_called()


def test_check_gpu_support_gpu_warning():
    """Test that the function falls back to CPU if GPU-related warnings are captured."""

    import warnings

    def mock_train_with_warning(*args, **kwargs):
        warnings.warn("GPU-related warning")

    with patch("xgboost.train", side_effect=mock_train_with_warning):
        params = check_gpu_support()
        assert params == {"tree_method": "hist"}
