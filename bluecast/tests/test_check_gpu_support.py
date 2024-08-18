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
            {"device": "cuda", "tree_method": "gpu_hist"},
            {"device": "cuda"},
            {"tree_method": "gpu_hist"},
        ]

        # Ensure that xgb.train was called at least once
        mock_train.assert_called()


def test_check_gpu_support_gpu_warning():
    """Test that the function falls back to CPU if GPU-related warnings are captured."""

    # Mock xgb.train to raise a warning with 'GPU' in the message, simulating GPU issues
    with patch("xgboost.train"):
        with patch("warnings.warn") as mock_warn:
            mock_warn.side_effect = lambda *args, **kwargs: mock_warn.message
            mock_warn.message = "GPU-related warning"

            params = check_gpu_support()
            assert params == {"tree_method": "gpu_hist"}
