from unittest.mock import ANY, patch

import numpy as np
import xgboost as xgb

from bluecast.general_utils.general_utils import check_gpu_support


def test_check_gpu_support_with_gpu():
    data = np.random.rand(50, 2)
    label = np.random.randint(2, size=50)
    xgb.DMatrix(data, label=label)

    # Mock xgb.train to simulate GPU support
    with patch("xgboost.train") as mock_train:
        mock_train.return_value = None  # Simulate successful training
        params = check_gpu_support()
        assert params["device"] == "cuda"
        assert "tree_method" in params


def test_check_gpu_support_without_gpu():
    data = np.random.rand(50, 2)
    label = np.random.randint(2, size=50)
    xgb.DMatrix(data, label=label)

    # Mock xgb.train to raise an error to simulate no GPU support
    with patch("xgboost.train", side_effect=xgb.core.XGBoostError("GPU not found")):
        params = check_gpu_support()
        assert params["device"] == "cpu"
        assert params["tree_method"] == "hist"


def test_check_gpu_support_with_xgboost_error():
    data = np.random.rand(50, 2)
    label = np.random.randint(2, size=50)
    xgb.DMatrix(data, label=label)

    with patch(
        "xgboost.train", side_effect=xgb.core.XGBoostError("Some XGBoost error")
    ):
        with patch("logging.Logger.warning") as mock_logger_warning:
            params = check_gpu_support()
            mock_logger_warning.assert_any_call(
                "Failed with params %s. Error: %s", ANY, "Some XGBoost error"
            )
            assert params["device"] == "cpu"
            assert params["tree_method"] == "hist"
