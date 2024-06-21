import logging

import numpy as np
import xgboost as xgb

from bluecast.general_utils.general_utils import check_gpu_support


def test_check_gpu_support(monkeypatch):
    """
    Test check_gpu_support function.

    We use the monkeypatch fixture provided by pytest to mock the xgb.train function and numpy's random functions.
    The mock_train function is defined as a replacement for xgb.train and asserts that the expected parameters are
    passed and the labels are set correctly. We then patch numpy's rand function to return a predefined array for data
    and randint function to return a predefined array for labels.

    Finally, we call check_gpu_support and assert that it returns the expected output, which in this case is "gpu_hist".
    """

    def mock_train(params, d_train):
        assert params == {"device": "cuda"}
        assert np.array_equal(d_train.get_label(), np.array([0, 1, 0, 1, 0]))
        return None

    monkeypatch.setattr(xgb, "train", mock_train)
    monkeypatch.setattr(
        np.random,
        "rand",
        lambda *args: np.array(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 0.1]]
        ),
    )
    monkeypatch.setattr(
        np.random, "randint", lambda low, size: np.array([0, 1, 0, 1, 0])
    )

    assert check_gpu_support() == {"tree_method": "hist", "device": "cpu"}


def test_check_gpu_support_logging(caplog):
    with caplog.at_level(logging.INFO):
        params = check_gpu_support()

        # Verify the logging output
        assert "Start checking if GPU is available for usage." in caplog.text

        if "gpu" in params["tree_method"]:
            assert "Xgboost uses GPU." in caplog.text
            assert "Can use {'tree_method': 'gpu'}." in caplog.text
        else:
            assert "Xgboost uses CPU." in caplog.text
            assert (
                "Can use {'tree_method': 'hist', 'device': 'cpu'} for Xgboost"
                in caplog.text
            )

        # Check the returned params are correctly logged
        if params["device"] == "cuda":
            assert params == {
                "device": "cuda",
                "tree_method": "gpu",
                "predictor": "gpu_predictor",
            }
        elif "gpu" in params["tree_method"]:
            assert params == {"tree_method": "gpu"}
        else:
            assert params == {"tree_method": "hist", "device": "cpu"}
