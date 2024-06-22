import logging

import numpy as np
import xgboost as xgb

from bluecast.general_utils.general_utils import check_gpu_support


def test_check_gpu_support_cuda(monkeypatch):
    """
    Test check_gpu_support function for CUDA support.
    """

    def mock_train(params, d_train, num_boost_round):
        assert params == {
            "device": "cuda",
            "tree_method": "gpu",
            "predictor": "gpu_predictor",
        }
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
        np.random, "randint", lambda low, high, size: np.array([0, 1, 0, 1, 0])
    )

    assert check_gpu_support() == {
        "device": "cuda",
        "tree_method": "gpu",
        "predictor": "gpu_predictor",
    }


def test_check_gpu_support_gpu_hist(monkeypatch):
    """
    Test check_gpu_support function for GPU hist support.
    """

    def mock_train(params, d_train, num_boost_round):
        assert params == {
            "tree_method": "gpu",
        }
        return None

    # Trigger an exception for the first try block
    def mock_train_fail(params, d_train, num_boost_round):
        raise Exception("GPU not supported")

    monkeypatch.setattr(xgb, "train", mock_train_fail)
    monkeypatch.setattr(xgb, "train", mock_train, raising=False)
    monkeypatch.setattr(
        np.random,
        "rand",
        lambda *args: np.array(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 0.1]]
        ),
    )
    monkeypatch.setattr(
        np.random, "randint", lambda low, high, size: np.array([0, 1, 0, 1, 0])
    )

    assert check_gpu_support() == {"tree_method": "gpu"}


def test_check_gpu_support_cpu(monkeypatch):
    """
    Test check_gpu_support function for CPU support.
    """

    def mock_train(params, d_train, num_boost_round):
        raise Exception("GPU not supported")

    monkeypatch.setattr(xgb, "train", mock_train)
    monkeypatch.setattr(
        np.random,
        "rand",
        lambda *args: np.array(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 0.1]]
        ),
    )
    monkeypatch.setattr(
        np.random, "randint", lambda low, high, size: np.array([0, 1, 0, 1, 0])
    )

    assert check_gpu_support() == {"tree_method": "hist", "device": "cpu"}


def test_check_gpu_support_logging(caplog, monkeypatch):
    with caplog.at_level(logging.INFO):

        def mock_train(params, d_train, num_boost_round):
            if params.get("device") == "cuda":
                return None
            elif params.get("tree_method") == "gpu":
                return None
            raise Exception("GPU not supported")

        monkeypatch.setattr(xgb, "train", mock_train)
        monkeypatch.setattr(
            np.random,
            "rand",
            lambda *args: np.array(
                [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 0.1]]
            ),
        )
        monkeypatch.setattr(
            np.random, "randint", lambda low, high, size: np.array([0, 1, 0, 1, 0])
        )

        params = check_gpu_support()

        # Verify the logging output
        assert "Start checking if GPU is available for usage." in caplog.text

        if params.get("device") == "cuda":
            assert "Xgboost uses GPU." in caplog.text
            assert (
                "Can use {'device': 'cuda', 'tree_method': 'gpu', 'predictor': 'gpu_predictor'} for Xgboost"
                in caplog.text
            )
        elif params.get("tree_method") == "gpu":
            assert "Xgboost uses GPU." in caplog.text
            assert "Can use {'tree_method': 'gpu'}." in caplog.text
        else:
            assert "Xgboost uses CPU." in caplog.text
            assert (
                "Can use {'tree_method': 'hist', 'device': 'cpu'} for Xgboost"
                in caplog.text
            )
