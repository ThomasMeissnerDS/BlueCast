import numpy as np
import pytest

from bluecast.ensemble.nelder_mead import NelderMeadEnsemble


def test_nelder_mead_classification():
    y_true = np.array([0, 1, 0, 1])
    preds1 = np.array([0.1, 0.8, 0.2, 0.9])
    preds2 = np.array([0.4, 0.6, 0.4, 0.6])

    ensemble = NelderMeadEnsemble(is_classification=True)
    ensemble.fit([preds1, preds2], y_true)

    assert ensemble.is_fitted
    assert len(ensemble.weights) == 2

    blended = ensemble.predict([preds1, preds2])
    assert len(blended) == 4
    assert np.all((blended >= 0.0) & (blended <= 1.0))


def test_nelder_mead_regression():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    preds1 = np.array([11.0, 19.0, 31.0, 39.0])
    preds2 = np.array([15.0, 25.0, 25.0, 35.0])

    ensemble = NelderMeadEnsemble(is_classification=False)
    ensemble.fit([preds1, preds2], y_true)

    assert ensemble.is_fitted
    assert len(ensemble.weights) == 2

    blended = ensemble.predict([preds1, preds2])
    assert len(blended) == 4


def test_nelder_mead_rank():
    y_true = np.array([0, 1, 0, 1])
    preds1 = np.array([0.1, 0.8, 0.2, 0.9])

    ensemble = NelderMeadEnsemble(is_classification=True, blending_method="rank")
    ensemble.fit([preds1], y_true)

    blended = ensemble.predict([preds1])
    assert len(blended) == 4


def test_nelder_mead_not_fitted():
    ensemble = NelderMeadEnsemble(is_classification=True)
    with pytest.raises(RuntimeError):
        ensemble.predict([np.array([0.5])])
