"""Test save/load round-trip with actual BlueCast instances."""

import os
import tempfile

import numpy as np

from bluecast.general_utils.general_utils import load_for_production, save_to_production


class TestSaveLoadBlueCastRoundtrip:
    def test_save_and_load_binary_model(
        self, trained_bluecast_binary, synthetic_binary_data
    ):
        df_test = synthetic_binary_data.drop("target", axis=1).head(5)
        y_probs_before, y_classes_before = trained_bluecast_binary.predict(df_test)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_model.dat")
            save_to_production(trained_bluecast_binary, file_name=path, file_type="")
            loaded = load_for_production(file_name=path, file_type="")

        y_probs_after, y_classes_after = loaded.predict(df_test)
        np.testing.assert_array_almost_equal(y_probs_before, y_probs_after)
        np.testing.assert_array_equal(y_classes_before, y_classes_after)

    def test_save_and_load_regression_model(
        self, trained_bluecast_regression, synthetic_regression_data
    ):
        df_test = synthetic_regression_data.drop("target", axis=1).head(5)
        y_before = trained_bluecast_regression.predict(df_test)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_reg_model.dat")
            save_to_production(
                trained_bluecast_regression, file_name=path, file_type=""
            )
            loaded = load_for_production(file_name=path, file_type="")

        y_after = loaded.predict(df_test)
        np.testing.assert_array_almost_equal(y_before, y_after)

    def test_loaded_model_has_all_attributes(self, trained_bluecast_binary):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_attrs.dat")
            save_to_production(trained_bluecast_binary, file_name=path, file_type="")
            loaded = load_for_production(file_name=path, file_type="")

        assert hasattr(loaded, "ml_model")
        assert hasattr(loaded, "conf_training")
        assert hasattr(loaded, "feat_type_detector")
        assert hasattr(loaded, "prediction_mode")
        assert loaded.prediction_mode is True
