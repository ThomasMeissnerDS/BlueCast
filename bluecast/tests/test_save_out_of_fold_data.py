import os
import tempfile

import pandas as pd

from bluecast.config.training_config import TrainingConfig
from bluecast.general_utils.general_utils import save_out_of_fold_data


def test_save_out_of_fold_data():
    # Create mock data for oof_data and y_hat
    oof_data = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    y_hat = pd.Series([0.1, 0.2, 0.3])

    # Create a temporary directory to store the file
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a TrainingConfig instance with the temporary directory as the path
        training_config = TrainingConfig(
            global_random_state=42, out_of_fold_dataset_store_path=tmpdir + "/"
        )

        # Call the function to save out of fold data
        save_out_of_fold_data(oof_data, y_hat, training_config)

        # Construct the expected file path
        expected_file_path = os.path.join(
            training_config.out_of_fold_dataset_store_path,
            f"oof_data_{training_config.global_random_state}.parquet",
        )

        # Check if the file was created
        assert os.path.exists(expected_file_path), "Output file was not created."

        # Read the file back and check its contents
        saved_oof_data = pd.read_parquet(expected_file_path)
        expected_oof_data = oof_data.copy()
        expected_oof_data["preditions"] = y_hat

        pd.testing.assert_frame_equal(saved_oof_data, expected_oof_data)
