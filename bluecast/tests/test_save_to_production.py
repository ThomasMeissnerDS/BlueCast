import os
import pickle

from bluecast.general_utils.general_utils import save_to_production


class DummyClass:
    def __init__(self, data):
        self.data = data


def test_save_to_production():
    # Create a temporary directory to store the test file
    # Create a dummy instance
    dummy_instance = DummyClass("test_data")
    # Define the expected file path and name
    file_name = "test_instance"
    file_type = ".dat"
    # Call the function to save the instance
    save_to_production(
        dummy_instance,
        file_name=file_name,
        file_type=file_type,
    )
    # Construct the expected file path
    expected_file_path = os.path.join(file_name + file_type)
    # Check if the file was created
    assert os.path.exists(expected_file_path)
    # Load the saved instance
    with open(expected_file_path, "rb") as file:
        loaded_instance = pickle.load(file)
    # Check if the loaded instance is equal to the original instance
    assert loaded_instance.data == dummy_instance.data
