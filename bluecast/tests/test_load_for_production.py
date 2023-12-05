import os
import pickle

from bluecast.general_utils.general_utils import load_for_production


def test_load_for_production():
    # Define the file path, name, and type for the test file
    file_name = "test_instance"
    file_type = ".dat"

    # Create a dummy instance to save
    dummy_instance = "test_data"

    # Save the dummy instance to the file
    with open(file_name + file_type, "wb") as file:
        pickle.dump(dummy_instance, file)

    # Call the function to load the saved instance
    loaded_instance = load_for_production(file_name=file_name, file_type=file_type)

    # Check if the loaded instance is loaded
    assert loaded_instance

    # Delete the test file
    os.remove(file_name + file_type)


def test_load_for_production_with_default_file_type():
    # Define the file path and name for the test file
    file_name = "test_instance"

    # Create a dummy instance to save
    dummy_instance = "test_data"

    # Save the dummy instance to the file with the default file type
    with open(file_name + ".dat", "wb") as file:
        pickle.dump(dummy_instance, file)

    # Call the function to load the saved instance without specifying the file type
    loaded_instance = load_for_production(file_name=file_name)

    # Check if the loaded instance is loaded
    assert loaded_instance

    # Delete the test file
    os.remove(file_name + ".dat")
