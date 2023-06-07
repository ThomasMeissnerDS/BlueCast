"""General utilities."""
import logging
from typing import Any, Optional

import dill as pickle
import numpy as np
import xgboost as xgb


def check_gpu_support() -> str:
    data = np.random.rand(50, 2)
    label = np.random.randint(2, size=50)
    d_train = xgb.DMatrix(data, label=label)
    params = {"tree_method": "gpu_hist", "steps": 2}
    try:
        xgb.train(params, d_train)
        print("Xgboost uses GPU.")
        return "gpu_hist"
    except Exception:
        print("Xgboost uses CPU.")
        return "exact"


def logger(message: str) -> None:
    logging.info(message)
    print(message)


def save_to_production(
    class_instance: Any,
    file_path: Optional[str] = None,
    file_name: str = "automl_instance",
    file_type: str = ".dat",
) -> None:
    """
    Takes a pretrained model and saves it via dill.
    :param class_instance: Takes instance of a BlueCast class.
    :param file_path: Takes a string containing the full absolute path.
    :param file_name: Takes a string containing the whole file name.
    :param file_type: Takes the expected type of file to export.
    :return:
    """
    if file_path:
        full_path = file_path + file_name + file_type
    else:
        full_path = file_name + file_type
    filehandler = open(full_path, "wb")
    pickle.dump(class_instance, filehandler)
    filehandler.close()


def load_for_production(
    file_path: Optional[str] = None,
    file_name: str = "automl_instance",
    file_type: str = ".dat",
) -> Any:
    """
    Load in a pretrained auto ml model. This function will try to load the model as provided.
    It has a fallback logic to impute .dat as file_type in case the import fails initially.
    :param file_path: Takes a string containing the full absolute path.
    :param file_name: Takes a string containing the whole file name.
    :param file_type: Takes the expected type of file to import.
    :return:
    """
    if file_path:
        full_path = file_path + file_name
    else:
        full_path = file_name
    try:
        filehandler = open(full_path, "rb")
    except Exception:
        filehandler = open(full_path + file_type, "rb")
    automl_model = pickle.load(filehandler)
    filehandler.close()
    return automl_model
