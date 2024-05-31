"""General utilities."""

import logging
from typing import Any, Dict, Optional

import dill as pickle
import numpy as np
import xgboost as xgb


def check_gpu_support() -> Dict[str, str]:
    logging.info("Start checking if GPU is available for usage.")
    data = np.random.rand(50, 2)
    label = np.random.randint(2, size=50)
    d_train = xgb.DMatrix(data, label=label)

    try:
        params = {"device": "cuda"}
        xgb.train(params, d_train, num_boost_round=2)
        logging.info("Xgboost uses GPU.")
        return params
    except Exception:
        pass

    try:
        params = {"tree_method": "gpu_hist"}
        xgb.train(params, d_train, num_boost_round=2)
        logging.info("Xgboost uses GPU.")
        return params
    except Exception:
        pass

    try:
        params = {"tree_method": "gpu"}
        xgb.train(params, d_train, num_boost_round=2)
        logging.info("Xgboost uses GPU.")
        return params
    except Exception as e:
        print(e)
        params = {"tree_method": "exact"}
        logging.info("Xgboost uses CPU.")
        return params


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
    logging.info("Start saving class instance.")
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
    :return: The loaded model object
    """
    logging.info("Start loading class instance.")
    if file_path:
        full_path = file_path + file_name
    else:
        full_path = file_name
    try:
        with open(full_path, "rb") as filehandler:
            automl_model = pickle.load(filehandler)
    except Exception:
        with open(full_path + file_type, "rb") as filehandler:
            automl_model = pickle.load(filehandler)
    return automl_model


def log_sampling(nb_rows: int, alpha: float = 2.0) -> int:
    """Return a number of samples.

    :param nb_rows: Number of rows in the dataset.
    :param alpha: Alpha value for sampling weight. Higher alpha values will result in less samples.
    """
    perc_reduction = np.log10(nb_rows) ** alpha / 100

    nb_samples = int(round(nb_rows * (1 - perc_reduction), 0))
    logging.info(f"Down sampling from {nb_rows} to {nb_samples} samples.")
    return nb_samples
