"""General utilities."""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import dill as pickle
import numpy as np
import xgboost as xgb


def check_gpu_support() -> Dict[str, str]:
    logger(f"{datetime.utcnow()}: Start checking if GPU is available for usage.")
    data = np.random.rand(50, 2)
    label = np.random.randint(2, size=50)
    d_train = xgb.DMatrix(data, label=label)

    try:
        params = {"device": "cuda"}
        xgb.train(params, d_train, num_boost_round=2)
        print("Xgboost uses GPU.")
        return params
    except Exception:
        pass

    try:
        params = {"tree_method": "gpu_hist"}
        xgb.train(params, d_train, num_boost_round=2)
        print("Xgboost uses GPU.")
        return params
    except Exception:
        pass

    try:
        params = {"tree_method": "gpu"}
        xgb.train(params, d_train, num_boost_round=2)
        print("Xgboost uses GPU.")
        return params
    except Exception as e:
        print(e)
        params = {"tree_method": "exact"}
        print("Xgboost uses CPU.")
        return params


def logger(message: str) -> None:
    logging.basicConfig(
        filename="bluecast.log",
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
    )
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
    logger(f"{datetime.utcnow()}: Start saving class instance.")
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
    logger(f"{datetime.utcnow()}: Start loading class instance.")
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


def log_sampling(nb_rows: int, alpha: float = 0.1) -> int:
    """Return number of samples based on log sampling.

    With the default value of alpha, the function will return:
    * 96 samples for 100 rows.
    * 960 samples for 1000 rows.
    * 9416 samples for 10000 rows.
    * 87262 samples for 100000 rows.
    * 615326 samples for 1000000 rows.
    :param nb_rows: Number of rows in the dataset.
    :param alpha: Alpha value for log sampling. Higher alpha values will result in less samples.
    """
    nb_samples = int(
        round(nb_rows - (np.log10(nb_rows) ** np.log10(nb_rows) ** (1 + alpha)), 0)
    )
    logger(f"Down sampling from {nb_rows} to {nb_samples} samples.")
    return nb_samples
