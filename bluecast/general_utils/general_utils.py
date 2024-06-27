"""General utilities."""

import logging
import warnings
from typing import Any, Dict, Optional

import dill as pickle
import numpy as np
import xgboost as xgb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


# Capture warnings and redirect them to logging
def warning_to_logger(message, category, filename, lineno, file=None, line=None):
    logger.warning(f"{filename}:{lineno}: {category.__name__}: {message}")


warnings.showwarning = warning_to_logger


def check_gpu_support() -> Dict[str, str]:
    logger.info("Start checking if GPU is available for usage.")
    data = np.random.rand(50, 2)
    label = np.random.randint(2, size=50)
    d_train = xgb.DMatrix(data, label=label)

    params_list = [
        {"device": "cuda", "tree_method": "gpu_hist"},
        {"device": "cuda"},
        {"tree_method": "gpu_hist"},
    ]

    for params in params_list:
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                xgb.train(params, d_train, num_boost_round=2)

                # Check if any captured warnings indicate GPU-related issues
                gpu_warning = any(
                    "GPU" in str(warn.message) or "gpu" in str(warn.message)
                    for warn in w
                )
                if gpu_warning:
                    logger.warning(f"GPU-related warning captured: {w}")

                # If no warnings or no GPU-related warnings, consider GPU support confirmed
                if not gpu_warning:
                    logger.info("Xgboost is using GPU with parameters: %s", params)
                    return params
                else:
                    logger.warning(
                        "GPU settings applied but GPU-related warning detected: %s",
                        params,
                    )
        except xgb.core.XGBoostError as e:
            logger.warning("Failed with params %s. Error: %s", params, str(e))

    # If no GPU parameters work, fall back to CPU
    params = {"tree_method": "hist", "device": "cpu"}
    logger.info("No GPU detected. Xgboost will use CPU with parameters: %s", params)
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
