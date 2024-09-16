"""General utilities."""

import gc
import logging
import warnings
from typing import Any, Dict, Literal, Optional, Union

import dill as pickle
import numpy as np
import pandas as pd
import s3fs
import xgboost as xgb

from bluecast.config.training_config import TrainingConfig
from bluecast.preprocessing.encode_target_labels import TargetLabelEncoder

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
        {"tree_method": "gpu_hist"},
    ]

    for params in params_list:
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                xgb.train(params, d_train, num_boost_round=2)

                # Check if any captured warnings indicate GPU-related issues
                gpu_warning = any(
                    "GPU" in str(warn.message)
                    or "gpu" in str(warn.message)
                    or "device" in str(warn.message)
                    or "Device" in str(warn.message)
                    or "AllVisibleGPUs" in str(warn.message)
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
    params = {"tree_method": "hist"}
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


def save_out_of_fold_data(
    oof_data: pd.DataFrame,
    y_hat: Union[pd.Series, np.ndarray],
    y_classes: Optional[Union[pd.Series, np.ndarray]],
    y_true: Union[pd.Series, np.ndarray],
    target_column: str,
    class_problem: Literal["binary", "multiclass", "regression"],
    training_config: TrainingConfig,
    target_label_encoder: Optional[TargetLabelEncoder] = None,
) -> None:
    """Save out of fold data.

    :param oof_data: Data to save.
    :param y_hat: Predictions. Will be appended to oof_data and saved together. When class_problem is "binary", only the
        target class score is expected.
    :param y_classes: Predicted classes. Will be appended to oof_data and saved together. Only required for class_problem
        'binary' or 'multiclass'.
    :param y_true: True targets.
    :param target_column: String specifying name of the target column.
    :param class_problem: Takes a string containing the class problem type. Either "binary", "multiclass" or
        "regression".
    :param training_config: Training configuration.
    :param target_label_encoder: (Optional) TargetLabelEncoder object. This object will be created during classification
        tasks automatically when the target label is a string. It can be retrieved from the BlueCast and BlueCastCV
        instances via bluecast_obj.target_label_encoder or
        bluecast_cv_onj.bluecast_models[idx_of_model].target_label_encoder. Adding this argument will reverse translate
        the targets from numerical encodings back to the original strings for the column nam representation.
    """
    logging.info("Start saving out of fold data.")
    oof_data_copy = oof_data.copy()

    if isinstance(target_label_encoder, TargetLabelEncoder):
        reverse_target_mapping = {
            v: k for k, v in target_label_encoder.target_label_mapping.items()
        }
    else:
        reverse_target_mapping = {}

    if class_problem == "binary":
        if not isinstance(y_classes, (pd.Series, np.ndarray, list)):
            raise ValueError(
                "For 'class_problem binary and multiclass the array for y_classes has to be provided"
            )
        elif isinstance(y_classes, list):
            y_classes = np.asarray(y_classes).astype(int)

        y_true = y_true.astype(int)

        oof_data_copy["predicted_class"] = y_classes
        oof_data_copy["target_class_predicted_probas"] = [
            1 - preds if cls == 0 else preds for preds, cls in zip(y_hat, y_classes)
        ]
        oof_data_copy[f"predictions_class_{reverse_target_mapping.get(0, 0)}"] = (
            1 - y_hat
        )
        oof_data_copy[f"predictions_class_{reverse_target_mapping.get(1, 1)}"] = y_hat
    elif class_problem == "multiclass":
        if not isinstance(y_classes, (pd.Series, np.ndarray, list)):
            raise ValueError(
                "For 'class_problem binary and multiclass the array for y_classes has to be provided"
            )
        elif isinstance(y_classes, list):
            y_classes = np.asarray(y_classes).astype(int)

        if isinstance(y_true, pd.DataFrame):
            y_true = y_true[target_column].values

        y_true = y_true.astype(int)

        oof_data_copy["predicted_class"] = y_classes
        oof_data_copy["target_class_predicted_probas"] = np.asarray(
            [pred[target_cls] for pred, target_cls in zip(y_hat, y_true)]
        )
        for cls_idx in range(y_hat.shape[1]):
            oof_data_copy[
                f"predictions_class_{reverse_target_mapping.get(cls_idx, cls_idx)}"
            ] = y_hat[:, cls_idx]
    else:
        oof_data_copy["predictions"] = y_hat

    oof_data_copy[target_column] = y_true

    if (
        isinstance(training_config.out_of_fold_dataset_store_path, str)
        and "s3://" in training_config.out_of_fold_dataset_store_path
    ):
        fs = s3fs.S3FileSystem()
        with fs.open(
            training_config.out_of_fold_dataset_store_path
            + f"oof_data_{training_config.global_random_state}.parquet",
            "wb",
        ) as f:
            oof_data_copy.to_parquet(f)

    if isinstance(training_config.out_of_fold_dataset_store_path, str):
        oof_data_copy.to_parquet(
            training_config.out_of_fold_dataset_store_path
            + f"oof_data_{training_config.global_random_state}.parquet"
        )

    del oof_data_copy
    _ = gc.collect()
