"""Module to calculate SHAP values for a trained ML model.

The implementation is flexible and can be used for almost any ML model. The implementation is based on the SHAP library.
"""

from typing import List, Literal, Tuple

import numpy as np
import pandas as pd
import shap

from bluecast.general_utils.general_utils import logger


def shap_explanations(model, df: pd.DataFrame) -> Tuple[np.ndarray, shap.Explainer]:
    """
    See explanations under:
    https://medium.com/rapids-ai/gpu-accelerated-shap-values-with-xgboost-1-3-and-rapids-587fad6822
    :param model: Trained ML model
    :param df: Test data to predict on.
    :return: Shap values
    """
    shap.initjs()
    try:
        explainer = shap.TreeExplainer(model)
        model_shap_values = explainer.shap_values(df)
        explainer = explainer(df)
        explainer = shap.Explanation(
            explainer.values[:, :, 1],
            explainer.base_values[:, 1],
            data=df.values,
            feature_names=df.columns,
        )
        shap.summary_plot(model_shap_values, df, plot_type="bar", show=True)
    except (AssertionError, shap.utils._exceptions.InvalidModelError):
        explainer = shap.KernelExplainer(model.predict, df)
        model_shap_values = explainer.shap_values(df)
        explainer = explainer(df)
        shap.summary_plot(model_shap_values, df, show=True)
    return model_shap_values, explainer


def shap_waterfall_plot(
    explainer: shap.Explainer,
    indices: List[int],
    class_problem: Literal["binary", "multiclass", "regression"],
) -> None:
    """
    Plot the SHAP waterfall plot.
    :param explainer: SHAP Explainer instance
    :param indices: List of sample indices to plot. Each index represents a sample.
    :param class_problem: Class problem type (i.e.: binary, multiclass, regression)
    :return: None
    """

    try:
        _, unique_classes = explainer[0].shape
    except ValueError:
        _, unique_classes = explainer.shape

    for idx in indices:
        if class_problem == "regression":
            logger(f"Show SHAP waterfall plot for idx {idx}.")
            explainer_values = explainer[idx]
            shap.waterfall_plot(
                explainer_values,
                show=True,
            )

        elif class_problem == "binary":
            logger(f"Show SHAP waterfall plot for idx {idx} and target class.")
            # try/except catches differences between base estimators like Xgboost and RandomForestClassifier
            try:
                explainer_values = explainer[idx, :, 1]
                shap.waterfall_plot(
                    explainer_values,
                    show=True,
                )
            except IndexError:
                explainer_values = explainer[idx, :]
                shap.waterfall_plot(
                    explainer_values,
                    show=True,
                )

        elif class_problem == "multiclass":
            for class_idx in range(unique_classes):
                logger(f"Show SHAP waterfall plot for idx {idx} and class {class_idx}.")
                explainer_values = explainer[idx, :, class_idx]
                shap.waterfall_plot(
                    explainer_values,
                    show=True,
                )
