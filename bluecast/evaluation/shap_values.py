"""Module to calculate SHAP values for a trained ML model.

The implementation is flexible and can be used for almost any ML model. The implementation is based on the SHAP library.
"""
import numpy as np
import pandas as pd
import shap


def shap_explanations(model, df: pd.DataFrame, explainer: str = "tree") -> np.ndarray:
    """
    See explanations under:
    https://medium.com/rapids-ai/gpu-accelerated-shap-values-with-xgboost-1-3-and-rapids-587fad6822
    :param model: Trained ML model
    :param df: Test data to predict on.
    :param explainer: Set "tree" for TreeExplainer. Otherwise uses KernelExplainer.
    :return: Shap values
    """
    shap.initjs()
    if explainer == "tree":
        try:
            tree_explainer = shap.TreeExplainer(model)
            model_shap_values = tree_explainer.shap_values(df)
            shap.summary_plot(model_shap_values, df, plot_type="bar", show=True)
        except AssertionError:
            model_shap_explainer = shap.KernelExplainer(model.predict, df)
            model_shap_values = model_shap_explainer.shap_values(df)
            shap.summary_plot(model_shap_values, df, show=True)
    else:
        model_shap_explainer = shap.KernelExplainer(model.predict, df)
        model_shap_values = model_shap_explainer.shap_values(df)
        shap.summary_plot(model_shap_values, df, show=True)
        return model_shap_values
