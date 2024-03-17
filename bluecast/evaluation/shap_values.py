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
            explainer.base_values[:, :, 1],
            data=df.values,
            feature_names=df.columns,
        )
        shap.summary_plot(model_shap_values, df, plot_type="bar", show=True)
    except IndexError:
        explainer = shap.TreeExplainer(model)
        model_shap_values = explainer.shap_values(df)
        explainer = explainer(df)
        explainer = shap.Explanation(
            explainer.values,
            explainer.base_values,
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
            explainer_values = explainer[idx, :]
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


def get_most_important_features_by_shap_values(
    shap_values: np.ndarray, df: pd.DataFrame
) -> pd.DataFrame:
    """
    Get the most important features by SHAP values.
    :param shap_values: Numpy ndarray holding Shap values
    :param df: Pandas DataFrame
    :return: Pandas DataFrame with columns 'col_name' and 'feature_importance_vals'
    """

    feature_names = df.columns

    # shap values might arrive as (nb_rows, nb_cols, nb_classes) or (nb_classes, nb_rows, nb_cols)
    if (
        np.asarray(shap_values).shape[0] != df.shape[0]
        or np.asarray(shap_values).shape[1] != df.shape[1]
    ):
        dfs = []
        for class_shap_values in shap_values:  # Loop through classes
            class_df = pd.DataFrame(class_shap_values, columns=feature_names)
            dfs.append(class_df)
        result_df = pd.concat(dfs)
    else:
        try:
            result_df = pd.DataFrame(shap_values, columns=feature_names)
        except Exception:
            result_df = pd.DataFrame(
                np.array(shap_values)[:, :, 1], columns=feature_names
            )

    vals = np.abs(result_df.values).mean(0)
    shap_importance = pd.DataFrame(
        list(zip(feature_names, vals)), columns=["col_name", "feature_importance_vals"]
    )
    shap_importance.sort_values(
        by=["feature_importance_vals"], ascending=False, inplace=True
    )
    return shap_importance


def shap_dependence_plots(
    shap_values: np.ndarray,
    df: pd.DataFrame,
    show_dependence_plots_of_top_n_features: int,
) -> None:
    """
    Plot the SHAP dependence plots for each column in the dataframe.
    :param shap_values: Numpy ndarray holding Shap values
    :param df: Pandas DataFrame
    :param show_dependence_plots_of_top_n_features: Number of features to show the dependence plots for.
    """

    logger("Plotting interactions of most important features by global SHAP values...")
    if show_dependence_plots_of_top_n_features > len(df.columns):
        show_dependence_plots_of_top_n_features = len(df.columns)

    sorted_shap_df = get_most_important_features_by_shap_values(shap_values, df)
    # We can also use the special "rank(i)" systax to specify the i'th most important feature

    for col in sorted_shap_df["col_name"].values[
        :show_dependence_plots_of_top_n_features
    ]:
        if len(np.asarray(shap_values).shape) == 2:
            try:
                shap.dependence_plot(col, shap_values, df, feature_names=df.columns)
            except (IndexError, AssertionError):
                shap.dependence_plot(col, shap_values[0], df, feature_names=df.columns)
        else:
            try:
                for idx, class_shap_values in enumerate(shap_values):
                    logger(
                        f"Showing dependence plots f top features for class {idx}..."
                    )
                    shap.dependence_plot(
                        col, class_shap_values, df, feature_names=df.columns
                    )
            except (IndexError, AssertionError):
                shap.dependence_plot(
                    col, np.array(shap_values)[:, :, 1], df, feature_names=df.columns
                )
