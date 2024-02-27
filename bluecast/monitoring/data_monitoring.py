"""
Module containing classes and function to monitor data drifts.

This is meant for pipelines on production.
"""

import numbers
from datetime import datetime
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from bluecast.general_utils.general_utils import logger


class DataDrift:
    """
    Monitor data drift.

    Class holding various functions to measure and visualize data drift.
    This is suitable for batch models and not recommended for online models.
    """

    def __init__(self):
        self.kolmogorov_smirnov_flags: Dict[str, bool] = {}
        self.population_stability_index_values: Dict[str, float] = {}
        self.population_stability_index_flags: Dict[str, Any] = {}

    def kolmogorov_smirnov_test(
        self,
        data: pd.DataFrame,
        new_data: pd.DataFrame,
        threshold: float = 0.05,
    ):
        """
        Checks for data drift in new data based on K-S test.

        OThe K-S test is a nonparametric test that compares the cumulative distributions of two numerical data sets.
        Only columns falling under pd.api.types.is_numeric_dtype will be considered.

        :param data: Pandas DataFrame with the original data
        :param new_data: Pandas DataFrame containing new data to compare against
        :param threshold: Threshold for the Kolmogorov-Smirnov test (default is 0.05)
        :return drift_flags: Dictionary containing flags indicating data drift for each column
        """
        logger(
            f"{datetime.utcnow()}: Start checking for data drift via Kolmogorov-Smirnov test."
        )

        for column in new_data.columns:
            # Check for numerical columns
            if pd.api.types.is_numeric_dtype(new_data[column]):
                # Perform Kolmogorov-Smirnov test for numerical columns
                #  test the null hypothesis that two samples were drawn from the same distribution
                ks_stat, p_value = ks_2samp(
                    new_data[column],
                    data[column],
                )

                if p_value < threshold:
                    self.kolmogorov_smirnov_flags[column] = (
                        True  # not drawn from same distribution
                    )
                else:
                    self.kolmogorov_smirnov_flags[column] = (
                        False  # drawn from same distribution
                    )

    def _calculate_psi(self, expected, actual, buckets=10) -> float:
        def scale_range(input, min_val, max_val):
            input += -(np.min(input))
            input /= np.max(input) / (max_val - min_val)
            input += min_val
            return input

        def sub_psi(e_perc, a_perc):
            a_perc = max(a_perc, 0.0001)
            e_perc = max(e_perc, 0.0001)
            return (e_perc - a_perc) * np.log(e_perc / a_perc)

        def psi(expected_array, actual_array, buckets):
            breakpoints = np.arange(0, buckets + 1) / buckets * 100
            breakpoints = scale_range(
                breakpoints, np.min(expected_array), np.max(expected_array)
            )
            expected_percents = np.histogram(expected_array, breakpoints)[0] / len(
                expected_array
            )
            actual_percents = np.histogram(actual_array, breakpoints)[0] / len(
                actual_array
            )
            return np.sum(
                sub_psi(expected_percents[i], actual_percents[i])
                for i in range(len(expected_percents))
            )

        psi_values = psi(expected, actual, buckets)

        return psi_values

    def population_stability_index(
        self, data: pd.DataFrame, new_data: pd.DataFrame
    ) -> Dict[str, bool]:
        """
        Checks for data drift in new, categorical data based on population stability index.

        Interpretation of PSI scores:
        - psi <= 0.1: no change or shift in the distributions of both datasets.
        - psi 0.1 < PSI <0.2: indicates a slight change or shift has occurred.
        - psi > 0.2: indicates a large shift in the distribution has occurred between both datasets

        :param data: Pandas DataFrame with the original data
        :param new_data: Pandas DataFrame containing new data to compare against
        :return drift_flags: Dictionary containing flags indicating data drift for each column
        """
        logger(
            f"{datetime.utcnow()}: Start checking for data drift via population stability index."
        )
        top_feature_list = data.columns
        for column in top_feature_list:
            if pd.api.types.is_numeric_dtype(new_data[column]):
                # Assuming you have a validation and training set
                psi_t = self._calculate_psi(data[column], new_data[column])
                self.population_stability_index_values[column] = psi_t
                self.population_stability_index_flags[column] = psi_t > 0.1

        return self.population_stability_index_flags

    def qqplot_two_samples(
        self,
        x,
        y,
        x_label: str = "X",
        y_label: str = "Y",
        quantiles=None,
        interpolation="nearest",
        ax=None,
        rug=True,
        rug_length=0.05,
        rug_kwargs=None,
        **kwargs,
    ):
        """
        Draw a quantile-quantile plot for `x` versus `y`.

        :param x: array-like one-dimensional numeric array or Pandas series
        :param y: array-like one-dimensional numeric array or Pandas series
        :param x_label: String defining the x-axis label
        :param y_label: String defining the y-axis label
        :param ax : matplotlib.axes.Axes, optional
            Axes on which to plot. If not provided, the current axes will be used.
        :param quantiles : int or array-like, optional
            Quantiles to include in the plot. This can be an array of quantiles, in
            which case only the specified quantiles of `x` and `y` will be plotted.
            If this is an int `n`, then the quantiles will be `n` evenly spaced
            points between 0 and 1. If this is None, then `min(len(x), len(y))`
            evenly spaced quantiles between 0 and 1 will be computed.
        :param interpolation: {‘linear’, ‘lower’, ‘higher’, ‘midpoint’, ‘nearest’}
            Specify the interpolation method used to find quantiles when `quantiles`
            is an int or None. See the documentation for numpy.quantile().
        :param rug: bool, optional
            If True, draw a rug plot representing both samples on the horizontal and
            vertical axes. If False, no rug plot is drawn.
        :param rug_length: float in [0, 1], optional
            Specifies the length of the rug plot lines as a fraction of the total
            vertical or horizontal length.
        :param rug_kwargs: dict of keyword arguments
            Keyword arguments to pass to matplotlib.axes.Axes.axvline() and
            matplotlib.axes.Axes.axhline() when drawing rug plots.
        :param kwargs: dict of keyword arguments
            Keyword arguments to pass to matplotlib.axes.Axes.scatter() when drawing
            the q-q plot.
        """
        # Get current axes if none are provided
        if ax is None:
            ax = plt.gca()

        if quantiles is None:
            quantiles = min(len(x), len(y))

        # Compute quantiles of the two samples
        if isinstance(quantiles, numbers.Integral):
            quantiles = np.linspace(start=0, stop=1, num=int(quantiles))
        else:
            quantiles = np.atleast_1d(np.sort(quantiles))
        x_quantiles = np.quantile(x, quantiles, interpolation=interpolation)
        y_quantiles = np.quantile(y, quantiles, interpolation=interpolation)

        # Draw the rug plots if requested
        if rug:
            # Default rug plot settings
            rug_x_params = dict(ymin=0, ymax=rug_length, c="gray", alpha=0.5)
            rug_y_params = dict(xmin=0, xmax=rug_length, c="gray", alpha=0.5)

            # Override default setting by any user-specified settings
            if rug_kwargs is not None:
                rug_x_params.update(rug_kwargs)
                rug_y_params.update(rug_kwargs)

            # Draw the rug plots
            for point in x:
                ax.axvline(point, **rug_x_params)
            for point in y:
                ax.axhline(point, **rug_y_params)

        # Draw the q-q plot
        ax.scatter(x_quantiles, y_quantiles, **kwargs)
        # Add a line representing the theoretical perfect relationship
        lims = [
            np.min(
                [ax.get_xlim(), ax.get_ylim()]
            ),  # Get the minimum of the x and y limits
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]  # Get the maximum of the x and y limits
        ax.plot(lims, lims, "k-", alpha=0.75, zorder=0)  # Plot the diagonal line
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f"QQplot of {x_label} & {y_label}")
        plt.show()
        plt.close()
