import math
from collections import Counter
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def univariate_plots(df: pd.DataFrame, target: str) -> None:
    """
    Plots univariate plots for all the columns in the dataframe.
    The target column must be part of the provided DataFrame.

    Expects numeric columns only.
    """
    for col in df.columns:
        # Check if the col is the target column (EC1 or EC2)
        if col == target:
            continue  # Skip target columns in univariate analysis

        plt.figure(figsize=(8, 4))

        # Histogram
        plt.subplot(1, 2, 1)
        sns.histplot(data=df, x=col, kde=True)
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.title("Histogram")

        # Box plot
        plt.subplot(1, 2, 2)
        sns.boxplot(data=df, y=col)
        plt.ylabel(col)
        plt.title("Box Plot")

        # Adjust spacing between subplots
        plt.tight_layout()

        # Show the plots
        plt.show()


def bi_variate_plots(df: pd.DataFrame, target: str) -> None:
    """
    Plots bivariate plots for all column combinations in the dataframe.
    The target column must be part of the provided DataFrame.

    Expects numeric columns only.
    """
    # Get the list of column names except for the target column
    variables = [col for col in df.columns if col != target]

    # Define the grid layout based on the number of variables
    num_variables = len(variables)
    num_cols = 4  # Number of columns in the grid
    num_rows = (
        num_variables + num_cols - 1
    ) // num_cols  # Calculate the number of rows needed

    # Set the size of the figure
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4 * num_rows))

    # Generate violin plots for each variable with respect to EC1
    for i, variable in enumerate(variables):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row][col]

        sns.violinplot(data=df, x=target, y=variable, ax=ax)
        ax.set_xlabel(target)
        ax.set_ylabel(variable)
        ax.set_title(f"Violin Plot: {variable} vs {target}")

    # Remove any empty subplots
    if num_variables < num_rows * num_cols:
        for i in range(num_variables, num_rows * num_cols):
            fig.delaxes(axes.flatten()[i])

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()


def correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Plots half of the heatmap showing correlations of all features.

    Expects numeric columns only.
    """
    # Calculate the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=0.3,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )

    plt.show()


def correlation_to_target(df: pd.DataFrame, target: str) -> None:
    """
    Plots correlations for all the columns in the dataframe in relation to the target column.
    The target column must be part of the provided DataFrame.

    Expects numeric columns only.
    """
    # Calculate the correlation matrix
    corr = df.corr()

    # Get correlations without 'EC1' and 'EC2'
    corrs = corr[target].drop([target])

    # Sort correlation values in descending order
    corrs_sorted = corrs.sort_values(ascending=False)

    # Create a heatmap of the correlations with EC1
    sns.set(font_scale=0.8)
    sns.set_style("white")
    sns.set_palette("PuBuGn_d")
    sns.heatmap(corrs_sorted.to_frame(), cmap="coolwarm", annot=True, fmt=".2f")
    plt.title("Correlation with EC1")
    plt.show()


def plot_pca(df: pd.DataFrame, target: str) -> None:
    """
    Plots PCA for the dataframe. The target column must be part of the provided DataFrame.

    Expects numeric columns only.
    """
    pca = PCA(n_components=2)

    pca_df = pd.DataFrame(pca.fit_transform(df.drop([target], axis=1)))
    pca_df[target] = df[target].values
    fig, ax = plt.subplots(ncols=1, figsize=(10, 5))
    explained_variance = round(sum(pca.explained_variance_ratio_), 2)

    # Define a custom color palette with distinct colors for the target variable
    target_palette = sns.color_palette("hls", len(df[target].unique()))

    for _i, col in enumerate([target]):
        sns.scatterplot(data=pca_df, x=0, y=1, hue=col, ax=ax, palette=target_palette)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

    fig.suptitle(f"PCA \n explained variance :{explained_variance}", y=1.1)
    plt.show()


def plot_tsne(df: pd.DataFrame, target: str, perplexity=50, random_state=42) -> None:
    """
    Plots t-SNE for the dataframe. The target column must be part of the provided DataFrame.

    Expects numeric columns only.
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)

    tsne_df = pd.DataFrame(tsne.fit_transform(df.drop([target], axis=1)))
    tsne_df[target] = df[target].values
    fig, ax = plt.subplots(ncols=1, figsize=(10, 5))

    # Define a custom color palette with distinct colors for the target variable
    target_palette = sns.color_palette("hls", len(df[target].unique()))

    for _i, col in enumerate([target]):
        sns.scatterplot(data=tsne_df, x=0, y=1, hue=col, ax=ax, palette=target_palette)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

    fig.suptitle("t-SNE", y=1.1)
    plt.show()


def conditional_entropy(x, y):
    # entropy of x given y
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x, y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y / p_xy)
    return entropy


def theil_u(x, y):
    s_xy = conditional_entropy(x, y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n / total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x


def plot_theil_u_heatmap(data: pd.DataFrame, columns: List[Union[str, int, float]]):
    """Plot a heatmap for categorical data using Theil's U."""
    theil_matrix = np.zeros((len(columns), len(columns)))

    for i in range(len(columns)):
        for j in range(len(columns)):
            theil_matrix[i, j] = theil_u(data[columns[i]], data[columns[j]])

    plt.figure(figsize=(len(columns), len(columns)))
    sns.heatmap(
        theil_matrix,
        annot=True,
        xticklabels=columns,
        yticklabels=columns,
        cmap="coolwarm",
    )
    plt.title("Theil's U Heatmap")

    plt.show()
    return theil_matrix


def plot_null_percentage(dataframe: pd.DataFrame) -> None:
    # Calculate the percentage of null values for each column
    null_percentage = (dataframe.isnull().mean() * 100).round(2)

    # Create a bar plot to visualize the null percentages
    plt.figure(figsize=(12, 6))
    bars = plt.bar(null_percentage.index, null_percentage.values)
    plt.title("Percentage of Null Values in Each Column")
    plt.xlabel("Columns")
    plt.ylabel("Percentage of Null Values")
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Add annotations to the bars
    for bar, percentage in zip(bars, null_percentage.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2 - 0.15,  # Adjust x-position for centering
            bar.get_height() + 1,  # Adjust y-position for vertical alignment
            f"{percentage}%",  # Display the percentage
            ha="center",  # Horizontal alignment
            va="bottom",  # Vertical alignment
        )

    # Show the plot
    plt.show()


def check_unique_values(
    df: pd.DataFrame, columns: List[Union[str, int, float]], threshold: float
) -> List[Union[str, int, float]]:
    """
    Check if the columns have an amount of unique values that is almost the number of total rows (being above the defined threshold)

    :param df: The pandas DataFrame to check
    :param columns: A list of column names to check
    :param threshold: The threshold to check against
    :returns: A list of column names that have a high amount of unique values
    """
    total_rows = len(df.index)
    lots_uniques = []
    for column in columns:
        unique_values = len(df[column].unique())
        if unique_values / total_rows >= threshold:
            lots_uniques.append(column)
    return lots_uniques
