import math
from collections import Counter
from typing import List, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def plot_pie_chart(
    df: pd.DataFrame,
    column: str,
    explode: Optional[List[float]] = None,
    colors: Optional[List[str]] = None,
) -> None:
    """
    Create a pie chart with labels, sizes, and optional explosion.

    Parameters:
    - df: Pandas DataFrame holding the column of nterest
    - column: The column to be plottted
    - explode: (Optional) List of numerical values, representing the explosion distance for each segment.
    - colors: (Optional) List with hexadecimal representations of colors in the RGB color model
    """
    value_counts = df[column].value_counts()
    sizes = value_counts.to_list()
    labels = value_counts.index.to_list()

    if explode is None:
        explode = [0.1] * len(labels)  # No explosion by default

    if not colors and len(labels) <= 50:
        colors = [
            "#ff6666",
            "#ff9966",
            "#ffb366",
            "#ffcc66",
            "#ffd966",
            "#ffeb66",
            "#ffff66",
            "#ebff66",
            "#d9ff66",
            "#b3ff66",
            "#99ff66",
            "#66ff66",
            "#66ff99",
            "#66ffcc",
            "#66ffff",
            "#66ebff",
            "#66d9ff",
            "#66b3ff",
            "#6699ff",
            "#6666ff",
            "#9966ff",
            "#cc66ff",
            "#ff66ff",
            "#ff66cc",
            "#ff6699",
            "#ff6666",
            "#ff9999",
            "#ffcc99",
            "#ffff99",
            "#ccff99",
            "#99ff99",
            "#99ffcc",
            "#99ffff",
            "#99ccff",
            "#9999ff",
            "#cc99ff",
            "#ff99ff",
            "#ff99cc",
            "#ff9999",
            "#66b3ff",
            "#66ccff",
            "#66e6ff",
            "#66f3ff",
            "#66feff",
            "#66fffb",
            "#66ffec",
            "#66ffe1",
            "#66ffd5",
            "#66ffc8",
            "#66ffba",
        ]

    # Create a pie chart
    plt.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        explode=explode,
        shadow=True,
        pctdistance=0.85,
        colors=colors,
    )

    centre_circle = plt.Circle((0, 0), 0.70, fc="white")
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    # Add a title
    plt.title(f"Distribution of column {column}")

    # Show the plot
    plt.show()


def plot_count_pair(
    df_1: pd.DataFrame,
    df_2: pd.DataFrame,
    df_aliases: Optional[List[str]],
    feature: str,
    order: Optional[List[str]] = None,
    palette: Optional[List[str]] = None,
) -> None:
    """
    Compare the counts between two DataFrames of the chosen provided categorical column.

    :param df_1: Pandas DataFrame. I.e.: df_1 dataset
    :param df_2: Pandas DataFrame. I.e.: Test dataset
    :param df_aliases: List with names of DataFrames that shall be shown on the count plots to represent them.
        Format: [df_1 representation, df_2 representation]
    :param feature: String indicating categorical column to plot
    :param hue: Read the sns.countplot
    :param order: List with category names to define the order they appear in the plot
    :param palette:  List with hexadecimal representations of colors in the RGB color model
    """
    if not df_aliases:
        df_aliases = ["train", "test"]
    data_df = df_1.copy()
    data_df["set"] = df_aliases[0]
    data_df = pd.concat([data_df, df_2.copy()]).fillna(df_aliases[1])
    f, ax = plt.subplots(1, 1, figsize=(8, 6))  # Increased height to 6

    # Create countplot
    sns.countplot(x=feature, data=data_df, hue="set", palette=palette, order=order)

    # Rotate x-axis labels by 90 degrees
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=90
    )  # ax.set_xticks(ax.get_xticks())

    # Add annotations above the bars
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
        )

    # Customize the plot
    plt.grid(color="black", linestyle="-.", linewidth=0.5, axis="y", which="major")
    ax.set_title(f"Paired {df_aliases[0]}/{df_aliases[1]}frequencies of {feature}")
    plt.show()


def plot_count_pairs(
    df_1: pd.DataFrame,
    df_2: pd.DataFrame,
    cat_cols: List[str],
    df_aliases: Optional[List[str]] = None,
    palette: Optional[List[str]] = None,
) -> None:
    """
    Compare the counts between two DataFrames of each categorical column in the provided list.

    :param df_1: Pandas DataFrame. I.e.: Train dataset
    :param df_2: Pandas DataFrame. I.e.: Test dataset
    :param df_aliases: List with names of DataFrames that shall be shown on the count plots to represent them.
        Format: [df_1 representation, df_2 representation]
    :param cat_cols: List with strings indicating categorical column names to plot
    :param palette:  List with hexadecimal representations of colors in the RGB color model
    """
    if isinstance(df_aliases, List):
        assert len(df_aliases) == 2

    for feature in cat_cols:
        order = sorted(df_1[feature].unique())
        plot_count_pair(
            df_1,
            df_2,
            df_aliases=df_aliases,
            feature=feature,
            order=order,
            palette=palette,
        )


def univariate_plots(df: pd.DataFrame) -> None:
    """
    Plots univariate plots for all the columns in the dataframe. Only numerical columns are expected.
    The target column does not need to be part of the provided DataFrame.

    Expects numeric columns only.
    """
    for col in df.columns:
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


def bi_variate_plots(df: pd.DataFrame, target: str, num_cols_grid: int = 4) -> None:
    """
    Plots bivariate plots for all column combinations in the dataframe.
    The target column must be part of the provided DataFrame.
    Param num_cols_grid specifies how many columns the grid shall have.

    Expects numeric columns only.
    """
    if target not in df.columns.to_list():
        raise ValueError("Target column must be part of the provided DataFrame")
    # Get the list of column names except for the target column
    variables = [col for col in df.columns if col != target]

    # Define the grid layout based on the number of variables
    num_variables = len(variables)
    num_cols = num_cols_grid  # Number of columns in the grid
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
    if target not in df.columns.to_list():
        raise ValueError("Target column must be part of the provided DataFrame")
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
    plt.title(f"Correlation with {target}")
    plt.show()


def plot_pca(df: pd.DataFrame, target: str, scale_data: bool = True) -> None:
    """
    Plots PCA for the dataframe. The target column must be part of the provided DataFrame.

    Expects numeric columns only.
    :param df: Pandas DataFrame. Should not include the target variable.
    :param scale_data: If true, standard scaling will be performed before applying PCA, otherwise the raw data is used.
    """
    if target not in df.columns.to_list():
        raise ValueError("Target column must be part of the provided DataFrame")

    pca = PCA(n_components=2)

    if scale_data:
        scaler = StandardScaler()
        df.drop([target], axis=1).loc[:, :] = scaler.fit_transform(
            df.drop([target], axis=1).loc[:, :]
        )
    else:
        df = df.copy()

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


def plot_pca_cumulative_variance(
    df: pd.DataFrame, scale_data: bool = True, n_components: int = 10
) -> None:
    """
    Plot the cumulative variance of principal components.

    :param df: Pandas DataFrame. Should not include the target variable.
    :param scale_data: If true, standard scaling will be performed before applying PCA, otherwise the raw data is used.
    :param n_components: Number of total components to compute.
    """
    if scale_data:
        scaler = StandardScaler()
        data_standardized = scaler.fit_transform(df.copy())
    else:
        data_standardized = df.copy()

    # Perform PCA to create components
    pca = PCA(n_components=n_components)
    pca.fit(data_standardized)
    explained_variances = pca.explained_variance_ratio_

    # Individual explained variances for 10 components
    individual_variances = explained_variances.tolist()

    # Compute the cumulative explained variance
    cumulative_variances = np.cumsum(individual_variances)
    # Create the bar plot for individual variances
    plt.figure(figsize=(12, 7))
    plot_bar = plt.bar(
        range(1, n_components + 1),
        individual_variances,
        alpha=0.6,
        color="g",
        label="Individual Explained Variance",
    )

    # Create the line plot for cumulative variance
    plt.plot(
        range(1, n_components + 1),
        cumulative_variances,
        marker="o",
        linestyle="-",
        color="r",
        label="Cumulative Explained Variance",
    )

    # Adding percentage values on top of bars and dots
    for i, (bar, cum_val) in enumerate(zip(plot_bar, cumulative_variances)):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{individual_variances[i]*100:.1f}%",
            ha="center",
            va="bottom",
        )
        plt.text(i + 1, cum_val, f"{cum_val*100:.1f}%", ha="center", va="bottom")

    # Aesthetics for the plot
    plt.xlabel("Principal Components")
    plt.ylabel("Explained Variance")
    plt.title("Explained Variance by Different Principal Components")
    plt.xticks(range(1, n_components + 1))
    plt.legend(loc="upper left")
    plt.ylim(0, 1.1)  # extend y-axis limit to accommodate text labels
    plt.grid(True)
    plt.show()


def plot_tsne(
    df: pd.DataFrame,
    target: str,
    perplexity=50,
    random_state=42,
    scale_data: bool = True,
) -> None:
    """
    Plots t-SNE for the dataframe. The target column must be part of the provided DataFrame.

    Expects numeric columns only.
    :param df: Pandas DataFrame. Should not include the target variable.
    :param target: String indicating which column is the target column. Must be part of the provided DataFrame.
    :param perplexity: The perplexity parameter for t-SNE
    :param random_state: The random state for t-SNE
    :param scale_data: If true, standard scaling will be performed before applying t-SNE, otherwise the raw data is used.
    """
    if target not in df.columns.to_list():
        raise ValueError("Target column must be part of the provided DataFrame")

    if scale_data:
        scaler = StandardScaler()
        df.drop([target], axis=1).loc[:, :] = scaler.fit_transform(
            df.drop([target], axis=1).loc[:, :]
        )
    else:
        df = df.copy()

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


def mutual_info_to_target(
    df: pd.DataFrame,
    target: str,
    class_problem: Literal["binary", "multiclass", "regression"],
    **mut_params,
) -> None:
    """
    Plots mutual information scores for all the categorical columns in the DataFrame in relation to the target column.
    The target column must be part of the provided DataFrame.
    :param df: DataFrame containing all columns including target column. Features are expected to be numerical.
    :param target: String indicating which column is the target column.
    :param class_problem: Any of ["binary", "multiclass", "regression"]
    :param mut_params: Dictionary passing additional arguments into sklearn's mutual_info_classif function.

    To be used for classification only.
    """
    if target not in df.columns.to_list():
        raise ValueError("Target column must be part of the provided DataFrame")
    if class_problem in ["binary", "multiclass"]:
        mi_scores = mutual_info_classif(
            X=df.drop(columns=[target]), y=df[target], **mut_params
        )
    else:
        mi_scores = mutual_info_regression(
            X=df.drop(columns=[target]), y=df[target], **mut_params
        )

    # Sort features by MI score descending
    sorted_features = df.drop(columns=[target]).columns[np.argsort(-mi_scores)]

    # Sort MI scores in descending order
    mi_scores_sorted = mi_scores[np.argsort(-mi_scores)]

    # Create a bar chart of the mutual information scores
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=mi_scores_sorted, y=sorted_features, order=sorted_features, ax=ax)
    ax.set_title("Mutual Information Scores with Target")
    ax.set_xlabel("Mutual Information Score")
    ax.set_ylabel("Features")
    for i, v in enumerate(mi_scores_sorted):
        ax.text(v + 0.01, i, str(round(v, 2)), color="blue", fontweight="bold")
    plt.show()


def plot_ecdf(
    df: pd.DataFrame,
    columns: List[Union[str, int, float]],
    plot_all_at_once: bool = False,
) -> None:
    """
    Plot the empirical cumulative density function.

    Matplotlib contains a direct implementation at version 3.8 and higher, but
    this might run into dependency issues in environments with older data.

    :param df: DataFrame containing all columns including target column. Features are expected to be numerical.
    :param columns: A list of column names to check.
    :param plot_all_at_once: If True, plot all eCDFs in one plot. If False, plot each eCDF separately.
    """
    if plot_all_at_once:
        fig, ax = plt.subplots()
        for col in columns:
            sorted_col = np.sort(df[col])
            y = np.arange(1, len(sorted_col) + 1) / len(sorted_col)
            ax.plot(sorted_col, y, label=col)
        ax.set_xlabel("Value")
        ax.set_ylabel("ECDF")
        ax.legend()
        plt.show()
    else:
        for col in columns:
            plt.plot(
                np.sort(df[col]),
                np.linspace(0, 1, len(df[col]), endpoint=False),
                label=col,
            )
            plt.xlabel("Value")
            plt.ylabel("ECDF")
            plt.legend()
            plt.show()
