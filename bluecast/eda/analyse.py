import math
import warnings
from collections import Counter
from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as ss
import statsmodels.api as sm
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")


def find_bind_with_with_freedman_diaconis(data: np.ndarray):
    # Calculate the IQR
    iqr = np.percentile(data, 75) - np.percentile(data, 25)

    # Calculate the bin width using the Freedman-Diaconis rule
    bin_width_fd = 2 * iqr / np.power(len(data), 1 / 3)
    return bin_width_fd


def plot_pie_chart(
    df: pd.DataFrame,
    column: str,
    explode: Optional[List[float]] = None,
    colors: Optional[List[str]] = None,
    show: bool = True,
) -> go.Figure:
    """
    Create a pie chart with labels, sizes, and optional explosion.

    Parameters:
    - df: Pandas DataFrame holding the column of interest
    - column: The column to be plotted
    - explode: (Optional) List of numerical values (not used in plotly version)
    - colors: (Optional) List with hexadecimal representations of colors in the RGB color model
    - show: Whether to display the plot

    Returns:
    - plotly.graph_objects.Figure: The pie chart figure
    """
    value_counts = df[column].value_counts()
    sizes = value_counts.to_list()
    labels = value_counts.index.to_list()

    if not colors and len(labels) <= 50:
        colors = px.colors.qualitative.Set3

    # Create a pie chart
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=sizes,
                hole=0.3,
                textposition="inside",
                textinfo="percent+label",
            )
        ]
    )

    if colors:
        fig.update_traces(marker=dict(colors=colors))

    # Add a title
    fig.update_layout(title=f"Distribution of column {column}")

    if show:
        fig.show()
    return fig


def plot_count_pair(
    df_1: pd.DataFrame,
    df_2: pd.DataFrame,
    df_aliases: Optional[List[str]],
    feature: str,
    order: Optional[List[str]] = None,
    palette: Optional[List[str]] = None,
    show: bool = True,
) -> go.Figure:
    """
    Compare the counts between two DataFrames of the chosen provided categorical column.

    :param df_1: Pandas DataFrame. I.e.: df_1 dataset
    :param df_2: Pandas DataFrame. I.e.: Test dataset
    :param df_aliases: List with names of DataFrames that shall be shown on the count plots to represent them.
        Format: [df_1 representation, df_2 representation]
    :param feature: String indicating categorical column to plot
    :param order: List with category names to define the order they appear in the plot
    :param palette: List with hexadecimal representations of colors in the RGB color model
    :param show: Whether to display the plot

    Returns:
    - plotly.graph_objects.Figure: The count plot figure
    """
    if not df_aliases:
        df_aliases = ["train", "test"]

    data_df_1 = df_1.copy()
    data_df_1["set"] = df_aliases[0]
    data_df_2 = df_2.copy()
    data_df_2["set"] = df_aliases[1]
    data_df = pd.concat([data_df_1, data_df_2]).reset_index(drop=True)

    if not palette:
        palette = px.colors.qualitative.Set1

    fig = px.histogram(
        data_df,
        x=feature,
        color="set",
        barmode="group",
        color_discrete_sequence=palette,
        category_orders={feature: order} if order else None,
    )

    # Customize the plot
    fig.update_layout(
        title=f"Paired {df_aliases[0]}/{df_aliases[1]} frequencies of {feature}",
        xaxis_title=feature,
        yaxis_title="Count",
        xaxis_tickangle=90,
    )

    if show:
        fig.show()
    return fig


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
    :param palette: List with hexadecimal representations of colors in the RGB color model
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


def univariate_plots(df: pd.DataFrame, col_requires_at_least_n_values: int = 5) -> None:
    """
    Plots univariate plots for all the columns in the dataframe. Only numerical columns are expected.
    The target column does not need to be part of the provided DataFrame.

    Expects numeric columns only. The number of bins will be determined using the Freedman-Diaconis rule.

    :param df: DataFrame holding the features.
    :param col_requires_at_least_n_values: Minimum number of unique values required to plot the feature.
        If number of unique features is less, the column will be skipped.
    """
    for col in df.columns.to_list():
        if df[col].nunique() >= col_requires_at_least_n_values:
            nb_bins = len(
                np.arange(
                    min(df[col]),
                    max(df[col]),
                    max(find_bind_with_with_freedman_diaconis(df[col].values), 0.1),
                )
            )

            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=("Histogram", "Box Plot"),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]],
            )

            # Histogram
            fig.add_trace(
                go.Histogram(
                    x=df[col],
                    nbinsx=min(nb_bins, 50),
                    name="Histogram",
                    marker_color="lightblue",
                    opacity=0.7,
                ),
                row=1,
                col=1,
            )

            # Box plot
            fig.add_trace(
                go.Box(
                    y=df[col],
                    name="Box Plot",
                    marker_color="lightcoral",
                    opacity=0.7,
                ),
                row=1,
                col=2,
            )

            fig.update_xaxes(title_text=col, row=1, col=1)
            fig.update_yaxes(title_text="Frequency", row=1, col=1)
            fig.update_yaxes(title_text=col, row=1, col=2)

            fig.update_layout(
                title_text=f"Univariate Analysis: {col}", showlegend=False
            )

            fig.show()


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
    num_rows = (num_variables + num_cols - 1) // num_cols

    # Create subplots
    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=variables,
        specs=[
            [{"secondary_y": False} for _ in range(num_cols)] for _ in range(num_rows)
        ],
    )

    # Define a color palette for the categories
    unique_categories = df[target].unique()
    colors = px.colors.qualitative.Set1[: len(unique_categories)]

    # Generate violin plots for each variable with respect to the target
    for i, variable in enumerate(variables):
        row = i // num_cols + 1
        col = i % num_cols + 1

        for j, category in enumerate(unique_categories):
            data_subset = df[df[target] == category]
            fig.add_trace(
                go.Violin(
                    y=data_subset[variable],
                    name=f"{category}",
                    line_color=colors[j],
                    showlegend=(i == 0),  # Only show legend for first subplot
                ),
                row=row,
                col=col,
            )

        fig.update_xaxes(title_text=target, row=row, col=col)
        fig.update_yaxes(title_text=variable, row=row, col=col)

    fig.update_layout(
        title_text="Bivariate Analysis", height=300 * num_rows, showlegend=True
    )

    fig.show()


def correlation_heatmap(df: pd.DataFrame, show: bool = True) -> go.Figure:
    """
    Plots half of the heatmap showing correlations of all features.

    Expects numeric columns only.

    Returns:
    - plotly.graph_objects.Figure: The correlation heatmap figure
    """
    # Calculate the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Apply mask
    corr_masked = corr.copy()
    corr_masked[mask] = np.nan

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_masked.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu",
            zmid=0,
            text=corr_masked.round(2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title="Correlation Heatmap", xaxis_title="Features", yaxis_title="Features"
    )

    if show:
        fig.show()
    return fig


def correlation_to_target(
    df: pd.DataFrame, target: str, show: bool = True
) -> go.Figure:
    """
    Plots correlations for all the columns in the dataframe in relation to the target column.
    The target column must be part of the provided DataFrame.

    Expects numeric columns only.

    Returns:
    - plotly.graph_objects.Figure: The correlation to target figure
    """
    if target not in df.columns.to_list():
        raise ValueError("Target column must be part of the provided DataFrame")

    # Calculate the correlation matrix
    corr = df.corr()

    # Get correlations without target
    corrs = corr[target].drop([target])

    # Sort correlation values in descending order
    corrs_sorted = corrs.sort_values(ascending=False)

    # Create a heatmap of the correlations
    fig = go.Figure(
        data=go.Heatmap(
            z=[corrs_sorted.values],
            x=corrs_sorted.index,
            y=[f"Correlation with {target}"],
            colorscale="RdBu",
            zmid=0,
            text=[corrs_sorted.round(2).values],
            texttemplate="%{text}",
            textfont={"size": 10},
        )
    )

    fig.update_layout(
        title=f"Correlation with {target}",
        xaxis_title="Features",
        yaxis_title="",
        height=200,
    )

    if show:
        fig.show()
    return fig


def plot_against_target_for_regression(
    df: pd.DataFrame,
    num_columns: List[Union[int, float, str]],
    target_col: str,
    show: bool = True,
) -> go.Figure:
    """
    Creates scatter plots for each column in num_columns against the target_col.
    Draws a regression line and shows the p-value for the regression line.

    Parameters:
    - df: pd.DataFrame -> The input dataframe containing the data.
    - num_columns: List[Union[int, float, str]] -> List of column names to plot against the target column.
    - target_col: str -> The target column name for regression.
    - show: Whether to display the plot

    Returns:
    - plotly.graph_objects.Figure: The regression plots figure
    """

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' must be part of the provided DataFrame"
        )

    num_cols_grid = 2  # Set the number of columns for the grid layout
    num_variables = len(num_columns)
    num_rows = (num_variables + num_cols_grid - 1) // num_cols_grid

    # Create subplots
    fig = make_subplots(
        rows=num_rows,
        cols=num_cols_grid,
        subplot_titles=[f"{col} vs {target_col}" for col in num_columns],
    )

    for i, column in enumerate(num_columns):
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        row = i // num_cols_grid + 1
        col = i % num_cols_grid + 1

        x = df[column]
        y = df[target_col]

        # Scatter plot
        fig.add_trace(
            go.Scatter(x=x, y=y, mode="markers", name=f"{column}", showlegend=False),
            row=row,
            col=col,
        )

        # Fit a regression line
        X = sm.add_constant(x)  # Adds a constant term to the predictor
        model = sm.OLS(y, X).fit()
        prediction = model.predict(X)

        # Plot the regression line
        sorted_indices = np.argsort(x)
        fig.add_trace(
            go.Scatter(
                x=x.iloc[sorted_indices],
                y=prediction[sorted_indices],
                mode="lines",
                name="Regression Line",
                line=dict(color="red"),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        # Calculate and show the p-value
        p_value = model.pvalues[1]
        fig.add_annotation(
            text=f"p-value: {p_value:.4f}",
            xref=f"x{i + 1}",
            yref=f"y{i + 1}",
            x=0.05,
            y=0.95,
            showarrow=False,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            row=row,
            col=col,
        )

        fig.update_xaxes(title_text=column, row=row, col=col)
        fig.update_yaxes(title_text=target_col, row=row, col=col)

    fig.update_layout(
        title_text="Regression Analysis", height=400 * num_rows, showlegend=False
    )

    if show:
        fig.show()
    return fig


def plot_pca(
    df: pd.DataFrame, target: str, scale_data: bool = True, show: bool = True
) -> go.Figure:
    """
    Plots PCA for the dataframe. The target column must be part of the provided DataFrame.

    Expects numeric columns only.
    :param df: Pandas DataFrame. Should include the target variable.
    :param target: String indicating the target column.
    :param scale_data: If true, standard scaling will be performed before applying PCA, otherwise the raw data is used.
    :param show: Whether to display the plot

    Returns:
    - plotly.graph_objects.Figure: The PCA plot figure
    """
    if target not in df.columns.to_list():
        raise ValueError("Target column must be part of the provided DataFrame")

    df_features = df.drop([target], axis=1)

    if scale_data:
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df_features),
            columns=df_features.columns,
            index=df_features.index,
        )
    else:
        df_scaled = df_features.copy()

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_scaled)

    explained_variance = round(sum(pca.explained_variance_ratio_), 2)

    # Create PCA plot
    fig = px.scatter(
        x=pca_result[:, 0],
        y=pca_result[:, 1],
        color=df[target],
        title=f"PCA - Explained Variance: {explained_variance}",
        labels={"x": "Component 1", "y": "Component 2"},
    )

    if show:
        fig.show()
    return fig


def plot_pca_cumulative_variance(
    df: pd.DataFrame, scale_data: bool = True, n_components: int = 10, show: bool = True
) -> go.Figure:
    """
    Plot the cumulative variance of principal components.

    :param df: Pandas DataFrame. Should not include the target variable.
    :param scale_data: If true, standard scaling will be performed before applying PCA, otherwise the raw data is used.
    :param n_components: Number of total components to compute.
    :param show: Whether to display the plot

    Returns:
    - plotly.graph_objects.Figure: The PCA cumulative variance figure
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

    # Individual explained variances
    individual_variances = explained_variances.tolist()

    # Compute the cumulative explained variance
    cumulative_variances = np.cumsum(individual_variances)

    components = list(range(1, n_components + 1))

    # Create figure
    fig = go.Figure()

    # Add bar chart for individual variances
    fig.add_trace(
        go.Bar(
            x=components,
            y=individual_variances,
            name="Individual Explained Variance",
            marker_color="lightblue",
            opacity=0.7,
            text=[f"{var * 100:.1f}%" for var in individual_variances],
            textposition="outside",
        )
    )

    # Add line chart for cumulative variances
    fig.add_trace(
        go.Scatter(
            x=components,
            y=cumulative_variances,
            mode="lines+markers",
            name="Cumulative Explained Variance",
            line=dict(color="red"),
            text=[f"{var * 100:.1f}%" for var in cumulative_variances],
            textposition="top center",
        )
    )

    fig.update_layout(
        title="Explained Variance by Different Principal Components",
        xaxis_title="Principal Components",
        yaxis_title="Explained Variance",
        yaxis=dict(range=[0, 1.1]),
        showlegend=True,
    )

    if show:
        fig.show()
    return fig


def plot_pca_biplot(
    df: pd.DataFrame, target: str, scale_data: bool = True, show: bool = True
) -> go.Figure:
    """
    Plots PCA biplot for the dataframe.

    Expects numeric columns only.

    :param df: Pandas DataFrame.
    :param target: String indicating the target column. Will be dropped if part of the DataFrame.
    :param scale_data: If true, standard scaling will be performed before applying PCA, otherwise the raw data is used.
    :param show: Whether to display the plot

    Returns:
    - plotly.graph_objects.Figure: The PCA biplot figure
    """
    if target in df.columns.to_list():
        df_features = df.drop(target, axis=1)
    else:
        df_features = df.copy()

    if scale_data:
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df_features), columns=df_features.columns
        )
    else:
        df_scaled = df_features.copy()

    pca = PCA(n_components=2)
    pca.fit(df_scaled)

    labels = df_features.columns
    coeff = np.transpose(pca.components_)

    fig = go.Figure()

    # Add arrows and labels
    for i, label in enumerate(labels):
        fig.add_trace(
            go.Scatter(
                x=[0, coeff[i, 0]],
                y=[0, coeff[i, 1]],
                mode="lines",
                line=dict(color="black", width=2),
                showlegend=False,
            )
        )

        fig.add_annotation(
            x=coeff[i, 0] * 1.15,
            y=coeff[i, 1] * 1.15,
            text=label,
            showarrow=False,
            font=dict(size=12),
        )

    # Add unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    fig.add_trace(
        go.Scatter(
            x=np.cos(theta),
            y=np.sin(theta),
            mode="lines",
            line=dict(color="gray", dash="dash"),
            showlegend=False,
        )
    )

    fig.update_layout(
        title="Dataset PCA Biplot",
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        xaxis=dict(range=[-1.2, 1.2]),
        yaxis=dict(range=[-1.2, 1.2]),
        width=600,
        height=600,
    )

    # Add grid lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")

    if show:
        fig.show()
    return fig


def plot_tsne(
    df: pd.DataFrame,
    target: str,
    perplexity=50,
    random_state=42,
    scale_data: bool = True,
    show: bool = True,
) -> go.Figure:
    """
    Plots t-SNE for the dataframe. The target column must be part of the provided DataFrame.

    Expects numeric columns only.
    :param df: Pandas DataFrame. Should include the target variable.
    :param target: String indicating which column is the target column. Must be part of the provided DataFrame.
    :param perplexity: The perplexity parameter for t-SNE
    :param random_state: The random state for t-SNE
    :param scale_data: If true, standard scaling will be performed before applying t-SNE, otherwise the raw data is used.
    :param show: Whether to display the plot

    Returns:
    - plotly.graph_objects.Figure: The t-SNE plot figure
    """
    if target not in df.columns.to_list():
        raise ValueError("Target column must be part of the provided DataFrame")

    df_features = df.drop([target], axis=1)

    if scale_data:
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_features)
    else:
        df_scaled = df_features.values

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    tsne_result = tsne.fit_transform(df_scaled)

    # Create t-SNE plot
    fig = px.scatter(
        x=tsne_result[:, 0],
        y=tsne_result[:, 1],
        color=df[target],
        title="t-SNE Visualization",
        labels={"x": "Component 1", "y": "Component 2"},
    )

    if show:
        fig.show()
    return fig


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


def plot_theil_u_heatmap(
    data: pd.DataFrame, columns: List[Union[str, int, float]], show: bool = True
) -> go.Figure:
    """Plot a heatmap for categorical data using Theil's U.

    Returns:
    - plotly.graph_objects.Figure: The Theil's U heatmap figure
    """
    theil_matrix = np.zeros((len(columns), len(columns)))

    for i in range(len(columns)):
        for j in range(len(columns)):
            theil_matrix[i, j] = theil_u(data[columns[i]], data[columns[j]])

    fig = go.Figure(
        data=go.Heatmap(
            z=theil_matrix,
            x=columns,
            y=columns,
            colorscale="RdBu",
            text=np.round(theil_matrix, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
        )
    )

    fig.update_layout(
        title="Theil's U Heatmap", xaxis_title="Features", yaxis_title="Features"
    )

    if show:
        fig.show()
    return fig, theil_matrix


def plot_null_percentage(dataframe: pd.DataFrame, show: bool = True) -> go.Figure:
    """
    Plot the percentage of null values in each column.

    Returns:
    - plotly.graph_objects.Figure: The null percentage plot figure
    """
    # Calculate the percentage of null values for each column
    null_percentage = (dataframe.isnull().mean() * 100).round(2)

    # Create a bar plot to visualize the null percentages
    fig = go.Figure(
        data=[
            go.Bar(
                x=null_percentage.index,
                y=null_percentage.values,
                text=[f"{val}%" for val in null_percentage.values],
                textposition="outside",
            )
        ]
    )

    fig.update_layout(
        title="Percentage of Null Values in Each Column",
        xaxis_title="Columns",
        yaxis_title="Percentage of Null Values",
        xaxis_tickangle=90,
    )

    if show:
        fig.show()
    return fig


def check_unique_values(
    df: pd.DataFrame, columns: List[Union[str, int, float]], threshold: float = 0.9
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


def plot_classification_target_distribution_within_categories(
    df: pd.DataFrame, cat_columns: List[str], target_col: str
) -> None:
    """
    Plot distribution of target across categorical features.

    This suitable for classification tasks only.
    :param df: Pandas dataFrame. Must include the target column.
    :param cat_columns: List of categorical column names.
    :param target_col: String indicating the target column name.
    :return:
    """
    if target_col not in df.columns.to_list():
        raise KeyError("Target column must be part of the provided DataFrame")

    for col in cat_columns:
        contingency_table = pd.crosstab(df[col], df[target_col], normalize="index")

        fig = go.Figure()

        # Add traces for each target class
        for _i, target_class in enumerate(contingency_table.columns):
            fig.add_trace(
                go.Bar(
                    x=contingency_table.index,
                    y=contingency_table[target_class],
                    name=f"Class {target_class}",
                    text=contingency_table[target_class].round(2),
                    textposition="auto",
                )
            )

        fig.update_layout(
            title=f"Percentage Distribution of Target across {col}",
            xaxis_title=col,
            yaxis_title="Percentage",
            barmode="stack",
            showlegend=True,
        )

        fig.show()


def mutual_info_to_target(
    df: pd.DataFrame,
    target: str,
    class_problem: Literal["binary", "multiclass", "regression"],
    show: bool = True,
    **mut_params,
) -> go.Figure:
    """
    Plots mutual information scores for all the categorical columns in the DataFrame in relation to the target column.
    The target column must be part of the provided DataFrame.
    :param df: DataFrame containing all columns including target column. Features are expected to be numerical.
    :param target: String indicating which column is the target column.
    :param class_problem: Any of ["binary", "multiclass", "regression"]
    :param show: Whether to display the plot
    :param mut_params: Dictionary passing additional arguments into sklearn's mutual_info_classif function.

    Returns:
    - plotly.graph_objects.Figure: The mutual information plot figure
    """
    if target not in df.columns.to_list():
        raise ValueError("Target column must be part of the provided DataFrame")

    # Compute the mutual information scores
    if class_problem in ["binary", "multiclass"]:
        mi_scores = mutual_info_classif(
            X=df.drop(columns=[target]), y=df[target], **mut_params
        )
    else:
        mi_scores = mutual_info_regression(
            X=df.drop(columns=[target]), y=df[target], **mut_params
        )

    # Sort features by MI score descending
    feature_names = df.drop(columns=[target]).columns
    sorted_indices = np.argsort(-mi_scores)
    sorted_features = feature_names[sorted_indices]
    mi_scores_sorted = mi_scores[sorted_indices]

    # Create a horizontal bar chart
    fig = go.Figure(
        data=[
            go.Bar(
                x=mi_scores_sorted,
                y=sorted_features,
                orientation="h",
                text=np.round(mi_scores_sorted, 3),
                textposition="outside",
            )
        ]
    )

    fig.update_layout(
        title="Mutual Information Scores with Target",
        xaxis_title="Mutual Information Score",
        yaxis_title="Features",
        height=max(400, len(sorted_features) * 25),
    )

    if show:
        fig.show()
    return fig


def plot_ecdf(
    df: pd.DataFrame,
    columns: List[Union[str, int, float]],
    plot_all_at_once: bool = False,
    show: bool = True,
) -> Union[go.Figure, List[go.Figure]]:
    """
    Plot the empirical cumulative density function (ECDF) and histogram.

    :param df: DataFrame containing all columns including target column. Features are expected to be numerical.
    :param columns: A list of column names to check.
    :param plot_all_at_once: If True, plot all eCDFs in one plot. If False, plot each eCDF separately.
    :param show: Whether to display the plot

    Returns:
    - plotly.graph_objects.Figure or List[plotly.graph_objects.Figure]: The ECDF figure(s)
    """
    if plot_all_at_once:
        fig = go.Figure()

        for col in columns:
            sorted_col = np.sort(df[col])
            y = np.arange(1, len(sorted_col) + 1) / len(sorted_col)

            fig.add_trace(go.Scatter(x=sorted_col, y=y, mode="lines", name=col))

        fig.update_layout(
            title="Empirical Cumulative Distribution Function",
            xaxis_title="Value",
            yaxis_title="ECDF",
        )

        if show:
            fig.show()
        return fig
    else:
        figures = []
        for col in columns:
            nb_bins = len(
                np.arange(
                    min(df[col]),
                    max(df[col]),
                    max(find_bind_with_with_freedman_diaconis(df[col].values), 0.1),
                )
            )

            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=("ECDF", "Histogram"),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]],
            )

            # Plot ECDF
            sorted_col = np.sort(df[col])
            ecdf_y = np.arange(1, len(sorted_col) + 1) / len(sorted_col)

            fig.add_trace(
                go.Scatter(
                    x=sorted_col,
                    y=ecdf_y,
                    mode="lines",
                    name="ECDF",
                    line=dict(color="blue"),
                ),
                row=1,
                col=1,
            )

            # Add histogram
            fig.add_trace(
                go.Histogram(
                    x=df[col],
                    nbinsx=min(nb_bins, 50),
                    name="Histogram",
                    opacity=0.7,
                    marker_color="lightcoral",
                ),
                row=1,
                col=2,
            )

            fig.update_xaxes(title_text="Value", row=1, col=1)
            fig.update_yaxes(title_text="ECDF", row=1, col=1)
            fig.update_xaxes(title_text="Value", row=1, col=2)
            fig.update_yaxes(title_text="Frequency", row=1, col=2)

            fig.update_layout(
                title_text=f"ECDF and Histogram for {col}", showlegend=False
            )

            if show:
                fig.show()
            figures.append(fig)

        return figures


def plot_distribution_by_time(
    df: pd.DataFrame,
    col_to_plot: str,
    date_col: str,
    xlabel: str = "Week",
    ylabel: str = "Feature distribution",
    title: str = "Weekly distribution of the feature",
    freq: str = "W",
    show: bool = True,
) -> go.Figure:
    """
    Plot the distribution of a feature over time.

    :param df: Pandas DataFrame
    :param col_to_plot: String indicating which column to plot
    :param date_col: String indicating which column to use as date
    :param xlabel: String indicating the x-axis label
    :param ylabel: String indicating the y-axis label
    :param title: String indicating the title of the plot
    :param freq: Label indicating the frequency of the time grouping. Must be one of Pandas' Offset aliases.
    :param show: Whether to display the plot
    :return: plotly.graph_objects.Figure: The time distribution figure
    """
    # Convert date column to datetime and set as index
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    df_copy = df_copy.set_index(date_col)

    # Group by time frequency and calculate distribution
    grouped = df_copy.groupby(pd.Grouper(freq=freq))[col_to_plot].apply(list)

    # Prepare data for box plot
    fig = go.Figure()

    for _i, (date, values) in enumerate(grouped.items()):
        if len(values) > 0:  # Only plot if there are values
            fig.add_trace(
                go.Box(y=values, name=date.strftime("%Y-%m-%d"), showlegend=False)
            )

    fig.update_layout(
        title=title, xaxis_title=xlabel, yaxis_title=ylabel, xaxis_tickangle=45
    )

    if show:
        fig.show()
    return fig


def plot_error_distributions(
    df: pd.DataFrame,
    target: str,
    prediction_error: str,
    num_cols_grid: int = 1,
    max_x_elements: int = 5,
) -> None:
    """
    Plots bivariate plots for each column in the dataframe with respect to the target.
    Each subplot represents unique values of the target column.
    The 'prediction_error' is plotted using unique values of the target column as the hue.
    Param num_cols_grid specifies how many columns the grid shall have.
    max_x_elements determines the maximum number of unique values on the x-axis per plot.
    """
    if target not in df.columns.to_list():
        raise ValueError("Target column must be part of the provided DataFrame")
    if prediction_error not in df.columns.to_list():
        raise ValueError(
            "Prediction error column must be part of the provided DataFrame"
        )

    # Get the list of column names except for the target and prediction error columns
    variables = [
        col
        for col in df.columns
        if col not in [target, prediction_error, "prediction", "predictions"]
    ]

    # Generate plots for each variable
    for variable in variables:
        unique_values = sorted(df[variable].unique())
        unique_values_count = len(unique_values)

        # Split the plot into multiple figures if x-axis elements exceed the threshold
        if unique_values_count > max_x_elements:
            num_splits = (unique_values_count + max_x_elements - 1) // max_x_elements

            for split_index in range(num_splits):
                subset_values = unique_values[
                    split_index * max_x_elements : (split_index + 1) * max_x_elements
                ]
                df_subset = df[df[variable].isin(subset_values)]

                fig = px.violin(
                    df_subset,
                    x=variable,
                    y=prediction_error,
                    color=target,
                    title=f"Violin Plot: {variable} vs {prediction_error} (Split {split_index + 1})",
                )

                fig.update_xaxes(tickangle=90)
                fig.show()

        else:
            # If the number of unique values is within the limit, plot normally
            fig = px.violin(
                df,
                x=variable,
                y=prediction_error,
                color=target,
                title=f"Violin Plot: {variable} vs {prediction_error}",
            )

            fig.update_xaxes(tickangle=90)
            fig.show()


def plot_andrews_curve(
    df: pd.DataFrame,
    target: str,
    n_samples: Optional[int] = 200,
    random_state=500,
    show: bool = True,
) -> go.Figure:
    """
    Plot Andrews curve.

    Andrews Curve helps visualize if there are inherent groupings of the numerical features based on a given grouping.

    :param df: Pandas DataFrame
    :param target: String indicating the target column
    :param n_samples: Int indicating how many samples shall be shown. If None, the full DataFrame is taken.
    :param random_state: Random seed determining the DataFrame sampling.
    :param show: Whether to display the plot
    :return: plotly.graph_objects.Figure: The Andrews curve figure
    """
    if target not in df.columns.to_list():
        raise KeyError("Target column must be part of the provided DataFrame")

    if isinstance(n_samples, int):
        if n_samples >= len(df.index):
            n_samples = len(df.index)
    else:
        n_samples = len(df.index)

    df_sample = df.sample(n_samples, random_state=random_state)

    # Andrews curves implementation for plotly
    # This is a simplified version - for full Andrews curves, use pandas.plotting.andrews_curves
    fig = go.Figure()

    numeric_cols = (
        df_sample.select_dtypes(include=[np.number]).drop(columns=[target]).columns
    )
    t = np.linspace(-np.pi, np.pi, 150)

    for target_val in df_sample[target].unique():
        subset = df_sample[df_sample[target] == target_val]

        for idx in subset.index[: min(50, len(subset))]:  # Limit for performance
            row = subset.loc[idx, numeric_cols].values
            # Simplified Andrews function
            y = row[0] / np.sqrt(2)
            for i in range(1, len(row)):
                if i % 2 == 1:
                    y += row[i] * np.sin((i + 1) // 2 * t)
                else:
                    y += row[i] * np.cos(i // 2 * t)

            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=y,
                    mode="lines",
                    name=f"Class {target_val}",
                    showlegend=bool(
                        idx == subset.index[0]
                    ),  # Only show legend for first trace of each class
                    opacity=0.7,
                )
            )

    fig.update_layout(
        title="Andrews Curves",
        xaxis_title="t",
        yaxis_title="f(t)",
        xaxis=dict(range=[-3.2, 3.2]),
    )

    if show:
        fig.show()
    return fig


def plot_distribution_pairs(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    feature: str,
    palette: Optional[List[str]] = None,
    show: bool = True,
) -> go.Figure:
    """
    Compare distributions of two datasets for a given feature.

    Only the central 95% of the data is considered for the histogram.

    :param df1: DataFrame containing the feature.
    :param df2: Second DataFrame containing the feature for comparison.
    :param feature: String indicating the feature name
    :param palette: List of colors to use for the plots.
    :param show: Whether to display the plot

    Returns:
    - plotly.graph_objects.Figure: The distribution comparison figure
    """
    if not palette:
        palette = px.colors.qualitative.Set1

    # Prepare data
    df1_copy = df1.copy()
    df1_copy["dataset"] = "Dataset 1"
    df2_copy = df2.copy()
    df2_copy["dataset"] = "Dataset 2"

    combined_df = pd.concat([df1_copy, df2_copy])

    # Filter to central 95% for each dataset
    filtered_data = []
    for dataset_name in ["Dataset 1", "Dataset 2"]:
        subset = combined_df[combined_df["dataset"] == dataset_name][feature]
        q_025, q_975 = np.percentile(subset, [2.5, 97.5])
        filtered_subset = subset[(subset >= q_025) & (subset <= q_975)]
        filtered_data.append(filtered_subset)

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Distribution Comparison", "Box Plot Comparison"),
    )

    # Add histograms
    for i, (data, name) in enumerate(zip(filtered_data, ["Dataset 1", "Dataset 2"])):
        fig.add_trace(
            go.Histogram(x=data, name=name, opacity=0.7, marker_color=palette[i]),
            row=1,
            col=1,
        )

    # Add box plots
    fig.add_trace(
        go.Box(y=filtered_data[0], name="Dataset 1", marker_color=palette[0]),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Box(y=filtered_data[1], name="Dataset 2", marker_color=palette[1]),
        row=1,
        col=2,
    )

    fig.update_layout(
        title=f"Distribution Comparison: {feature}", showlegend=True, barmode="overlay"
    )

    fig.update_xaxes(title_text=feature, row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text=feature, row=1, col=2)

    if show:
        fig.show()
    return fig


# Dashboard functionality
def create_eda_dashboard(
    df: pd.DataFrame, target_col: str, port: int = 8050, run_server: bool = True
):
    """
    Create a Dash dashboard for exploratory data analysis.

    :param df: DataFrame to analyze
    :param target_col: Target column name
    :param port: Port number for the dashboard
    :param run_server: Whether to start the server (set to False for testing)
    """
    try:
        import dash
        from dash import Input, Output, dcc, html
    except ImportError:
        raise ImportError(
            "Dash is required for dashboard functionality. Install with: pip install dash"
        )

    app = dash.Dash(__name__)

    # Get numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)

    app.layout = html.Div(
        [
            html.H1(
                "Exploratory Data Analysis Dashboard", style={"textAlign": "center"}
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Select Plot Type:"),
                            dcc.Dropdown(
                                id="plot-type",
                                options=[
                                    {
                                        "label": "Correlation Heatmap",
                                        "value": "correlation",
                                    },
                                    {
                                        "label": "Distribution Plot",
                                        "value": "distribution",
                                    },
                                    {"label": "PCA Analysis", "value": "pca"},
                                    {"label": "Box Plot", "value": "boxplot"},
                                    {"label": "Scatter Plot", "value": "scatter"},
                                ],
                                value="correlation",
                            ),
                        ],
                        style={"width": "48%", "display": "inline-block"},
                    ),
                    html.Div(
                        [
                            html.Label("Select Feature:"),
                            dcc.Dropdown(
                                id="feature-dropdown",
                                options=[
                                    {"label": col, "value": col}
                                    for col in numeric_cols + categorical_cols
                                ],
                                value=numeric_cols[0] if numeric_cols else None,
                            ),
                        ],
                        style={
                            "width": "48%",
                            "float": "right",
                            "display": "inline-block",
                        },
                    ),
                ]
            ),
            dcc.Graph(id="main-plot"),
            html.Div([html.H3("Dataset Summary"), html.Div(id="summary-stats")]),
        ]
    )

    @app.callback(
        Output("main-plot", "figure"),
        [Input("plot-type", "value"), Input("feature-dropdown", "value")],
    )
    def update_plot(plot_type, selected_feature):
        if plot_type == "correlation" and len(numeric_cols) > 1:
            return correlation_heatmap(df[numeric_cols + [target_col]], show=False)
        elif plot_type == "distribution" and selected_feature:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df[selected_feature], name=selected_feature))
            fig.update_layout(title=f"Distribution of {selected_feature}")
            return fig
        elif plot_type == "pca" and len(numeric_cols) > 1:
            return plot_pca(df[numeric_cols + [target_col]], target_col, show=False)
        elif plot_type == "boxplot" and selected_feature:
            fig = go.Figure()
            fig.add_trace(go.Box(y=df[selected_feature], name=selected_feature))
            fig.update_layout(title=f"Box Plot of {selected_feature}")
            return fig
        elif plot_type == "scatter" and selected_feature and len(numeric_cols) > 1:
            other_col = [col for col in numeric_cols if col != selected_feature][0]
            fig = px.scatter(df, x=selected_feature, y=other_col, color=target_col)
            return fig
        else:
            # Default empty plot
            fig = go.Figure()
            fig.update_layout(title="Select valid options to display plot")
            return fig

    @app.callback(
        Output("summary-stats", "children"), [Input("feature-dropdown", "value")]
    )
    def update_summary(selected_feature):
        if selected_feature:
            if selected_feature in numeric_cols:
                stats = df[selected_feature].describe()
                return html.Table(
                    [
                        html.Tr([html.Td(stat), html.Td(f"{value:.2f}")])
                        for stat, value in stats.items()
                    ]
                )
            else:
                value_counts = df[selected_feature].value_counts()
                return html.Table(
                    [
                        html.Tr([html.Td(value), html.Td(count)])
                        for value, count in value_counts.head(10).items()
                    ]
                )
        return "Select a feature to see summary statistics"

    if run_server:
        print(f"Starting dashboard on http://localhost:{port}")
        app.run(debug=True, port=port)

    return app
