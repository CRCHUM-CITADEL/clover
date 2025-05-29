from typing import Optional, Tuple, Union, List  # standard library
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt  # 3rd party packages
import numpy as np
from matplotlib.axes import Axes
import seaborn as sns
import pandas as pd

import config  # local packages


def save_figure(
    save_folder: Union[str, Path], filename: str, figure_format: str = "pdf"
) -> None:
    """
    Save the current plot in the specified folder.

    :param save_folder: the path of the folder to save the figure
    :param filename: the name of the file, the date is automatically added
    :param figure_format: the format of the figure
    :return: *None*
    """
    date = datetime.today().strftime("%Y-%m-%d")
    plt.savefig(
        Path(save_folder) / f"{date}_{filename}.{figure_format}",
        bbox_inches="tight",
    )


def identity_line(
    ax: Axes = None, ls: str = "--", *args: Optional, **kwargs: Optional
) -> Axes:
    """
    Draw the identity line on a plot.

    From https://stackoverflow.com/q/22104256/3986320.

    :param ax: the *Axes* object to draw the plot onto, otherwise use the current *Axes*
    :param ls: the plot linestyle
    :param args: matplotlib *plot* parameters *fmt* and *data*
    :param kwargs: matplotlib properties
    :return: the *Axes* object with the plot drawn into it
    """
    ax = ax or plt.gca()
    (identity,) = ax.plot([], [], ls=ls, *args, **kwargs)

    def callback(axes: Axes) -> None:
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = min(low_x, low_y)
        high = max(high_x, high_y)
        identity.set_data([low, high], [low, high])

    callback(ax)
    ax.callbacks.connect("xlim_changed", callback)
    ax.callbacks.connect("ylim_changed", callback)
    return ax


def scatter_plot(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    xlim: list = None,
    ylim: list = None,
    ax: Axes = None,
) -> Axes:
    """
    Simple seaborn scatterplot.

    :param x: the input data on the x-axis
    :param y: the input data on the y-axis
    :param xlabel: the x-axis label to display on the plot
    :param ylabel: the y-axis label to display on the plot
    :param xlim: the x-axis bounds
    :param ylim: the y-axis bounds
    :param ax: the *Axes* object to draw the plot onto, otherwise use the current *Axes*
    :return: the *Axes* object with the plot drawn into it
    """
    axis = sns.scatterplot(x=x, y=y, ax=ax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    return axis


def heat_map(
    data: pd.DataFrame,
    title: str,
    annot: bool = True,
    xrotation: bool = True,
    vmin: float = None,
    vmax: float = None,
    ax: Axes = None,
) -> Axes:
    """
    Simple seaborn heatmap with optional rotated xlabels.

    :param data: 2D pandas dataframe, the index and columns names are used to label the x and y axes
    :param title: the title of the plot
    :param annot: if True, the data values are written in each cell
    :param xrotation: if True, the xlabels are rotated
    :param vmin: the colormap minimum bound
    :param vmax: the colormap maximal bound
    :param ax: the *Axes* object to draw the plot onto, otherwise use the current *Axes*
    :return: the *Axes* object with the plot drawn into it
    """
    axis = sns.heatmap(
        data=data,
        annot=annot,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        cmap=config.SEABORN_PALETTE,
    )
    if xrotation:
        plt.setp(
            axis.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor"
        )

    axis.set_title(title)
    axis.set_xlabel("")
    axis.set_ylabel("")

    return axis


def box_plot(
    data: pd.DataFrame, title: str, orient: str = "v", ax: Axes = None
) -> Axes:
    """
    Simple seaborn boxplot.

    :param data: the input pandas dataframe
    :param title: the title of the plot
    :param orient: the orientation of the plot, *h* for horizontal and *v* for vertical
    :param ax: the *Axes* object to draw the plot onto, otherwise use the current *Axes*
    :return: the *Axes* object with the plot drawn into it
    """
    axis = sns.boxplot(
        data=data,
        orient=orient,
        ax=ax,
        palette=config.SEABORN_PALETTE,
    )

    axis.set_title(title)

    return axis


def box_plot_per_column_hue(
    df: Union[pd.DataFrame, pd.Series],
    df_nested: Union[pd.DataFrame, pd.Series],
    original_name: str,
    nested_name: str,
    hue_name: str,
    title: str,
    value_name: str = "",
    orient: str = "v",
    xrotation: bool = True,
    ax: Axes = None,
) -> Axes:
    """
    Plot a seaborn nested boxplot per variable.

    :param df: the original data
    :param df_nested: the nested data
    :param original_name: the name of the original data
    :param nested_name: the name of the nested data
    :param hue_name: the name of the nested group
    :param title: the title of the plot
    :param value_name: the name of the value axis
    :param orient: the orientation of the plot, *h* for horizontal and *v* for vertical
    :param xrotation: if True, the xlabels are rotated
    :param ax: the *Axes* object to draw the plot onto, otherwise use the current *Axes*
    :return: the *Axes* object with the plot drawn into it
    """

    if isinstance(df, pd.DataFrame):
        assert set(df.columns) == set(df_nested.columns)
    else:
        assert isinstance(df, pd.Series) and isinstance(df_nested, pd.Series)
    assert orient in ["h", "v"], "orient must be h (horizontal) or v (vertical)"

    original = df.to_frame() if isinstance(df, pd.Series) else df.copy()
    original[hue_name] = original_name
    nested = (
        df_nested.to_frame() if isinstance(df_nested, pd.Series) else df_nested.copy()
    )
    nested[hue_name] = nested_name

    # Wide to long format
    df_long_format = (
        pd.concat([original, nested]).melt(id_vars=hue_name).sort_values(by="value")
    )

    x = "value" if orient == "h" else "variable"
    y = "variable" if orient == "h" else "value"

    axis = sns.boxplot(
        data=df_long_format,
        x=x,
        y=y,
        hue=hue_name,
        hue_order=[original_name, nested_name],
        orient=orient,
        palette=config.SEABORN_PALETTE,
        fliersize=2,
        ax=ax,
    )

    axis.set_title(title)
    if orient == "h":
        axis.set_xlabel(value_name)
        axis.set_ylabel("")
    else:
        axis.set_xlabel("")
        axis.set_ylabel(value_name)

    if xrotation:
        plt.setp(
            axis.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor"
        )

    return axis


def bar_plot_per_column_hue(
    df: pd.DataFrame,
    df_nested: pd.DataFrame,
    original_name: str,
    nested_name: str,
    hue_name: str,
    title: str,
    order: List[str] = None,
    value_name: str = "",
    orient: str = "v",
    xrotation: bool = True,
    ax: Axes = None,
) -> Axes:
    """
    Plot a seaborn nested catplot per variable.

    :param df: the original dataframe
    :param df_nested: the nested dataframe
    :param original_name: the name of the original dataframe
    :param nested_name: the name of the nested dataframe
    :param hue_name: the name of the nested group
    :param title: the title of the plot
    :param order: the order of the columns
    :param value_name: the name of the value axis
    :param orient: the orientation of the plot, *h* for horizontal and *v* for vertical
    :param xrotation: if True, the xlabels are rotated
    :param ax: the *Axes* object to draw the plot onto, otherwise use the current *Axes*
    :return: the *Axes* object with the plot drawn into it
    """

    assert set(df.columns) == set(df_nested.columns)
    assert orient in ["h", "v"], "orient must be h (horizontal) or v (vertical)"

    original = df.copy()
    original[hue_name] = original_name
    nested = df_nested.copy()
    nested[hue_name] = nested_name

    # Wide to long format
    df_long_format = (
        pd.concat([original, nested]).melt(id_vars=hue_name).sort_values(by="value")
    )

    x = "value" if orient == "h" else "variable"
    y = "variable" if orient == "h" else "value"

    axis = sns.barplot(
        data=df_long_format,
        x=x,
        y=y,
        order=order,
        hue=hue_name,
        hue_order=[original_name, nested_name],
        orient=orient,
        palette=config.SEABORN_PALETTE,
        ax=ax,
    )
    axis.set_title(title)

    if orient == "h":
        axis.set_ylabel("")
        axis.set_xlabel(value_name)
    else:
        axis.set_ylabel(value_name)
        axis.set_xlabel("")

    if xrotation:
        plt.setp(
            axis.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor"
        )

    # Display counts on top of the bars
    for container in axis.containers:
        axis.bar_label(container, fmt="%0.1f")

    return axis


def bar_plot(
    data: pd.DataFrame,
    title: str,
    value_name: str = "",
    orient: str = "v",
    xrotation: bool = True,
    lim: Tuple[float, float] = None,
    ax: Axes = None,
) -> Axes:
    """
    Simple seaborn barplot.

    :param data: the input pandas dataframe
    :param title: the title of the plot
    :param value_name: the name of the value axis
    :param orient: the orientation of the plot, *h* for horizontal and *v* for vertical
    :param xrotation: if True, the xlabels are rotated
    :param lim: the bounds
    :param ax: the *Axes* object to draw the plot onto, otherwise use the current *Axes*
    :return: the *Axes* object with the plot drawn into it
    """

    axis = sns.barplot(data=data, orient=orient, palette=config.SEABORN_PALETTE, ax=ax)

    plt.title(title)

    if lim is not None:
        mini = lim[0] if not np.isinf(lim[0]) else None
        maxi = lim[1] if not np.isinf(lim[1]) else None
        if orient == "h":
            axis.set_xlim(left=mini, right=maxi)
        else:
            axis.set_ylim(bottom=mini, top=maxi)

    if orient == "h":
        axis.set_xlabel(value_name)
    else:
        axis.set_ylabel(value_name)

    if xrotation:
        plt.setp(
            axis.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor"
        )

    # Display counts on top of the bars
    for container in axis.containers:
        axis.bar_label(container, fmt="%0.1f")

    return axis


def bar_plot_hue(
    s: pd.Series,
    s_nested: pd.Series,
    original_name: str,
    nested_name: str,
    hue_name: str,
    title: str,
    value_name: str = "",
    orient: str = "v",
    xrotation: bool = True,
    ax: Axes = None,
) -> Axes:
    """
    Plot a seaborn nested barplot.

    :param s: the original series
    :param s_nested: the nested series
    :param original_name: the name of the original dataframe
    :param nested_name: the name of the nested dataframe
    :param hue_name: the name of the nested group
    :param title: the title of the plot
    :param value_name: the name of the value axis
    :param orient: the orientation of the plot, *h* for horizontal and *v* for vertical
    :param xrotation: if True, the xlabels are rotated
    :param ax: the *Axes* object to draw the plot onto, otherwise use the current *Axes*
    :return: the *Axes* object with the plot drawn into it
    """

    assert orient in ["h", "v"], "orient must be h (horizontal) or v (vertical)"

    original = s.rename(original_name)
    nested = s_nested.rename(nested_name)

    df_concat = (
        pd.concat([original, nested], axis=1)
        .fillna(0)
        .rename_axis("category")
        .reset_index()
        .melt(id_vars="category")
        .rename(columns={"variable": hue_name})
        .sort_values(by="value")
    )

    x = "value" if orient == "h" else "category"
    y = "category" if orient == "h" else "value"

    axis = sns.barplot(
        data=df_concat,
        x=x,
        y=y,
        hue=hue_name,
        hue_order=[original_name, nested_name],
        orient=orient,
        palette=config.SEABORN_PALETTE,
        ax=ax,
    )

    # Display counts on top of the bars
    for container in axis.containers:
        axis.bar_label(container, fmt="%0.1f")

    axis.set_title(title)
    if orient == "h":
        axis.set_xlabel(value_name)
        axis.set_ylabel("")
    else:
        axis.set_xlabel("")
        axis.set_ylabel(value_name)

    if xrotation:
        plt.setp(
            axis.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor"
        )

    return axis


def pair_plot(
    df: pd.DataFrame,
    df_nested: pd.DataFrame,
    original_name: str,
    nested_name: str,
    hue_name: str,
    title: str,
) -> None:
    """
    Draw a seaborn pairplot.

    :param df: the original dataframe
    :param df_nested: the nested dataframe
    :param original_name: the name of the original dataframe
    :param nested_name: the name of the nested dataframe
    :param hue_name: the name of the nested group
    :param title: the title of the plot
    :return: *None*
    """
    assert set(df.columns) == set(df_nested.columns)

    original = df.copy()
    original[hue_name] = original_name
    nested = df_nested.copy()
    nested[hue_name] = nested_name
    df_concat = pd.concat([original, nested])

    sns.pairplot(data=df_concat, hue=hue_name, palette=config.SEABORN_PALETTE)
    plt.title(title)


def kde_plot_hue_plot_per_col(
    df: pd.DataFrame,
    df_nested: pd.DataFrame,
    original_name: str,
    nested_name: str,
    hue_name: str,
    title: str,
    axes: np.ndarray = None,
) -> None:
    """
    Draw a seaborn kdeplot with one plot per variable.

    :param df: the original dataframe
    :param df_nested: the nested dataframe
    :param original_name: the name of the original dataframe
    :param nested_name: the name of the nested dataframe
    :param hue_name: the name of the nested group
    :param title: the title of the plot
    :param axes: the *Axes* list to draw the plot onto, otherwise use the current *Axes*
    :return: *None*
    """
    assert set(df.columns) == set(df_nested.columns)
    if axes is not None:
        assert len(axes) >= df.shape[1]

    original = df.copy()
    original[hue_name] = original_name
    nested = df_nested.copy()
    nested[hue_name] = nested_name
    df_concat = pd.concat([original, nested])

    for i, col in enumerate(df.columns):
        sns.kdeplot(
            data=df_concat[[col, hue_name]],
            x=col,
            hue=hue_name,
            palette=config.SEABORN_PALETTE,
            fill=True,
            ax=axes[i] if axes is not None else None,
        )

        sns.move_legend(axes[i], "upper left", bbox_to_anchor=(1, 1), frameon=False)

    plt.suptitle(title)


def count_plot_hue_plot_per_col(
    df: pd.DataFrame,
    df_nested: pd.DataFrame,
    original_name: str,
    nested_name: str,
    hue_name: str,
    title: str,
    axes: np.ndarray = None,
) -> None:
    """
    Draw a categorical variable distribution (normalized to 1) with one plot per variable.

    :param df: the original dataframe
    :param df_nested: the nested dataframe
    :param original_name: the name of the original dataframe
    :param nested_name: the name of the nested dataframe
    :param hue_name: the name of the nested group
    :param title: the title of the plot
    :param axes: the *Axes* list to draw the plot onto, otherwise use the current *Axes*
    :return: *None*
    """
    assert set(df.columns) == set(df_nested.columns)
    if axes is not None:
        assert len(axes) >= df.shape[1]

    for i, col in enumerate(df.columns):
        vc_original = df[col].value_counts(normalize=True)
        vc_nested = df_nested[col].value_counts(normalize=True)

        # Combine value counts into a single DataFrame
        df_combined = pd.DataFrame(
            {original_name: vc_original, nested_name: vc_nested}
        ).fillna(0)

        df_combined.plot(
            kind="bar",
            colormap="tab20",
            alpha=0.8,
            ax=axes[i] if axes is not None else None,
        )
        axes[i].set_xlabel(col)
        axes[i].tick_params(axis="x", rotation=45)
        axes[i].set_ylabel("Proportion")
        axes[i].legend(
            loc="upper left", bbox_to_anchor=(1, 1), frameon=False, title=hue_name
        )

    plt.suptitle(title)


def histplot_hue(
    s: pd.Series,
    s_nested: pd.Series,
    original_name: str,
    nested_name: str,
    hue_name: str,
    title: str,
    value_name: str = "",
    xrotation: bool = True,
    counts: bool = False,
    ax: Axes = None,
) -> Axes:
    """
    Plot a seaborn nested histplot.

    :param s: the original series
    :param s_nested: the nested series
    :param original_name: the name of the original dataframe
    :param nested_name: the name of the nested dataframe
    :param hue_name: the name of the nested group
    :param title: the title of the plot
    :param value_name: the name of the value axis
    :param xrotation: if True, the xlabels are rotated
    :param counts: display counts on top of the bars
    :param ax: the *Axes* object to draw the plot onto, otherwise use the current *Axes*
    :return: the *Axes* object with the plot drawn into it
    """

    df_concat = (
        pd.concat([s, s_nested], axis=0)
        .to_frame(value_name)
        .assign(
            **{
                hue_name: [original_name for _ in range(len(s))]
                + [nested_name for _ in range(len(s_nested))]
            }
        )
    )

    colors = sns.color_palette(config.SEABORN_PALETTE, n_colors=10).as_hex()
    colors = [colors[3], colors[-1]]

    axis = sns.histplot(
        data=df_concat,
        x=value_name,
        bins="auto",
        hue=hue_name,
        hue_order=[original_name, nested_name],
        multiple="layer",
        palette=colors,
        ax=ax,
    )

    # Display counts on top of the bars
    if counts:
        for container in axis.containers:
            axis.bar_label(container, fmt="%0.1f")

    axis.set_title(title)

    if xrotation:
        plt.setp(
            axis.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor"
        )

    return axis


def histplot_plot(
    s: pd.Series,
    title: str,
    value_name: str = "",
    xrotation: bool = True,
    stat: str = "count",
    bins: any = "auto",
    binrange: Tuple[float] = None,
    counts: bool = False,
    ax: Axes = None,
) -> Axes:
    """
    Plot a seaborn histplot.

    :param s: the original series
    :param title: the title of the plot
    :param value_name: the name of the value axis
    :param xrotation: if True, the xlabels are rotated
    :param stat: the aggregate statistic to compute in each bin
    :param bins: the number of bins
    :param binrange: the upper and lower bounds to include in the bins
    :param counts: display counts on top of the bars
    :param ax: the *Axes* object to draw the plot onto, otherwise use the current *Axes*
    :return: the *Axes* object with the plot drawn into it
    """

    axis = sns.histplot(
        x=s,
        bins=bins,
        binrange=binrange,
        alpha=0.5,
        color=sns.color_palette(config.SEABORN_PALETTE, n_colors=10).as_hex()[0],
        ax=ax,
        stat=stat,
    )

    # Display counts on top of the bars
    if counts:
        for container in axis.containers:
            axis.bar_label(container, fmt="%0.1f")

    axis.set_title(title)
    axis.set_xlabel(value_name)

    if xrotation:
        plt.setp(
            axis.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor"
        )

    return axis


def plot_log_scale(
    data: list, title: str, labels: list, x_label: str, y_label: str
) -> None:
    """Plot a log-log graph

    :param data: the data to be plotted in the form of[(x1, y1), (x2, y2)...]
    :param title: title of the plot
    :param labels: the list of labels
    :param x_label: the x label of the plot
    :param y_label: the y label of the plot
    :return: None
    """

    for idx, (x, y) in enumerate(data):
        label = labels[idx]
        plt.plot(x, y, label=label)

    plt.semilogx()
    plt.semilogy()
    plt.xlim(1e-5, 1)
    plt.ylim(1e-5, 1)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot([0, 1], [0, 1], ls="--", color="gray")
    plt.legend()


def line_plot(data: list, title: str, x_label: str, y_label: str) -> None:
    """Plot a log-log graph

    :param data: the data to be plotted in the form of[(y1, x1), (y2, x2)...]
    :param title: title of the plot
    :param x_label: the x label of the plot
    :param y_label: the y label of the plot
    :return: None
    """

    for idx, (y, x) in enumerate(data):
        label = f"trial {idx+1}"
        plt.plot(x, y, label=label)

    plt.xlim(0, 1)
    # plt.ylim(0, 1)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
