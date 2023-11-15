from pathlib import Path
from typing import Literal, NamedTuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
from seaborn import axes_style

__all__ = [
    "plot_multi_two_way",
    "plot_two_way",
    "plot_one_way",
    "plot_two_way_violin",
    "multiline_plot",
    "Group",
    "Subgroup",
]


class Group(NamedTuple):
    mapping: dict


class Subgroup(NamedTuple):
    mapping: dict


def pval_ycoord(y, ax, spacing=0.06):
    tp = ax.transLimits.transform((0.5, y))
    bar = ax.transLimits.inverted().transform((tp[0], tp[1] + spacing))[1]
    symbol = ax.transLimits.inverted().transform((tp[0], tp[1] + spacing + 0.04))[1]
    return bar, symbol


def pval_xcoord(p_value, coord):
    if p_value <= 0.05 and p_value > 0.01:
        p_coor_x = [coord]
    elif p_value <= 0.01 and p_value > 0.005:
        p_coor_x = [coord - 0.025, coord + 0.025]
    elif p_value <= 0.005 and p_value > 0.001:
        p_coor_x = [coord - 0.045, coord, coord + 0.045]
    elif p_value <= 0.001:
        p_coor_x = [coord - 0.075, coord - 0.025, coord + 0.025, coord + 0.075]
    return p_coor_x


def plot_multi_two_way(
    df: pd.DataFrame,
    group: str,
    subgroup: str,
    col_list: list[str],
    y_label: list[str],
    **kwargs,
):
    for i, j in zip(col_list, y_label):
        plot_two_way(df=df, group=group, subgroup=subgroup, y=i, y_label=j, **kwargs)


def plot_two_way(
    df: pd.DataFrame,
    group: str,
    subgroup: str,
    y: str,
    order: Union[list, None],
    hue_order: list,
    y_label: str,
    title: str = "",
    x_pval: float = 0.1,
    color: Union[list, None] = None,
    alpha: int = 0.5,
    color_pval: float = 0.1,
    legend=False,
    y_lim: Union[list, None] = None,
    aspect: Union[int, float] = 1,
    figsize: Union[None, tuple[int, int]] = None,
    y_scale: Literal["linear", "log", "symlog"] = "linear",
    steps: int = 5,
    edgecolor: str = "none",
    pointsize: int = 8,
    meansize: int = 40,
    marker: Union[Group, Subgroup, str] = "o",
    decimals: int = 1,
    jitter: Union[tuple[int, float], None] = None,
    margins: float = 0.05,
    gap: int = 0,
    path: Union[None, str, Path] = None,
    labelsize: int = 20,
    filetype: str = "svg",
):
    # if isinstance(marker, dict):
    if jitter is None:
        jitter = (42, 0)
    # plt.rcParams["axes.autolimit_mode"] = "round_numbers"
    if y_lim is None:
        y_lim = [None, None]
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["font.size"] = labelsize
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots(subplot_kw=dict(box_aspect=aspect), figsize=figsize)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if order is not None:
        df["group_1"] = df[group].replace({i: ind for ind, i in enumerate(order)})
        df["group_2"] = df[subgroup].replace(
            {i: ind for ind, i in enumerate(hue_order)}
        )
        df.sort_values(["group_1", "group_2"], inplace=True)
        df.drop(labels=["group_1", "group_2"], axis=1, inplace=True)
    if isinstance(marker, str):
        p = so.Plot(df, x=group, y=y, color=subgroup)
        p = p.add(
            so.Dots(
                artist_kws={"edgecolor": edgecolor},
                pointsize=pointsize,
                fillalpha=alpha,
                marker=marker,
            ),
            so.Jitter(seed=jitter[0], width=jitter[1]),
            so.Dodge(gap=gap),
            legend=legend,
        ).scale(fillcolor=color)
    else:
        if isinstance(marker, Group):
            marker_group = group
        else:
            marker_group = subgroup
        p = so.Plot(df, x=group, y=y, color=subgroup, marker=marker_group)
        p = p.add(
            so.Dots(
                artist_kws={"edgecolor": edgecolor},
                pointsize=pointsize,
                fillalpha=alpha,
            ),
            so.Jitter(seed=jitter[0], width=jitter[1]),
            so.Dodge(gap=gap),
            legend=legend,
        ).scale(fillcolor=color, marker=marker.mapping)
    p = (
        p.add(
            so.Dot(pointsize=meansize, stroke=2, marker="_", color="Black"),
            so.Agg(),
            so.Dodge(gap=gap),
            legend=False,
        )
        .add(
            so.Range(artist_kws={"capstyle": "round"}, linewidth=2, color="Black"),
            so.Est(errorbar="se"),
            so.Dodge(gap=gap),
            legend=False,
        )
        .label(
            x="",
            y=y_label,
            title=title,
            frameon=False,
        )
    )
    if color is not None:
        p = p.scale(color=color, y=y_scale)
    p.on(ax).plot()
    pos = []
    if color_pval <= 0.05:
        maxes = df.groupby(group)[y].max().to_numpy()
        x_coords = [[i - 0.19, i + 0.21] for i, _ in enumerate(maxes)]
        for i, j, k in zip(maxes, x_coords, [0, 1]):
            bar, mark = pval_ycoord(i, ax)
            pos.append(mark)
            pval_x = pval_xcoord(color_pval, k)
            pval_y = [mark for i in pval_x]
            ax.plot(j, [bar, bar], color="black")
            ax.plot(pval_x, pval_y, marker=(5, 2), markersize=8, c="black", lw=0)
    if x_pval <= 0.05:
        if len(pos) > 0:
            bar, mark = pval_ycoord(max(pos), ax)
        else:
            coord = df[y].max()
            bar, mark = pval_ycoord(coord, ax)
        pval_x = pval_xcoord(x_pval, 0.5)
        pval_y = [mark for i in pval_x]
        ax.plot([0, 1], [bar, bar], color="black")
        ax.plot(pval_x, pval_y, marker=(5, 2), markersize=8, c="black", lw=0)
    if "/" in y:
        y = y.replace("/", "_")
    if y_scale not in ["log", "symlog"]:
        ticks = ax.get_yticks()
        if y_lim[0] is None:
            y_lim[0] = ticks[0]
        if y_lim[1] is None:
            y_lim[1] = ticks[-1]
        print(ticks)
        ax.set_ylim(bottom=y_lim[0], top=y_lim[1])
        ticks = np.round(np.linspace(y_lim[0], y_lim[1], steps), decimals=decimals)
        ax.set_yticks(ticks)
    else:
        ticks = ax.get_yticks()
        if y_lim[0] is None:
            y_lim[0] = ticks[0]
        if y_lim[1] is None:
            y_lim[1] = ticks[-1]
        ax.set_ylim(bottom=y_lim[0], top=y_lim[1])
    ax.margins(margins)
    ax.tick_params(axis="both", which="major", labelsize=20, width=2)
    if path is not None:
        plt.savefig(f"{path}/{y}.{filetype}", format=filetype, bbox_inches="tight")
    return fig, ax


def plot_one_way(
    df: pd.DataFrame,
    group: str,
    y: str,
    order: Union[list, list],
    y_label: str,
    title: str = "",
    x_pval: float = 0.1,
    color: Union[list, None] = None,
    alpha: int = 0.5,
    color_pval: float = 0.1,
    legend=False,
    y_lim: Union[list, None] = None,
    aspect: Union[float, int] = 1,
    figsize: Union[None, tuple] = None,
    y_scale: Literal["linear", "log", "symlog"] = "linear",
    steps: int = 5,
    edgecolor: str = "none",
    pointsize: int = 8,
    marker: Union[Group, str] = "o",
    jitter: Union[tuple[int, float], None] = None,
    decimals: int = 1,
    gap: float = 0.1,
    path: Union[None, str, Path] = None,
    filetype: str = "svg",
):
    if jitter is None:
        jitter = (42, 0)
    # plt.rcParams["axes.autolimit_mode"] = "round_numbers"
    if y_lim is None:
        y_lim = [None, None]
    plt.rcParams["legend.frameon"] = False
    # plt.rcParams["xtick.labelsize"] = 20
    # plt.rcParams["ytick.labelsize"] = 20
    plt.rcParams["font.size"] = 20
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots(subplot_kw=dict(box_aspect=aspect), figsize=figsize)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    # ax.tick_params(width=2)
    if order is not None:
        df["group"] = df[group].replace({i: ind for ind, i in enumerate(order)})
        df.sort_values(["group"], inplace=True)
        df.drop(labels=["group"], axis=1, inplace=True)
    if isinstance(marker, str):
        p = so.Plot(df, x=group, y=y, color=group)
        p = p.add(
            so.Dots(
                artist_kws={"edgecolor": edgecolor},
                pointsize=pointsize,
                fillalpha=alpha,
                marker=marker,
            ),
            so.Jitter(seed=jitter[0], width=jitter[1]),
            legend=legend,
        ).scale(fillcolor=color)
    else:
        p = so.Plot(df, x=group, y=y, color=group, marker=marker)
        p = p.add(
            so.Dots(
                artist_kws={"edgecolor": edgecolor},
                pointsize=pointsize,
                fillalpha=alpha,
            ),
            so.Jitter(seed=jitter[0], width=jitter[1]),
            so.Dodge(gap=gap),
            legend=legend,
        ).scale(fillcolor=color, marker=marker.mapping)
    p = (
        p.add(
            so.Dot(pointsize=40, stroke=2, marker="_", color="Black"),
            so.Agg(),
            legend=False,
        )
        .add(
            so.Range(artist_kws={"capstyle": "round"}, linewidth=2, color="Black"),
            so.Est(errorbar="se"),
            legend=False,
        )
        .label(x="", y=y_label, title=title, frameon=False)
    )
    if color is not None:
        p = p.scale(color=color)
    p.on(ax).plot()
    pos = []
    if color_pval <= 0.05:
        maxes = df.groupby(group)[y].max().to_numpy()
        x_coords = [[i - 0.19, i + 0.21] for i, _ in enumerate(maxes)]
        for i, j, k in zip(maxes, x_coords, [0, 1]):
            bar, mark = pval_ycoord(i, ax)
            pos.append(mark)
            pval_x = pval_xcoord(color_pval, k)
            pval_y = [mark for i in pval_x]
            ax.plot(j, [bar, bar], color="black")
            ax.plot(pval_x, pval_y, marker=(5, 2), markersize=8, c="black", lw=0)
    if x_pval <= 0.05:
        if len(pos) > 0:
            bar, mark = pval_ycoord(max(pos), ax)
        else:
            coord = df[y].max()
            bar, mark = pval_ycoord(coord, ax)
        pval_x = pval_xcoord(x_pval, 0.5)
        pval_y = [mark for i in pval_x]
        ax.plot([0, 1], [bar, bar], color="black")
        ax.plot(pval_x, pval_y, marker=(5, 2), markersize=8, c="black", lw=0)
    if "/" in y:
        y = y.replace("/", "_")
    if y_scale not in ["log", "symlog"]:
        ticks = ax.get_yticks()
        if y_lim[0] is None:
            y_lim[0] = ticks[0]
        if y_lim[1] is None:
            y_lim[1] = ticks[-1]
        ax.set_ylim(bottom=y_lim[0], top=y_lim[1])
        ticks = np.round(np.linspace(y_lim[0], y_lim[1], steps), decimals=decimals)
        ax.set_yticks(ticks)
    else:
        ticks = ax.get_yticks()
        if y_lim[0] is None:
            y_lim[0] = ticks[0]
        if y_lim[1] is None:
            y_lim[1] = ticks[-1]
        ax.set_ylim(bottom=y_lim[0], top=y_lim[1])
    ax.tick_params(axis="both", which="major", labelsize=20, width=2)
    if path is not None:
        plt.savefig(f"{path}/{y}.{filetype}", format=filetype, bbox_inches="tight")
    return fig, ax


def plot_two_way_violin(
    df: pd.DataFrame,
    group: str,
    subgroup: str,
    y: str,
    order: Union[list, list],
    hue_order: list,
    y_label: str,
    title: str = "",
    color: Union[list, None] = None,
    inner: str = "quart",
    alpha: int = 1,
    legend=False,
    y_lim: Union[list, None] = None,
    aspect: Union[float, int] = 1,
    figsize: Union[None, tuple] = None,
    y_scale: Literal["linear", "log", "symlog"] = "linear",
    steps: int = 5,
    decimals: int = 1,
    margins: float = 0.05,
    gap: float = 0.1,
    path: Union[None, str, Path] = None,
    filetype: str = "svg",
):
    if y_lim is None:
        y_lim = [None, None]
    plt.rcParams["font.size"] = 20
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots(subplot_kw=dict(box_aspect=aspect), figsize=figsize)
    ax.tick_params(axis="both", which="major", labelsize=20, width=2)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    sns.violinplot(
        df,
        y=y,
        x=group,
        order=order,
        hue_order=hue_order,
        hue=subgroup,
        palette=color,
        split=True,
        inner=inner,
        alpha=alpha,
        gap=gap,
        legend=legend,
    )
    if "/" in y:
        y = y.replace("/", "_")
    if y_scale not in ["log", "symlog"]:
        ticks = ax.get_yticks()
        if y_lim[0] is None:
            y_lim[0] = ticks[0]
        if y_lim[1] is None:
            y_lim[1] = ticks[-1]
        ax.set_ylim(bottom=y_lim[0], top=y_lim[1])
        ticks = np.round(np.linspace(y_lim[0], y_lim[1], steps), decimals=decimals)
        ax.set_yticks(ticks)
    else:
        ticks = ax.get_yticks()
        if y_lim[0] is None:
            y_lim[0] = ticks[0]
        if y_lim[1] is None:
            y_lim[1] = ticks[-1]
        ax.set_ylim(bottom=y_lim[0], top=y_lim[1])
    ax.margins(margins)
    ax.set_xlabel("")
    ax.set_ylabel(y_label, fontsize=20, fontweight="bold")
    ax.set_title(title)
    if path is not None:
        plt.savefig(f"{path}/{y}.{filetype}", format=filetype, bbox_inches="tight")
    return fig, ax


def multiline_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str,
    linestyle: str,
    order: Union[list, None],
    colors: Union[None, list] = None,
    x_label: str = "",
    y_label: str = "",
    y_lim: Union[None, list] = None,
    y_scale: Literal["linear", "log", "symlog"] = "linear",
    y_steps: int = 5,
    y_decimals: int = 1,
    x_lim: Union[None, list] = None,
    x_scale: Literal["linear", "log", "symlog"] = "linear",
    x_steps: int = 5,
    x_decimals: int = 1,
    aspect: float = 1.0,
    path: Union[str, None] = None,
    filetype: str = "svg",
):
    if y_lim is None:
        y_lim = [None, None]
    if x_lim is None:
        x_lim = [None, None]
    plt.rcParams["legend.frameon"] = False
    # plt.rcParams["xtick.labelsize"] = 20
    # plt.rcParams["ytick.labelsize"] = 20
    plt.rcParams["font.size"] = 20
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots(subplot_kw=dict(box_aspect=aspect))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    # ax.tick_params(width=2)
    if order is not None:
        df["group"] = df[linestyle].replace({i: ind for ind, i in enumerate(order)})
        df.sort_values(["group"], inplace=True)
        df.drop(labels=["group"], axis=1, inplace=True)
    p = (
        so.Plot(
            df,
            x=x,
            y=y,
            color=color,
            linestyle=linestyle,
        )
        .add(so.Line(), so.Agg(), legend=False)
        .add(so.Band(), so.Est(errorbar="se"), group=linestyle, legend=False)
        .label(x=x_label, y=y_label, textsize=20)
        .theme(
            {
                **axes_style("ticks"),
                "axes.facecolor": "w",
                "axes.edgecolor": "k",
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.autolimit_mode": "round_numbers",
                "axes.labelsize": 22,
                "xtick.labelsize": 20,
                "ytick.labelsize": 20,
            }
        )
    )
    if colors is not None:
        p = p.scale(color=colors)
    p.on(ax).plot()
    if y_scale is not None:
        ax.set_yscale(y_scale)
    if y_scale not in ["log", "symlog"]:
        ticks = ax.get_yticks()
        if y_lim[0] is None:
            y_lim[0] = ticks[0]
        if y_lim[1] is None:
            y_lim[1] = ticks[-1]
        ax.set_ylim(bottom=y_lim[0], top=y_lim[1])
        ticks = np.round(np.linspace(y_lim[0], y_lim[1], y_steps), decimals=y_decimals)
        ax.set_yticks(ticks)
    else:
        ticks = ax.get_yticks()
        if y_lim[0] is None:
            y_lim[0] = ticks[0]
        if y_lim[1] is None:
            y_lim[1] = ticks[-1]
        ax.set_ylim(bottom=y_lim[0], top=y_lim[1])
    if x_scale not in ["log", "sxmlog"]:
        ticks = ax.get_xticks()
        if x_lim[0] is None:
            x_lim[0] = ticks[0]
        if x_lim[1] is None:
            x_lim[1] = ticks[-1]
        ax.set_xlim(left=x_lim[0], right=x_lim[1])
        ticks = np.round(np.linspace(x_lim[0], x_lim[1], x_steps), decimals=x_decimals)
        ax.set_xticks(ticks)
    else:
        ticks = ax.get_xticks()
        if x_lim[0] is None:
            x_lim[0] = ticks[0]
        if x_lim[1] is None:
            x_lim[1] = ticks[-1]
        ax.set_xlim(left=x_lim[0], right=x_lim[1])
    ax.tick_params(axis="both", which="major", labelsize=20, width=2)
    plt.show()
    if path is not None:
        plt.savefig(f"{path}/{y}.{filetype}", format=filetype, bbox_inches="tight")
    return fig, ax
