from typing import Union, Literal
import inspect

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib._enums import CapStyle


def process_args(arg, group_order, subgroup_order):
    if isinstance(arg, str):
        arg = {key: arg for key in group_order}

    output_dict = {}
    for s in group_order:
        for b in subgroup_order:
            key = rf"{s}" + rf"{b}"
            if s in arg:
                output_dict[key] = arg[s]
            else:
                output_dict[key] = arg[b]
    return output_dict


def transform_func(a, transform=None):
    if transform is not None:
        return transform(a)
    else:
        return a


def get_valid_kwargs(args_list, **kwargs):
    output_args = {}
    for i in args_list:
        if i in kwargs:
            output_args[i] = kwargs[i]
    return output_args


def sem(a):
    return np.std(a) / np.sqrt(len(a))


def ci(a):
    ci_interval = stats.t.interval(
        confidence=0.95, df=len(a) - 1, loc=np.mean(a), scale=stats.sem(a)
    )
    return ci_interval[1] - ci_interval[0]


def get_func(input):
    if input == "sem":
        return sem
    elif input == "ci":
        return ci
    elif input == "mean":
        return np.mean
    elif input == "median":
        return np.median
    elif input == "std":
        return np.std
    else:
        return lambda a: 0


def _jitter_plot(
    df,
    y,
    group,
    subgroup,
    group_order,
    subgroup_order,
    color="black",
    marker="o",
    edgecolor="none",
    alpha=1,
    group_spacing=0.75,
    subgroup_spacing=0.15,
    jitter=1,
    seed=42,
    transform=None,
    ax=None,
):
    if ax is None:
        ax = plt.gca()
    group_loc = {key: group_spacing * index for index, key in enumerate(group_order)}
    temp_loc = np.linspace(-subgroup_spacing, subgroup_spacing, len(subgroup_order))
    width = np.abs(temp_loc[1] - temp_loc[0])

    marker_dict = process_args(marker, group_order, subgroup_order)
    color_dict = process_args(color, group_order, subgroup_order)
    edgecolor_dict = process_args(edgecolor, group_order, subgroup_order)

    subgroup_loc = {key: value for key, value in zip(subgroup_order, temp_loc)}
    loc_dict = {}
    for i in group_order:
        for j in subgroup_order:
            key = rf"{i}" + rf"{j}"
            loc_dict[key] = group_loc[i] + subgroup_loc[j]

    unique_groups = df[group].astype(str) + df[subgroup].astype(str)
    rng = default_rng(seed)
    jitter_values = rng.random(unique_groups.size)
    jitter_values *= width
    jitter_values *= jitter
    jitter_values -= (jitter * width) / 2
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    for i in unique_groups.unique():
        indexes = np.where(unique_groups == i)[0]
        x = np.array([loc_dict[i]] * indexes.size)
        x += jitter_values[indexes]
        ax.scatter(
            x,
            transform_func(df[y].iloc[indexes], transform),
            marker=marker_dict[i],
            c=color_dict[i],
            edgecolors=edgecolor_dict[i],
            alpha=alpha,
        )
    return ax


def _summary_plot(
    df,
    y,
    group,
    subgroup,
    group_order=None,
    subgroup_order=None,
    func="mean",
    capsize=0,
    capstyle="round",
    width=1.0,
    err_func="sem",
    linewidth=2,
    group_spacing=0.75,
    subgroup_spacing=0.15,
    transform=None,
    ax=None,
):
    if ax is None:
        ax = plt.gca()

    group_loc = {key: group_spacing * index for index, key in enumerate(group_order)}
    temp_loc = np.linspace(-subgroup_spacing, subgroup_spacing, len(subgroup_order))
    subgroup_loc = {key: value for key, value in zip(subgroup_order, temp_loc)}
    width *= (np.abs(temp_loc[1] - temp_loc[0])) / 2

    loc_dict = {}
    for i in group_order:
        for j in subgroup_order:
            key = rf"{i}" + rf"{j}"
            loc_dict[key] = group_loc[i] + subgroup_loc[j]

    unique_groups = df[group].astype(str) + df[subgroup].astype(str)

    for i in unique_groups.unique():
        indexes = np.where(unique_groups == i)[0]
        tdata = get_func(func)(transform_func(df[y].iloc[indexes], transform))
        err_data = get_func(err_func)(transform_func(df[y].iloc[indexes], transform))
        _, caplines, bars = ax.errorbar(
            x=loc_dict[i],
            y=tdata,
            # xerr=width,
            yerr=err_data,
            c="black",
            fmt="none",
            linewidth=linewidth,
            capsize=capsize,
        )
        for cap in caplines:
            cap.set_solid_capstyle(capstyle)
            cap.set_markeredgewidth(linewidth)
            cap._marker._capstyle = CapStyle(capstyle)
        for b in bars:
            b.set_capstyle(capstyle)
        _, _, bars = ax.errorbar(
            x=loc_dict[i],
            y=tdata,
            xerr=width,
            # yerr=err_data,
            c="black",
            fmt="none",
            linewidth=linewidth,
        )
        for b in bars:
            b.set_capstyle(capstyle)
    return ax


def paired_plot():
    pass


def plot_two_way(
    df,
    y,
    group,
    subgroup,
    group_order=None,
    subgroup_order=None,
    plot_type="jitter",
    group_spacing=0.75,
    y_label="",
    title="",
    y_lim: Union[list, None] = None,
    y_scale: Literal["linear", "log", "symlog"] = "linear",
    steps: int = 5,
    margins=0.05,
    aspect: Union[int, float] = 1,
    figsize: Union[None, tuple[int, int]] = None,
    labelsize=20,
    path=None,
    filetype="svg",
    decimals=None,
    **kwargs,
):
    if group_order is None:
        group_order = df[group].unique()
    else:
        if len(group_order) != len(df[group].unique()):
            raise AttributeError(
                "The number groups does not match the number in group_order"
            )
    if subgroup_order is None:
        subgroup_order = df[subgroup].unique()
    else:
        if len(subgroup_order) != len(df[subgroup].unique()):
            raise AttributeError(
                "The number subgroups does not match the number in subgroup_order"
            )
    group_loc = {key: group_spacing * index for index, key in enumerate(group_order)}

    fig, ax = plt.subplots(subplot_kw=dict(box_aspect=aspect), figsize=figsize)

    if y_lim is None:
        y_lim = [None, None]

    if isinstance(plot_type, str):
        plot_type = [plot_type]

    for plot in plot_type:
        if plot == "jitter":
            args = inspect.getfullargspec(_jitter_plot).args
            valid_kwargs = get_valid_kwargs(args, **kwargs)
            ax = _jitter_plot(
                df=df,
                y=y,
                group=group,
                subgroup=subgroup,
                group_spacing=group_spacing,
                ax=ax,
                group_order=group_order,
                subgroup_order=subgroup_order,
                **valid_kwargs,
            )
        if plot == "summary":
            args = inspect.getfullargspec(_summary_plot).args
            valid_kwargs = get_valid_kwargs(args, **kwargs)
            ax = _summary_plot(
                df=df,
                y=y,
                group=group,
                subgroup=subgroup,
                group_spacing=group_spacing,
                ax=ax,
                group_order=group_order,
                subgroup_order=subgroup_order,
                **valid_kwargs,
            )

    if decimals is None:
        decimals = np.abs(int(np.max(np.round(np.log10(np.abs(df[y])))))) + 2
    ax.set_xticks(list(group_loc.values()), group_order)
    ax.margins(margins)
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
    ax.set_ylabel(y_label, fontsize=labelsize)
    ax.set_title(title, fontsize=labelsize)
    ax.tick_params(axis="both", which="major", labelsize=labelsize, width=2)
    if path is not None:
        plt.savefig(f"{path}/{y}.{filetype}", format=filetype, bbox_inches="tight")
