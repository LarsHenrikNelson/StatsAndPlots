from typing import Literal, Optional, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib._enums import CapStyle
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, to_rgba

# from matplotlib.container import BarContainer
from numpy.random import default_rng
from sklearn import decomposition, preprocessing

from ..stats import kde, ecdf
from .plot_utils import process_args, process_duplicates, _bin_data
from ..utils import DataHolder, get_transform


# Reorder the filled matplotlib markers to choose the most different
MARKERS = [
    "o",
    "X",
    "^",
    "s",
    "*",
    "d",
    "h",
    "p",
    "<",
    "H",
    "D",
    "v",
    "P",
    ".",
    ">",
    "8",
]
CB6 = ["#0173B2", "#029E73", "#D55E00", "#CC78BC", "#ECE133", "#56B4E9"]
HATCHES = [
    None,
    "/",
    "o",
    "-",
    "*",
    "+",
    "\\",
    "|",
    "O",
    ".",
    "x",
]


def _make_legend_patches(color_dict, alpha, group, subgroup):
    legend_patches = []
    # for j in group:
    #     if j in color_dict:
    #         legend_patches.append(
    #             mpatches.Patch(color=to_rgba(color_dict[j], alpha=alpha), label=j)
    #         )
    # for j in subgroup:
    #     if j in color_dict:
    #         legend_patches.append(
    #             mpatches.Patch(color=to_rgba(color_dict[j], alpha=alpha), label=j)
    #         )
    for key, value in color_dict.items():
        legend_patches.append(
            mpatches.Patch(color=to_rgba(value, alpha=alpha), label=key)
        )
    return legend_patches


def _add_rectangles(
    tops,
    bottoms,
    loc,
    bw,
    fillcolors,
    edgecolors,
    hatches,
    linewidth,
    ax,
    axis="x",
):
    if axis == "x":
        ax.bar(
            x=loc,
            height=tops,
            bottom=bottoms,
            width=bw,
            color=fillcolors,
            edgecolor=edgecolors,
            linewidth=linewidth,
            hatch=hatches,
        )
    else:
        ax.barh(
            y=loc,
            width=tops,
            left=bottoms,
            height=bw,
            color=fillcolors,
            edgecolor=edgecolors,
            linewidth=linewidth,
            hatch=hatches,
        )
    return ax


def _jitter_plot(
    data,
    y,
    levels,
    unique_groups,
    loc_dict,
    width,
    color_dict,
    marker_dict,
    edgecolor_dict,
    alpha=1,
    seed=42,
    markersize=2,
    transform=None,
    ax=None,
    unique_id=None,
):
    if ax is None:
        ax = plt.gca()

    transform = get_transform(transform)

    rng = default_rng(seed)
    jitter_values = rng.random(data[y].size)
    jitter_values *= width
    jitter_values -= width / 2

    if len(levels) == 0:
        pass
    else:
        groups = data.groups(levels)
        if unique_id is not None:
            uid_groups = data.groups(levels + [unique_id])
        for i in unique_groups:
            if unique_id is None:
                indexes = groups[i]
                x = np.array([loc_dict[i]] * len(indexes))
                x += jitter_values[indexes]
                ax.plot(
                    x,
                    transform(data[indexes, y]),
                    marker_dict[i],
                    markerfacecolor=color_dict[i],
                    markeredgecolor=edgecolor_dict[i],
                    alpha=alpha,
                    markersize=markersize,
                )
            else:
                unique_ids_sub = np.unique(data[groups[i], unique_id])
                for index, ui_group in enumerate(unique_ids_sub):
                    sub_indexes = uid_groups[i + (ui_group,)]
                    x = np.full(len(sub_indexes), loc_dict[i])
                    x += jitter_values[sub_indexes]
                    ax.plot(
                        x,
                        transform(data[sub_indexes, y]),
                        MARKERS[index],
                        markerfacecolor=color_dict[i],
                        markeredgecolor=edgecolor_dict[i],
                        alpha=alpha,
                        markersize=markersize,
                    )
    return ax


def _jitteru_plot(
    data,
    y,
    levels,
    unique_groups,
    unique_id,
    loc_dict,
    width,
    color_dict,
    marker_dict,
    edgecolor_dict,
    alpha=1,
    duplicate_offset=0.0,
    markersize=2,
    agg_func=None,
    transform=None,
    ax=None,
):
    if ax is None:
        ax = plt.gca()

    transform = get_transform(transform)
    temp = width / 2

    if len(levels) == 0:
        pass
    else:
        groups = data.groups(levels)
        if unique_id is not None:
            uid_groups = data.groups(levels + [unique_id])
        for i in unique_groups:
            unique_ids_sub = np.unique(data[groups[i], unique_id])
            if len(unique_ids_sub) > 1:
                dist = np.linspace(-temp, temp, num=len(unique_ids_sub) + 1)
                dist = np.round((dist[1:] + dist[:-1]) / 2, 10)
            else:
                dist = [0]
            for index, ui_group in enumerate(unique_ids_sub):
                sub_indexes = uid_groups[i + (ui_group,)]
                x = np.full(len(sub_indexes), loc_dict[i]) + dist[index]
                if duplicate_offset > 0.0:
                    output = (
                        process_duplicates(data[sub_indexes, y])
                        * duplicate_offset
                        * temp
                    )
                    x += output
                if agg_func is None:
                    x = get_transform(agg_func)(x)
                else:
                    x = x[0]
                ax.plot(
                    x,
                    get_transform(agg_func)(transform(data[sub_indexes, y])),
                    marker_dict[i],
                    markerfacecolor=color_dict[i],
                    markeredgecolor=edgecolor_dict[i],
                    alpha=alpha,
                    markersize=markersize,
                )
    return ax


def _summary_plot(
    data,
    y,
    levels,
    unique_groups,
    loc_dict,
    func,
    capsize,
    capstyle,
    barwidth,
    err_func,
    linewidth,
    color_dict,
    alpha,
    transform=None,
    ax=None,
):
    if ax is None:
        ax = plt.gca()

    transform = get_transform(transform)
    y_data = []
    errs = []
    colors = []
    x_data = []
    widths = []

    if len(levels) == 0:
        pass
    else:
        groups = data.groups(levels)
        for i in unique_groups:
            x_data.append(loc_dict[i])
            colors.append(color_dict[i])
            widths.append(barwidth)
            y_data.append(get_transform(func)(transform(data[groups[i], y])))
            if err_func is not None:
                errs.append(get_transform(err_func)(transform(data[groups[i], y])))
            else:
                errs.append(None)
        ax = _plot_summary(
            x_data=x_data,
            y_data=y_data,
            errs=errs,
            widths=widths,
            colors=colors,
            linewidth=linewidth,
            alpha=alpha,
            capstyle=capstyle,
            capsize=capsize,
            ax=ax,
        )
    return ax


def _summaryu_plot(
    data,
    y,
    levels,
    unique_groups,
    unique_id,
    loc_dict,
    func,
    capsize,
    capstyle,
    barwidth,
    err_func,
    linewidth,
    color_dict,
    alpha,
    agg_func=None,
    agg_width=1,
    transform=None,
    ax=None,
):
    if ax is None:
        ax = plt.gca()

    transform = get_transform(transform)
    y_data = []
    errs = []
    colors = []
    x_data = []
    widths = []

    if len(levels) == 0:
        pass
    else:
        groups = data.groups(levels)
        if unique_id is not None:
            uid_groups = data.groups(levels + [unique_id])
        for i in unique_groups:
            uids = np.unique(data[groups[i], unique_id])
            if agg_func is None:
                temp = barwidth / 2
                if len(uids) > 1:
                    dist = np.linspace(-temp, temp, num=len(uids) + 1)
                    centers = np.round((dist[1:] + dist[:-1]) / 2, 10)
                else:
                    centers = [0]
                w = agg_width / len(uids)
                for index, j in enumerate(uids):
                    widths.append(w)
                    vals = transform(data[uid_groups[i + (j,)], y])
                    x_data.append(loc_dict[i] + centers[index])
                    colors.append(color_dict[i])
                    y_data.append(get_transform(func)(vals))
                    if err_func is not None:
                        errs.append(get_transform(err_func)(vals))
                    else:
                        errs.append(None)
            else:
                temp_vals = []
                for index, j in enumerate(uids):
                    vals = transform(data[uid_groups[j], y])
                    temp_vals.append(get_transform(func)(vals))
                x_data.append(loc_dict[i])
                colors.append(color_dict[i])
                widths.append(barwidth)
                y_data.append(get_transform(func)(np.array(temp_vals)))
                if err_func is not None:
                    errs.append(get_transform(err_func)(np.array(temp_vals)))
                else:
                    errs.append(None)

    ax = _plot_summary(
        x_data=x_data,
        y_data=y_data,
        errs=errs,
        widths=widths,
        colors=colors,
        linewidth=linewidth,
        alpha=alpha,
        capstyle=capstyle,
        capsize=capsize,
        ax=ax,
    )
    return ax


def _plot_summary(
    x_data: list,
    y_data: list,
    errs: list,
    widths: list,
    colors: list,
    linewidth: float,
    alpha: float,
    capstyle: str,
    capsize: float,
    ax,
):
    for xd, yd, e, c, w in zip(x_data, y_data, errs, colors, widths):
        _, caplines, bars = ax.errorbar(
            x=xd,
            y=yd,
            yerr=e,
            xerr=w / 2,
            c=to_rgba(c, alpha=alpha),
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
    return ax


def _boxplot(
    data,
    y,
    levels,
    unique_groups,
    loc_dict,
    color_dict,
    linecolor_dict,
    fliers="",
    box_width: float = 1.0,
    linewidth=1,
    showmeans: bool = False,
    show_ci: bool = False,
    alpha: float = 1.0,
    line_alpha=1.0,
    transform=None,
    ax=None,
):
    if ax is None:
        ax = plt.gca()

    transform = get_transform(transform)

    if len(levels) == 0:
        pass
    else:
        groups = data.groups(levels)
        for i in unique_groups:
            props = {
                "boxprops": {
                    "facecolor": to_rgba(color_dict[i], alpha=alpha),
                    "edgecolor": to_rgba(linecolor_dict[i], alpha=line_alpha),
                },
                "medianprops": {"color": to_rgba(linecolor_dict[i], alpha=line_alpha)},
                "whiskerprops": {"color": to_rgba(linecolor_dict[i], alpha=line_alpha)},
                "capprops": {"color": to_rgba(linecolor_dict[i], alpha=line_alpha)},
            }
            if showmeans:
                props["meanprops"] = {
                    "color": to_rgba(linecolor_dict[i], alpha=line_alpha)
                }
            bplot = ax.boxplot(
                transform(data[groups[i], y]),
                positions=[loc_dict[i]],
                sym=fliers,
                widths=box_width,
                notch=show_ci,
                patch_artist=True,
                showmeans=showmeans,
                meanline=showmeans,
                **props,
            )
            for i in bplot["boxes"]:
                i.set_linewidth(linewidth)

    return ax


def _violin_plot(
    data,
    y,
    levels,
    unique_groups,
    loc_dict,
    color_dict,
    edge_dict,
    alpha=1,
    linewidth=1,
    showextrema: bool = False,
    width: float = 1.0,
    showmeans: bool = True,
    showmedians: bool = False,
    transform=None,
    ax=None,
):
    if ax is None:
        ax = plt.gca()

    transform = get_transform(transform)

    if len(levels) == 0:
        pass
    else:
        groups = data.groups(levels)
        for i in unique_groups:
            parts = ax.violinplot(
                transform(data[groups[i], y]),
                positions=[loc_dict[i]],
                widths=width,
                showmeans=showmeans,
                showmedians=showmedians,
                showextrema=showextrema,
            )
            for body in parts["bodies"]:
                body.set_alpha(alpha)
                body.set_facecolor(color_dict[i])
                body.set_edgecolor(edge_dict[i])
                body.set_linewidth(linewidth)
            if showmeans:
                parts["cmeans"].set_color(edge_dict[i])
                parts["cmeans"].set_linewidth(linewidth)
            if showmedians:
                parts["cmedians"].set_color(edge_dict[i])
                parts["cmedians"].set_linewidth(linewidth)
    return ax


def paired_plot():
    pass


def _calc_hist(data, bins, stat):
    if stat == "probability":
        data = np.histogram(data, bins)
        return data / data.sum()
    elif stat == "count":
        data = np.histogram(data, bins)
        return data
    elif stat == "density":
        data, _ = np.histogram(data, bins, density=True)
        return data


def _hist_plot(
    data,
    y,
    x,
    unique_groups,
    levels,
    color_dict,
    facet_dict,
    hatch=None,
    hist_type: Literal["bar", "step", "stepfilled"] = "bar",
    fillalpha=1.0,
    linealpha=1.0,
    bin_limits=None,
    nbins=None,
    stat="probability",
    ax=None,
    agg_func=None,
    projection="rectilinear",
    unique_id=None,
    ytransform=None,
    xtransfrom=None,
    **kwargs,
):
    if ax is None:
        ax = plt.gca()

    y = y if x is None else x
    transform = ytransform if xtransfrom is None else xtransfrom
    axis = "y" if x is None else "x"

    if bin_limits is None:
        bins = np.linspace(
            get_transform(transform)(data[y].min()),
            get_transform(transform)(data[y].max()),
            num=nbins + 1,
        )
        # x = np.linspace(data[y].min(), data[y].max(), num=nbins)
        x = (bins[1:] + bins[:-1]) / 2
    else:
        # x = np.linspace(bin_limits[0], bin_limits[1], num=nbins)
        bins = np.linspace(
            get_transform(transform)(bin_limits[0]),
            get_transform(transform)(bin_limits[1]),
            num=nbins + 1,
        )
        x = (bins[1:] + bins[:-1]) / 2
    bottom = np.zeros(nbins)
    bw = np.full(
        nbins,
        bins[1] - bins[0],
    )
    plot_data = []
    count = 0
    colors = []
    axes1 = []
    edgec = []

    if len(levels) == 0:
        pass
    else:
        groups = data.groups(levels)
        if unique_id is not None:
            unique_id_indexes = data.groups(levels + [unique_id])
        for i in unique_groups:
            if unique_id is not None:
                uids = np.unique(data[unique_groups == i, unique_id])
                if agg_func is not None:
                    temp_list = np.zeros((len(uids), nbins))
                else:
                    temp_list = []
                for index, j in enumerate(uids):
                    temp_data = np.sort(data[unique_id_indexes[i + (j,)], y])
                    poly = _calc_hist(get_transform(transform)(temp_data), bins, stat)
                    if agg_func is not None:
                        temp_list[index] = poly
                    else:
                        plot_data.append(poly)
                        colors.append([to_rgba(color_dict[i], fillalpha)] * nbins)
                        edgec.append([to_rgba(color_dict[i], linealpha)] * nbins)
                        axes1.append(ax[facet_dict[i]])
                        count += 1
                if agg_func is not None:
                    plot_data.append(get_transform(agg_func)(temp_list, axis=0))
                    colors.append([to_rgba(color_dict[i], fillalpha)] * nbins)
                    edgec.append([to_rgba(color_dict[i], linealpha)] * nbins)
                    axes1.append(ax[facet_dict[i]])
                    count += 1
            else:
                temp_data = np.sort(data[groups[i], y])
                poly = _calc_hist(get_transform(transform)(temp_data), bins, stat)
                plot_data.append(poly)
                colors.append([to_rgba(color_dict[i], fillalpha)] * nbins)
                edgec.append([to_rgba(color_dict[i], linealpha)] * nbins)
                axes1.append(ax[facet_dict[i]])
                count += 1
    bottom = [bottom for _ in range(count)]
    bins = [bins[:-1] for _ in range(count)]
    bw = [bw for _ in range(count)]
    hatches = [[hatch] * nbins] * count
    linewidth = [np.full(nbins, 0) for _ in range(count)]
    for d, b, x, w, c, e, h, ln, sub_ax in zip(
        plot_data, bottom, bins, bw, colors, edgec, hatches, linewidth, axes1
    ):
        _add_rectangles(d, b, x, w, c, e, h, ln, sub_ax, axis=axis)
    return ax


def _scatter_plot(
    data,
    y,
    x,
    unique_groups,
    levels,
    markers,
    markercolors,
    edgecolors,
    markersizes,
    facetgroup,
    ax=None,
    facet_dict=None,
    xtransform=None,
    ytransform=None,
):
    if ax is None:
        ax = plt.gca()
        ax = [ax]
    for key, value in facet_dict.items():
        indexes = np.array([index for index, j in enumerate(facetgroup) if value == j])
        ax[value].scatter(
            get_transform(xtransform)(data[indexes, x]),
            get_transform(ytransform)(data[indexes, y]),
            marker=markers,
            color=[markercolors[i] for i in indexes],
            edgecolors=[edgecolors[i] for i in indexes],
            s=[markersizes[i] for i in indexes],
        )
    return ax


def _plot_agg_line(
    x_data,
    y_data,
    ed,
    value,
    ax,
    marker=None,
    linecolor=None,
    linewidth=None,
    linestyle=None,
    markerfacecolor=None,
    markeredgecolor=None,
    fill_between=False,
    markersize=None,
    fillalpha=None,
    linealpha=None,
):
    if not fill_between:
        ax[value].errorbar(
            x_data,
            y_data,
            yerr=ed,
            marker=marker,
            color=linecolor,
            elinewidth=linewidth,
            linewidth=linewidth,
            markerfacecolor=markerfacecolor,
            markeredgecolor=markeredgecolor,
            markersize=markersize,
            alpha=linealpha,
        )
    else:
        ax[value].fill_between(
            x_data,
            y_data - ed,
            y_data + ed,
            color=linecolor,
            alpha=fillalpha,
            linewidth=0,
        )
        ax[value].plot(
            x_data,
            y_data,
            linestyle=linestyle,
            linewidth=linewidth,
            color=linecolor,
            alpha=linealpha,
        )


def _agg_line(
    data,
    x,
    y,
    levels,
    marker,
    markersize,
    markerfacecolor,
    markeredgecolor,
    linestyle,
    linewidth,
    linecolor,
    linealpha,
    func,
    err_func,
    facet_dict,
    fill_between=False,
    fillalpha=1.0,
    agg_func=None,
    ytransform=None,
    xtransform=None,
    unique_id=None,
    ax=None,
    sort=True,
    unique_groups=None,
):
    err_data = None
    new_levels = (levels + [x]) if unique_id is None else (levels + [x, unique_id])
    new_data = data.groupby(y, new_levels).agg(get_transform(func)).reset_index()
    if unique_id is None:
        if err_func is not None:
            err_data = DataHolder(
                data.groupby(y, new_levels).agg(get_transform(err_func)).reset_index()
            )
    else:
        if agg_func is not None:
            if err_func is not None:
                err_data = DataHolder(
                    new_data[levels + [x, y]]
                    .groupby(levels + [x])
                    .agg(get_transform(err_func))
                    .reset_index()
                )
        new_data = (
            new_data[levels + [x, y]]
            .groupby(levels + [x])
            .agg(get_transform(func))
            .reset_index()
        )
    new_data = DataHolder(new_data)
    if unique_id is not None and agg_func is None:
        ugrps = list(set(zip(*[new_data[i] for i in levels + [unique_id]])))
    else:
        ugrps = list(set(zip(*[new_data[i] for i in levels])))
    temp_levels = (levels + [unique_id]) if unique_id is not None else levels
    if len(ugrps) != 0:
        for i in ugrps:
            value = facet_dict[i]
            indexes = new_data.get_data(temp_levels, i)
            y_data = new_data[indexes, y]
            x_data = new_data[indexes, x]
            ed = err_data[indexes, y] if err_func is not None else np.zeros(y_data.size)
            _plot_agg_line(
                get_transform(xtransform)(x_data),
                get_transform(ytransform)(y_data),
                ed,
                value,
                ax,
                marker=marker[i],
                linecolor=linecolor[i],
                linewidth=linewidth,
                linestyle=linestyle[i],
                markerfacecolor=markerfacecolor[i],
                markeredgecolor=markeredgecolor[i],
                markersize=markersize,
                fill_between=fill_between,
                linealpha=linealpha,
                fillalpha=fillalpha,
            )
    else:
        value = facet_dict[("",)]
        y_data = new_data[y]
        x_data = new_data[x]
        ed = err_data[y] if err_func is not None else np.zeros(y_data.size)
        _plot_agg_line(
            get_transform(xtransform)(x_data),
            get_transform(ytransform)(y_data),
            ed,
            value,
            ax,
            marker=marker[("",)],
            linecolor=linecolor[("",)],
            linewidth=linewidth,
            linestyle=linestyle[("",)],
            markerfacecolor=markerfacecolor[("",)],
            markeredgecolor=markeredgecolor[("",)],
            markersize=markersize,
            fill_between=fill_between,
            linealpha=linealpha,
            fillalpha=fillalpha,
        )
    return ax


def _kde_plot(
    data,
    y,
    x,
    unique_groups,
    levels,
    linecolor_dict,
    facet_dict,
    linestyle_dict,
    linewidth,
    alpha,
    fillalpha,
    fill_under,
    kernel: Literal[
        "gaussian",
        "exponential",
        "box",
        "tri",
        "epa",
        "biweight",
        "triweight",
        "tricube",
        "cosine",
    ] = "gaussian",
    bw: Literal["ISJ", "silverman", "scott"] = "ISJ",
    tol: Union[float, int] = 1e-3,
    common_norm: bool = True,
    unique_id=None,
    ax=None,
    agg_func=None,
    err_func=None,
    xtransform=None,
    ytransform=None,
    kde_type="fft",
):
    if ax is None:
        ax = plt.gca()
        ax = [ax]
    size = data.shape[0]

    x_data = []
    y_data = []
    linestyle_data = []
    linecolor_data = []
    facet_list = []
    errs = []

    column = y if x is None else x
    transform = ytransform if xtransform is None else xtransform

    if len(levels) == 0:
        y_values = data[column].to_numpy().flatten()
        temp_size = size
        x_kde, y_kde = kde(
            get_transform(transform)(y_values), bw=bw, kernel=kernel, tol=tol
        )
        if common_norm:
            multiplier = float(temp_size / size)
            y_kde *= multiplier
        if y is not None:
            y_kde, x_kde = x_kde, y_kde
        y_data.append(y_kde)
        x_data.append(x_kde)
        linecolor_data.append(linecolor_dict[("",)])
        linestyle_data.append(linestyle_dict[("",)])
        facet_list.append(facet_dict[("",)])
    else:
        groups = data.groups(levels)

        if unique_id is not None:
            uid_groups = data.groups(levels + [unique_id])
        for u in unique_groups:
            if unique_id is None:
                y_values = data[groups[u], column].to_numpy().flatten()
                temp_size = y_values.size
                x_kde, y_kde = kde(
                    get_transform(transform)(y_values), bw=bw, kernel=kernel, tol=tol
                )
                if common_norm:
                    multiplier = float(temp_size / size)
                    y_kde *= multiplier
                if y is not None:
                    y_kde, x_kde = x_kde, y_kde
                y_data.append(y_kde)
                x_data.append(x_kde)
                linecolor_data.append(linecolor_dict[u])
                linestyle_data.append(linestyle_dict[u])
                facet_list.append(facet_dict[u])
            else:
                subgroups, count = np.unique(
                    data[groups[u], unique_id], return_counts=True
                )
                temp_data = data[groups[u], column]
                min_data = get_transform(transform)(temp_data.min())
                max_data = get_transform(transform)(temp_data.max())
                min_data = min_data - np.abs((min_data * tol))
                max_data = max_data + np.abs((max_data * tol))
                min_data = min_data if min_data != 0 else -1e-10
                max_data = max_data if max_data != 0 else 1e-10
                if kde_type == "fft":
                    power2 = int(np.ceil(np.log2(len(temp_data))))
                    x = np.linspace(min_data, max_data, num=(1 << power2))
                else:
                    max_len = np.max(count)
                    x = np.linspace(min_data, max_data, num=int(max_len * 1.5))

                if agg_func is not None:
                    y_hold = np.zeros((len(subgroups), x.size))

                for hi, s in enumerate(subgroups):
                    s_indexes = uid_groups[u + (s,)]
                    y_values = data[s_indexes, column].to_numpy().flatten()
                    temp_size = y_values.size
                    if agg_func is None:
                        x_kde, y_kde = kde(
                            get_transform(transform)(y_values),
                            bw=bw,
                            kernel=kernel,
                            tol=tol,
                        )
                        if y is not None:
                            y_kde, x_kde = x_kde, y_kde
                        y_data.append(y_kde)
                        x_data.append(x_kde)
                        linecolor_data.append(linecolor_dict[u])
                        linestyle_data.append(linestyle_dict[u])
                        facet_list.append(facet_dict[u])
                    else:
                        _, y_kde = kde(
                            get_transform(transform)(y_values),
                            bw=bw,
                            kernel=kernel,
                            tol=tol,
                            x=x,
                            kde_type="fft",
                        )
                        y_hold[hi, :] = y_kde
                if agg_func is not None:
                    y_data.append(get_transform(agg_func)(y_hold, axis=0))
                    x_data.append(x)
                    linecolor_data.append(linecolor_dict[u])
                    linestyle_data.append(linestyle_dict[u])
                    facet_list.append(facet_dict[u])
                if err_func is not None:
                    errs.append(get_transform(err_func)(y_hold, axis=0))
    for plot_index, (x, y, lc, ls, fax) in enumerate(
        zip(x_data, y_data, linecolor_data, linestyle_data, facet_list)
    ):
        ax[fax].plot(
            x,
            y,
            c=lc,
            linestyle=ls,
            alpha=alpha,
            linewidth=linewidth,
        )
        if fill_under:
            ax[fax].fill_between(x, y, color=lc, alpha=fillalpha, linewidth=0)
        if err_func is not None:
            ax[fax].fill_between(
                x,
                y - errs[plot_index],
                y + errs[plot_index],
                color=lc,
                alpha=fillalpha,
                linewidth=0,
            )
    return ax


def _ecdf(
    data,
    y,
    x,
    levels,
    unique_groups,
    linewidth,
    color_dict,
    facet_dict,
    linestyle_dict,
    alpha,
    unique_id=None,
    agg_func=None,
    err_func=None,
    ecdf_type: Literal["spline", "bootstrap"] = "spline",
    ecdf_args=None,
    ax=None,
    xtransform=None,
    ytransform=None,
):
    if ax is None:
        ax = plt.gca()
        ax = [ax]

    column = y if x is None else x
    transform = ytransform if xtransform is None else xtransform

    x_data = []
    y_data = []
    linestyle_data = []
    linecolor_data = []
    facet_list = []
    errs = []

    etypes = {"spline", "bootstrap"}

    if len(levels) > 0:
        ugroups = data.groups(levels)

        if unique_id is not None:
            uid_groups = data.groups(levels + [unique_id])

    for u in unique_groups:
        if u == ("",):
            y_values = data[column].to_numpy.flatten()
            x_ecdf, y_ecdf = ecdf.ecdf(
                get_transform(transform)(y_values), ecdf_type=ecdf_type, **ecdf_args
            )
            y_data.append(y_ecdf)
            x_data.append(x_ecdf)
            linecolor_data.append(color_dict[u])
            linestyle_data.append(linestyle_dict[u])
            facet_list.append(facet_dict[u])
        elif unique_id is None:
            y_values = data[ugroups[u], column].to_numpy().flatten()
            x_ecdf, y_ecdf = ecdf.ecdf(
                get_transform(transform)(y_values), ecdf_type=ecdf_type, **ecdf_args
            )
            y_data.append(y_ecdf)
            x_data.append(x_ecdf)
            linecolor_data.append(color_dict[u])
            linestyle_data.append(linestyle_dict[u])
            facet_list.append(facet_dict[u])
        else:
            subgroups, counts = np.unique(
                data[ugroups[u], unique_id], return_counts=True
            )
            if agg_func is not None:
                if ecdf_type not in etypes:
                    raise ValueError(
                        "ecdf_type must be spline or bootstrap when using an agg_func"
                    )
                if "size" not in ecdf_args:
                    ecdf_args["size"] = np.max(counts)
                y_ecdf = np.arange(ecdf_args["size"]) / ecdf_args["size"]
                x_hold = np.zeros((len(subgroups), ecdf_args["size"]))
            for hi, s in enumerate(subgroups):
                y_values = data[uid_groups[u + (s,)], column].to_numpy().flatten()
                if agg_func is None:
                    x_ecdf, y_ecdf = ecdf.ecdf(
                        get_transform(transform)(y_values),
                        ecdf_type=ecdf_type,
                        **ecdf_args,
                    )
                    y_data.append(y_ecdf)
                    x_data.append(x_ecdf)
                    linecolor_data.append(color_dict[u])
                    linestyle_data.append(linestyle_dict[u])
                    facet_list.append(facet_dict[u])
                else:
                    x_ecdf, _ = ecdf.ecdf(
                        get_transform(transform)(y_values),
                        ecdf_type=ecdf_type,
                        **ecdf_args,
                    )
                    x_hold[hi, :] = x_ecdf
            if agg_func is not None:
                x_data.append(get_transform(agg_func)(x_hold, axis=0))
                y_data.append(y_ecdf)
                linecolor_data.append(color_dict[u])
                linestyle_data.append(linestyle_dict[u])
                facet_list.append(facet_dict[u])
            if err_func is not None:
                errs.append(get_transform(err_func)(x_hold, axis=0))

    for plot_index, (x, y, lc, ls, fax) in enumerate(
        zip(x_data, y_data, linecolor_data, linestyle_data, facet_list)
    ):
        ax[fax].plot(
            x,
            y,
            c=lc,
            linestyle=ls,
            alpha=alpha,
            linewidth=linewidth,
        )
        if err_func is not None:
            ax[fax].fill_between(
                x,
                y - errs[plot_index],
                y + errs[plot_index],
                color=lc,
                alpha=alpha,
                linewidth=0,
            )


def _poly_hist(
    data,
    y,
    x,
    unique_groups,
    levels,
    color_dict,
    facet_dict,
    linestyle_dict,
    linewidth,
    unique_id=None,
    density=True,
    bin_limits=None,
    nbins=50,
    func="mean",
    err_func="sem",
    fit_func=None,
    alpha=1,
    ax=None,
    xtransform=None,
    ytransform=None,
):
    y = y if x is None else x
    transform = ytransform if xtransform is None else xtransform
    transform = get_transform(transform)
    if bin_limits is None:
        bins = np.linspace(
            transform(data[y]).min(), transform(data[y]).max(), num=nbins + 1
        )
        x = np.linspace(transform(data[y]).min(), transform(data[y]).max(), num=nbins)
    else:
        x = np.linspace(bin[0], bin[1], num=nbins)
        bins = np.linspace(bin[0], bin[1], num=nbins + 1)
    if ax is None:
        ax = plt.gca()
        ax = [ax]

    if unique_id is not None:
        func = get_transform(func)
        if err_func is not None:
            err_func = get_transform(err_func)
        ugrp = np.unique(unique_groups)
        for i in ugrp:
            indexes = np.where(unique_groups == i)[0]
            uids = np.unique(data[indexes, unique_id])
            temp_list = np.zeros((len(uids), bins))
            for index, j in enumerate(uids):
                temp = np.where(data[unique_id] == j)[0]
                temp_data = np.sort(transform(data[temp, y]))
                poly, _ = np.histogram(temp_data, bins)
                if density:
                    poly = poly / poly.sum()
                if fit_func is not None:
                    poly = fit_func(x, poly)
                temp_list[index] = poly
            mean_data = func(temp_list, axis=0)
            ax[facet_dict[i]].plot(
                x,
                mean_data,
                c=color_dict[i],
                linestyle=linestyle_dict[i],
                alpha=alpha,
                linewidth=linewidth,
            )
            if err_func is not None:
                ax[facet_dict[i]].fill_between(
                    x=x,
                    y1=mean_data - err_func(temp_list, axis=0),
                    y2=mean_data + err_func(temp_list, axis=0),
                    alpha=alpha / 2,
                    color=color_dict[i],
                )
    else:
        ugrp = np.unique(unique_groups)
        for i in ugrp:
            indexes = np.where(unique_groups == i)[0]
            temp = np.sort(transform(data[indexes, y]))
            poly, _ = np.histogram(temp, bins)
            if fit_func is not None:
                poly = fit_func(x, poly)
            if density:
                poly = poly / poly.sum()
            ax[facet_dict[i]].plot(
                x,
                poly,
                c=color_dict[i],
                linestyle=linestyle_dict[i],
            )
    return ax


def _line_plot(
    data,
    y,
    x,
    unique_groups,
    levels,
    color_dict,
    facet_dict,
    linestyle_dict,
    linewidth=2,
    unique_id=None,
    func="mean",
    err_func="sem",
    fit_func=None,
    alpha=1,
    ax=None,
    xtransform=None,
    ytransform=None,
):
    if ax is None:
        ax = plt.gca()
    if ax is None:
        ax = plt.gca()
        ax = [ax]

    groups = data.groups(levels)
    if unique_id is not None:
        uid_groups = data.groups(levels + [unique_id])
        func = get_transform(func)
        if err_func is not None:
            err_func = get_transform(err_func)
        for i in unique_groups:
            indexes = groups[i]
            uids = np.unique(data[indexes])
            temp_list_y = None
            temp_list_x = None
            for index, j in enumerate(uids):
                temp = uid_groups[i + (j,)]
                temp_y = data[temp, y].to_numpy()
                temp_x = data[temp, x].to_numpy()
                if temp_list_y is None:
                    temp_list_y = np.zeros((len(uids), temp_x.size))
                if temp_list_x is None:
                    temp_list_x = np.zeros((len(uids), temp_x.size))
                if fit_func is not None:
                    poly = fit_func(temp_x, temp_y)
                    temp_list_y[index] = poly
                else:
                    temp_list_y[index] = temp_y
                temp_list_x[index] = temp_x
            mean_x = np.nanmean(temp_list_x, axis=0)
            mean_y = func(temp_list_y, axis=0)
            ax[facet_dict[i]].plot(
                mean_x,
                mean_y,
                c=color_dict[i],
                linestyle=linestyle_dict[i],
                linewidth=linewidth,
                alpha=alpha,
            )
            if err_func is not None:
                mean_y = func(temp_list_y)
                err = err_func(temp_list_y, axis=0)
                ax[facet_dict[i]].fill_between(
                    x=mean_x,
                    y1=mean_y - err,
                    y2=mean_y + err,
                    alpha=alpha / 2,
                    color=color_dict[i],
                )
    else:
        for i in unique_groups:
            indexes = groups[i]
            temp_y = data[indexes, y].to_numpy()
            temp_x = data[indexes, x].to_numpy()
            if fit_func is not None:
                temp_y = fit_func(temp_x, temp_y)
            ax[facet_dict[i]].plot(
                temp_x, temp_y, c=color_dict[i], linestyle=linestyle_dict[i]
            )
    return ax


def biplot(
    data,
    columns,
    group,
    subgroup=None,
    group_order=None,
    subgroup_order=None,
    plot_pca=False,
    plot_loadings=True,
    marker="o",
    color="black",
    components=None,
    alpha=0.8,
    labelsize=20,
    axis=True,
):
    if components is None:
        components = (0, 1)
    X = preprocessing.scale(data[columns])
    pca = decomposition.PCA(n_components=np.max(components) + 1)
    X = pca.fit_transform(X)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    fig, ax = plt.subplots()

    if plot_pca:
        if group_order is None:
            group_order = np.unique(data[group])
        if subgroup is None:
            subgroup_order = [""]
        if subgroup_order is None:
            subgroup_order = np.unique(data[subgroup])

        unique_groups = []
        for i in group_order:
            for j in subgroup_order:
                unique_groups.append(i + j)
        if subgroup is None:
            ug_list = data[group]
        else:
            ug_list = data[group] + data[subgroup]

        marker_dict = process_args(marker, group_order, subgroup_order)
        color_dict = process_args(color, group_order, subgroup_order)

        if components is None:
            components = [0, 1]
        xs = X[:, components[0]]
        ys = X[:, components[1]]
        scalex = 1.0 / (xs.max() - xs.min())
        scaley = 1.0 / (ys.max() - ys.min())
        for ug in unique_groups:
            indexes = np.where(ug_list == ug)[0]
            ax.scatter(
                xs[indexes] * scalex,
                ys[indexes] * scaley,
                alpha=alpha,
                marker=marker_dict[ug],
                c=color_dict[ug],
            )
        ax.legend(
            marker,
        )
    if plot_loadings:
        width = -0.005 * np.min(
            [np.subtract(*ax.get_xlim()), np.subtract(*ax.get_ylim())]
        )
        for i in range(loadings.shape[0]):
            ax.arrow(
                0,
                0,
                loadings[i, 0],
                loadings[i, 1],
                color="grey",
                alpha=0.5,
                width=width,
            )
            ax.text(
                loadings[i, 0] * 1.15,
                loadings[i, 1] * 1.15,
                columns[i],
                color="grey",
                ha="center",
                va="center",
            )
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )
    ax.set_xlabel(
        f"PC{components[0]} ({np.round(pca.explained_variance_ratio_[components[0]] * 100,decimals=2)}%)",
        fontsize=labelsize,
    )
    ax.set_ylabel(
        f"PC{components[1]} ({np.round(pca.explained_variance_ratio_[components[1]] * 100,decimals=2)}%)",
        fontsize=labelsize,
    )
    ax.spines["top"].set_visible(axis)
    ax.spines["right"].set_visible(axis)
    ax.spines["left"].set_visible(axis)
    ax.spines["bottom"].set_visible(axis)


def _percent_plot(
    data: DataHolder,
    y,
    levels,
    unique_groups,
    loc_dict,
    color_dict,
    linecolor_dict,
    cutoff: Union[None, float, int, list[Union[float, int]]],
    include_bins: list[bool],
    barwidth: float = 1.0,
    linewidth=1,
    alpha: float = 1.0,
    line_alpha=1.0,
    hatch=None,
    unique_id=None,
    ax=None,
    transform=None,
    invert=False,
    axis_type="density",
):
    if ax is None:
        ax = plt.gca()

    if cutoff is not None:
        bins = np.zeros(len(cutoff) + 2)
        bins[-1] = data[y].max() + 1e-6
        bins[0] = data[y].min() - 1e-6
        for i in range(len(cutoff)):
            bins[i + 1] = cutoff[i]

        if include_bins is None:
            include_bins = [True] * (len(bins) - 1)
    else:
        bins = np.unique(data[y])
        if include_bins is None:
            include_bins = [True] * len(bins)

    plot_bins = sum(include_bins)

    if hatch is True:
        hs = HATCHES[:plot_bins]
    else:
        hs = [None] * plot_bins

    tops = []
    bottoms = []
    lw = []
    edgecolors = []
    fillcolors = []
    x_loc = []
    hatches = []
    bw = []

    groups = data.groups(levels)
    if unique_id is not None:
        uid_groups = data.groups(levels + [unique_id])
    for gr in unique_groups:
        if unique_id is None:
            bw.extend([barwidth] * plot_bins)
            lw.extend([linewidth] * plot_bins)
            top, bottom = _bin_data(
                data[groups[gr], y], bins, axis_type, invert, cutoff
            )
            tops.extend(top[include_bins])
            bottoms.extend(bottom[include_bins])
            fc = [
                to_rgba(color_dict[gr], alpha=alpha),
            ] * plot_bins
            fillcolors.extend(fc)
            ec = [
                to_rgba(linecolor_dict[gr], alpha=line_alpha),
            ] * plot_bins
            edgecolors.extend(ec)
            x_s = [loc_dict[gr]] * plot_bins
            x_loc.extend(x_s)
            hatches.extend(hs)
        else:
            unique_ids_sub = np.unique(data[data[gr], unique_id])
            temp_width = barwidth / len(unique_ids_sub)
            bw.extend([temp_width] * plot_bins * len(unique_ids_sub))
            lw.extend([linewidth] * plot_bins * len(unique_ids_sub))
            if len(unique_ids_sub) > 1:
                dist = np.linspace(
                    -barwidth / 2, barwidth / 2, num=len(unique_ids_sub) + 1
                )
                dist = (dist[1:] + dist[:-1]) / 2
            else:
                dist = [0]
            for index, ui_group in enumerate(unique_ids_sub):
                top, bottom = _bin_data(
                    data[uid_groups[ui_group], y], bins, axis_type, invert, cutoff
                )

                tops.extend(top[include_bins])
                bottoms.extend(bottom[include_bins])
                fc = [
                    to_rgba(color_dict[gr], alpha=alpha),
                ] * plot_bins
                fillcolors.extend(fc)
                ec = [
                    to_rgba(linecolor_dict[gr], alpha=line_alpha),
                ] * plot_bins
                edgecolors.extend(ec)
                x_s = [loc_dict[gr] + dist[index]] * plot_bins
                x_loc.extend(x_s)
                hatches.extend(hs)
    ax = _add_rectangles(
        tops,
        bottoms,
        x_loc,
        bw,
        fillcolors,
        edgecolors,
        hatches,
        lw,
        ax,
    )
    return ax


def _count_plot(
    data: DataHolder,
    y: str,
    levels: list[str],
    unique_groups: dict,
    loc_dict: dict,
    color_dict,
    linecolor_dict,
    hatch,
    barwidth,
    linewidth,
    alpha,
    line_alpha,
    axis_type,
    unique_id=None,
    invert=False,
    agg_func=None,
    err_func=None,
    ax=None,
    transform=None,
):

    bw = []
    bottoms = []
    tops = []
    fillcolors = []
    edgecolors = []
    x_loc = []
    hatches = []
    lws = []

    multiplier = 100 if axis_type == "percent" else 1

    groups = data.groups(levels)
    for gr in unique_groups:
        unique_groups_sub, counts = np.unique(data[groups[gr], y], return_counts=True)
        size = sum(counts)
        temp_width = barwidth / len(unique_groups_sub)
        if len(unique_groups_sub) > 1:
            dist = np.linspace(
                -barwidth / 2, barwidth / 2, num=len(unique_groups_sub) + 1
            )
            dist = (dist[1:] + dist[:-1]) / 2
        else:
            dist = [0]
        bw.extend([temp_width] * len(unique_groups_sub))
        for index, ui_group, count in enumerate(zip(unique_groups_sub, counts)):
            if unique_id is None:
                bottoms.append(0)
                tops.append(
                    (count / size if axis_type != "count" else count) * multiplier
                )
                fillcolors.append(to_rgba(color_dict[str(ui_group)], alpha=alpha))
                edgecolors.append(
                    to_rgba(linecolor_dict[str(ui_group)], alpha=line_alpha)
                )
                x_loc.append(loc_dict[gr] + dist[index])
                hatches.append(HATCHES[index] if hatch else None)
                lws.append(linewidth)
            else:
                pass

    ax = _add_rectangles(
        tops,
        bottoms,
        x_loc,
        bw,
        fillcolors,
        edgecolors,
        hatches,
        lws,
        ax,
    )
    ax.autoscale()
    return ax


def _plot_network(
    graph,
    marker_alpha: float = 0.8,
    line_alpha: float = 0.1,
    markersize: int = 2,
    marker_scale: int = 1,
    linewidth: int = 1,
    edge_color: str = "k",
    marker_color: str = "red",
    marker_attr: Optional[str] = None,
    cmap: str = "gray",
    seed: int = 42,
    scale: int = 50,
    plot_max_degree: bool = False,
    layout: Literal["spring", "circular", "communities"] = "spring",
):

    if isinstance(cmap, str):
        cmap = plt.colormaps[cmap]
    _, ax = plt.subplots()
    Gcc = graph.subgraph(
        sorted(nx.connected_components(graph), key=len, reverse=True)[0]
    )
    if layout == "spring":
        pos = nx.spring_layout(Gcc, seed=seed, scale=scale)
    elif layout == "circular":
        pos = nx.circular_layout(Gcc, scale=scale)
    elif layout == "random":
        pos = nx.random_layout(Gcc, seed=seed)
    elif layout == "communities":
        communities = nx.community.greedy_modularity_communities(Gcc)
        # Compute positions for the node clusters as if they were themselves nodes in a
        # supergraph using a larger scale factor
        _ = nx.cycle_graph(len(communities))
        superpos = nx.spring_layout(Gcc, scale=scale, seed=seed)

        # Use the "supernode" positions as the center of each node cluster
        centers = list(superpos.values())
        pos = {}
        for center, comm in zip(centers, communities):
            pos.update(
                nx.spring_layout(nx.subgraph(Gcc, comm), center=center, seed=seed)
            )

    nodelist = list(Gcc)
    markersize = np.array([Gcc.degree(i) for i in nodelist])
    markersize = markersize * marker_scale
    xy = np.asarray([pos[v] for v in nodelist])

    edgelist = list(Gcc.edges(data=True))
    edge_pos = np.asarray([(pos[e0], pos[e1]) for (e0, e1, _) in edgelist])
    _, _, data = edgelist[0]
    if edge_color in data:
        edge_color = [data["weight"] for (_, _, data) in edgelist]
        edge_vmin = min(edge_color)
        edge_vmax = max(edge_color)
        color_normal = Normalize(vmin=edge_vmin, vmax=edge_vmax)
        edge_color = [cmap(color_normal(e)) for e in edge_color]
    edge_collection = LineCollection(
        edge_pos,
        colors=edge_color,
        linewidths=linewidth,
        antialiaseds=(1,),
        linestyle="solid",
        alpha=line_alpha,
    )
    edge_collection.set_cmap(cmap)
    edge_collection.set_clim(edge_vmin, edge_vmax)
    edge_collection.set_zorder(0)  # edges go behind nodes
    edge_collection.set_label("edges")
    ax.add_collection(edge_collection)

    if isinstance(marker_color, dict):
        if marker_attr is not None:
            mcolor = [
                marker_color[data[marker_attr]] for (_, data) in Gcc.nodes(data=True)
            ]
        else:
            mcolor = "red"
    else:
        mcolor = marker_color

    path_collection = ax.scatter(
        xy[:, 0], xy[:, 1], s=markersize, alpha=marker_alpha, c=mcolor
    )
    path_collection.set_zorder(1)
    ax.axis("off")
