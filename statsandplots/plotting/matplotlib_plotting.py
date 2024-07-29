from typing import Literal, Union, Optional

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib._enums import CapStyle
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, to_rgba
from matplotlib.container import BarContainer
from numpy.random import default_rng
from sklearn import decomposition, preprocessing

from ..stats import kde
from .plot_utils import bin_data, get_func, process_args, process_duplicates

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


def _jitter_plot(
    df,
    y,
    unique_groups,
    loc_dict,
    width,
    color_dict,
    marker_dict,
    edgecolor_dict,
    alpha=1,
    seed=42,
    marker_size=2,
    transform=None,
    ax=None,
    unique_id=None,
):
    if ax is None:
        ax = plt.gca()

    transform = get_func(transform)

    rng = default_rng(seed)
    jitter_values = rng.random(unique_groups.size)
    jitter_values *= width
    jitter_values -= width / 2

    for i in unique_groups.unique():
        indexes = np.where(unique_groups == i)[0]
        if unique_id is None:
            x = np.array([loc_dict[i]] * indexes.size)
            x += jitter_values[indexes]
            ax.plot(
                x,
                transform(df[y].iloc[indexes]),
                marker_dict[i],
                markerfacecolor=color_dict[i],
                markeredgecolor=edgecolor_dict[i],
                alpha=alpha,
                markersize=marker_size,
            )
        else:
            unique_ids_sub = np.unique(df[unique_id].iloc[indexes])
            for index, ui_group in enumerate(unique_ids_sub):
                sub_indexes = np.where(
                    np.logical_and(df[unique_id] == ui_group, unique_groups == i)
                )[0]
                x = np.array([loc_dict[i]] * sub_indexes.size)
                x += jitter_values[sub_indexes]
                ax.plot(
                    x,
                    transform(df[y].iloc[sub_indexes]),
                    MARKERS[index],
                    markerfacecolor=color_dict[i],
                    markeredgecolor=edgecolor_dict[i],
                    alpha=alpha,
                    markersize=marker_size,
                )
    return ax


def _jitteru_plot(
    df,
    y,
    unique_groups,
    unique_id,
    loc_dict,
    width,
    color_dict,
    marker_dict,
    edgecolor_dict,
    alpha=1,
    duplicate_offset=0.0,
    marker_size=2,
    transform=None,
    ax=None,
):
    if ax is None:
        ax = plt.gca()

    transform = get_func(transform)
    temp = width / 2

    for i in unique_groups.unique():
        indexes = np.where(unique_groups == i)[0]
        unique_ids_sub = np.unique(df[unique_id].iloc[indexes])
        if len(unique_ids_sub) > 1:
            dist = np.linspace(-temp, temp, num=len(unique_ids_sub))
        else:
            dist = [0]
        for index, ui_group in enumerate(unique_ids_sub):
            sub_indexes = np.where(
                np.logical_and(df[unique_id] == ui_group, unique_groups == i)
            )[0]
            x = np.full(sub_indexes.size, loc_dict[i]) + dist[index]
            if duplicate_offset > 0.0:
                output = (
                    process_duplicates(df[y].iloc[sub_indexes])
                    * duplicate_offset
                    * temp
                )
                x += output
            ax.plot(
                x,
                transform(df[y].iloc[sub_indexes]),
                marker_dict[i],
                markerfacecolor=color_dict[i],
                markeredgecolor=edgecolor_dict[i],
                alpha=alpha,
                markersize=marker_size,
            )
    return ax


def _summary_plot(
    df,
    y,
    unique_groups,
    loc_dict,
    func,
    capsize,
    capstyle,
    bar_width,
    err_func,
    linewidth,
    color_dict,
    alpha,
    transform=None,
    ax=None,
):
    if ax is None:
        ax = plt.gca()

    transform = get_func(transform)

    for i in unique_groups.unique():
        indexes = np.where(unique_groups == i)[0]
        tdata = get_func(func)(transform(df[y].iloc[indexes]))
        if err_func is not None:
            err_data = get_func(err_func)(transform(df[y].iloc[indexes]))
        else:
            err_data = None
        _, caplines, bars = ax.errorbar(
            x=loc_dict[i],
            y=tdata,
            # xerr=width,
            yerr=err_data,
            c=color_dict[i],
            fmt="none",
            linewidth=linewidth,
            capsize=capsize,
            alpha=alpha,
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
            xerr=bar_width / 2,
            # yerr=err_data,
            c=to_rgba(color_dict[i], alpha=alpha),
            fmt="none",
            linewidth=linewidth,
        )
        for b in bars:
            b.set_capstyle(capstyle)
    return ax


def _boxplot(
    df,
    y,
    unique_groups,
    loc_dict,
    color_dict,
    linecolor_dict,
    fliers="",
    box_width: float = 1.0,
    linewidth=1,
    show_means: bool = False,
    show_ci: bool = False,
    alpha: float = 1.0,
    line_alpha=1.0,
    transform=None,
    ax=None,
):
    if ax is None:
        ax = plt.gca()

    transform = get_func(transform)

    for i in unique_groups.unique():
        props = {
            "boxprops": {
                "facecolor": mpl.colors.to_rgba(color_dict[i], alpha=alpha),
                "edgecolor": mpl.colors.to_rgba(linecolor_dict[i], alpha=line_alpha),
            },
            "medianprops": {
                "color": mpl.colors.to_rgba(linecolor_dict[i], alpha=line_alpha)
            },
            "whiskerprops": {
                "color": mpl.colors.to_rgba(linecolor_dict[i], alpha=line_alpha)
            },
            "capprops": {
                "color": mpl.colors.to_rgba(linecolor_dict[i], alpha=line_alpha)
            },
        }
        if show_means:
            props["meanprops"] = {
                "color": mpl.colors.to_rgba(linecolor_dict[i], alpha=line_alpha)
            }
        indexes = np.where(unique_groups == i)[0]
        bplot = ax.boxplot(
            transform(df[y].iloc[indexes]),
            positions=[loc_dict[i]],
            sym=fliers,
            widths=box_width,
            notch=show_ci,
            patch_artist=True,
            showmeans=show_means,
            meanline=show_means,
            **props,
        )
        for i in bplot["boxes"]:
            i.set_linewidth(linewidth)

    return ax


def _violin_plot(
    df,
    y,
    unique_groups,
    loc_dict,
    color_dict,
    edge_dict,
    alpha=1,
    showextrema: bool = False,
    violin_width: float = 1.0,
    show_means: bool = True,
    show_medians: bool = False,
    transform=None,
    ax=None,
):
    if ax is None:
        ax = plt.gca()

    transform = get_func(transform)

    for i in unique_groups.unique():
        indexes = np.where(unique_groups == i)[0]
        indexes = indexes

        parts = ax.violinplot(
            transform(df[y].iloc[indexes]),
            positions=[loc_dict[i]],
            widths=violin_width,
            showmeans=show_means,
            showmedians=show_medians,
            showextrema=showextrema,
        )
        for body in parts["bodies"]:
            body.set_alpha(alpha)
            body.set_facecolor(color_dict[i])
            body.set_edgecolor(edge_dict[i])
        if show_means:
            parts["cmeans"].set_color(edge_dict[i])
        if show_medians:
            parts["cmedians"].set_color(edge_dict[i])

    return ax


def paired_plot():
    pass


def _hist_plot(
    df,
    y,
    unique_groups,
    color_dict,
    facet_dict,
    hist_type: Literal["bar", "barstacked", "step", "stepfilled"] = "bar",
    bins=None,
    density=True,
    ax=None,
):
    if ax is None:
        ax = plt.gca()
    for i in unique_groups.unique():
        indexes = np.where(unique_groups == i)[0]
        temp = df[y].iloc[indexes]
        ax[facet_dict[i]].hist(
            x=temp,
            histtype=hist_type,
            bins=bins,
            color=color_dict[i],
            density=density,
        )
    return ax


def _kde_plot(
    df,
    y,
    unique_groups,
    line_color_dict,
    facet_dict,
    linestyle_dict,
    linewidth,
    alpha,
    fill_under,
    fill_color_dict,
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
    axis="y",
    unique_id=None,
    ax=None,
):
    if ax is None:
        ax = plt.gca()
        ax = [ax]
    ugroups = np.unique(unique_groups)
    size = df[y].size
    for i in ugroups:
        if i == "none" and ugroups == 1:
            y_values = df[y].to_numpy.flatten()
            temp_size = size
        else:
            indexes = np.where(unique_groups == i)[0]
            y_values = df[y].iloc[indexes].to_numpy().flatten()
            temp_size = indexes.size
        x_kde, y_kde = kde(y_values, bw=bw, kernel=kernel, tol=tol)
        if common_norm:
            multiplier = float(temp_size / size)
            y_kde *= multiplier
        if axis == "x":
            y_kde, x_kde = x_kde, y_kde
        ax[facet_dict[i]].plot(
            x_kde,
            y_kde,
            c=line_color_dict[i],
            linestyle=linestyle_dict[i],
            alpha=alpha,
            linewidth=linewidth,
        )
        if fill_under:
            ax[facet_dict[i]].fill_between(
                x_kde, y_kde, color=fill_color_dict[i], alpha=alpha
            )
    return ax


def _ecdf(
    df,
    y,
    unique_groups,
    unique_id,
    linewidth,
    color_dict,
    facet_dict,
    linestyle_dict,
    alpha,
    ax,
):
    if ax is None:
        ax = plt.gca()

    for i in unique_groups.unique():
        indexes = np.where(unique_groups == i)[0]
        temp = df[y].iloc[indexes]
        ax[facet_dict[i]].ecdf(
            x=temp,
            color=color_dict[i],
            alpha=alpha,
            linestyle=linestyle_dict[i],
            linewidth=linewidth,
        )


def _poly_hist(
    df,
    y,
    unique_groups,
    color_dict,
    facet_dict,
    linestyle_dict,
    unique_id=None,
    density=True,
    bin=None,
    steps=50,
    func="mean",
    err_func="sem",
    fit_func=None,
    alpha=1,
    ax=None,
):
    if bin is None:
        bins = np.linspace(df[y].min(), df[y].max(), num=steps + 1)
    else:
        bins = np.linspace(bin[0], bin[1], num=steps + 1)
    if ax is None:
        ax = plt.gca()
        ax = [ax]
    if unique_id is not None:
        func = get_func(func)
        if err_func is not None:
            err_func = get_func(err_func)
        x = np.linspace(bin[0], bin[1], num=steps)
        for i in unique_groups.unique():
            indexes = np.where(unique_groups == i)[0]
            temp_df = df.iloc[indexes]
            uids = temp_df[unique_id].unique()
            temp_list = np.zeros((len(uids), steps))
            for index, j in enumerate(uids):
                temp = np.where(df[unique_id] == j)[0]
                temp_data = df[y].iloc[temp].to_numpy()
                poly = bin_data(np.sort(temp_data), bins)
                if density:
                    poly = poly / poly.sum()
                if fit_func is not None:
                    poly = fit_func(x, poly)
                temp_list[index] = poly
            mean_data = func(temp_list, axis=0)
            ax[facet_dict[i]].plot(
                x, mean_data, c=color_dict[i], linestyle=linestyle_dict[i], alpha=alpha
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
        for i in unique_groups.unique():
            indexes = np.where(unique_groups == i)[0]
            temp = df[y].iloc[indexes].sort_values()
            poly = bin_data(temp, bins)
            if fit_func is not None:
                poly = fit_func(x, poly)
            if density:
                poly /= poly.sum()
            ax[facet_dict[i]].plot(poly, c=color_dict[i], linestyle=linestyle_dict[i])
    return ax


def _line_plot(
    df,
    y,
    x,
    unique_groups,
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
):
    if ax is None:
        ax = plt.gca()
    if ax is None:
        ax = plt.gca()
        ax = [ax]
    if unique_id is not None:
        func = get_func(func)
        if err_func is not None:
            err_func = get_func(err_func)
        for i in unique_groups.unique():
            indexes = np.where(unique_groups == i)[0]
            temp_df = df.iloc[indexes]
            uids = temp_df[unique_id].unique()
            temp_list_y = None
            temp_list_x = None
            for index, j in enumerate(uids):
                temp = np.where(df[unique_id] == j)[0]
                temp_y = df[y].iloc[temp].to_numpy()
                temp_x = df[x].iloc[temp].to_numpy()
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
        for i in unique_groups.unique():
            indexes = np.where(unique_groups == i)[0]
            temp_y = df[y].iloc[temp].to_numpy()
            temp_x = df[x].iloc[temp].to_numpy()
            if fit_func is not None:
                temp_y = fit_func(temp_x, temp_y)
            ax[facet_dict[i]].plot(
                temp_x, temp_y, c=color_dict[i], linestyle=linestyle_dict[i]
            )
    return ax


def biplot(
    df,
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
    X = preprocessing.scale(df[columns])
    pca = decomposition.PCA(n_components=np.max(components) + 1)
    X = pca.fit_transform(X)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    fig, ax = plt.subplots()

    if plot_pca:
        if group_order is None:
            group_order = df[group].unique()
        if subgroup is None:
            subgroup_order = [""]
        if subgroup_order is None:
            subgroup_order = df[subgroup].unique()

        unique_groups = []
        for i in group_order:
            for j in subgroup_order:
                unique_groups.append(i + j)
        if subgroup is None:
            ug_list = df[group]
        else:
            ug_list = df[group] + df[subgroup]

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
    df,
    y,
    unique_groups,
    loc_dict,
    color_dict,
    linecolor_dict,
    cutoff: Union[float, int, list[Union[float, int]]],
    include_bins: list[bool],
    fill: bool = True,
    bar_width: float = 1.0,
    linewidth=1,
    alpha: float = 1.0,
    line_alpha=1.0,
    hatch=None,
    unique_id=None,
    ax=None,
):
    if ax is None:
        ax = plt.gca()

    bins = np.zeros(len(cutoff) + 3)
    bins[0] = df[y].min() - 1
    bins[1] = df[y].min()
    bins[-1] = df[y].max() + 1
    for i in range(len(cutoff)):
        bins[i + 2] = cutoff[i]

    groups = unique_groups.unique()
    plot_bins = sum(include_bins)

    if unique_id is not None:
        uids = np.unique(df[unique_id])
        multiplier = len(groups) * len(uids)
    else:
        multiplier = len(groups) * plot_bins

    if hatch is True:
        hs = HATCHES[:plot_bins]
    else:
        hs = [None] * plot_bins

    tops = []
    bottoms = []
    linewidth = [linewidth] * multiplier
    fill = [fill] * multiplier
    edgecolors = []
    fillcolors = []
    x_loc = []
    hatches = []
    bw = []

    for gr in groups:
        indexes = np.where(unique_groups == gr)[0]
        if unique_id is None:
            bar_width = [bar_width] * multiplier
            bw.extend(bar_width)
            temp = df[y].iloc[indexes].sort_values()
            binned_data = bin_data(temp, bins)
            binned_data = binned_data / binned_data.sum()
            top = binned_data[1:]
            bottom = binned_data[:-1]
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
            unique_ids_sub = np.unique(df[unique_id].iloc[indexes])
            temp_width = bar_width / len(unique_ids_sub)
            if len(unique_ids_sub) > 1:
                dist = np.linspace(
                    -bar_width / 2, bar_width / 2, num=len(unique_ids_sub) + 1
                )
                dist = (dist[1:] + dist[:-1]) / 2
            else:
                dist = [0]
            bw.extend([temp_width] * len(unique_ids_sub))
            for index, ui_group in enumerate(unique_ids_sub):
                sub_indexes = np.where(
                    np.logical_and(df[unique_id] == ui_group, unique_groups == gr)
                )[0]
                temp = df[y].iloc[sub_indexes].sort_values()
                binned_data = bin_data(temp, bins)
                binned_data = binned_data / binned_data.sum()
                top = binned_data[1:]
                bottom = binned_data[:-1]
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
    patches = []
    for t, b, x, width, fc, ec, h, lw in zip(
        tops, bottoms, x_loc, bw, fillcolors, edgecolors, hatches, linewidth
    ):
        left = x - width / 2
        r = mpatches.Rectangle(
            xy=(left, b),
            width=width,
            height=t,
            facecolor=fc,
            edgecolor=ec,
            hatch=h,
            linewidth=lw,
        )
        # r._internal_update()
        r.get_path()._interpolation_steps = 100
        r.sticky_edges.y.append(b)
        ax.add_patch(r)
        patches.append(r)
    bar_container = BarContainer(patches, datavalues=tops)
    ax.add_container(bar_container)
    return ax


def _plot_network(
    graph,
    marker_alpha: float = 0.8,
    line_alpha: float = 0.1,
    marker_size: int = 2,
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
    marker_size = np.array([Gcc.degree(i) for i in nodelist])
    marker_size = marker_size * marker_scale
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
        xy[:, 0], xy[:, 1], s=marker_size, alpha=marker_alpha, c=mcolor
    )
    path_collection.set_zorder(1)
    ax.axis("off")
