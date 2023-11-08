from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib._enums import CapStyle
from matplotlib.lines import Line2D
from numpy.random import default_rng
from sklearn import decomposition, preprocessing

from .plot_utils import get_func, transform_func, process_args, bin_data

MARKERS = Line2D.filled_markers
CB6 = ["#0173B2", "#029E73", "#D55E00", "#CC78BC", "#ECE133", "#56B4E9"]


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
        x = np.array([loc_dict[i]] * indexes.size)
        x += jitter_values[indexes]
        ax.scatter(
            x,
            transform_func(df[y].iloc[indexes], transform),
            marker=marker_dict[i],
            c=color_dict[i],
            edgecolors=edgecolor_dict[i],
            alpha=alpha,
            s=marker_size,
        )
    return ax


def _summary_plot(
    df,
    y,
    unique_groups,
    loc_dict,
    func="mean",
    capsize=0,
    capstyle="round",
    bar_width=1.0,
    err_func="sem",
    linewidth=2,
    transform=None,
    ax=None,
):
    if ax is None:
        ax = plt.gca()

    transform = get_func(transform)

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
            xerr=bar_width / 2,
            # yerr=err_data,
            c="black",
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
    fliers="",
    box_width: float = 1.0,
    linewidth=1,
    show_means: bool = False,
    show_ci: bool = False,
    transform=None,
    ax=None,
):
    if ax is None:
        ax = plt.gca()

    transform = get_func(transform)

    for i in unique_groups.unique():
        props = {
            "boxprops": {"facecolor": color_dict[i], "edgecolor": "black"},
            "medianprops": {"color": "black"},
            "whiskerprops": {"color": "black"},
            "capprops": {"color": "black"},
        }
        if show_means:
            props["meanprops"] = {"color": "black"}
        indexes = np.where(unique_groups == i)[0]
        indexes = indexes
        bplot = ax.boxplot(
            transform_func(df[y].iloc[indexes], transform),
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
            # i.set_alpha(alpha)
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
            transform_func(df[y].iloc[indexes], transform),
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
        parts["cmeans"].set_color(edge_dict[i])

    return ax


def paired_plot():
    pass


def _hist_plot(
    df,
    y,
    unique_groups,
    color_dict,
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
        ax.hist(
            x=temp,
            histtype=hist_type,
            bins=bins,
            color=color_dict[i],
            density=density,
        )
    return ax


def _poly_hist(
    df,
    y,
    unique_groups,
    color_dict,
    facet_dict,
    linestyle_dict,
    bins=None,
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
            indexes = np.where(unique_groups == i)
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
    linestyle_dict,
    unique_id,
    facet_dict,
    fit_func,
    stat_func,
    ax=None,
):
    if ax is None:
        ax = plt.gca()
    if unique_id:
        pass
        # for i in unique_groups.unique():
        #     indexes = np.where(unique_groups == i)[0]
        #     temp_df = df.iloc[indexes]
    else:
        for i in unique_groups.unique():
            indexes = np.where(unique_groups == i)[0]
            temp_y = df[y].iloc[indexes]
            temp_x = df[x].loc[indexes]
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
