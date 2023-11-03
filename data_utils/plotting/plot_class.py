from typing import Literal, Union

import matplotlib.pyplot as plt
import numpy as np

from .custom_plotting import _jitter_plot, _summary_plot, _boxplot, _violin_plot

PLOTS = {
    "jitter": _jitter_plot,
    "summary": _summary_plot,
    "boxplot": _boxplot,
    "violin": _violin_plot,
}


class CategoricalPlot:
    def __init__(
        self,
        df,
        y,
        group,
        subgroup=None,
        group_order=None,
        subgroup_order=None,
        group_spacing=0.75,
        subgroup_spacing=0.15,
        y_label="",
        title="",
        y_lim: Union[list, None] = None,
        y_scale: Literal["linear", "log", "symlog"] = "linear",
        steps: int = 5,
        margins=0.05,
        aspect: Union[int, float] = 1,
        figsize: Union[None, tuple[int, int]] = None,
        labelsize=20,
        linewidth=2,
        ticksize=2,
        ticklabel=20,
        decimals=None,
    ):
        if y_lim is None:
            y_lim = [None, None]

        group_order, subgroup_order = self._process_groups(
            df, group, subgroup, group_order, subgroup_order
        )
        self.plot_dict = {
            "df": df,
            "y": y,
            "group": group,
            "subgroup": subgroup,
            "group_order": group_order,
            "subgroup_order": subgroup_order,
            "group_spacing": group_spacing,
            "subgroup_spacing": subgroup_spacing,
            "y_label": y_label,
            "title": title,
            "y_lim": y_lim,
            "y_scale": y_scale,
            "steps": steps,
            "margins": margins,
            "aspect": aspect,
            "figsize": figsize,
            "labelsize": labelsize,
            "ticksize": ticksize,
            "ticklabel": ticklabel,
            "decimals": decimals,
            "linewidth": linewidth,
        }
        self.plots = {}
        self.plot_list = []

    def _process_groups(self, df, group, subgroup, group_order, subgroup_order):
        if group_order is None:
            group_order = df[group].unique()
        else:
            if len(group_order) != len(df[group].unique()):
                raise AttributeError(
                    "The number groups does not match the number in group_order"
                )
        if subgroup is not None:
            if subgroup_order is None:
                subgroup_order = df[subgroup].unique()
            elif len(subgroup_order) != len(df[subgroup].unique()):
                raise AttributeError(
                    "The number subgroups does not match the number in subgroup_order"
                )
        else:
            subgroup_order = [""]
        return group_order, subgroup_order

    def jitter(
        self,
        color="black",
        marker="o",
        edgecolor="none",
        alpha=1,
        jitter=1,
        seed=42,
        marker_size=2,
        transform=None,
    ):
        jitter_plot = {
            "color": color,
            "marker": marker,
            "edgecolor": edgecolor,
            "alpha": alpha,
            "jitter": jitter,
            "seed": seed,
            "marker_size": marker_size,
            "transform": transform,
        }
        self.plots["jitter"] = jitter_plot
        self.plot_list.append("jitter")

    def summary(
        self,
        func="mean",
        capsize=0,
        capstyle="round",
        width=1.0,
        err_func="sem",
        linewidth=2,
        transform=None,
    ):
        summary_plot = {
            "func": func,
            "capsize": capsize,
            "capstyle": capstyle,
            "width": width,
            "err_func": err_func,
            "linewidth": linewidth,
            "transform": transform,
        }
        self.plots["summary"] = summary_plot
        self.plot_list.append("summary")

    def boxplot(
        self,
        facecolor="none",
        fliers="",
        width: float = 1.0,
        transform=None,
        linewidth=1,
        alpha: float = 1.0,
        show_means: bool = False,
        show_ci: bool = False,
    ):
        boxplot = {
            "facecolor": facecolor,
            "fliers": fliers,
            "width": width,
            "show_means": show_means,
            "show_ci": show_ci,
            "transform": transform,
            "alpha": alpha,
            "linewidth": linewidth,
        }
        self.plots["boxplot"] = boxplot
        self.plot_list.append("boxplot")

    def violin(
        self,
        facecolor="none",
        edgecolor="black",
        alpha=1,
        showextrema: bool = False,
        width: float = 1.0,
        show_means: bool = True,
        show_medians: bool = False,
        transform=None,
    ):
        violin = {
            "facecolor": facecolor,
            "edgecolor": edgecolor,
            "alpha": alpha,
            "showextrema": showextrema,
            "width": width,
            "show_means": show_means,
            "show_medians": show_medians,
            "transform": transform,
        }
        self.plots["violin"] = violin
        self.plot_list.append("violin")

    def plot(
        self, savefig: bool = False, path=None, filetype="svg", backend="matplotlib"
    ):
        if backend == "matplotlib":
            output = self._matplotlib_backend(
                savefig=savefig, path=path, filetype=filetype
            )
            return output
        else:
            raise AttributeError("Backend not implemented")

    def _matplotlib_backend(
        self,
        savefig: bool = False,
        path: str = "",
        filetype: str = "svg",
        transparent=False,
    ):
        group_loc = {
            key: self.plot_dict["group_spacing"] * index
            for index, key in enumerate(self.plot_dict["group_order"])
        }
        fig, ax = plt.subplots(
            subplot_kw=dict(box_aspect=self.plot_dict["aspect"]),
            figsize=self.plot_dict["figsize"],
        )

        for i in self.plot_list:
            plot_func = PLOTS[i]
            plot_func(
                df=self.plot_dict["df"],
                y=self.plot_dict["y"],
                group=self.plot_dict["group"],
                subgroup=self.plot_dict["subgroup"],
                group_order=self.plot_dict["group_order"],
                subgroup_order=self.plot_dict["subgroup_order"],
                group_spacing=self.plot_dict["group_spacing"],
                subgroup_spacing=self.plot_dict["subgroup_spacing"],
                **self.plots[i],
            )

        if self.plot_dict["decimals"] is None:
            decimals = (
                np.abs(
                    int(
                        np.max(
                            np.round(
                                np.log10(
                                    np.abs(self.plot_dict["df"][self.plot_dict["y"]])
                                )
                            )
                        )
                    )
                )
                + 2
            )
        else:
            decimals = self.plot_dict["decimals"]
        ax.set_xticks(list(group_loc.values()), self.plot_dict["group_order"])
        ax.margins(self.plot_dict["margins"])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_linewidth(self.plot_dict["linewidth"])
        ax.spines["bottom"].set_linewidth(self.plot_dict["linewidth"])
        if "/" in self.plot_dict["y"]:
            self.plot_dict["y"] = self.plot_dict["y"].replace("/", "_")
        if self.plot_dict["y_scale"] not in ["log", "symlog"]:
            ticks = ax.get_yticks()
            if self.plot_dict["y_lim"][0] is None:
                self.plot_dict["y_lim"][0] = ticks[0]
            if self.plot_dict["y_lim"][1] is None:
                self.plot_dict["y_lim"][1] = ticks[-1]
            ax.set_ylim(
                bottom=self.plot_dict["y_lim"][0], top=self.plot_dict["y_lim"][1]
            )
            ticks = np.round(
                np.linspace(
                    self.plot_dict["y_lim"][0],
                    self.plot_dict["y_lim"][1],
                    self.plot_dict["steps"],
                ),
                decimals=decimals,
            )
            ax.set_yticks(ticks)
        else:
            ticks = ax.get_yticks()
            if self.plot_dict["y_lim"][0] is None:
                self.plot_dict["y_lim"][0] = ticks[0]
            if self.plot_dict["y_lim"][1] is None:
                self.plot_dict["y_lim"][1] = ticks[-1]
            ax.set_ylim(
                bottom=self.plot_dict["y_lim"][0], top=self.plot_dict["y_lim"][1]
            )
        ax.set_ylabel(self.plot_dict["y_label"], fontsize=self.plot_dict["labelsize"])
        ax.set_title(self.plot_dict["title"], fontsize=self.plot_dict["labelsize"])
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=self.plot_dict["ticklabel"],
            width=self.plot_dict["ticksize"],
        )
        if savefig:
            plt.savefig(
                f"{path}/{self.plot_dict['y']}.{filetype}",
                format=filetype,
                bbox_inches="tight",
                transparent=transparent,
            )
