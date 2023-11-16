from typing import Literal, Union
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from . import matplotlib_plotting as mp
from . import plotly_plotting as plp
from .plot_utils import (
    _process_positions,
    process_args,
    _process_groups,
    decimals,
    get_ticks,
)

MP_PLOTS = {
    "jitter": mp._jitter_plot,
    "summary": mp._summary_plot,
    "boxplot": mp._boxplot,
    "violin": mp._violin_plot,
    "line_plot": mp._line_plot,
    "poly_hist": mp._poly_hist,
    "hist": mp._hist_plot,
}
PLP_PLOTS = {
    "jitter": plp._jitter_plot,
    # "summary": plp._summary_plot,
    # "boxplot": plp._boxplot,
    # "violin": plp._violin_plot,
}

SAVE_TYPES = {"svg", "png", "jpeg", "html"}


# %%
class LinePlot:
    def __init__(
        self,
        df,
        y,
        group,
        subgroup=None,
        group_order=None,
        subgroup_order=None,
        unique_id=None,
        y_label="",
        x_label="",
        title="",
        y_lim: Union[list, None] = None,
        x_lim: Union[list, None] = None,
        y_scale: Literal["linear", "log", "symlog"] = "linear",
        x_scale: Literal["linear", "log", "symlog"] = "linear",
        margins=0.05,
        aspect: Union[int, float] = 1,
        figsize: Union[None, tuple[int, int]] = None,
        labelsize=20,
        linewidth=2,
        ticksize=2,
        ticklabel=20,
        steps=5,
        y_decimals=None,
        x_decimals=None,
        facet=False,
    ):
        self.plots = {}
        self.plot_list = []
        if y_lim is None:
            y_lim = [None, None]
        if x_lim is None:
            x_lim = [None, None]

        if subgroup is not None:
            unique_groups = df[group].astype(str) + df[subgroup].astype(str)
        else:
            unique_groups = df[group].astype(str) + ""

        group_order, subgroup_order = _process_groups(
            df, group, subgroup, group_order, subgroup_order
        )
        if isinstance(title, str) and not facet:
            title = [title]
        elif isinstance(title, str) and facet:
            title = [title] * len(group_order)
        elif isinstance(title, list) and facet:
            if len(title) != len(group_order):
                raise ValueError(
                    "Length of title must be the same a the number of groups."
                )
            title = title
        else:
            title = group_order

        if facet:
            facet_length = list(range(len(group_order)))
        else:
            facet_length = 0
        facet_dict = process_args(
            facet_length,
            group_order,
            subgroup_order,
        )

        self.plot_dict = {
            "df": df,
            "y": y,
            "group": group,
            "subgroup": subgroup,
            "group_order": group_order,
            "subgroup_order": subgroup_order,
            "unique_groups": unique_groups,
            "y_label": y_label,
            "x_label": x_label,
            "title": title,
            "y_lim": y_lim,
            "x_lim": x_lim,
            "y_scale": y_scale,
            "x_scale": x_scale,
            "margins": margins,
            "aspect": aspect,
            "figsize": figsize,
            "labelsize": labelsize,
            "ticksize": ticksize,
            "ticklabel": ticklabel,
            "linewidth": linewidth,
            "facet": facet,
            "facet_dict": facet_dict,
            "unique_id": unique_id,
            "y_decimals": y_decimals,
            "x_decimals": x_decimals,
            "steps": steps,
        }
        self.plots = {}
        self.plot_list = []

    def line(
        self,
        x,
        color="black",
        linestyle="-",
        func="mean",
        err_func="sem",
        fit_func=None,
        alpha=1,
    ):
        color_dict = process_args(
            color, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
        )
        linestyle_dict = process_args(
            linestyle, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
        )

        line_plot = {
            "color_dict": color_dict,
            "linestyle_dict": linestyle_dict,
            "func": func,
            "err_func": err_func,
            "fit_func": fit_func,
            "x": x,
            "alpha": alpha,
        }
        self.plots["line_plot"] = line_plot
        self.plot_list.append("line_plot")

    def polyhist(
        self,
        color="black",
        linestyle="-",
        bin=None,
        density=True,
        steps=50,
        func="mean",
        err_func="sem",
        fit_func=None,
        alpha=1,
    ):
        color_dict = process_args(
            color, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
        )
        linestyle_dict = process_args(
            linestyle, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
        )

        poly_hist = {
            "color_dict": color_dict,
            "linestyle_dict": linestyle_dict,
            "density": density,
            "bin": bin,
            "steps": steps,
            "func": func,
            "err_func": err_func,
            "fit_func": fit_func,
            "alpha": alpha,
        }
        self.plots["poly_hist"] = poly_hist
        self.plot_list.append("poly_hist")

    def plot(
        self, savefig: bool = False, path=None, filetype="svg", backend="matplotlib"
    ):
        if backend == "matplotlib":
            output = self._matplotlib_backend(
                savefig=savefig, path=path, filetype=filetype
            )
            return output
        elif backend == "plotly":
            output = self._plotly_backend(savefig=savefig, path=path, filetype=filetype)
        else:
            raise AttributeError("Backend not implemented")
        return output

    def _matplotlib_backend(
        self,
        savefig: bool = False,
        path: str = "",
        filetype: str = "svg",
        transparent=False,
    ):
        if self.plot_dict["facet"]:
            fig, ax = plt.subplots(
                subplot_kw=dict(box_aspect=self.plot_dict["aspect"]),
                figsize=self.plot_dict["figsize"],
                ncols=len(self.plot_dict["group_order"]),
            )
        else:
            fig, ax = plt.subplots(
                subplot_kw=dict(box_aspect=self.plot_dict["aspect"]),
                figsize=self.plot_dict["figsize"],
            )
            ax = [ax]
        for i in self.plot_list:
            plot_func = MP_PLOTS[i]
            plot_func(
                df=self.plot_dict["df"],
                y=self.plot_dict["y"],
                unique_groups=self.plot_dict["unique_groups"],
                unique_id=self.plot_dict["unique_id"],
                facet_dict=self.plot_dict["facet_dict"],
                ax=ax,
                **self.plots[i],
            )

        if self.plot_dict["y_decimals"] is None:
            y_decimals = decimals(self.plot_dict["df"][self.plot_dict["y"]])
        else:
            y_decimals = self.plot_dict["y_decimals"]
        if self.plot_dict["x_decimals"] is None:
            x_decimals = decimals(self.plot_dict["df"][self.plot_dict["y"]])
        else:
            x_decimals = self.plot_dict["x_decimals"]
        for index, i in enumerate(ax):
            i.margins(self.plot_dict["margins"])
            i.spines["right"].set_visible(False)
            i.spines["top"].set_visible(False)
            i.spines["left"].set_linewidth(self.plot_dict["linewidth"])
            i.spines["bottom"].set_linewidth(self.plot_dict["linewidth"])
            if "/" in self.plot_dict["y"]:
                self.plot_dict["y"] = self.plot_dict["y"].replace("/", "_")
            if self.plot_dict["y_scale"] not in ["log", "symlog"]:
                ticks = i.get_yticks()
                lim, ticks = get_ticks(
                    self.plot_dict["y_lim"], ticks, self.plot_dict["steps"], y_decimals
                )
                i.set_ylim(bottom=lim[0], top=lim[1])
                i.set_yticks(ticks)
            else:
                ticks = i.get_yticks()
                lim, _ = get_ticks(
                    self.plot_dict["y_lim"], ticks, self.plot_dict["steps"], y_decimals
                )
                i.set_ylim(bottom=lim[0], top=lim[1])
            if self.plot_dict["x_scale"] not in ["log", "symlog"]:
                ticks = i.get_xticks()
                lim, ticks = get_ticks(
                    self.plot_dict["x_lim"], ticks, self.plot_dict["steps"], x_decimals
                )
                i.set_xlim(left=lim[0], right=lim[1])
                i.set_xticks(ticks)
            else:
                ticks = i.get_xticks()
                lim, _ = get_ticks(
                    self.plot_dict["x_lim"], ticks, self.plot_dict["steps"], x_decimals
                )
                i.set_xlim(left=lim[0], right=lim[1])
            i.set_ylabel(
                self.plot_dict["y_label"], fontsize=self.plot_dict["labelsize"]
            )
            i.set_title(
                self.plot_dict["title"][index], fontsize=self.plot_dict["labelsize"]
            )
            i.tick_params(
                axis="both",
                which="major",
                labelsize=self.plot_dict["ticklabel"],
                width=self.plot_dict["ticksize"],
            )
        fig.tight_layout()
        if savefig:
            path = Path(path)
            if path.suffix[1:] not in SAVE_TYPES:
                path = path / f"{self.plot_dict['y']}.{filetype}"
            else:
                filetype = path.suffix[1:]
            plt.savefig(
                path,
                format=filetype,
                bbox_inches="tight",
                transparent=transparent,
            )
        return fig, ax


class CategoricalPlot:
    def __init__(
        self,
        df,
        y,
        group,
        subgroup=None,
        group_order=None,
        subgroup_order=None,
        group_spacing=1,
        subgroup_spacing=0.6,
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

        if subgroup is not None:
            unique_groups = df[group].astype(str) + df[subgroup].astype(str)
        else:
            unique_groups = df[group].astype(str) + ""

        group_order, subgroup_order = _process_groups(
            df, group, subgroup, group_order, subgroup_order
        )

        loc_dict, width = _process_positions(
            subgroup=subgroup,
            group_order=group_order,
            subgroup_order=subgroup_order,
            group_spacing=group_spacing,
            subgroup_spacing=subgroup_spacing,
        )

        x_ticks = [group_spacing * index for index, _ in enumerate(group_order)]
        self.plot_dict = {
            "df": df,
            "y": y,
            "group": group,
            "subgroup": subgroup,
            "group_order": group_order,
            "subgroup_order": subgroup_order,
            "unique_groups": unique_groups,
            "x_ticks": x_ticks,
            "loc_dict": loc_dict,
            "width": width,
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

    def jitter(
        self,
        color="black",
        marker="o",
        edgecolor="",
        alpha=1,
        jitter=1,
        seed=42,
        marker_size=2,
        transform=None,
    ):
        marker_dict = process_args(
            marker, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
        )
        color_dict = process_args(
            color, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
        )
        edgecolor_dict = process_args(
            edgecolor, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
        )

        jitter_plot = {
            "color_dict": color_dict,
            "marker_dict": marker_dict,
            "edgecolor_dict": edgecolor_dict,
            "alpha": alpha,
            "width": jitter * self.plot_dict["width"],
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
        bar_width=1.0,
        err_func="sem",
        linewidth=2,
        transform=None,
    ):
        summary_plot = {
            "func": func,
            "capsize": capsize,
            "capstyle": capstyle,
            "bar_width": bar_width * self.plot_dict["width"],
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
        box_width: float = 1.0,
        transform=None,
        linewidth=1,
        show_means: bool = False,
        show_ci: bool = False,
    ):
        color_dict = process_args(
            facecolor, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
        )
        boxplot = {
            "color_dict": color_dict,
            "fliers": fliers,
            "box_width": box_width * self.plot_dict["width"],
            "show_means": show_means,
            "show_ci": show_ci,
            "transform": transform,
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
        violin_width: float = 1.0,
        show_means: bool = True,
        show_medians: bool = False,
        transform=None,
    ):
        color_dict = process_args(
            facecolor, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
        )
        edge_dict = process_args(
            edgecolor, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
        )
        violin = {
            "color_dict": color_dict,
            "edge_dict": edge_dict,
            "alpha": alpha,
            "showextrema": showextrema,
            "violin_width": violin_width * self.plot_dict["width"],
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
        elif backend == "plotly":
            output = self._plotly_backend(savefig=savefig, path=path, filetype=filetype)
        else:
            raise AttributeError("Backend not implemented")
        return output

    def _matplotlib_backend(
        self,
        savefig: bool = False,
        path: str = "",
        filetype: str = "svg",
        transparent=False,
    ):
        fig, ax = plt.subplots(
            subplot_kw=dict(box_aspect=self.plot_dict["aspect"]),
            figsize=self.plot_dict["figsize"],
        )

        for i in self.plot_list:
            plot_func = MP_PLOTS[i]
            plot_func(
                df=self.plot_dict["df"],
                y=self.plot_dict["y"],
                loc_dict=self.plot_dict["loc_dict"],
                unique_groups=self.plot_dict["unique_groups"],
                ax=ax,
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
        ax.set_xticks(self.plot_dict["x_ticks"], self.plot_dict["group_order"])
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
            path = Path(path)
            if path.suffix[1:] not in SAVE_TYPES:
                path = path / f"{self.plot_dict['y']}.{filetype}"
            else:
                filetype = path.suffix[1:]
            plt.savefig(
                path,
                format=filetype,
                bbox_inches="tight",
                transparent=transparent,
            )
        return fig, ax

    def _plotly_backend(self, savefig=False, path="", filetype="svg"):
        fig = go.Figure()
        for i in self.plot_list:
            plot_func = PLP_PLOTS[i]
            plot_func(
                df=self.plot_dict["df"],
                y=self.plot_dict["y"],
                loc_dict=self.plot_dict["loc_dict"],
                unique_groups=self.plot_dict["unique_groups"],
                **self.plots[i],
                fig=fig,
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

        if self.plot_dict["y_lim"][0] is None:
            self.plot_dict["y_lim"][0] = (
                self.plot_dict["df"][self.plot_dict["y"]].min() * 0.9
            )
        if self.plot_dict["y_lim"][1] is None:
            self.plot_dict["y_lim"][1] = (
                self.plot_dict["df"][self.plot_dict["y"]].max() * 1.1
            )
        ticks = np.round(
            np.linspace(
                self.plot_dict["y_lim"][0],
                self.plot_dict["y_lim"][1],
                self.plot_dict["steps"],
            ),
            decimals=decimals,
        )
        fig.update_layout(
            xaxis=dict(
                showline=True,
                linecolor="black",
                showgrid=False,
                tickmode="array",
                tickvals=self.plot_dict["x_ticks"],
                ticktext=self.plot_dict["group_order"],
                ticks="outside",
                color="black",
                tickfont=dict(size=self.plot_dict["ticklabel"]),
                tickwidth=self.plot_dict["ticksize"],
                linewidth=self.plot_dict["linewidth"],
                automargin=True,
            ),
            yaxis=dict(
                titlefont=dict(size=self.plot_dict["labelsize"]),
                title=dict(text=self.plot_dict["y_label"]),
                nticks=self.plot_dict["steps"],
                showline=True,
                tickmode="array",
                linecolor="black",
                tickvals=ticks,
                showgrid=False,
                ticks="outside",
                color="black",
                tickfont=dict(size=self.plot_dict["ticklabel"]),
                tickwidth=self.plot_dict["ticksize"],
                linewidth=self.plot_dict["linewidth"],
                automargin=True,
                range=[ticks[0], ticks[-1]],
                constrain="range",
                ticklabeloverflow="hide past div",
            ),
            plot_bgcolor="white",
        )
        fig.show()
        return fig
