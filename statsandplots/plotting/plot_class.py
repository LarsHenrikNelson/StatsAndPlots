from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Annotated, Callable, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..utils import (
    AGGREGATE,
    BACK_TRANSFORM_DICT,
    ERROR,
    TRANSFORM,
    DataHolder,
)
from . import matplotlib_plotting as mp
from . import plotly_plotting as plp
from .plot_utils import (
    _decimals,
    _process_groups,
    _process_positions,
    get_ticks,
    process_args,
    process_args_alt,
    process_scatter_args,
)


@dataclass
class ValueRange:
    lo: float
    hi: float


AlphaRange = Annotated[float, ValueRange(0.0, 1.0)]
ColorDict = Union[str, dict[str, str]]

MP_PLOTS = {
    "boxplot": mp._boxplot,
    "hist": mp._hist_plot,
    "jitter": mp._jitter_plot,
    "jitteru": mp._jitteru_plot,
    "line_plot": mp._line_plot,
    "poly_hist": mp._poly_hist,
    "summary": mp._summary_plot,
    "summaryu": mp._summaryu_plot,
    "violin": mp._violin_plot,
    "kde": mp._kde_plot,
    "percent": mp._percent_plot,
    "ecdf": mp._ecdf,
    "count": mp._count_plot,
    "scatter": mp._scatter_plot,
    "aggline": mp._agg_line,
}
PLP_PLOTS = {
    "jitter": plp._jitter_plot,
    # "summary": plp._summary_plot,
    # "boxplot": plp._boxplot,
    # "violin": plp._violin_plot,
}

MPL_SAVE_TYPES = {"svg", "png", "jpeg"}
PLOTLY_SAVE_TYPES = {"html"}


# %%
class LinePlot:

    def __init__(
        self,
        data: pd.DataFrame,
        y: str,
        group: Optional[str] = None,
        subgroup: Optional[str] = None,
        group_order: Optional[list[str]] = None,
        subgroup_order: Optional[list[str]] = None,
        ylabel: str = "",
        xlabel: str = "",
        title: str = "",
        facet: bool = False,
        facet_title: bool = False,
        nrows: int = None,
        ncols: int = None,
        inplace: bool = False,
    ):
        self.inplace = inplace
        self.plots = []
        self.plot_list = []
        self._plot_settings_run = False

        data = DataHolder(data)

        if group is None:
            unique_groups = np.array(["none"] * data.shape[0])
        else:
            if subgroup is not None:
                unique_groups = data[group].astype(str) + data[subgroup].astype(str)
            else:
                unique_groups = data[group].astype(str) + ""

        group_order, subgroup_order = _process_groups(
            data, group, subgroup, group_order, subgroup_order
        )

        ugs = {
            key: value
            for value, key in enumerate(list(product(group_order, subgroup_order)))
        }
        mapping_dict = {
            key: value
            for key, value in enumerate(list(product(group_order, subgroup_order)))
        }

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
            "data": data,
            "y": y,
            "group": group,
            "subgroup": subgroup,
            "group_order": group_order,
            "subgroup_order": subgroup_order,
            "unique_groups": unique_groups,
            "ylabel": ylabel,
            "xlabel": xlabel,
            "title": title,
            "facet": facet,
            "facet_dict": facet_dict,
            "nrows": nrows,
            "ncols": ncols,
            "facet_title": facet_title,
            "xtransform": None,
            "xback_transform_ticks": False,
            "ytransform": None,
            "yback_transform_ticks": False,
            "ugs": ugs,
            "mapping_dict": mapping_dict,
            "levels": [group, subgroup],
        }
        self.lines = {}

    def plot_settings(
        self,
        style: str = "default",
        ylim: Optional[list] = None,
        xlim: Optional[list] = None,
        yscale: Literal["linear", "log", "symlog"] = "linear",
        xscale: Literal["linear", "log", "symlog"] = "linear",
        xlabel_rotation: Union[Literal["horizontal", "vertical"], float] = "horizontal",
        margins: float = 0.05,
        aspect: Union[int, float] = 1,
        figsize: Union[None, tuple[int, int]] = None,
        labelsize: int = 20,
        linewidth: int = 2,
        ticksize: int = 2,
        ticklabel: int = 12,
        steps: int = 7,
        tickstyle: Literal["all", "middle"] = "all",
        ydecimals: int = None,
        xdecimals: int = None,
    ):
        self._plot_settings_run = True
        if ylim is None:
            ylim = [None, None]
        if xlim is None:
            xlim = [None, None]

        plot_settings = {
            "yscale": yscale,
            "xscale": xscale,
            "xlabel_rotation": xlabel_rotation,
            "margins": margins,
            "aspect": aspect,
            "figsize": figsize,
            "labelsize": labelsize,
            "ticksize": ticksize,
            "ticklabel": ticklabel,
            "linewidth": linewidth,
            "ylim": ylim,
            "xlim": xlim,
            "ydecimals": ydecimals,
            "xdecimals": xdecimals,
            "steps": steps,
            "tickstyle": tickstyle,
        }
        self.plot_dict.update(plot_settings)

        plt.style.use(style)
        self.style = style

        # Quick check just for dark and default backgrounds
        for index, val in enumerate(self.plot_list):
            if val == "line":
                if (
                    self.style == "dark_background"
                    and self.plots[index]["color"] == "black"
                ):
                    self.plots[index]["color"] = "white"
                elif self.style == "default" and self.plots[index]["color"] == "white":
                    self.plots[index]["color"] = "black"

        if not self.inplace:
            return self

    def transform(
        self,
        ytransform: Optional[TRANSFORM] = (None,),
        yback_transform_ticks: bool = False,
        xtransform: Optional[TRANSFORM] = (None,),
        xback_transform_ticks: bool = False,
    ):
        self.plot_dict["ytransform"] = ytransform
        self.plot_dict["yback_transform_ticks"] = yback_transform_ticks
        self.plot_dict["xtransform"] = xtransform
        self.plot_dict["xback_transform_ticks"] = xback_transform_ticks

        if not self.inplace:
            return self

    def add_line(
        self,
        linetype: Literal["hline", "vline"],
        lines: list,
        linestyle="solid",
        linealpha=1,
        linecolor="black",
    ):
        if linetype not in ["hline", "vline"]:
            raise AttributeError("linetype must by hline or vline")
        if isinstance(lines, (float, int)):
            lines = [lines]
        self.plot_dict[linetype] = {
            "line": lines,
            linestyle: linestyle,
            linealpha: linealpha,
            linecolor: linecolor,
        }

    def _plot_lines(line_dict, ax):
        for key, item in line_dict.items():
            for ap in ax:
                for ll in item["lines"]:
                    if key == "vline":
                        ap.axvline(
                            ll,
                            linestyle=item["linestyle"],
                            color=item["linecolor"],
                            alpha=item["linealpha"],
                        )
                    else:
                        ap.axhline(
                            ll,
                            linestyle=item["linestyle"],
                            color=item["linecolor"],
                            alpha=item["linealpha"],
                        )

    def line(
        self,
        x: str,
        color: ColorDict = "black",
        linestyle: str = "-",
        linewidth: int = 2,
        func: str = "mean",
        err_func: str = "sem",
        fit_func: Optional[Callable[[Union[np.ndarray, pd.Series]], np.ndarray]] = None,
        alpha: AlphaRange = 1.0,
        unique_id: Optional[str] = None,
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
            "linewidth": linewidth,
            "func": func,
            "err_func": err_func,
            "fit_func": fit_func,
            "x": x,
            "alpha": alpha,
            "unique_id": unique_id,
        }
        self.plots.append(line_plot)
        self.plot_list.append("line_plot")

        if not self.inplace:
            return self

    def aggline(
        self,
        x,
        marker: str = "none",
        markerfacecolor: Union[ColorDict, tuple[str, str]] = "black",
        markeredgecolor: Union[ColorDict, tuple[str, str]] = "black",
        markersize: Union[float, str] = 1,
        linecolor: ColorDict = "black",
        linewidth: float = 1.0,
        linestyle: str = "-",
        linealpha: float = 1.0,
        func="mean",
        err_func="sem",
        agg_func=None,
        fill_between=False,
        fillalpha: float = 1.0,
        sort=True,
        colorall: ColorDict = None,
        unique_id=None,
    ):
        if colorall is None:
            linecolor_dict = process_args_alt(
                self.plot_dict["mapping_dict"],
                linecolor,
            )
            markerfacecolor_dict = process_args_alt(
                self.plot_dict["mapping_dict"],
                markerfacecolor,
            )
            markeredgecolor_dict = process_args_alt(
                self.plot_dict["mapping_dict"], markeredgecolor
            )
        else:
            temp_dict = process_args_alt(
                self.plot_dict["mapping_dict"],
                colorall,
            )
            markeredgecolor_dict = temp_dict
            markerfacecolor_dict = temp_dict
            linecolor_dict = temp_dict

        marker_dict = process_args_alt(self.plot_dict["mapping_dict"], marker)
        linestyle_dict = process_args_alt(self.plot_dict["mapping_dict"], linestyle)

        line_plot = {
            "linecolor": linecolor_dict,
            "linestyle": linestyle_dict,
            "linewidth": linewidth,
            "func": func,
            "err_func": err_func,
            "x": x,
            "linealpha": linealpha,
            "fill_between": fill_between,
            "fillalpha": fillalpha,
            "sort": sort,
            "marker": marker_dict,
            "markerfacecolor": markerfacecolor_dict,
            "markeredgecolor": markeredgecolor_dict,
            "markersize": markersize,
            "unique_id": unique_id,
            "levels": self.plot_dict["levels"],
            "ugs": self.plot_dict["ugs"],
            "agg_func": agg_func,
        }
        facet_dict = {
            key: value for value, key in enumerate(self.plot_dict["group_order"])
        }
        facet_dict = process_args_alt(
            self.plot_dict["mapping_dict"],
            facet_dict,
        )
        self.plot_dict["facet_dict"] = facet_dict
        self.plots.append(line_plot)
        self.plot_list.append("aggline")

        if not self.inplace:
            return self

    def kde(
        self,
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
        linecolor: ColorDict = "black",
        linestyle: str = "-",
        linewidth: int = 2,
        fill_under: bool = False,
        fillcolor: ColorDict = "black",
        alpha: AlphaRange = 1.0,
        axis: Literal["x", "y"] = "y",
        unique_id: Optional[str] = None,
    ):
        linecolor_dict = process_args(
            linecolor, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
        )
        if fill_under:
            fillcolor_dict = process_args(
                fillcolor,
                self.plot_dict["group_order"],
                self.plot_dict["subgroup_order"],
            )
        else:
            fillcolor_dict = {}

        linestyle_dict = process_args(
            linestyle, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
        )

        kde_plot = {
            "linecolor_dict": linecolor_dict,
            "linestyle_dict": linestyle_dict,
            "linewidth": linewidth,
            "alpha": alpha,
            "fill_under": fill_under,
            "fillcolor_dict": fillcolor_dict,
            "axis": axis,
            "kernel": kernel,
            "bw": bw,
            "tol": tol,
            "common_norm": common_norm,
            "unique_id": unique_id,
        }

        self.plots.append(kde_plot)
        self.plot_list.append("kde")

        if not self.inplace:
            return self

    def polyhist(
        self,
        color: ColorDict = "black",
        linestyle: str = "-",
        linewidth: int = 2,
        bin=None,
        density=True,
        bins=50,
        func="mean",
        err_func="sem",
        fit_func=None,
        alpha: AlphaRange = 1.0,
        unique_id: Optional[str] = None,
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
            "linewidth": linewidth,
            "density": density,
            "bin": bin,
            "bins": bins,
            "func": func,
            "err_func": err_func,
            "fit_func": fit_func,
            "unique_id": unique_id,
            "alpha": alpha,
        }
        self.plots.append(poly_hist)
        self.plot_list.append("poly_hist")

        if not self.inplace:
            return self

    def hist():
        pass

    def ecdf(
        self,
        color: ColorDict = "black",
        linestyle: str = "-",
        linewidth: int = 2,
        alpha: AlphaRange = 1.0,
        unique_id: Optional[str] = None,
    ):
        color_dict = process_args(
            color, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
        )
        linestyle_dict = process_args(
            linestyle, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
        )

        ecdf = {
            "color_dict": color_dict,
            "linestyle_dict": linestyle_dict,
            "linewidth": linewidth,
            "alpha": alpha,
            "unique_id": unique_id,
        }
        self.plots.append(ecdf)
        self.plot_list.append("ecdf")

        if not self.inplace:
            return self

    def scatter(
        self,
        x,
        marker: str = ".",
        markercolor: Union[ColorDict, tuple[str, str]] = "black",
        edgecolor: ColorDict = "black",
        markersize: Union[float, str] = 1,
        alpha: float = 1.0,
    ):
        # if isinstance(marker, tuple):
        #     marker0 = marker[0]
        #     marker1 = marker[1]
        # else:
        #     marker0 = marker
        #     marker1 = None
        if isinstance(markercolor, tuple):
            markercolor0 = markercolor[0]
            markercolor1 = markercolor[1]
        else:
            markercolor0 = markercolor
            markercolor1 = None

        if isinstance(edgecolor, tuple):
            edgecolor0 = edgecolor[0]
            edgecolor1 = edgecolor[1]
        else:
            edgecolor0 = edgecolor
            edgecolor1 = None

        # markers = process_scatter_args(
        #     marker0,
        #     self.plot_dict["data"],
        #     self.plot_dict["group_order"],
        #     self.plot_dict["subgroup_order"],
        #     self.plot_dict["unique_groups"],
        #     marker1,
        # )
        # markers = markers.to_list()
        colors = process_scatter_args(
            markercolor0,
            self.plot_dict["data"],
            self.plot_dict["group_order"],
            self.plot_dict["subgroup_order"],
            self.plot_dict["unique_groups"],
            markercolor1,
            alpha=alpha,
        )
        edgecolors = process_scatter_args(
            edgecolor0,
            self.plot_dict["data"],
            self.plot_dict["group_order"],
            self.plot_dict["subgroup_order"],
            self.plot_dict["unique_groups"],
            edgecolor1,
            alpha=alpha,
        )
        markersize = process_scatter_args(
            markersize,
            self.plot_dict["data"],
            self.plot_dict["group_order"],
            self.plot_dict["subgroup_order"],
            self.plot_dict["unique_groups"],
            None,
        )
        plot_data = {
            "x": x,
            "markers": marker,
            "markercolors": colors,
            "edgecolors": edgecolors,
            "markersizes": markersize,
        }

        self.plot_list.append("scatter")
        self.plots.append(plot_data)

        if not self.inplace:
            return self

    def plot(
        self, savefig: bool = False, path=None, filetype="svg", backend="matplotlib"
    ):
        if not self._plot_settings_run:
            if self.inplace:
                self.plot_settings()
            else:
                self.inplace = True
                self.plot_settings()
                self.inplace = False
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
        if self.plot_dict["nrows"] is None and self.plot_dict["ncols"] is None:
            nrows = len(self.plot_dict["group_order"])
            ncols = 1
        elif self.plot_dict["nrows"] is None:
            nrows = 1
            ncols = self.plot_dict["ncols"]
        elif self.plot_dict["ncols"] is None:
            nrows = self.plot_dict["nrows"]
            ncols = 1
        else:
            nrows = self.plot_dict["nrows"]
            ncols = self.plot_dict["ncols"]
        if self.plot_dict["facet"]:
            fig, ax = plt.subplots(
                subplot_kw=dict(box_aspect=self.plot_dict["aspect"]),
                figsize=self.plot_dict["figsize"],
                ncols=ncols,
                nrows=nrows,
            )
            ax = ax.flatten()
        else:
            fig, ax = plt.subplots(
                subplot_kw=dict(box_aspect=self.plot_dict["aspect"]),
                figsize=self.plot_dict["figsize"],
            )
            ax = [ax]
        for i, j in zip(self.plot_list, self.plots):
            plot_func = MP_PLOTS[i]
            plot_func(
                data=self.plot_dict["data"],
                y=self.plot_dict["y"],
                unique_groups=self.plot_dict["unique_groups"],
                facet_dict=self.plot_dict["facet_dict"],
                ax=ax,
                ytransform=self.plot_dict["ytransform"],
                xtransform=self.plot_dict["xtransform"],
                **j,
            )

        if self.plot_dict["ydecimals"] is None:
            ydecimals = _decimals(self.plot_dict["data"][self.plot_dict["y"]])
        else:
            ydecimals = self.plot_dict["ydecimals"]
        if self.plot_dict["xdecimals"] is None:
            xdecimals = _decimals(self.plot_dict["data"][self.plot_dict["y"]])
        else:
            xdecimals = self.plot_dict["xdecimals"]
        num_plots = len(self.plot_dict["group_order"])
        for index, i in enumerate(ax[:num_plots]):
            if "kde" in self.plot_list and all(
                v is None for v in self.plot_dict["ylim"]
            ):
                if j["axis"] == "y":
                    self.plot_dict["ylim"] = [0, None]
            if "kde" in self.plot_list and all(
                v is None for v in self.plot_dict["xlim"]
            ):
                index = self.plot_list.index("kde")
                if self.plots[index]["axis"] == "x":
                    self.plot_dict["xlim"] = [0, None]
            i.spines["right"].set_visible(False)
            i.spines["top"].set_visible(False)
            i.spines["left"].set_linewidth(self.plot_dict["linewidth"])
            i.spines["bottom"].set_linewidth(self.plot_dict["linewidth"])
            if "/" in str(self.plot_dict["y"]):
                self.plot_dict["y"] = self.plot_dict["y"].replace("/", "_")

            if self.plot_dict["yscale"] not in ["log", "symlog"]:
                ticks = i.get_yticks()
                lim, ticks = get_ticks(
                    self.plot_dict["ylim"],
                    ticks,
                    self.plot_dict["steps"],
                    ydecimals,
                    tickstyle=self.plot_dict["tickstyle"],
                )
                i.set_ylim(bottom=lim[0], top=lim[1])
                i.set_yticks(ticks)
            else:
                i.set_yscale(self.plot_dict["yscale"])
                ticks = i.get_yticks()
                lim, _ = get_ticks(
                    self.plot_dict["ylim"], ticks, self.plot_dict["steps"], ydecimals
                )
                i.set_ylim(bottom=lim[0], top=lim[1])
            if self.plot_dict["xscale"] not in ["log", "symlog"]:
                ticks = i.get_xticks()
                lim, ticks = get_ticks(
                    self.plot_dict["xlim"],
                    ticks,
                    self.plot_dict["steps"],
                    xdecimals,
                    tickstyle=self.plot_dict["tickstyle"],
                )
                i.set_xlim(left=lim[0], right=lim[1])
                i.set_xticks(ticks)
            else:
                i.set_xscale(self.plot_dict["xscale"])
                ticks = i.get_xticks()
                lim, _ = get_ticks(
                    self.plot_dict["xlim"], ticks, self.plot_dict["steps"], xdecimals
                )
                i.set_xlim(left=lim[0], right=lim[1])
            i.set_ylabel(self.plot_dict["ylabel"], fontsize=self.plot_dict["labelsize"])
            if self.plot_dict["facet_title"]:
                i.set_title(
                    self.plot_dict["group_order"][index],
                    fontsize=self.plot_dict["labelsize"],
                )
            else:
                i.set_title(
                    self.plot_dict["title"], fontsize=self.plot_dict["labelsize"]
                )

            i.tick_params(
                axis="both",
                which="major",
                labelsize=self.plot_dict["ticklabel"],
                width=self.plot_dict["ticksize"],
            )
            i.margins(self.plot_dict["margins"])
        if self.plot_dict["title"] is not None:
            fig.suptitle(self.plot_dict["title"], fontsize=self.plot_dict["labelsize"])
        fig.tight_layout()
        if savefig:
            path = Path(path)
            if path.suffix[1:] not in MPL_SAVE_TYPES:
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
        data: Union[pd.DataFrame, np.ndarray, dict],
        y: Union[str, int, float],
        group: Union[str, int, float] = None,
        subgroup: Union[str, int, float] = None,
        group_order: Union[None, list[Union[str, int, float]]] = None,
        subgroup_order: Union[None, list[Union[str, int, float]]] = None,
        group_spacing: Union[float, int] = 1.0,
        ylabel: str = "",
        title: str = "",
        inplace: bool = True,
    ):

        self._plot_settings_run = False
        self.inplace = inplace
        self.style = "default"

        data = DataHolder(data)

        if subgroup is not None:
            if group not in data:
                raise ValueError(f"{group} must be supplied if subgroup is used")
            unique_groups = data[group].astype(str) + data[subgroup].astype(str)
        else:
            if group is None:
                unique_groups = pd.Series([""] * data.shape[0])
            else:
                unique_groups = data[group].astype(str) + ""

        if group is not None:
            group_order, subgroup_order = _process_groups(
                data, group, subgroup, group_order, subgroup_order
            )

            loc_dict, width = _process_positions(
                subgroup=subgroup,
                group_order=group_order,
                subgroup_order=subgroup_order,
                group_spacing=group_spacing,
            )
        else:
            group_order = [""]
            subgroup_order = [""]
            loc_dict = {}
            loc_dict[""] = 0.0
            width = 1

        x_ticks = [index for index, _ in enumerate(group_order)]
        self.plot_dict = {
            "data": data,
            "y": y,
            "group": group,
            "subgroup": subgroup,
            "group_order": group_order,
            "subgroup_order": subgroup_order,
            "unique_groups": unique_groups,
            "x_ticks": x_ticks,
            "loc_dict": loc_dict,
            "width": width,
            "ylabel": ylabel,
            "title": title,
            "transform": None,
            "back_transform_ticks": False,
        }
        self.plots = []
        self.plot_list = []

    def plot_settings(
        self,
        style: str = "default",
        legend_loc: tuple = "upper left",
        legend_anchor: tuple = (1, 1),
        ylim: Optional[list] = None,
        yscale: Literal["linear", "log", "symlog"] = "linear",
        xlabel_rotation: Union[Literal["horizontal", "vertical"], float] = "horizontal",
        steps: int = 5,
        margins=0.05,
        aspect: Union[int, float] = 1,
        figsize: Union[None, tuple[int, int]] = None,
        labelsize: int = 20,
        linewidth: int = 2,
        ticksize: int = 2,
        ticklabel: int = 20,
        decimals: int = None,
    ):
        self._plot_settings_run = True
        if "ylim" in self.plot_dict:
            ylim = self.plot_dict["ylim"]
        elif ylim is None:
            ylim = [None, None]

        plot_settings = {
            "ylim": ylim,
            "yscale": yscale,
            "steps": steps,
            "margins": margins,
            "aspect": aspect,
            "figsize": figsize,
            "labelsize": labelsize,
            "ticksize": ticksize,
            "ticklabel": ticklabel,
            "xlabel_rotation": xlabel_rotation,
            "decimals": decimals,
            "linewidth": linewidth,
            "legend_loc": legend_loc,
            "legend_anchor": legend_anchor,
        }
        self.plot_dict.update(plot_settings)

        plt.style.use(style)
        self.style = style

        # Quick check just for dark and default backgrounds
        for index, val in enumerate(self.plot_list):
            if val == "summary":
                if (
                    self.style == "dark_background"
                    and self.plots[index]["color_dict"] == "black"
                ):
                    self.plots[index]["color_dict"] = "white"
                elif (
                    self.style == "default"
                    and self.plots[index]["color_dict"] == "white"
                ):
                    self.plots[index]["color_dict"] = "black"

        if not self.inplace:
            return self

    def transform(
        self,
        transform: Optional[TRANSFORM] = (None,),
        back_transform_ticks: bool = False,
    ):
        self.plot_dict["transform"] = transform
        self.plot_dict["back_transform_ticks"] = back_transform_ticks

        if not self.inplace:
            return self

    def jitter(
        self,
        color: ColorDict = "black",
        marker: Union[str, dict[str, str]] = "o",
        edgecolor: ColorDict = "none",
        alpha: AlphaRange = 1.0,
        jitter: Union[float, int] = 1.0,
        seed: int = 42,
        markersize: float = 2.0,
        unique_id: Union[None] = None,
        legend: bool = False,
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
            "markersize": markersize,
            "unique_id": unique_id,
        }
        self.plots.append(jitter_plot)
        self.plot_list.append("jitter")

        if legend:
            self.plot_dict["legend_dict"] = (color, alpha)

        if not self.inplace:
            return self

    def jitteru(
        self,
        unique_id: Union[str, int, float],
        color: ColorDict = "black",
        marker: Union[str, dict[str, str]] = "o",
        edgecolor: ColorDict = "none",
        alpha: AlphaRange = 1.0,
        width: Union[float, int] = 1.0,
        duplicate_offset=0.0,
        markersize: float = 2.0,
        agg_func: Optional[AGGREGATE] = None,
        legend: bool = False,
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

        jitteru_plot = {
            "color_dict": color_dict,
            "marker_dict": marker_dict,
            "edgecolor_dict": edgecolor_dict,
            "alpha": alpha,
            "width": width * self.plot_dict["width"],
            "markersize": markersize,
            "unique_id": unique_id,
            "duplicate_offset": duplicate_offset,
            "agg_func": agg_func,
        }
        self.plots.append(jitteru_plot)
        self.plot_list.append("jitteru")

        if legend:
            self.plot_dict["legend_dict"] = (color, alpha)

        if not self.inplace:
            return self

    def summary(
        self,
        func: AGGREGATE = "mean",
        capsize: int = 0,
        capstyle: str = "round",
        barwidth: float = 1.0,
        err_func: ERROR = "sem",
        linewidth: int = 2,
        color: ColorDict = "black",
        alpha: float = 1.0,
        legend: bool = False,
    ):
        if self.style == "dark_background" and color == "black":
            color = "white"

        color_dict = process_args(
            color, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
        )

        summary_plot = {
            "func": func,
            "capsize": capsize,
            "capstyle": capstyle,
            "barwidth": barwidth * self.plot_dict["width"],
            "err_func": err_func,
            "linewidth": linewidth,
            "color_dict": color_dict,
            "alpha": alpha,
        }
        self.plots.append(summary_plot)
        self.plot_list.append("summary")

        if legend:
            self.plot_dict["legend_dict"] = (color, alpha)

        if not self.inplace:
            return self

    def summaryu(
        self,
        unique_id,
        func: AGGREGATE = "mean",
        agg_func: AGGREGATE = None,
        agg_width: float = 1.0,
        capsize: int = 0,
        capstyle: str = "round",
        barwidth: float = 1.0,
        err_func: ERROR = "sem",
        linewidth: int = 2,
        color: ColorDict = "black",
        alpha: float = 1.0,
        legend: bool = False,
    ):
        if self.style == "dark_background" and color == "black":
            color = "white"

        color_dict = process_args(
            color, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
        )

        summary_plot = {
            "func": func,
            "unique_id": unique_id,
            "agg_func": agg_func,
            "capsize": capsize,
            "capstyle": capstyle,
            "barwidth": barwidth * self.plot_dict["width"],
            "err_func": err_func,
            "linewidth": linewidth,
            "color_dict": color_dict,
            "alpha": alpha,
            "agg_width": agg_width,
        }
        self.plots.append(summary_plot)
        self.plot_list.append("summaryu")

        if legend:
            self.plot_dict["legend_dict"] = (color, alpha)

        if not self.inplace:
            return self

    def boxplot(
        self,
        facecolor="none",
        linecolor: ColorDict = "black",
        fliers="",
        box_width: float = 1.0,
        linewidth=1,
        alpha: AlphaRange = 1.0,
        line_alpha: AlphaRange = 1.0,
        showmeans: bool = False,
        show_ci: bool = False,
        legend: bool = False,
    ):
        color_dict = process_args(
            facecolor, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
        )

        linecolor_dict = process_args(
            linecolor, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
        )
        boxplot = {
            "color_dict": color_dict,
            "linecolor_dict": linecolor_dict,
            "fliers": fliers,
            "box_width": box_width * self.plot_dict["width"],
            "showmeans": showmeans,
            "show_ci": show_ci,
            "linewidth": linewidth,
            "alpha": alpha,
            "line_alpha": line_alpha,
        }
        self.plots.append(boxplot)
        self.plot_list.append("boxplot")

        if legend:
            self.plot_dict["legend_dict"] = (facecolor, alpha)

        if not self.inplace:
            return self

    def violin(
        self,
        facecolor="none",
        edgecolor: ColorDict = "black",
        alpha: AlphaRange = 1.0,
        showextrema: bool = False,
        violin_width: float = 1.0,
        showmeans: bool = True,
        showmedians: bool = False,
        legend: bool = False,
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
            "showmeans": showmeans,
            "showmedians": showmedians,
        }
        self.plots.append(violin)
        self.plot_list.append("violin")

        if legend:
            self.plot_dict["legend_dict"] = (facecolor, alpha)

        if not self.inplace:
            return self

    def percent(
        self,
        unique_id=None,
        facecolor="none",
        linecolor: ColorDict = "black",
        hatch=None,
        barwidth: float = 1.0,
        linewidth=1,
        alpha: float = 1.0,
        line_alpha=1.0,
        axis_type: Literal["density", "percent"] = "density",
        cutoff: Union[None, float, int, list[Union[float, int]]] = None,
        include_bins: Optional[list[bool]] = None,
        invert: bool = False,
        legend: bool = False,
    ):
        if isinstance(cutoff, (float, int)):
            cutoff = [cutoff]

        color_dict = process_args(
            facecolor, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
        )

        linecolor_dict = process_args(
            linecolor, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
        )
        if cutoff is None:
            cutoff = [self.plot_dict["data"][self.plot_dict["y"]].mean()]

        if include_bins is None:
            include_bins = [True] * (len(cutoff) + 1)

        percent_plot = {
            "color_dict": color_dict,
            "linecolor_dict": linecolor_dict,
            "cutoff": cutoff,
            "hatch": hatch,
            "barwidth": barwidth * self.plot_dict["width"],
            "linewidth": linewidth,
            "alpha": alpha,
            "line_alpha": line_alpha,
            "include_bins": include_bins,
            "unique_id": unique_id,
            "invert": invert,
        }
        self.plots.append(percent_plot)
        self.plot_list.append("percent")
        if axis_type == "density":
            self.plot_dict["ylim"] = [0.0, 1.0]
        else:
            self.plot_dict["ylim"] = [0, 100]

        if legend:
            self.plot_dict["legend_dict"] = (facecolor, alpha)

        if not self.inplace:
            return self

    def count(
        self,
        facecolor: ColorDict = "none",
        linecolor: ColorDict = "black",
        hatch=None,
        barwidth: float = 1.0,
        linewidth=1,
        alpha: float = 1.0,
        line_alpha=1.0,
        axis_type: Literal["density", "count", "percent"] = "density",
        legend: bool = False,
    ):
        # color_dict = process_args(
        #     facecolor, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
        # )
        if isinstance(facecolor, str):
            unique_ids_sub = self.plot_dict["data"][self.plot_dict["y"]].unique()
            facecolor = {key: facecolor for key in unique_ids_sub}

        if isinstance(linecolor, str):
            unique_ids_sub = self.plot_dict["data"][self.plot_dict["y"]].unique()
            linecolor = {key: linecolor for key in unique_ids_sub}

        count_plot = {
            "color_dict": facecolor,
            "linecolor_dict": linecolor,
            "hatch": hatch,
            "barwidth": barwidth * self.plot_dict["width"],
            "linewidth": linewidth,
            "alpha": alpha,
            "line_alpha": line_alpha,
            "axis_type": axis_type,
        }
        self.plots.append(count_plot)
        self.plot_list.append("count")

        if legend:
            self.plot_dict["legend_dict"] = (facecolor, alpha)

        if not self.inplace:
            return self

    def plot_legend(self):
        fig, ax = plt.subplots()

        handles = mp._make_legend_patches(
            color_dict=self.plot_dict["legend_dict"][0],
            alpha=self.plot_dict["legend_dict"][1],
            group=self.plot_dict["group_order"],
            subgroup=self.plot_dict["subgroup_order"],
        )
        ax.plot()
        ax.axis("off")
        ax.legend(handles=handles, frameon=False)
        return fig, ax

    def plot(
        self, savefig: bool = False, path=None, filetype="svg", backend="matplotlib"
    ):
        if not self._plot_settings_run:
            if self.inplace:
                self.plot_settings()
            else:
                self.inplace = True
                self.plot_settings()
                self.inplace = False

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

        for i, j in zip(self.plot_list, self.plots):
            plot_func = MP_PLOTS[i]
            plot_func(
                data=self.plot_dict["data"],
                y=self.plot_dict["y"],
                loc_dict=self.plot_dict["loc_dict"],
                unique_groups=self.plot_dict["unique_groups"],
                ax=ax,
                transform=self.plot_dict["transform"],
                **j,
            )

        if "count" in self.plot_list:
            decimals = None
        elif self.plot_dict["decimals"] is None:

            # No better way around this mess at the moment
            decimals = _decimals(self.plot_dict["data"][self.plot_dict["y"]])
        else:
            decimals = self.plot_dict["decimals"]
        ax.set_xticks(
            ticks=self.plot_dict["x_ticks"],
            labels=self.plot_dict["group_order"],
            rotation=self.plot_dict["xlabel_rotation"],
        )
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_linewidth(self.plot_dict["linewidth"])
        ax.spines["bottom"].set_linewidth(self.plot_dict["linewidth"])
        if "/" in self.plot_dict["y"]:
            self.plot_dict["y"] = self.plot_dict["y"].replace("/", "_")
        if self.plot_dict["yscale"] not in ["log", "symlog"]:
            ticks = ax.get_yticks()
            if self.plot_dict["ylim"][0] is None:
                self.plot_dict["ylim"][0] = ticks[0]
            if self.plot_dict["ylim"][1] is None:
                self.plot_dict["ylim"][1] = ticks[-1]
            ax.set_ylim(bottom=self.plot_dict["ylim"][0], top=self.plot_dict["ylim"][1])
            if decimals is not None:
                ticks = np.round(
                    np.linspace(
                        self.plot_dict["ylim"][0],
                        self.plot_dict["ylim"][1],
                        self.plot_dict["steps"],
                    ),
                    decimals=decimals,
                )
            else:
                ticks = np.linspace(
                    self.plot_dict["ylim"][0],
                    self.plot_dict["ylim"][1],
                    self.plot_dict["steps"],
                )
            if self.plot_dict["back_transform_ticks"]:
                tick_labels = np.round(
                    BACK_TRANSFORM_DICT[self.plot_dict["transform"]](ticks),
                    decimals=decimals,
                )
            else:
                tick_labels = np.round(ticks, decimals=decimals)
            ax.set_yticks(ticks, labels=tick_labels)
        else:
            ax.set_yscale(self.plot_dict["yscale"])
            ticks = ax.get_yticks()
            if self.plot_dict["ylim"][0] is None:
                self.plot_dict["ylim"][0] = ticks[0]
            if self.plot_dict["ylim"][1] is None:
                self.plot_dict["ylim"][1] = ticks[-1]
            ax.set_ylim(bottom=self.plot_dict["ylim"][0], top=self.plot_dict["ylim"][1])
        ax.set_ylabel(self.plot_dict["ylabel"], fontsize=self.plot_dict["labelsize"])
        ax.set_title(self.plot_dict["title"], fontsize=self.plot_dict["labelsize"])
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=self.plot_dict["ticklabel"],
            width=self.plot_dict["ticksize"],
        )
        ax.margins(x=self.plot_dict["margins"])

        if "legend_dict" in self.plot_dict:
            handles = mp._make_legend_patches(
                color_dict=self.plot_dict["legend_dict"][0],
                alpha=self.plot_dict["legend_dict"][1],
                group=self.plot_dict["group_order"],
                subgroup=self.plot_dict["subgroup_order"],
            )
            ax.legend(
                handles=handles,
                bbox_to_anchor=self.plot_dict["legend_anchor"],
                loc=self.plot_dict["legend_loc"],
                frameon=False,
            )

        if savefig:
            path = Path(path)
            if path.suffix[1:] not in MPL_SAVE_TYPES:
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
                data=self.plot_dict["data"],
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
                                    np.abs(self.plot_dict["data"][self.plot_dict["y"]])
                                )
                            )
                        )
                    )
                )
                + 2
            )
        else:
            decimals = self.plot_dict["decimals"]

        if self.plot_dict["ylim"][0] is None:
            self.plot_dict["ylim"][0] = (
                self.plot_dict["data"][self.plot_dict["y"]].min() * 0.9
            )
        if self.plot_dict["ylim"][1] is None:
            self.plot_dict["ylim"][1] = (
                self.plot_dict["data"][self.plot_dict["y"]].max() * 1.1
            )
        ticks = np.round(
            np.linspace(
                self.plot_dict["ylim"][0],
                self.plot_dict["ylim"][1],
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
                title=dict(text=self.plot_dict["ylabel"]),
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
        if savefig:
            path = Path(path)
            if path.suffix[1:] not in PLOTLY_SAVE_TYPES:
                path = path / f"{self.plot_dict['y']}.{filetype}"
            else:
                filetype = path.suffix[1:]
            if filetype == "html":
                fig.write_html(path)
        return fig


class GraphPlot:

    def __init__(self, graph):
        self.plot_dict = {}
        self.plot_dict["graph"] = graph
        self.plots = []

    def graphplot(
        self,
        marker_alpha: float = 0.8,
        line_alpha: float = 0.1,
        markersize: int = 2,
        markerscale: int = 1,
        linewidth: int = 1,
        edgecolor: str = "k",
        markercolor: str = "red",
        marker_attr: Optional[str] = None,
        cmap: str = "gray",
        seed: int = 42,
        scale: int = 50,
        plot_max_degree: bool = False,
        layout: Literal["spring", "circular", "communities"] = "spring",
    ):
        graph_plot = {
            "marker_alpha": marker_alpha,
            "line_alpha": line_alpha,
            "markersize": markersize,
            "markerscale": markerscale,
            "linewidth": linewidth,
            "edgecolor": edgecolor,
            "markercolor": markercolor,
            "marker_attr": marker_attr,
            "cmap": cmap,
            "seed": seed,
            "scale": scale,
            "layout": layout,
            "plot_max_degree": plot_max_degree,
        }

        self.plots.append(graph_plot)
