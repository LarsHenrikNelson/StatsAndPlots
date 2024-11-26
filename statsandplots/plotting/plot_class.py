from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Callable, Literal, Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..utils import (
    AGGREGATE,
    ERROR,
    TRANSFORM,
    DataHolder,
    get_backtransform,
    get_transform,
)
from . import matplotlib_plotting as mp
from . import plotly_plotting as plp
from .plot_utils import (
    _decimals,
    _process_colors,
    _process_positions,
    get_ticks,
    process_args,
    process_scatter_args,
    radian_ticks,
    create_dict,
)

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"


@dataclass
class ValueRange:
    lo: float
    hi: float


AlphaRange = Annotated[float, ValueRange(0.0, 1.0)]
ColorDict = Union[str, dict[str, str], None]

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


class BasePlot:

    def __init__(self):
        self._plot_format = {}

        label_props = {
            "labelsize": 20,
            "titlesize": 22,
            "ticklabel_size": 12,
            "font": "DejaVu Sans",
            "title_fontweight": "regular",
            "label_fontweight": "regular",
            "tick_fontweight": "regular",
            "xlabel_rotation": "horizontal",
        }

        self._plot_format["labels"] = label_props

        axis = {
            "style": "default",
            "yscale": "linear",
            "xscale": "linear",
            "tickwidth": 2,
            "ticklength": 5.0,
            "linewidth": 2.0,
            "ylim": [None, None],
            "xlim": [None, None],
            "ydecimals": None,
            "xdecimals": None,
            "xsteps": 5,
            "ysteps": 5,
            "tickstyle": "all",
            "xunits": None,
            "yminorticks": False,
            "xminorticks": False,
            "minor_tickwidth": 1.5,
            "minor_ticklength": 1.5,
        }
        self._plot_format["axis"] = axis

        figure = {
            "gridspec_kw": None,
            "margins": 0.05,
            "aspect": 1,
            "figsize": None,
            "nrows": None,
            "ncols": None,
            "projection": "rectilinear",
        }

        self._plot_format["figure"] = figure

        grid = {
            "ygrid": False,
            "xgrid": False,
            "linestyle": "solid",
            "ylinewidth": 1,
            "xlinewidth": 1,
        }

        self._plot_format["grid"] = grid

    def _set_grid(self, sub_ax):
        if self.plot_dict["grid"]["ygrid"]:
            sub_ax.yaxis.grid(
                linewidth=self.plot_dict["grid"]["ylinewidth"],
                linestyle=self.plot_dict["grid"]["linestyle"],
            )

        if self.plot_dict["grid"]["xgrid"]:
            sub_ax.xaxis.grid(
                linewidth=self.plot_dict["grid"]["xlinewidth"],
                linestyle=self.plot_dict["grid"]["linestyle"],
            )

    def add_axline(
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
            "linetype": linetype,
            "lines": lines,
            "linestyle": linestyle,
            "linealpha": linealpha,
            "linecolor": linecolor,
        }

        if not self.inplace:
            return self

    def _plot_axlines(self, line_dict, ax):
        for ll in line_dict["lines"]:
            if line_dict["linetype"] == "vline":
                ax.axvline(
                    ll,
                    linestyle=line_dict["linestyle"],
                    color=line_dict["linecolor"],
                    alpha=line_dict["linealpha"],
                )
            else:
                ax.axhline(
                    ll,
                    linestyle=line_dict["linestyle"],
                    color=line_dict["linecolor"],
                    alpha=line_dict["linealpha"],
                )

    def _set_lims(self, ax, decimals, axis="x"):
        if axis == "y":
            if self._plot_format["axis"]["yscale"] not in ["log", "symlog"]:
                ticks = ax.get_yticks()
                lim, ticks = get_ticks(
                    self._plot_format["axis"]["ylim"],
                    ticks,
                    self._plot_format["axis"]["steps"],
                    tickstyle=self._plot_format["axis"]["tickstyle"],
                )
                ax.set_ylim(bottom=lim[0], top=lim[1])
                if (
                    "back_transform_yticks" in self.plot_dict
                    and self.plot_dict["back_transform_yticks"]
                ):
                    tick_labels = get_backtransform(self.plot_dict["ytransform"])(ticks)
                else:
                    tick_labels = ticks
                if decimals is not None:
                    if decimals == -1:
                        tick_labels = tick_labels.astype(int)
                    else:
                        tick_labels = np.round(tick_labels, decimals=decimals)
                ax.set_yticks(
                    ticks,
                    labels=tick_labels,
                    fontfamily=self._plot_format["labels"]["font"],
                    fontweight=self._plot_format["labels"]["tick_fontweight"],
                    fontsize=self._plot_format["labels"]["ticklabel_size"],
                )
            else:
                ax.set_yscale(self._plot_format["axis"]["yscale"])
                ticks = ax.get_yticks()
                lim, _ = get_ticks(
                    self._plot_format["axis"]["ylim"],
                    ticks,
                    self._plot_format["axis"]["steps"],
                    decimals,
                )
                ax.set_ylim(bottom=lim[0], top=lim[1])
        else:
            if self._plot_format["axis"]["xscale"] not in ["log", "symlog"]:
                ticks = ax.get_xticks()
                lim, ticks = get_ticks(
                    self._plot_format["axis"]["xlim"],
                    ticks,
                    self._plot_format["axis"]["steps"],
                    tickstyle=self._plot_format["axis"]["tickstyle"],
                )
                ax.set_xlim(left=lim[0], right=lim[1])
                if (
                    "back_transform_xticks" in self.plot_dict
                    and self.plot_dict["back_transform_xticks"]
                ):
                    tick_labels = get_backtransform(self.plot_dict["xtransform"])(ticks)
                else:
                    tick_labels = ticks
                if decimals is not None:
                    if decimals == -1:
                        tick_labels = tick_labels.astype(int)
                    else:
                        tick_labels = np.round(tick_labels, decimals=decimals)
                ax.set_xticks(
                    ticks,
                    labels=tick_labels,
                    fontfamily=self._plot_format["labels"]["font"],
                    fontweight=self._plot_format["labels"]["tick_fontweight"],
                    fontsize=self._plot_format["labels"]["ticklabel_size"],
                )

            else:
                ax.set_xscale(self._plot_format["axis"]["xscale"])
                ticks = ax.get_xticks()
                lim, _ = get_ticks(
                    self._plot_format["axis"]["xlim"],
                    ticks,
                    self._plot_format["axis"]["steps"],
                    decimals,
                )
                ax.set_xlim(left=lim[0], right=lim[1])

    def _set_minorticks(self, ax, transform: str, ticks: Literal["y", "x"]):
        if ticks == "y":
            yticks = ax.get_yticks()
        else:
            yticks = ax.get_xticks()
        yticks = get_backtransform(transform)(yticks)
        mticks = np.zeros((len(yticks) - 1) * 5)
        for index in range(yticks.size - 1):
            vals = np.linspace(yticks[index], yticks[index + 1], num=5, endpoint=False)
            start = index * 5
            end = index * 5 + 5
            mticks[start:end] = vals
        if ticks == "y":
            ax.set_yticks(
                get_transform(transform)(mticks),
                minor=True,
            )
            ax.tick_params(
                axis="y",
                which="minor",
                width=self._plot_format["axis"]["minor_tickwidth"],
                length=self._plot_format["axis"]["minor_ticklength"],
                labelfontfamily=self._plot_format["labels"]["font"],
            )

        else:
            ax.set_xticks(
                get_transform(transform)(mticks),
                minor=True,
            )
            ax.tick_params(
                axis="x",
                which="minor",
                width=self._plot_format["axis"]["minor_tickwidth"],
                length=self._plot_format["axis"]["minor_ticklength"],
                labelfontfamily=self._plot_format["labels"]["font"],
            )

    def labels(
        self,
        labelsize: float = 20,
        titlesize: float = 22,
        ticklabel_size: int = 12,
        font: str = "DejaVu Sans",
        fontweight: None | str | float = None,
        title_fontweight: str | float = "regular",
        label_fontweight: str | float = "regular",
        tick_fontweight: str | float = "regular",
        xlabel_rotation: Union[Literal["horizontal", "vertical"], float] = "horizontal",
    ):
        if fontweight is not None:
            title_fontweight = fontweight
            label_fontweight = fontweight
            tick_fontweight = fontweight

        label_props = {
            "labelsize": labelsize,
            "titlesize": titlesize,
            "font": font,
            "ticklabel_size": ticklabel_size,
            "title_fontweight": title_fontweight,
            "label_fontweight": label_fontweight,
            "tick_fontweight": tick_fontweight,
            "xlabel_rotation": xlabel_rotation,
        }
        self._plot_format["labels"] = label_props
        if not self.inplace:
            return self

    def axis(
        self,
        style: str = "default",
        ylim: Optional[list] = None,
        xlim: Optional[list] = None,
        yscale: Literal["linear", "log", "symlog"] = "linear",
        xscale: Literal["linear", "log", "symlog"] = "linear",
        linewidth: float = 2,
        tickwidth: float = 2,
        ticklength: float = 5.0,
        yminorticks: bool = False,
        xminorticks: bool = False,
        minor_tickwidth: float = 1.5,
        minor_ticklength: float = 2.5,
        ysteps: int = 5,
        xsteps: int = 5,
        tickstyle: Literal["all", "middle"] = "all",
        ydecimals: int = None,
        xdecimals: int = None,
        xunits: Optional[Literal["degree", "radian" "wradian"]] = None,
    ):
        if ylim is None:
            ylim = [None, None]
        if xlim is None:
            xlim = [None, None]

        plot_settings = {
            "yscale": yscale,
            "xscale": xscale,
            "tickwidth": tickwidth,
            "ticklength": ticklength,
            "linewidth": linewidth,
            "ylim": ylim,
            "xlim": xlim,
            "ydecimals": ydecimals,
            "xdecimals": xdecimals,
            "xsteps": xsteps,
            "ysteps": ysteps,
            "tickstyle": tickstyle,
            "xunits": xunits,
            "yminorticks": yminorticks,
            "xminorticks": xminorticks,
            "minor_tickwidth": minor_tickwidth,
            "minor_ticklength": minor_ticklength,
        }
        self._plot_format["axis"] = plot_settings

        plt.style.use(style)

        # Quick check just for dark and default backgrounds
        # for index, val in enumerate(self.plot_list):
        #     if val in ["line", "summary"]:
        #         if (
        #             self.style == "dark_background"
        #             and self.plots[index]["color"] == "black"
        #         ):
        #             self.plots[index]["color"] = "white"
        #         elif self.style == "default" and self.plots[index]["color"] == "white":
        #             self.plots[index]["color"] = "black"

        if not self.inplace:
            return self

    def figure(
        self,
        margins=0.05,
        aspect: Union[int, float] = 1,
        figsize: Union[None, tuple[int, int]] = None,
        gridspec_kw: dict[str, Union[str, int, float]] = None,
        nrows: int = None,
        ncols: int = None,
        projection: Literal["rectilinear", "polar"] = "rectilinear",
    ):

        figure = {
            "gridspec_kw": gridspec_kw,
            "margins": margins,
            "aspect": (aspect if projection == "rectilinear" else None),
            "figsize": figsize,
            "nrows": nrows,
            "ncols": ncols,
        }

        self._plot_format["figure"] = figure

        if not self.inplace:
            return self

    def grid_settings(
        self,
        ygrid: bool = False,
        xgrid: bool = False,
        linestyle: Union[str, tuple] = "solid",
        ylinewidth: Union[float, int] = 1,
        xlinewidth: Union[float, int] = 1,
    ):

        grid_settings = {
            "ygrid": ygrid,
            "xgrid": xgrid,
            "linestyle": linestyle,
            "xlinewidth": xlinewidth,
            "ylinewidth": ylinewidth,
        }
        self._plot_format["grid"] = grid_settings

        if not self.inplace:
            return self

    def plot(
        self,
        savefig: bool = False,
        path=None,
        filename="",
        filetype="svg",
        backend="matplotlib",
    ):

        if backend == "matplotlib":
            output = self._matplotlib_backend(
                savefig=savefig, path=path, filetype=filetype, filename=filename
            )
            return output
        elif backend == "plotly":
            output = self._plotly_backend(
                savefig=savefig, path=path, filetype=filetype, filename=filename
            )
        else:
            raise AttributeError("Backend not implemented")
        return output

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

    def get_format(self):
        return self._plot_format


class LinePlot(BasePlot):
    ecdf_args = {
        "spline": {"size": 1000, "bc_type": "natural"},
        "bootstrap": {"size": 1000, "repititions": 1000, "seed": 42},
    }

    def __init__(
        self,
        data: pd.DataFrame,
        y: Optional[str] = None,
        x: Optional[str] = None,
        group: Optional[str] = None,
        subgroup: Optional[str] = None,
        group_order: Optional[list[str]] = None,
        subgroup_order: Optional[list[str]] = None,
        ylabel: str = "",
        xlabel: str = "",
        title: str = "",
        facet: bool = False,
        facet_title: bool = False,
        inplace: bool = False,
    ):
        super().__init__()

        self.inplace = inplace
        self.plots = []
        self.plot_list = []
        self._plot_settings_run = False

        data = DataHolder(data)

        if group is None:
            unique_groups = {("",)}
            group_order = [""]
            levels = []
        elif subgroup is None:
            if group_order is None:
                group_order = np.unique(data[group])
            unique_groups = {(g,) for g in group_order}
            levels = [group]
        else:
            if group_order is None:
                group_order = np.unique(data[group])
            if subgroup_order is None:
                subgroup_order = np.unique(data[subgroup])
            unique_groups = set(zip(data[group], data[subgroup]))
            levels = [group, subgroup]

        if facet:
            facet_dict = create_dict(group_order, unique_groups)
        else:
            facet_dict = create_dict(0, unique_groups)

        self.plot_dict = {
            "data": data,
            "y": y,
            "x": x,
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
            "facet_title": facet_title,
            "xtransform": None,
            "ytransform": None,
            "levels": levels,
        }
        self.lines = {}

    def transform(
        self,
        ytransform: Optional[TRANSFORM] = (None,),
        back_transform_yticks: bool = False,
        xtransform: Optional[TRANSFORM] = (None,),
        back_transform_xticks: bool = False,
    ):
        self.plot_dict["ytransform"] = ytransform
        if callable(ytransform):
            self.plot_dict["back_transform_yticks"] = False
        else:
            self.plot_dict["back_transform_yticks"] = back_transform_yticks

        self.plot_dict["xtransform"] = xtransform
        if callable(xtransform):
            self.plot_dict["back_transform_xticks"] = False
        else:
            self.plot_dict["back_transform_xticks"] = back_transform_xticks

        if not self.inplace:
            return self

    def line(
        self,
        linecolor: ColorDict = "black",
        linestyle: str = "-",
        linewidth: int = 2,
        func: str = "mean",
        err_func: str = "sem",
        fit_func: Optional[Callable[[Union[np.ndarray, pd.Series]], np.ndarray]] = None,
        alpha: AlphaRange = 1.0,
        unique_id: Optional[str] = None,
    ):
        linecolor_dict = create_dict(linecolor, self.plot_dict["unique_groups"])
        linestyle_dict = create_dict(linestyle, self.plot_dict["unique_groups"])
        line_plot = {
            "color_dict": linecolor_dict,
            "linestyle_dict": linestyle_dict,
            "linewidth": linewidth,
            "func": func,
            "err_func": err_func,
            "fit_func": fit_func,
            "alpha": alpha,
            "unique_id": unique_id,
        }
        self.plots.append(line_plot)
        self.plot_list.append("line_plot")

        if not self.inplace:
            return self

    def aggline(
        self,
        marker: str = "none",
        markerfacecolor: Union[ColorDict, tuple[str, str]] = None,
        markeredgecolor: Union[ColorDict, tuple[str, str]] = None,
        markersize: Union[float, str] = 1,
        linecolor: ColorDict = None,
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
            linecolor = _process_colors(
                linecolor,
                self.plot_dict["group_order"],
                self.plot_dict["subgroup_order"],
            )
            linecolor_dict = create_dict(
                linecolor,
                self.plot_dict["unique_groups"],
            )
            markerfacecolor_dict = create_dict(
                markerfacecolor,
                self.plot_dict["unique_groups"],
            )
            markeredgecolor_dict = create_dict(
                markeredgecolor,
                self.plot_dict["unique_groups"],
            )
        else:
            temp_dict = create_dict(colorall, self.plot_dict["unique_groups"])
            markeredgecolor_dict = temp_dict
            markerfacecolor_dict = temp_dict
            linecolor_dict = temp_dict

        marker_dict = create_dict(marker, self.plot_dict["unique_groups"])
        linestyle_dict = create_dict(linestyle, self.plot_dict["unique_groups"])

        line_plot = {
            "linecolor": linecolor_dict,
            "linestyle": linestyle_dict,
            "linewidth": linewidth,
            "func": func,
            "err_func": err_func,
            "linealpha": linealpha,
            "fill_between": fill_between,
            "fillalpha": fillalpha,
            "sort": sort,
            "marker": marker_dict,
            "markerfacecolor": markerfacecolor_dict,
            "markeredgecolor": markeredgecolor_dict,
            "markersize": markersize,
            "unique_id": unique_id,
            "agg_func": agg_func,
        }
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
        bw: Literal["ISJ", "silverman", "scott"] = "silverman",
        tol: Union[float, int] = 1e-3,
        common_norm: bool = True,
        linecolor: Optional[ColorDict] = None,
        linestyle: str = "-",
        linewidth: int = 2,
        fill_under: bool = False,
        alpha: AlphaRange = 1.0,
        fillalpha=None,
        unique_id: Optional[str] = None,
        agg_func=None,
        err_func=None,
        kde_type: Literal["tree", "fft"] = "fft",
    ):

        linecolor = _process_colors(
            linecolor,
            self.plot_dict["group_order"],
            self.plot_dict["subgroup_order"],
        )
        linecolor_dict = create_dict(
            linecolor,
            self.plot_dict["unique_groups"],
        )

        linestyle_dict = create_dict(
            linestyle,
            self.plot_dict["unique_groups"],
        )

        kde_plot = {
            "linecolor_dict": linecolor_dict,
            "linestyle_dict": linestyle_dict,
            "linewidth": linewidth,
            "alpha": alpha,
            "fill_under": fill_under,
            "kernel": kernel,
            "bw": bw,
            "tol": tol,
            "common_norm": common_norm,
            "unique_id": unique_id,
            "agg_func": agg_func,
            "err_func": err_func,
            "kde_type": kde_type,
            "fillalpha": alpha / 2 if fillalpha is None else fillalpha,
        }

        self.plots.append(kde_plot)
        self.plot_list.append("kde")

        if not self.inplace:
            return self

    def polyhist(
        self,
        color: Optional[ColorDict] = None,
        linestyle: str = "-",
        linewidth: int = 2,
        bin_limits=None,
        density=True,
        nbins=50,
        func="mean",
        err_func="sem",
        fit_func=None,
        alpha: AlphaRange = 1.0,
        unique_id: Optional[str] = None,
    ):
        color_dict = process_args(
            _process_colors(
                color, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
            ),
            self.plot_dict["group_order"],
            self.plot_dict["subgroup_order"],
        )
        linestyle_dict = process_args(
            linestyle, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
        )

        if bin_limits is not None and len(bin_limits) != 2:
            raise AttributeError("bin_limits must be length 2")

        poly_hist = {
            "color_dict": color_dict,
            "linestyle_dict": linestyle_dict,
            "linewidth": linewidth,
            "density": density,
            "bin_limits": bin_limits,
            "nbins": nbins,
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

    def hist(
        self,
        hist_type: Literal["bar", "step", "stepfilled"] = "bar",
        color: ColorDict = None,
        linecolor: ColorDict = None,
        linewidth: int = 2,
        hatch=None,
        fillalpha: float = 1.0,
        linealpha: float = 1.0,
        bin_limits=None,
        stat: Literal["density", "probability", "count"] = "density",
        nbins=50,
        err_func=None,
        agg_func=None,
        unique_id=None,
    ):
        color = _process_colors(
            color,
            self.plot_dict["group_order"],
            self.plot_dict["subgroup_order"],
        )
        color_dict = create_dict(color, self.plot_dict["unique_groups"])
        linecolor = _process_colors(
            linecolor,
            self.plot_dict["group_order"],
            self.plot_dict["subgroup_order"],
        )
        linecolor_dict = create_dict(linecolor, self.plot_dict["unique_groups"])

        hist = {
            "color_dict": color_dict,
            "linecolor_dict": linecolor_dict,
            "linewidth": linewidth,
            "hatch": hatch,
            "stat": stat,
            "bin_limits": bin_limits,
            "nbins": nbins,
            "agg_func": agg_func,
            "err_func": err_func,
            "unique_id": unique_id,
            "fillalpha": fillalpha,
            "linealpha": linealpha,
            "projection": self.plot_dict["projection"],
        }
        self.plots.append(hist)
        self.plot_list.append("hist")

        if self._plot_format["figure"]["projection"] == "polar":
            self._plot_format["grid"]["ygrid"] = True
            self._plot_format["grid"]["xgrid"] = True

        if not self.inplace:
            return self

    def ecdf(
        self,
        color: ColorDict = None,
        linestyle: str = "-",
        linewidth: int = 2,
        alpha: AlphaRange = 1.0,
        fillalpha: AlphaRange = 0.5,
        unique_id: Optional[str] = None,
        agg_func=None,
        err_func=None,
        ecdf_type: Literal["spline", "bootstrap", "none"] = "none",
        ecdf_args=None,
    ):
        color = _process_colors(
            color, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
        )
        color_dict = create_dict(color, self.plot_dict["unique_groups"])
        linestyle_dict = create_dict(linestyle, self.plot_dict["unique_groups"])

        ecdf = {
            "color_dict": color_dict,
            "linestyle_dict": linestyle_dict,
            "linewidth": linewidth,
            "alpha": alpha,
            "unique_id": unique_id,
            "ecdf_type": ecdf_type,
            "ecdf_args": ecdf_args if ecdf_args is not None else {},
            "agg_func": agg_func,
            "err_func": err_func,
            "fillalpha": fillalpha,
        }
        self.plots.append(ecdf)
        self.plot_list.append("ecdf")

        if not self.inplace:
            return self

    def scatter(
        self,
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
            self.plot_dict["levels"],
            self.plot_dict["unique_groups"],
            markercolor1,
            alpha=alpha,
        )
        edgecolors = process_scatter_args(
            edgecolor0,
            self.plot_dict["data"],
            self.plot_dict["levels"],
            self.plot_dict["unique_groups"],
            edgecolor1,
            alpha=alpha,
        )
        markersize = process_scatter_args(
            markersize,
            self.plot_dict["data"],
            self.plot_dict["levels"],
            self.plot_dict["unique_groups"],
        )
        facetgroup = process_scatter_args(
            self.plot_dict["facet_dict"],
            self.plot_dict["data"],
            self.plot_dict["levels"],
            self.plot_dict["unique_groups"],
        )
        plot_data = {
            "markers": marker,
            "markercolors": colors,
            "edgecolors": edgecolors,
            "markersizes": markersize,
            "facetgroup": facetgroup,
        }

        self.plot_list.append("scatter")
        self.plots.append(plot_data)

        if not self.inplace:
            return self

    def _matplotlib_backend(
        self,
        savefig: bool = False,
        path: str = "",
        filetype: str = "svg",
        filename: str = "",
        transparent=False,
    ):
        if (
            self._plot_format["figure"]["nrows"] is None
            and self._plot_format["figure"]["ncols"] is None
        ):
            nrows = len(self.plot_dict["group_order"])
            ncols = 1
        elif self._plot_format["figure"]["nrows"] is None:
            nrows = 1
            ncols = self._plot_format["figure"]["ncols"]
        elif self._plot_format["figure"]["ncols"] is None:
            nrows = self._plot_format["figure"]["nrows"]
            ncols = 1
        else:
            nrows = self._plot_format["figure"]["nrows"]
            ncols = self._plot_format["figure"]["ncols"]
        if self.plot_dict["facet"]:
            fig, ax = plt.subplots(
                subplot_kw=dict(
                    box_aspect=self._plot_format["axis"]["aspect"],
                    projection=self._plot_format["figure"]["projection"],
                ),
                figsize=self._plot_format["figure"]["figsize"],
                gridspec_kw=self._plot_format["figure"]["gridspec_kw"],
                ncols=ncols,
                nrows=nrows,
            )
            ax = ax.flatten()
        else:
            fig, ax = plt.subplots(
                subplot_kw=dict(
                    box_aspect=self._plot_format["figure"]["aspect"],
                    projection=self._plot_format["figure"]["projection"],
                ),
                figsize=self._plot_format["figure"]["figsize"],
            )
            ax = [ax]
        for i, j in zip(self.plot_list, self.plots):
            plot_func = MP_PLOTS[i]
            plot_func(
                data=self.plot_dict["data"],
                y=self.plot_dict["y"],
                x=self.plot_dict["x"],
                unique_groups=self.plot_dict["unique_groups"],
                facet_dict=self.plot_dict["facet_dict"],
                ax=ax,
                ytransform=self.plot_dict["ytransform"],
                xtransform=self.plot_dict["xtransform"],
                levels=self.plot_dict["levels"],
                **j,
            )
            if i == "kde" or i == "hist":
                if self.plot_dict["x"] is not None:
                    if self._plot_format["axis"]["ylim"] is None:
                        self._plot_format["axis"]["ylim"] = [0, None]
                else:
                    if self._plot_format["axis"]["xlim"] is None:
                        self._plot_format["axis"]["xlim"] = [0, None]
            if i == "ecdf":
                self._plot_format["axis"]["ylim"] = [0.0, 1.0]

        if (
            self._plot_format["axis"]["ydecimals"] is None
            and self.plot_dict["data"][self.plot_dict["y"]] is not None
        ):
            ydecimals = _decimals(self.plot_dict["data"][self.plot_dict["y"]])
        else:
            ydecimals = self._plot_format["axis"]["ydecimals"]
        if (
            self._plot_format["axis"]["xdecimals"] is None
            and self.plot_dict["data"][self.plot_dict["x"]] is not None
        ):
            xdecimals = _decimals(self.plot_dict["data"][self.plot_dict["x"]])
        else:
            xdecimals = self._plot_format["axis"]["xdecimals"]
        # num_plots = len(self.plot_dict["group_order"])
        for index, sub_ax in enumerate(ax[: len(self.plot_dict["group_order"])]):
            if self._plot_format["figure"]["projection"] == "rectilinear":
                sub_ax.autoscale()
                sub_ax.spines["right"].set_visible(False)
                sub_ax.spines["top"].set_visible(False)
                sub_ax.spines["left"].set_linewidth(
                    self._plot_format["axis"]["linewidth"]
                )
                sub_ax.spines["bottom"].set_linewidth(
                    self._plot_format["axis"]["linewidth"]
                )

                self._set_lims(sub_ax, ydecimals, axis="y")
                self._set_lims(sub_ax, xdecimals, axis="x")

                if self._plot_format["axis"]["yminorticks"]:
                    self._set_minorticks(
                        sub_ax, self.plot_dict["ytransform"], ticks="y"
                    )

                if self._plot_format["axis"]["xminorticks"]:
                    self._set_minorticks(
                        sub_ax, self.plot_dict["xtransform"], ticks="x"
                    )

                sub_ax.margins(self._plot_format["figure"]["margins"])
                sub_ax.set_xlabel(
                    self.plot_dict["xlabel"],
                    fontsize=self._plot_format["labels"]["labelsize"],
                    fontweight=self._plot_format["labels"]["label_fontweight"],
                    fontfamily=self._plot_format["labels"]["font"],
                )
            else:
                if (
                    self._plot_format["axis"]["xunits"] == "radian"
                    or self._plot_format["axis"]["xunits"] == "wradian"
                ):
                    xticks = sub_ax.get_xticks()
                    labels = (
                        radian_ticks(xticks, rotate=False)
                        if self._plot_format["axis"]["xunits"] == "radian"
                        else radian_ticks(xticks, rotate=True)
                    )
                    sub_ax.set_xticks(
                        xticks,
                        labels,
                        fontfamily=self._plot_format["labels"]["font"],
                        fontweight=self._plot_format["labels"]["tick_fontweight"],
                        fontsize=self._plot_format["labels"]["ticklabel_size"],
                    )
                sub_ax.spines["polar"].set_visible(False)
                sub_ax.set_xlabel(
                    self.plot_dict["xlabel"],
                    fontsize=self._plot_format["labels"]["labelsize"],
                    fontweight=self._plot_format["labels"]["label_fontweight"],
                    fontfamily=self._plot_format["labels"]["font"],
                )
                sub_ax.set_rmax(sub_ax.dataLim.ymax)
                ticks = sub_ax.get_yticks()
                sub_ax.set_yticks(
                    ticks,
                    fontfamily=self._plot_format["labels"]["font"],
                    fontweight=self._plot_format["labels"]["tick_fontweight"],
                    fontsize=self._plot_format["labels"]["ticklabel_size"],
                )
            if "hline" in self.plot_dict:
                self._plot_axlines(self.plot_dict["hline"], sub_ax)

            if "vline" in self.plot_dict:
                self._plot_axlines(self.plot_dict["vline"], sub_ax)

            sub_ax.tick_params(
                axis="both",
                which="major",
                labelsize=self._plot_format["labels"]["ticklabel_size"],
                width=self._plot_format["axis"]["tickwidth"],
                length=self._plot_format["axis"]["ticklength"],
                labelfontfamily=self._plot_format["labels"]["font"],
            )

            self._set_grid(sub_ax)

            if "/" in str(self.plot_dict["y"]):
                self.plot_dict["y"] = self.plot_dict["y"].replace("/", "_")

            sub_ax.set_ylabel(
                self.plot_dict["ylabel"],
                fontsize=self._plot_format["labels"]["labelsize"],
                fontfamily=self._plot_format["labels"]["font"],
                fontweight=self._plot_format["labels"]["label_fontweight"],
            )
            if self.plot_dict["facet_title"]:
                sub_ax.set_title(
                    self.plot_dict["group_order"][index],
                    fontsize=self._plot_format["labels"]["labelsize"],
                    fontfamily=self._plot_format["labels"]["font"],
                )
            else:
                sub_ax.set_title(
                    self.plot_dict["title"],
                    fontsize=self._plot_format["labels"]["labelsize"],
                    fontfamily=self._plot_format["labels"]["font"],
                )

        if self.plot_dict["title"] is not None:
            fig.suptitle(
                self.plot_dict["title"],
                fontsize=self._plot_format["labels"]["titlesize"],
            )

        if self._plot_format["figure"]["projection"] == "rectilinear":
            if self._plot_format["figure"]["gridspec_kw"] is None:
                fig.tight_layout()

        if savefig:
            path = Path(path)
            if path.suffix[1:] not in MPL_SAVE_TYPES:
                filename = self.plot_dict["y"] if filename == "" else filename
                path = path / f"{filename}.{filetype}"
            else:
                filetype = path.suffix[1:]
            plt.savefig(
                path,
                format=filetype,
                bbox_inches="tight",
                transparent=transparent,
            )
        return fig, ax


class CategoricalPlot(BasePlot):
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
        inplace: bool = False,
    ):
        super().__init__()

        self._plot_settings_run = False
        self.inplace = inplace
        self.style = "default"
        self.plots = []
        self.plot_list = []

        data = DataHolder(data)

        if group is None:
            unique_groups = {("",)}
            group_order = [""]
            levels = []
        elif subgroup is None:
            if group_order is None:
                group_order = np.unique(data[group])
            unique_groups = {(g,) for g in group_order}
            levels = [group]
        else:
            if group_order is None:
                group_order = np.unique(data[group])
            if subgroup_order is None:
                subgroup_order = np.unique(data[subgroup])
            unique_groups = set(zip(data[group], data[subgroup]))
            levels = [group, subgroup]

        if group is not None:
            loc_dict, width = _process_positions(
                subgroup=subgroup,
                group_order=group_order,
                subgroup_order=subgroup_order,
                group_spacing=group_spacing,
            )
        else:
            group_order = [("",)]
            subgroup_order = [("",)]
            loc_dict = {("",): 0.0}
            loc_dict[("",)] = 0.0
            width = 1.0

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
            "ytransform": None,
            "back_transform_ticks": False,
            "levels": levels,
        }

    def transform(
        self,
        transform: Optional[TRANSFORM] = None,
        back_transform_ticks: bool = False,
    ):
        self.plot_dict["ytransform"] = transform

        if callable(transform):
            self.plot_dict["back_transform_yticks"] = False
        else:
            self.plot_dict["back_transform_yticks"] = back_transform_ticks

        if not self.inplace:
            return self

    def jitter(
        self,
        color: ColorDict = None,
        marker: Union[str, dict[str, str]] = "o",
        edgecolor: Optional[ColorDict] = "none",
        alpha: AlphaRange = 1.0,
        width: Union[float, int] = 1.0,
        seed: int = 42,
        markersize: float = 2.0,
        unique_id: Union[None] = None,
        legend: bool = False,
    ):
        marker_dict = create_dict(marker, self.plot_dict["unique_groups"])
        color = _process_colors(
            color, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
        )
        color_dict = create_dict(color, self.plot_dict["unique_groups"])

        edgecolor_dict = create_dict(edgecolor, self.plot_dict["unique_groups"])

        jitter_plot = {
            "color_dict": color_dict,
            "marker_dict": marker_dict,
            "edgecolor_dict": edgecolor_dict,
            "alpha": alpha,
            "width": width * self.plot_dict["width"],
            "seed": seed,
            "markersize": markersize,
            "unique_id": unique_id,
        }
        self.plots.append(jitter_plot)
        self.plot_list.append("jitter")

        if legend:
            if color is not None or edgecolor == "none":
                d = color
            else:
                d = edgecolor
            d = _process_colors(
                d, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
            )
            self.plot_dict["legend_dict"] = (d, alpha)

        if not self.inplace:
            return self

    def jitteru(
        self,
        unique_id: Union[str, int, float],
        color: ColorDict = None,
        marker: Union[str, dict[str, str]] = "o",
        edgecolor: ColorDict = None,
        alpha: AlphaRange = 1.0,
        edge_alpha: AlphaRange = None,
        width: Union[float, int] = 1.0,
        duplicate_offset=0.0,
        markersize: float = 2.0,
        agg_func: Optional[AGGREGATE] = None,
        legend: bool = False,
    ):
        marker_dict = create_dict(marker, self.plot_dict["unique_groups"])
        color = _process_colors(
            color, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
        )
        color_dict = create_dict(color, self.plot_dict["unique_groups"])

        if edgecolor is None:
            edgecolor_dict = color_dict
        else:
            edgecolor_dict = create_dict(edgecolor, self.plot_dict["unique_groups"])

        if edge_alpha is None:
            edge_alpha = alpha

        jitteru_plot = {
            "color_dict": color_dict,
            "marker_dict": marker_dict,
            "edgecolor_dict": edgecolor_dict,
            "alpha": alpha,
            "edge_alpha": edge_alpha,
            "width": width * self.plot_dict["width"],
            "markersize": markersize,
            "unique_id": unique_id,
            "duplicate_offset": duplicate_offset,
            "agg_func": agg_func,
        }
        self.plots.append(jitteru_plot)
        self.plot_list.append("jitteru")

        if legend:
            if color is not None or edgecolor == "none":
                d = color
            else:
                d = edgecolor
            d = _process_colors(
                d, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
            )
            self.plot_dict["legend_dict"] = (d, alpha)

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

        color = _process_colors(
            color, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
        )
        color_dict = create_dict(color, self.plot_dict["unique_groups"])

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
            d = _process_colors(
                color, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
            )
            self.plot_dict["legend_dict"] = (d, alpha)

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

        color = _process_colors(
            color, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
        )
        color_dict = create_dict(color, self.plot_dict["unique_groups"])

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
            d = _process_colors(
                color, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
            )
            self.plot_dict["legend_dict"] = (d, alpha)

        if not self.inplace:
            return self

    def boxplot(
        self,
        facecolor=None,
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
        color = _process_colors(
            facecolor, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
        )
        color_dict = create_dict(color, self.plot_dict["unique_groups"])

        linecolor = _process_colors(
            linecolor, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
        )
        linecolor_dict = create_dict(linecolor, self.plot_dict["unique_groups"])
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
            if facecolor is not None or linecolor == "black":
                d = facecolor
            else:
                d = linecolor
            d = _process_colors(
                d, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
            )
            self.plot_dict["legend_dict"] = (d, alpha)

        if not self.inplace:
            return self

    def violin(
        self,
        facecolor: ColorDict = None,
        edgecolor: ColorDict = "black",
        linewidth=1,
        alpha: AlphaRange = 1.0,
        showextrema: bool = False,
        width: float = 1.0,
        showmeans: bool = True,
        showmedians: bool = False,
        legend: bool = False,
    ):
        color = _process_colors(
            facecolor, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
        )
        color_dict = create_dict(color, self.plot_dict["unique_groups"])
        edgecolor = _process_colors(
            edgecolor, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
        )
        edge_dict = create_dict(edgecolor, self.plot_dict["unique_groups"])
        violin = {
            "color_dict": color_dict,
            "edge_dict": edge_dict,
            "alpha": alpha,
            "showextrema": showextrema,
            "width": width * self.plot_dict["width"],
            "showmeans": showmeans,
            "showmedians": showmedians,
            "linewidth": linewidth,
        }
        self.plots.append(violin)
        self.plot_list.append("violin")

        if legend:
            if facecolor is not None or edgecolor == "black":
                d = facecolor
            else:
                d = edgecolor
            d = _process_colors(
                d, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
            )
            self.plot_dict["legend_dict"] = (d, alpha)

        if not self.inplace:
            return self

    def percent(
        self,
        cutoff: Union[None, float, int, list[Union[float, int]]],
        unique_id=None,
        facecolor=None,
        linecolor: ColorDict = "black",
        hatch=None,
        barwidth: float = 1.0,
        linewidth=1,
        alpha: float = 1.0,
        line_alpha=1.0,
        axis_type: Literal["density", "percent"] = "density",
        include_bins: Optional[list[bool]] = None,
        invert: bool = False,
        legend: bool = False,
    ):
        if isinstance(cutoff, (float, int)):
            cutoff = [cutoff]
        facecolor = _process_colors(
            facecolor,
            self.plot_dict["group_order"],
            self.plot_dict["subgroup_order"],
        )
        color_dict = create_dict(facecolor, self.plot_dict["unique_groups"])

        linecolor = _process_colors(
            linecolor,
            self.plot_dict["group_order"],
            self.plot_dict["subgroup_order"],
        )
        linecolor_dict = create_dict(linecolor, self.plot_dict["unique_groups"])

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
            "axis_type": axis_type,
        }
        self.plots.append(percent_plot)
        self.plot_list.append("percent")
        if axis_type == "density":
            self._plot_format["axis"]["ylim"] = [0.0, 1.0]
        else:
            self._plot_format["axis"]["ylim"] = [0, 100]

        if legend:
            if facecolor is not None or linecolor == "black":
                d = facecolor
            else:
                d = linecolor
            d = _process_colors(
                d, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
            )
            self.plot_dict["legend_dict"] = (d, alpha)

        if not self.inplace:
            return self

    def count(
        self,
        facecolor: ColorDict = None,
        linecolor: ColorDict = "black",
        hatch=None,
        barwidth: float = 1.0,
        linewidth=1,
        alpha: float = 1.0,
        line_alpha=1.0,
        axis_type: Literal["density", "count", "percent"] = "density",
        legend: bool = False,
    ):

        facecolor = _process_colors(
            facecolor, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
        )
        color_dict = create_dict(facecolor, self.plot_dict["unique_groups"])

        linecolor = _process_colors(
            linecolor, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
        )
        linecolor_dict = create_dict(linecolor, self.plot_dict["unique_groups"])

        count_plot = {
            "color_dict": color_dict,
            "linecolor_dict": linecolor_dict,
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
            if facecolor is not None or linecolor == "black":
                d = facecolor
            else:
                d = linecolor
            d = _process_colors(
                d, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
            )
            self.plot_dict["legend_dict"] = (d, alpha)

        if not self.inplace:
            return self

    def _matplotlib_backend(
        self,
        savefig: bool = False,
        path: str = "",
        filename: str = "",
        filetype: str = "svg",
        transparent=False,
    ):
        fig, ax = plt.subplots(
            subplot_kw=dict(box_aspect=self._plot_format["figure"]["aspect"]),
            figsize=self._plot_format["figure"]["figsize"],
        )

        for i, j in zip(self.plot_list, self.plots):
            plot_func = MP_PLOTS[i]
            plot_func(
                data=self.plot_dict["data"],
                y=self.plot_dict["y"],
                loc_dict=self.plot_dict["loc_dict"],
                unique_groups=self.plot_dict["unique_groups"],
                ax=ax,
                transform=self.plot_dict["ytransform"],
                levels=self.plot_dict["levels"],
                **j,
            )

        decimals = self._plot_format["axis"]["ydecimals"]

        ax.set_xticks(
            ticks=self.plot_dict["x_ticks"],
            labels=self.plot_dict["group_order"],
            rotation=self._plot_format["labels"]["xlabel_rotation"],
            fontfamily=self._plot_format["labels"]["font"],
            fontweight=self._plot_format["labels"]["tick_fontweight"],
            fontsize=self._plot_format["labels"]["ticklabel_size"],
        )
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_linewidth(self._plot_format["axis"]["linewidth"])
        ax.spines["bottom"].set_linewidth(self._plot_format["axis"]["linewidth"])
        if "/" in self.plot_dict["y"]:
            self.plot_dict["y"] = self.plot_dict["y"].replace("/", "_")

        self._set_grid(ax)

        self._set_lims(ax, decimals, axis="y")

        if self._plot_format["axis"]["yminorticks"]:
            self._set_minorticks(ax, self.plot_dict["ytransform"], ticks="y")

        ax.set_ylabel(
            self.plot_dict["ylabel"],
            fontsize=self._plot_format["labels"]["labelsize"],
            fontfamily=self._plot_format["labels"]["font"],
            fontweight=self._plot_format["labels"]["label_fontweight"],
        )
        ax.set_title(
            self.plot_dict["title"],
            fontsize=self._plot_format["labels"]["titlesize"],
            fontfamily=self._plot_format["labels"]["font"],
            fontweight=self._plot_format["labels"]["label_fontweight"],
        )
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=self._plot_format["labels"]["ticklabel_size"],
            width=self._plot_format["axis"]["tickwidth"],
            length=self._plot_format["axis"]["ticklength"],
            labelfontfamily=self._plot_format["labels"]["font"],
        )
        ax.margins(x=self._plot_format["figure"]["margins"])

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
                filename = self.plot_dict["y"] if filename == "" else filename
                path = path / f"{filename}.{filetype}"
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
                tickfont=dict(size=self.plot_dict["ticklabel_size"]),
                tickwidth=self._plot_format["axis"]["tickwidth"],
                linewidth=self._plot_format["axis"]["linewidth"],
                automargin=True,
            ),
            yaxis=dict(
                titlefont=dict(size=self._plot_format["labels"]["labelsize"]),
                title=dict(text=self.plot_dict["ylabel"]),
                nticks=self.plot_dict["steps"],
                showline=True,
                tickmode="array",
                linecolor="black",
                tickvals=ticks,
                showgrid=False,
                ticks="outside",
                color="black",
                tickfont=dict(size=self.plot_dict["ticklabel_size"]),
                tickwidth=self._plot_format["axis"]["tickwidth"],
                linewidth=self._plot_format["axis"]["linewidth"],
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
