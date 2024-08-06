from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Optional, Union, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import matplotlib_plotting as mp
from . import plotly_plotting as plp
from .plot_utils import (
    _process_groups,
    _process_positions,
    decimals,
    get_ticks,
    process_args,
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
    "violin": mp._violin_plot,
    "kde": mp._kde_plot,
    "percent": mp._percent_plot,
    "ecdf": mp._ecdf,
    "count": mp._count_plot,
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
        df: pd.DataFrame,
        y: str,
        group: Optional[str] = None,
        x: Optional[str] = None,
        subgroup: Optional[str] = None,
        group_order: Optional[list[str]] = None,
        subgroup_order: Optional[list[str]] = None,
        unique_id: Optional[str] = None,
        y_label: str = "",
        x_label: str = "",
        title: str = "",
        facet: bool = False,
        facet_title: bool = False,
        cols_rows: Optional[tuple[int]] = None,
        inplace: bool = False,
    ):
        self.inplace = inplace
        self.plots = []
        self.plot_list = []
        self._plot_settings_run = False

        if group is None:
            unique_groups = np.array(["none"] * df.shape[0])
        else:
            if subgroup is not None:
                unique_groups = df[group].astype(str) + df[subgroup].astype(str)
            else:
                unique_groups = df[group].astype(str) + ""

        group_order, subgroup_order = _process_groups(
            df, group, subgroup, group_order, subgroup_order
        )
        # if isinstance(title, str) and not facet:
        #     title = [title]
        # elif isinstance(title, str) and facet:
        #     title = [title] * len(group_order)
        # elif isinstance(title, list) and facet:
        #     if len(title) != len(group_order):
        #         raise ValueError(
        #             "Number of titles must be the same a the number of groups."
        #         )
        #     title = title
        # else:
        #     title = group_order

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
            "facet": facet,
            "facet_dict": facet_dict,
            "unique_id": unique_id,
            "cols_rows": cols_rows,
            "facet_title": facet_title,
        }

    def plot_settings(
        self,
        style: str = "default",
        y_lim: Optional[list] = None,
        x_lim: Optional[list] = None,
        y_scale: Literal["linear", "log", "symlog"] = "linear",
        x_scale: Literal["linear", "log", "symlog"] = "linear",
        margins: float = 0.05,
        aspect: Union[int, float] = 1,
        figsize: Union[None, tuple[int, int]] = None,
        labelsize: int = 20,
        linewidth: int = 2,
        ticksize: int = 2,
        ticklabel: int = 12,
        steps: int = 7,
        tick_style: Literal["all", "middle"] = "all",
        y_decimals: int = None,
        x_decimals: int = None,
    ):
        self._plot_settings_run = True
        if y_lim is None:
            y_lim = [None, None]
        if x_lim is None:
            x_lim = [None, None]

        plot_settings = {
            "y_scale": y_scale,
            "x_scale": x_scale,
            "margins": margins,
            "aspect": aspect,
            "figsize": figsize,
            "labelsize": labelsize,
            "ticksize": ticksize,
            "ticklabel": ticklabel,
            "linewidth": linewidth,
            "y_lim": y_lim,
            "x_lim": x_lim,
            "y_decimals": y_decimals,
            "x_decimals": x_decimals,
            "steps": steps,
            "tick_style": tick_style,
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
        }
        self.plots.append(line_plot)
        self.plot_list.append("line_plot")

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
        line_color: ColorDict = "black",
        linestyle: str = "-",
        linewidth: int = 2,
        fill_under: bool = False,
        fill_color: ColorDict = "black",
        alpha: AlphaRange = 1.0,
        axis: Literal["x", "y"] = "y",
    ):
        line_color_dict = process_args(
            line_color, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
        )
        if fill_under:
            fill_color_dict = process_args(
                fill_color,
                self.plot_dict["group_order"],
                self.plot_dict["subgroup_order"],
            )
        else:
            fill_color_dict = {}

        linestyle_dict = process_args(
            linestyle, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
        )

        kde_plot = {
            "line_color_dict": line_color_dict,
            "linestyle_dict": linestyle_dict,
            "linewidth": linewidth,
            "alpha": alpha,
            "fill_under": fill_under,
            "fill_color_dict": fill_color_dict,
            "axis": axis,
            "kernel": kernel,
            "bw": bw,
            "tol": tol,
            "common_norm": common_norm,
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
        steps=50,
        func="mean",
        err_func="sem",
        fit_func=None,
        alpha: AlphaRange = 1.0,
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
            "steps": steps,
            "func": func,
            "err_func": err_func,
            "fit_func": fit_func,
            "alpha": alpha,
        }
        self.plots.append(poly_hist)
        self.plot_list.append("poly_hist")

        if not self.inplace:
            return self

    def ecdf(
        self,
        color: ColorDict = "black",
        linestyle: str = "-",
        linewidth: int = 2,
        alpha: AlphaRange = 1.0,
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
        }
        self.plots.append(ecdf)
        self.plot_list.append("ecdf")

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
        if self.plot_dict["cols_rows"] is None:
            nrows = len(self.plot_dict["group_order"])
            ncols = 1
        else:
            nrows = self.plot_dict["cols_rows"][1]
            ncols = self.plot_dict["cols_rows"][0]
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
                df=self.plot_dict["df"],
                y=self.plot_dict["y"],
                unique_groups=self.plot_dict["unique_groups"],
                unique_id=self.plot_dict["unique_id"],
                facet_dict=self.plot_dict["facet_dict"],
                ax=ax,
                **j,
            )

        if self.plot_dict["y_decimals"] is None:
            y_decimals = decimals(self.plot_dict["df"][self.plot_dict["y"]])
        else:
            y_decimals = self.plot_dict["y_decimals"]
        if self.plot_dict["x_decimals"] is None:
            x_decimals = decimals(self.plot_dict["df"][self.plot_dict["y"]])
        else:
            x_decimals = self.plot_dict["x_decimals"]
        num_plots = len(self.plot_dict["group_order"])
        for index, i in enumerate(ax[:num_plots]):
            if "kde" in self.plot_list and all(
                v is None for v in self.plot_dict["y_lim"]
            ):
                if j["axis"] == "y":
                    self.plot_dict["y_lim"] = [0, None]
            if "kde" in self.plot_list and all(
                v is None for v in self.plot_dict["x_lim"]
            ):
                index = self.plot_list.index("kde")
                if self.plots[index]["axis"] == "x":
                    self.plot_dict["x_lim"] = [0, None]
            i.spines["right"].set_visible(False)
            i.spines["top"].set_visible(False)
            i.spines["left"].set_linewidth(self.plot_dict["linewidth"])
            i.spines["bottom"].set_linewidth(self.plot_dict["linewidth"])
            if "/" in self.plot_dict["y"]:
                self.plot_dict["y"] = self.plot_dict["y"].replace("/", "_")

            if self.plot_dict["y_scale"] not in ["log", "symlog"]:
                ticks = i.get_yticks()
                lim, ticks = get_ticks(
                    self.plot_dict["y_lim"],
                    ticks,
                    self.plot_dict["steps"],
                    y_decimals,
                    tick_style=self.plot_dict["tick_style"],
                )
                i.set_ylim(bottom=lim[0], top=lim[1])
                i.set_yticks(ticks)
            else:
                i.set_yscale(self.plot_dict["y_scale"])
                ticks = i.get_yticks()
                lim, _ = get_ticks(
                    self.plot_dict["y_lim"], ticks, self.plot_dict["steps"], y_decimals
                )
                i.set_ylim(bottom=lim[0], top=lim[1])
            if self.plot_dict["x_scale"] not in ["log", "symlog"]:
                ticks = i.get_xticks()
                lim, ticks = get_ticks(
                    self.plot_dict["x_lim"],
                    ticks,
                    self.plot_dict["steps"],
                    x_decimals,
                    tick_style=self.plot_dict["tick_style"],
                )
                i.set_xlim(left=lim[0], right=lim[1])
                i.set_xticks(ticks)
            else:
                i.set_xscale(self.plot_dict["x_scale"])
                ticks = i.get_xticks()
                lim, _ = get_ticks(
                    self.plot_dict["x_lim"], ticks, self.plot_dict["steps"], x_decimals
                )
                i.set_xlim(left=lim[0], right=lim[1])
            i.set_ylabel(
                self.plot_dict["y_label"], fontsize=self.plot_dict["labelsize"]
            )
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
        df: pd.DataFrame,
        y: Union[str, int, float],
        group: Union[str, int, float] = None,
        subgroup: Union[str, int, float] = None,
        group_order: Union[None, list[Union[str, int, float]]] = None,
        subgroup_order: Union[None, list[Union[str, int, float]]] = None,
        group_spacing: Union[float, int] = 1.0,
        subgroup_spacing: Union[float, int] = 0.6,
        y_label: str = "",
        title: str = "",
        inplace: bool = True,
    ):

        self._plot_settings_run = False
        self.inplace = inplace
        self.style = "default"

        if subgroup is not None:
            if group not in df.columns:
                raise ValueError(f"{group} must be supplied if subgroup is used")
            unique_groups = df[group].astype(str) + df[subgroup].astype(str)
        else:
            if group is None:
                unique_groups = pd.Series([""] * df.shape[0])
            else:
                unique_groups = df[group].astype(str) + ""

        if group is not None:
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
        else:
            group_order = [""]
            subgroup_order = [""]
            loc_dict = {}
            loc_dict[""] = 0.0
            width = 1

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
        }
        self.plots = []
        self.plot_list = []

    def plot_settings(
        self,
        style: str = "default",
        y_lim: Optional[list] = None,
        y_scale: Literal["linear", "log", "symlog"] = "linear",
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
        if "y_lim" in self.plot_dict:
            y_lim = self.plot_dict["y_lim"]
        elif y_lim is None:
            y_lim = [None, None]

        plot_settings = {
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

    def jitter(
        self,
        color: ColorDict = "black",
        marker: Union[str, dict[str, str]] = "o",
        edgecolor: ColorDict = "none",
        alpha: AlphaRange = 1.0,
        jitter: Union[float, int] = 1.0,
        seed: int = 42,
        marker_size: float = 2.0,
        transform: Union[None, str] = None,
        unique_id: Union[None] = None,
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
            "unique_id": unique_id,
        }
        self.plots.append(jitter_plot)
        self.plot_list.append("jitter")

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
        marker_size: float = 2.0,
        transform: Union[None, str] = None,
        agg_func: Union[None, str] = None,
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
            "marker_size": marker_size,
            "transform": transform,
            "unique_id": unique_id,
            "duplicate_offset": duplicate_offset,
            "agg_func": agg_func,
        }
        self.plots.append(jitteru_plot)
        self.plot_list.append("jitteru")

        if not self.inplace:
            return self

    def summary(
        self,
        func: str = "mean",
        capsize: int = 0,
        capstyle: str = "round",
        bar_width: float = 1.0,
        err_func: str = "sem",
        linewidth: int = 2,
        color: ColorDict = "black",
        alpha: float = 1.0,
        transform=None,
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
            "bar_width": bar_width * self.plot_dict["width"],
            "err_func": err_func,
            "linewidth": linewidth,
            "transform": transform,
            "color_dict": color_dict,
            "alpha": alpha,
        }
        self.plots.append(summary_plot)
        self.plot_list.append("summary")

        if not self.inplace:
            return self

    def boxplot(
        self,
        facecolor="none",
        linecolor: ColorDict = "black",
        fliers="",
        box_width: float = 1.0,
        transform=None,
        linewidth=1,
        alpha: AlphaRange = 1.0,
        line_alpha: AlphaRange = 1.0,
        show_means: bool = False,
        show_ci: bool = False,
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
            "show_means": show_means,
            "show_ci": show_ci,
            "transform": transform,
            "linewidth": linewidth,
            "alpha": alpha,
            "line_alpha": line_alpha,
        }
        self.plots.append(boxplot)
        self.plot_list.append("boxplot")

        if not self.inplace:
            return self

    def violin(
        self,
        facecolor="none",
        edgecolor: ColorDict = "black",
        alpha: AlphaRange = 1.0,
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
        self.plots.append(violin)
        self.plot_list.append("violin")

        if not self.inplace:
            return self

    def percent(
        self,
        unique_id=None,
        facecolor="none",
        linecolor: ColorDict = "black",
        hatch=None,
        bar_width: float = 1.0,
        linewidth=1,
        alpha: float = 1.0,
        line_alpha=1.0,
        axis_type: Literal["density", "percent"] = "density",
        cutoff: Union[None, float, int, list[Union[float, int]]] = None,
        include_bins: Optional[list[bool]] = None,
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
            cutoff = [self.plot_dict["df"][self.plot_dict["y"]].mean()]

        if include_bins is None:
            include_bins = [True] * (len(cutoff) + 1)

        percent_plot = {
            "color_dict": color_dict,
            "linecolor_dict": linecolor_dict,
            "cutoff": cutoff,
            "hatch": hatch,
            "bar_width": bar_width * self.plot_dict["width"],
            "linewidth": linewidth,
            "alpha": alpha,
            "line_alpha": line_alpha,
            "include_bins": include_bins,
            "unique_id": unique_id,
        }
        self.plots.append(percent_plot)
        self.plot_list.append("percent")
        if axis_type == "density":
            self.plot_dict["y_lim"] = [0.0, 1.0]
        else:
            self.plot_dict["y_lim"] = [0, 100]

        if not self.inplace:
            return self

    def count(
        self,
        facecolor: ColorDict = "none",
        linecolor: ColorDict = "black",
        hatch=None,
        bar_width: float = 1.0,
        linewidth=1,
        alpha: float = 1.0,
        line_alpha=1.0,
        axis_type: Literal["density", "count", "percent"] = "density",
    ):
        # color_dict = process_args(
        #     facecolor, self.plot_dict["group_order"], self.plot_dict["subgroup_order"]
        # )
        if isinstance(facecolor, str):
            unique_ids_sub = self.plot_dict["df"][self.plot_dict["y"]].unique()
            facecolor = {key: facecolor for key in unique_ids_sub}

        if isinstance(linecolor, str):
            unique_ids_sub = self.plot_dict["df"][self.plot_dict["y"]].unique()
            linecolor = {key: linecolor for key in unique_ids_sub}

        count_plot = {
            "color_dict": facecolor,
            "linecolor_dict": linecolor,
            "hatch": hatch,
            "bar_width": bar_width * self.plot_dict["width"],
            "linewidth": linewidth,
            "alpha": alpha,
            "line_alpha": line_alpha,
            "axis_type": axis_type,
        }
        self.plots.append(count_plot)
        self.plot_list.append("count")

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
        fig, ax = plt.subplots(
            subplot_kw=dict(box_aspect=self.plot_dict["aspect"]),
            figsize=self.plot_dict["figsize"],
        )

        for i, j in zip(self.plot_list, self.plots):
            plot_func = MP_PLOTS[i]
            plot_func(
                df=self.plot_dict["df"],
                y=self.plot_dict["y"],
                loc_dict=self.plot_dict["loc_dict"],
                unique_groups=self.plot_dict["unique_groups"],
                ax=ax,
                **j,
            )

        if "count" in self.plot_list:
            decimals = None
        elif self.plot_dict["decimals"] is None:

            # No better way around this mess at the moment
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
            if decimals is not None:
                ticks = np.round(
                    np.linspace(
                        self.plot_dict["y_lim"][0],
                        self.plot_dict["y_lim"][1],
                        self.plot_dict["steps"],
                    ),
                    decimals=decimals,
                )
            else:
                ticks = np.linspace(
                    self.plot_dict["y_lim"][0],
                    self.plot_dict["y_lim"][1],
                    self.plot_dict["steps"],
                )
            ax.set_yticks(ticks)
        else:
            ax.set_yscale(self.plot_dict["y_scale"])
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
        ax.margins(x=self.plot_dict["margins"])
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
        graph_plot = {
            "marker_alpha": marker_alpha,
            "line_alpha": line_alpha,
            "marker_size": marker_size,
            "marker_scale": marker_scale,
            "linewidth": linewidth,
            "edge_color": edge_color,
            "marker_color": marker_color,
            "marker_attr": marker_attr,
            "cmap": cmap,
            "seed": seed,
            "scale": scale,
            "layout": layout,
            "plot_max_degree": plot_max_degree,
        }

        self.plots.append(graph_plot)
