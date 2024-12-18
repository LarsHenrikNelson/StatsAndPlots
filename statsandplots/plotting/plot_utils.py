from fractions import Fraction
from itertools import cycle

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, to_rgba


STANDARD_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def create_dict(grouping, unique_groups):
    if grouping is None or isinstance(grouping, (str, int)):
        output_dict = {key: grouping for key in unique_groups}
    else:
        if not isinstance(grouping, dict):
            grouping = {key: value for value, key in enumerate(grouping)}
        output_dict = {}
        for i in grouping:
            for j in unique_groups:
                if isinstance(i, tuple) and isinstance(j, tuple):
                    if len(i) != len(j):
                        if i == j[: len(i)]:
                            output_dict[j] = grouping[i]
                    elif i == j:
                        output_dict[j] = grouping[i]
                elif i in j:
                    output_dict[j] = grouping[i]
    return output_dict


def process_none(markefacecolor, unique_groups):
    if markefacecolor is None or markefacecolor == "none":
        return {key: None for key in unique_groups}
    else:
        return markefacecolor


def _process_colors(color, group, subgroup):
    if color is not None:
        return color
    elif group is not None:
        color = {}
        if subgroup is None:
            color = {key: value for key, value in zip(group, cycle(STANDARD_COLORS))}
        elif subgroup[0] != "":
            color = {key: value for key, value in zip(subgroup, cycle(STANDARD_COLORS))}
        else:
            color = {key: value for key, value in zip(group, cycle(STANDARD_COLORS))}
    else:
        color = STANDARD_COLORS[0]
    return color


def radian_ticks(ticks, rotate=False):
    pi_symbol = "\u03C0"
    mm = [int(180 * i / np.pi) for i in ticks]
    if rotate:
        mm = [deg if deg <= 180 else deg - 360 for deg in mm]
    jj = [Fraction(deg / 180) if deg != 0 else 0 for deg in mm]
    output = []
    for t in jj:
        sign = "-" if t < 0 else ""
        t = abs(t)
        if t.numerator == 0 or t == 0:
            output.append("0")
        elif t.numerator == 1 and t.denominator == 1:
            output.append(f"{sign}{pi_symbol}")
        elif abs(t.denominator) == 1:
            output.append(f"{t.numerator}{pi_symbol}")
        elif abs(t.numerator) == 1:
            output.append(f"{sign}{pi_symbol}/{t.denominator}")
        else:
            output.append(f"{sign}{t.numerator}{pi_symbol}/{t.denominator}")
    return output


def process_duplicates(values, output=None):
    vals, counts = np.unique(
        values,
        return_counts=True,
    )
    track_counts = {}
    if output is None:
        output = np.zeros(values.size)
    for key, val in zip(vals, counts):
        if val > 1:
            track_counts[key] = [0, np.linspace(-1, 1, num=val)]
        else:
            track_counts[key] = [0, [0]]
    for index, val in enumerate(values):

        output[index] += track_counts[val][1][track_counts[val][0]]
        track_counts[val][0] += 1
    return output


def _invert(array, invert):
    if invert:
        if isinstance(array, list):
            array.reverse()
        else:
            array = array[::-1]
        return array
    else:
        return array


def get_ticks(
    lim,
    ticks,
    steps,
    tickstyle=None,
):
    lim = lim.copy()
    if lim[0] is None:
        lim[0] = ticks[0]
    if lim[1] is None:
        lim[1] = ticks[-1]
    ticks = np.linspace(
        lim[0],
        lim[1],
        steps,
    )
    if tickstyle == "middle":
        ticks = ticks[1:-1]
    return lim, ticks


def _bin_data(data, bins, axis_type, invert, cutoff):
    if cutoff is not None:
        temp = np.sort(data)
        binned_data, _ = np.histogram(temp, bins)
    else:
        binned_data = np.zeros(len(bins))
        conv_dict = {key: value for value, key in enumerate(bins)}
        unames, ucounts = np.unique(data, return_counts=True)
        for un, uc in zip(unames, ucounts):
            binned_data[conv_dict[un]] = uc
    binned_data = binned_data / binned_data.sum()
    if axis_type == "percent":
        binned_data *= 100
    if invert:
        binned_data = binned_data[::-1]
    bottom = np.zeros(len(binned_data))
    bottom[1:] = binned_data[:-1]
    bottom = np.cumsum(bottom)
    top = binned_data
    return top, bottom


def _decimals(data):
    temp = np.abs(data)
    temp = temp[temp > 0.0]
    decimals = np.abs(int(np.max(np.round(np.log10(temp))))) + 2
    return decimals


def _process_groups(df, group, subgroup, group_order, subgroup_order):
    if group is None:
        return ["none"], [""]
    if group_order is None:
        group_order = sorted(df[group].unique())
    else:
        if len(group_order) != len(df[group].unique()):
            raise AttributeError(
                "The number groups does not match the number in group_order"
            )
    if subgroup is not None:
        if subgroup_order is None:
            subgroup_order = sorted(df[subgroup].unique())
        elif len(subgroup_order) != len(df[subgroup].unique()):
            raise AttributeError(
                "The number subgroups does not match the number in subgroup_order"
            )
    else:
        subgroup_order = [""] * len(group_order)
    return group_order, subgroup_order


def bin_data(data, bins):
    binned_data = np.zeros(bins.size - 1, dtype=int)
    index = 0
    for i in data:
        if index >= bins.size:
            binned_data[binned_data.size - 1] += 1
        elif i >= bins[index] and i < bins[int(index + 1)]:
            binned_data[index] += 1
        else:
            if index < binned_data.size - 1:
                index += 1
                binned_data[index] += 1
            elif index < binned_data.size:
                binned_data[index] += 1
                index += 1
            else:
                binned_data[binned_data.size - 1] += 1
    return binned_data


def process_args(arg, group, subgroup):
    if isinstance(arg, (str, int, float)):
        arg = {key: arg for key in group}
    elif isinstance(arg, list):
        arg = {key: arg for key, arg in zip(group, arg)}
    output_dict = {}
    for s in group:
        for b in subgroup:
            key = rf"{s}" + rf"{b}"
            if s in arg:
                output_dict[key] = arg[s]
            else:
                output_dict[key] = arg[b]
    return output_dict


def process_scatter_args(arg, data, levels, unique_groups, arg_cycle=None, alpha=None):
    if isinstance(arg_cycle, (np.ndarray, list)):
        if arg in data:
            if arg_cycle is not None:
                output = _discrete_cycler(arg, data, arg_cycle, alpha)
            else:
                output = data[arg]
        elif len(arg) < len(unique_groups):
            output = arg
    elif arg_cycle in mpl.colormaps:
        if arg not in data:
            raise AttributeError("arg[0] of arg must be in data passed to LinePlot")
        output = _continuous_cycler(arg, data, arg_cycle, alpha)
    else:
        output = create_dict(arg, unique_groups)
        if alpha:
            output = {key: to_rgba(value, alpha) for key, value in output.items()}
        output = [output[j] for j in zip(*[data[i] for i in levels])]
    return output


def _discrete_cycler(arg, data, arg_cycle, alpha=None):
    grps = np.unique(data[arg])
    ntimes = data.shape[0] // len(arg_cycle)
    markers = arg_cycle
    if ntimes > 0:
        markers = markers * (ntimes + 1)
        markers = markers[: data.shape[0]]
    mapping = {key: value for key, value in zip(grps, markers)}
    if alpha is not None:
        mapping = {key: to_rgba(value, alpha) for key, value in mapping.items()}
    output = data[arg].map(mapping).to_list()
    return output


def _continuous_cycler(arg, data, arg_cycle, alpha):
    cmap = mpl.colormaps[arg_cycle]
    if pd.api.types.is_string_dtype(data[arg]) or pd.api.types.is_object_dtype(
        data[arg]
    ):
        uvals = pd.unique(data[arg])
        vmin = 0
        vmax = len(uvals)
        color_normal = Normalize(vmin=vmin, vmax=vmax)
        mapping = {
            key: cmap(color_normal(value), alpha=alpha)
            for value, key in enumerate(uvals)
        }
        colors = data[arg].map(mapping).to_list()
    else:
        vmin = data[arg].min()
        vmax = data[arg].max()
        vals = data[arg]
        color_normal = Normalize(vmin=vmin, vmax=vmax)
        colors = [cmap(color_normal(e), alpha=alpha) for e in vals]
    return colors


def get_valid_kwargs(args_list, **kwargs):
    output_args = {}
    for i in args_list:
        if i in kwargs:
            output_args[i] = kwargs[i]
    return output_args


def _process_positions(group_spacing, group_order, subgroup=None, subgroup_order=None):
    group_loc = {key: float(index) for index, key in enumerate(group_order)}
    if subgroup is not None:
        width = group_spacing / len(subgroup_order)
        start = (group_spacing / 2) - (width / 2)
        sub_loc = np.linspace(-start, start, len(subgroup_order))
        subgroup_loc = {key: value for key, value in zip(subgroup_order, sub_loc)}
        loc_dict = {}
        for i, i_value in group_loc.items():
            for j, j_value in subgroup_loc.items():
                key = (i, j)
                loc_dict[key] = i_value + j_value

    else:
        loc_dict = {(key,): value for key, value in group_loc.items()}
        width = 1.0
    return loc_dict, width
