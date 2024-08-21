import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.colors import Normalize, to_rgba


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


def get_ticks(
    lim,
    ticks,
    steps,
    n_decimals,
    tickstyle=None,
):
    if lim[0] is None:
        lim[0] = ticks[0]
    if lim[1] is None:
        lim[1] = ticks[-1]
    ticks = np.round(
        np.linspace(
            lim[0],
            lim[1],
            steps,
        ),
        decimals=n_decimals,
    )
    if tickstyle == "middle":
        ticks = ticks[1:-1]
    return lim, ticks


def _decimals(data):
    decimals = np.abs(int(np.max(np.round(np.log10(np.abs(data)))))) + 2
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


def process_args(arg, group_order, subgroup_order):
    if isinstance(arg, (str, int, float)):
        arg = {key: arg for key in group_order}
    elif isinstance(arg, list):
        arg = {key: arg for key, arg in zip(group_order, arg)}
    output_dict = {}
    for s in group_order:
        for b in subgroup_order:
            key = rf"{s}" + rf"{b}"
            if s in arg:
                output_dict[key] = arg[s]
            else:
                output_dict[key] = arg[b]
    return output_dict


def process_scatter_args(
    arg, data, group_order, subgroup_order, unique_groups, arg_cycle=None, alpha=None
):
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
        output = process_args(arg, group_order, subgroup_order)
        if alpha:
            output = {key: to_rgba(value, alpha) for key, value in output.items()}
        output = unique_groups.map(output)
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
    output = data[arg].map(mapping)
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
        colors = data[arg].map(mapping)
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


def _process_positions(group_spacing, group_order, subgroup, subgroup_order):
    group_loc = {key: index for index, key in enumerate(group_order)}
    if subgroup is not None:
        width = group_spacing / len(subgroup_order)
        start = (group_spacing / 2) - (width / 2)
        sub_loc = np.linspace(-start, start, len(subgroup_order))
        subgroup_loc = {key: value for key, value in zip(subgroup_order, sub_loc)}

    else:
        subgroup_loc = {key: 0 for key in subgroup_order}
        width = float(group_spacing)
    loc_dict = {}
    for i, i_value in group_loc.items():
        for j, j_value in subgroup_loc.items():
            key = rf"{i}" + rf"{j}"
            loc_dict[key] = i_value + j_value
    return loc_dict, width
