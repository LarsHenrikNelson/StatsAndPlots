from scipy import stats
import numpy as np


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
    return lim, ticks


def decimals(data):
    decimals = np.abs(int(np.max(np.round(np.log10(np.abs(data)))))) + 2
    return decimals


def _process_groups(df, group, subgroup, group_order, subgroup_order):
    if group is None:
        return ["none"], [""]
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
        subgroup_order = [""] * len(group_order)
    return group_order, subgroup_order


def bin_data(data, bins):
    binned_data = np.zeros(bins.size - 1)
    index = 0
    for i in data:
        if i >= bins[index] and i < bins[int(index + 1)]:
            binned_data[index] += 1
        else:
            if index < binned_data.size:
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


def get_valid_kwargs(args_list, **kwargs):
    output_args = {}
    for i in args_list:
        if i in kwargs:
            output_args[i] = kwargs[i]
    return output_args


def sem(a, axis=None):
    if len(a.shape) == 2:
        shape = a.shape[1]
    else:
        shape = a.size
    return np.std(a, axis) / np.sqrt(shape)


def ci(a):
    ci_interval = stats.t.interval(
        confidence=0.95, df=len(a) - 1, loc=np.mean(a), scale=stats.sem(a)
    )
    return ci_interval[1] - ci_interval[0]


def get_func(input):
    if input == "sem":
        return sem
    elif input == "ci":
        return ci
    elif input == "mean":
        return np.mean
    elif input == "nanmean":
        return np.nanmean
    elif input == "nanmedian":
        return np.nanmedian
    elif input == "median":
        return np.median
    elif input == "std":
        return np.std
    elif input == "nanstd":
        return np.std
    elif input == "log10":
        return np.log10
    elif input == "log":
        return np.log
    elif input == "var":
        return np.var
    elif input == "nanvar":
        return np.nanvar
    else:
        return lambda a, axis=None: a


def _process_positions(
    group_spacing, group_order, subgroup, subgroup_order, subgroup_spacing
):
    group_loc = {key: group_spacing * index for index, key in enumerate(group_order)}
    if subgroup is not None:
        no_overlap_min = group_spacing / (len(group_order) + 2) * subgroup_spacing
        temp_loc = np.linspace(-no_overlap_min, no_overlap_min, len(subgroup_order))
        width = temp_loc[1] - temp_loc[0]
        subgroup_loc = {key: value for key, value in zip(subgroup_order, temp_loc)}

    else:
        subgroup_loc = {key: 0 for key in subgroup_order}
        width = float(group_spacing)
    loc_dict = {}
    for i, i_value in group_loc.items():
        for j, j_value in subgroup_loc.items():
            key = rf"{i}" + rf"{j}"
            loc_dict[key] = i_value + j_value
    return loc_dict, width
