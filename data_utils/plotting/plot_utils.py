from scipy import stats
import numpy as np


def process_args(arg, group_order, subgroup_order):
    if isinstance(arg, str):
        arg = {key: arg for key in group_order}
    elif isinstance(arg, list):
        arg = {key: arg for key in group_order}

    output_dict = {}
    for s in group_order:
        for b in subgroup_order:
            key = rf"{s}" + rf"{b}"
            if s in arg:
                output_dict[key] = arg[s]
            else:
                output_dict[key] = arg[b]
    return output_dict


def transform_func(a, transform=None):
    if transform is not None:
        return transform(a)
    else:
        return a


def get_valid_kwargs(args_list, **kwargs):
    output_args = {}
    for i in args_list:
        if i in kwargs:
            output_args[i] = kwargs[i]
    return output_args


def sem(a):
    return np.std(a) / np.sqrt(len(a))


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
    elif input == "median":
        return np.median
    elif input == "std":
        return np.std
    elif input == "log10":
        return np.log10
    elif input == "log":
        return np.log
    else:
        return lambda a: a


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
