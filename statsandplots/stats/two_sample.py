from typing import Union

import numpy as np
import pandas as pd
from numba import njit, prange
from numpy.random import default_rng
from scipy import stats
from scipy.stats import norm

from .stats_helpers import round_sig, BaseData

__all__ = [
    "BAc_confidence_intervals",
    "boostrap",
    "bootstrap_test",
    "bootstrap_two_sample",
    "BootStrapData",
    "find_counts",
    "permutation_test",
    "run_batch_bootstrap",
    "unpaired_ttest",
]


class BootStrapData(BaseData):
    replicates: np.ndarray


def run_batch_bootstrap(columns, data, group, iterations, base_stat_func, sig):
    output = {}
    for i in columns:
        output[i] = bootstrap_test(data, group, i, iterations, base_stat_func, sig)
    return output


def serialize_bootstrap(data):
    x = ""
    p_value = "P_value"
    cib = "Lower_CI"
    cit = "Upper_CI"
    x += f"p = {data[p_value].iloc[0]}, CI[{data[cib].iloc[0]}, {data[cit].iloc[0]}]\n"
    return x


def serialize_ttest(data):
    x = ""
    dof = data["Degrees of Freedom"].iloc[0]
    test_stat = data["Test Statistic"].iloc[0]
    p_value = data["P_value"].iloc[0]
    low = data["Lower_CI"].iloc[0]
    upper = data["Upper_CI"].iloc[0]
    x += f"t{dof} = {test_stat}, p = {p_value}, CI[{low}, {upper}]\n"
    return x


def _find_counts(data: dict[str, np.ndarray], column: str, indexes: list[str]):
    unique_ids = set()
    counts = {}
    for i in range(data[column].shape[0]):
        if data[column][i] not in unique_ids:
            unique_ids.add(data[column][i])
            temp = counts
            for j in range(len(indexes)):
                if j != len(indexes) - 1:
                    if data[indexes[j]][i] not in temp:
                        temp[data[indexes[j]][i]] = {}
                        temp = temp[data[indexes[j]][i]]
                    else:
                        temp = temp[data[indexes[j]][i]]
                else:
                    if data[indexes[j]][i] in temp:
                        temp[data[indexes[j]][i]] += 1
                    else:
                        temp[data[indexes[j]][i]] = 1
    return counts


def find_counts(
    data: Union[pd.DataFrame, np.ndarray],
    column: Union[str, int],
    indexes: list[Union[str, int]],
) -> dict:
    """Provides the n per group in a set of data.

    Parameters
    ----------
    data : pd.DataFrame, np.ndarray, dict or list
        Data for determining the number of counts per group
    column : str or int
        The index/name of data that contains the unique id of each element.
        Can map to sample, subject etc and works with nested data.
    indexes : list-like containing stings or integers
       The indexes/names of the column, index of data that contains the group
       assignments.

    Returns
    -------
    dict
        A nested dictionary containing the group counts.

    Raises
    ------
    ValueError
        If the data is not a pandas DataFrame, Numpy ndarray, dictionary, or list
    """
    data_dict = {}
    if isinstance(data, pd.DataFrame):
        data_dict[str(column)] = data[column].to_numpy()
        for i in indexes:
            data_dict[str(i)] = data[i].to_numpy()
    elif isinstance(data, np.ndarray):
        data_dict[str(column)] = data[:, column]
        for i in indexes:
            data_dict[str(i)] = data[:, i]
    elif isinstance(data, dict):
        data_dict = data
    elif isinstance(data, list):
        data_dict[str(column)] = data[column]
        for i in indexes:
            data_dict[str(i)] = data[i]
    else:
        raise ValueError(
            "data must be Pandas DataFrame, Numpy ndarray, dictionary, list"
        )
    indexes = [str(i) for i in indexes]
    counts = _find_counts(data_dict, column, indexes)
    return counts


def permutation_test(
    df: pd.DataFrame, column_for_group: str, column_for_analysis: str, iterations: int
):
    groups = df[column_for_group].unique()
    if len(groups) != 2:
        raise Exception("Permutation test will only run with two groups.")
    descriptive_stats = df.groupby([column_for_group])[column_for_analysis].agg(
        ["count", "mean", "median", "std", "sem"]
    )
    groups = df[column_for_group].unique()
    group_1 = groups[0]
    group_2 = groups[1]
    mean_1 = df.loc[df[column_for_group] == group_1, column_for_analysis].mean()
    mean_2 = df.loc[df[column_for_group] == group_2, column_for_analysis].mean()
    observed_mean = abs(mean_1 - mean_2)

    # Return the number of rows containing a specific
    # genotype that is used to split the data
    df_rows = df.loc[df[column_for_group] == group_1, column_for_analysis].dropna()
    rows = len(df_rows)

    dataset = np.array(df[column_for_analysis])
    dataset = dataset[~np.isnan(dataset)]

    # Permutation test
    np.random.seed(0)
    mean_diff = []
    for i in range(iterations):
        np.random.shuffle(dataset)
        array_split = np.split(dataset, [0, rows], axis=0)
        array_1_split = array_split[1]
        array_2_split = array_split[2]
        diff = array_1_split.mean() - array_2_split.mean()
        mean_diff.append(diff)

    # Two-sided permutation p-value
    per = 0
    for item in mean_diff:
        if abs(item) >= observed_mean:
            per = per + 1
    p_value = per / iterations
    print("Permutation p-value: " + str(p_value))
    return descriptive_stats, p_value, mean_diff


@njit()
def numba_bootstrap_loop(
    sample_1, sample_2, rng, iterations, base_stat_func: str = "mean"
):
    sample_1_bs = np.zeros(len(sample_1))
    sample_2_bs = np.zeros(len(sample_2))
    bs_replicates = np.zeros(iterations)
    size_1 = np.int(sample_1_bs.size)
    size_2 = np.int(sample_1_bs.size)
    for i in range(iterations):
        for i in range(sample_1_bs.size):
            sample_1_bs[i] = sample_1[rng.integers(0, high=size_1)]
            sample_2_bs[i] = sample_2[rng.integers(0, high=size_2)]
        # sample_1_bs = rng.choice(sample_1, len(sample_1))
        # sample_2_bs = rng.choice(sample_2, len(sample_2))
        if base_stat_func == "mean":
            bs_replicates[i] = np.mean(sample_1_bs) - np.mean(sample_2_bs)
        else:
            bs_replicates[i] = np.median(sample_1_bs) - np.median(sample_2_bs)
    return bs_replicates


@njit(parallel=True)
def bac_inner_loop(bs_replicates: np.ndarray):
    partial_estimates = np.zeros(bs_replicates.size)
    for i in prange(bs_replicates.size):
        mean = 0.0
        for j in range(bs_replicates.size):
            if j != i:
                mean += bs_replicates[j]
            else:
                mean += 0.0
        partial_estimates[i] = mean / (bs_replicates.size - 1)
    return partial_estimates


def BAc_confidence_intervals(bs_replicates: np.ndarray, observed_mean: float):
    bias = norm.ppf((1 - stats.percentileofscore(bs_replicates, observed_mean) / 100))
    partial_estimates = bac_inner_loop(bs_replicates)
    partial_mean = np.mean(partial_estimates)
    numerator = np.sum(np.power((partial_mean - partial_estimates), 3))
    denominator = np.sum(np.power((partial_mean - partial_estimates), 2))
    acceleration_factor = numerator / np.power((np.sqrt(6 * denominator)), 3)
    ci_l = (-1.96 - bias) / (1 - acceleration_factor * (-1.96 - bias)) - bias
    ci_u = (1.96 - bias) / (1 - acceleration_factor * (1.96 - bias)) - bias
    p_ci_l = norm.cdf(ci_l)
    p_ci_u = norm.cdf(ci_u)
    ci_lower = np.percentile(bs_replicates, p_ci_l * 100)
    ci_upper = np.percentile(bs_replicates, p_ci_u * 100)
    bs_confidence_intervals = [ci_lower, ci_upper]
    return bs_confidence_intervals


def bootstrap_test(
    df: pd.DataFrame,
    column_for_group: str,
    column_for_analysis: str,
    iterations: int,
    base_stat_func: str,
    sig: int = 3,
):
    """Boostrap_test function needs a dataframe input, the column that contains your
    groups, the column that contains the data you want to analyze, the iterations,
    and what you want to test. In most cases you will want to compare the mean
    so base_stat_funct will be mean. This test was taken from
    https://towardsdatascience.com/bootstrapping-vs-permutation-testing-a30237795970.
    There several different ways to run a boostrap test.


    Args:
        df (pd.DataFrame): _description_
        column_for_group (str): _description_
        column_for_analysis (str): _description_
        iterations (int): _description_
        base_stat_funct (str): _description_

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """
    groups = df[column_for_group].unique()
    if len(groups) != 2:
        raise Exception("Permutation test will only run with two groups.")
    descriptive_stats = df.groupby([column_for_group])[column_for_analysis].agg(
        ["count", "mean", "median", "std", "sem"]
    )
    group_1 = groups[0]
    group_2 = groups[1]
    sample_1 = np.array(df.loc[df[column_for_group] == group_1, column_for_analysis])
    sample_1 = sample_1[~np.isnan(sample_1)]
    sample_2 = np.array(df.loc[df[column_for_group] == group_2, column_for_analysis])
    sample_2 = sample_2[~np.isnan(sample_2)]
    observed_mean = np.mean(sample_1) - np.mean(sample_2)

    rng = default_rng(42)

    # bs_replicates = numba_bootstrap_loop(
    #     sample_1, sample_2, rng, iterations, base_stat_func
    # )
    if base_stat_func == "mean":
        base_stat_func = np.mean
    else:
        base_stat_func = np.median
    bs_replicates = np.zeros(iterations)
    for i in range(iterations):
        sample_1_bs = rng.choice(sample_1, len(sample_1))
        sample_2_bs = rng.choice(sample_2, len(sample_2))
        bs_replicates[i] = base_stat_func(sample_1_bs) - base_stat_func(sample_2_bs)

    bs_mean_diff = np.mean(bs_replicates)
    bs_replicates_shifted = bs_replicates - bs_mean_diff - 0
    emp_diff_pctile_rnk = stats.percentileofscore(bs_replicates_shifted, observed_mean)
    auc_right = emp_diff_pctile_rnk / 100
    auc_left = 1 - emp_diff_pctile_rnk / 100
    auc_tail = auc_left if auc_left < auc_right else auc_right
    p_value = auc_tail * 2

    confidence_intervals = BAc_confidence_intervals(bs_replicates, bs_mean_diff)
    statistics = pd.DataFrame(
        {
            "P_value": [p_value],
            "Mean_diff": [bs_mean_diff],
            "Lower_CI": [confidence_intervals[0]],
            "Upper_CI": [confidence_intervals[1]],
        }
    )
    text = serialize_bootstrap(statistics.map(lambda x: round_sig(x, sig)))
    output = BootStrapData(
        table=statistics.map(lambda x: round_sig(x, sig)),
        descriptive_stats=descriptive_stats,
        text=text,
        replicates=bs_replicates,
    )
    return output


def bootstrap_two_sample(
    df, column_for_group, column_for_analysis, iterations, base_stat_func: str
):
    """Alternative bootstrap test, the confidence intervals don't work.

    Args:
        df (pd.DataFrame): _description_
        column_for_group (str): _description_
        column_for_analysis (str): _description_
        iterations (int): _description_
        base_stat_funct (np.ufunc): _description_

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """
    groups = df[column_for_group].unique()
    if len(groups) != 2:
        raise Exception("Permutation test will only run with two groups.")
    descriptive_stats = df.groupby([column_for_group])[column_for_analysis].agg(
        ["count", "mean", "median", "std", "sem"]
    )
    group_1 = groups[0]
    group_2 = groups[1]
    sample_1 = np.array(df.loc[df[column_for_group] == group_1, column_for_analysis])
    sample_1 = sample_1[~np.isnan(sample_1)]
    sample_2 = np.array(df.loc[df[column_for_group] == group_2, column_for_analysis])
    sample_2 = sample_2[~np.isnan(sample_2)]
    observed_mean = np.mean(sample_1) - np.mean(sample_2)

    pooled_mean = np.mean(np.concatenate((sample_1, sample_2), axis=None))

    sample_1_shifted = sample_1 - np.mean(sample_1) + pooled_mean

    sample_2_shifted = sample_2 - np.mean(sample_2) + pooled_mean

    if base_stat_func == "mean":
        base_stat_func = np.mean
    else:
        base_stat_func = np.median

    bs_diff = np.zeros(iterations)
    rng = default_rng(0)
    for i in range(iterations):
        sample_1_bs = rng.choice(sample_1_shifted, len(sample_1))
        sample_2_bs = rng.choice(sample_2_shifted, len(sample_2))
        bs_diff[i] = base_stat_func(sample_1_bs) - base_stat_func(sample_2_bs)

    # bs_diff = sample_1_bs - sample_2_bs

    bs_mean_diff = np.mean(bs_diff)

    p_value = np.sum(bs_diff >= observed_mean) / len(bs_diff)

    confidence_intervals = BAc_confidence_intervals(bs_diff, bs_mean_diff)
    statistics = pd.DataFrame(
        {
            "P_value": [p_value],
            "Mean_diff": [bs_mean_diff],
            "Lower_CI": [confidence_intervals[0]],
            "Upper_CI": [confidence_intervals[1]],
        }
    )
    return descriptive_stats, statistics, bs_diff


def boostrap(
    df,
    group,
    column_for_group,
    column_for_analysis,
    iterations,
    base_stat_funct,
    seed: int = 42,
):
    """This is a general utility function that will calculate the boostrap of a
    column of values in a dataframe. The output can be used to calculate the BCa
    confidence intervals using the BCa_confiedence_intervals function.

    Args:
        df (_type_): _description_
        group (_type_): _description_
        column_for_group (_type_): _description_
        column_for_analysis (_type_): _description_
        iterations (_type_): _description_
        base_stat_funct (_type_): _description_
        seed (int, optional): _description_. Defaults to 42.

    Returns:
        _type_: _description_
    """
    sample_1 = np.array(df.loc[df[column_for_group] == group, column_for_analysis])
    sample_1 = sample_1[~np.isnan(sample_1)]
    rng = default_rng(seed)
    bs_replicates = np.zeros(iterations)
    for i in range(iterations):
        sample_bs = rng.choice(sample_1, len(sample_1))
        bs_replicates[i] = base_stat_funct(sample_bs)
    return bs_replicates


def unpaired_ttest(
    df,
    column_for_group,
    column_for_data,
    sig: int = 3,
):
    """Runs a Welch's t-test since most sample sizes will be unequal
    or have unequal variance. The Welch's t-test converges to a Student's
    t-test when the sample sizes are equal.

    Args:
        df (_type_): _description_
        column_for_group (_type_): _description_
        column_for_data (_type_): _description_

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """
    groups = df[column_for_group].unique()
    if len(groups) != 2:
        raise Exception("Unpaired_t_test will only run with two groups.")
    descriptive_stats = df.groupby([column_for_group])[column_for_data].agg(
        ["count", "mean", "median", "std", "sem"]
    )
    groups = df[column_for_group].unique()
    x1 = np.array(df.loc[(df[column_for_group] == groups[0]), column_for_data])
    x1 = x1[~np.isnan(x1)]
    x2 = np.array(df.loc[(df[column_for_group] == groups[1]), column_for_data])
    x2 = x2[~np.isnan(x2)]
    # dof = (x1.var() / x1.size + x2.var() / x2.size) ** 2 / (
    #     (x1.var() / x1.size) ** 2 / (x1.size - 1)
    #     + (x2.var() / x2.size) ** 2 / (x2.size - 1)
    # )
    result = stats.ttest_ind(x1, x2, equal_var=False)
    ci = result.confidence_interval()
    stat_table = pd.DataFrame(
        {
            "Degrees of Freedom": [round_sig(result.df, sig)],
            "Test Statistic": [round_sig(result.statistic, sig)],
            "P_value": [round_sig(result.pvalue, sig)],
            "Lower_CI": round_sig(ci.low, sig),
            "Upper_CI": round_sig(ci.high, sig),
        }
    )
    txt = serialize_ttest(stat_table)
    output = BaseData(data=stat_table, descriptive_stats=descriptive_stats, text=txt)
    return output
