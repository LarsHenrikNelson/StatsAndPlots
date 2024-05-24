from io import StringIO

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import t
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import MultiComparison

from .stats_helpers import round_sig, BaseData

__all__ = ["run_batch_aov", "two_way_anova", "three_way_anova"]


class TwoWayAnovaData(BaseData):
    descriptive_stats: pd.DataFrame
    table: pd.DataFrame
    text: str
    posthoc: pd.DataFrame


def run_batch_aov(columns, data, group, subgroup):
    output = {}
    for i in columns:
        output[i] = two_way_anova(data, group, subgroup, i, "bonferroni")
    return output


def serialize_aov(data):
    groups = data.index.to_list()
    df = data["df"]
    F = data["F"]
    p = data["PR(>F)"]
    n2 = data["eta_sq"]
    cit = data["CI_upper"]
    cib = data["CI_lower"]
    x = ""
    for i in range(3):
        x += f"{groups[i]}: F{int(df.iloc[i]),int(df.iloc[-1])} = {F.iloc[i]}, p = {p.iloc[i]}, η2 = {n2.iloc[i]}, 95% CI[{cib.iloc[i]}, {cit.iloc[i]}];"
        x += " "
    x += "\n"
    return x


def eta_squared(aov):
    aov["eta_sq"] = "NaN"
    aov["eta_sq"] = aov[:-1]["sum_sq"] / sum(aov["sum_sq"])
    return aov


def omega_squared(aov):
    mse = aov["sum_sq"].iloc[-1] / aov["df"].iloc[-1]
    aov["omega_sq"] = "NaN"
    aov["omega_sq"] = (aov[:-1]["sum_sq"] - (aov[:-1]["df"] * mse)) / (
        sum(aov["sum_sq"]) + mse
    )
    return aov


def two_way_ci(aov, group_by_table):
    aov["mse"] = aov["sum_sq"] / aov["df"]

    # group_by_table = df.groupby(
    #     [column_group_1, column_group_2])[column_for_analysis].agg(
    #         ['count','mean'])
    mean_one_two_1 = np.average(
        (group_by_table["mean"].iloc[0], group_by_table["mean"].iloc[1])
    )
    mean_one_two_2 = np.average(
        (group_by_table["mean"].iloc[2], group_by_table["mean"].iloc[3])
    )
    group_1_difference = mean_one_two_1 - mean_one_two_2

    mean_two_one_1 = np.average(
        (group_by_table["mean"].iloc[0], group_by_table["mean"].iloc[2])
    )
    mean_two_one_2 = np.average(
        (group_by_table["mean"].iloc[1], group_by_table["mean"].iloc[3])
    )
    group_2_difference = mean_two_one_1 - mean_two_one_2

    mean_diff_1 = group_by_table["mean"].iloc[0] - group_by_table["mean"].iloc[2]
    mean_diff_2 = group_by_table["mean"].iloc[1] - group_by_table["mean"].iloc[3]
    interaction_diff = mean_diff_1 - mean_diff_2

    group_se = (
        np.sqrt(aov.loc["Residual", "mse"] * np.sum(1 / group_by_table["count"])) / 2
    )
    t_crit = t.ppf(0.975, aov.loc["Residual", "df"])
    ci = group_se * t_crit

    upper_one_two = group_1_difference + ci
    lower_one_two = group_1_difference - ci

    upper_two_one = group_2_difference + ci
    lower_two_one = group_2_difference - ci

    upper_interaction = interaction_diff + (ci * 2)
    lower_interaction = interaction_diff - (ci * 2)

    aov["mean_diff"] = [
        group_1_difference,
        group_2_difference,
        interaction_diff,
        np.nan,
    ]
    aov["CI_upper"] = [upper_one_two, upper_two_one, upper_interaction, np.nan]
    aov["CI_lower"] = [lower_one_two, lower_two_one, lower_interaction, np.nan]
    return aov


def two_way_anova(
    df: pd.DataFrame,
    column_group_1: str,
    column_group_2: str,
    column_for_analysis: str,
    post_hoc_test: str,
    sig: int = 3,
):
    """Two-way ANOVA that outputs everything GraphPad outputs.
    A Type-III ANOVA is run automatically since the output is
    the same as a Type-I ANOVA when the sample sizes are equal.

    Args:
        df (pd.DataFrame): _description_
        column_group_1 (str): _description_
        column_group_2 (str): _description_
        column_for_analysis (str): _description_
        post_hoc_test (str): _description_

    Returns:
        _type_: _description_
    """
    df_copy = df.copy()
    pd.options.mode.chained_assignment = None
    df_copy[column_for_analysis].dropna(inplace=True)
    descriptive_stats = df_copy.groupby([column_group_1, column_group_2])[
        column_for_analysis
    ].agg(["count", "mean", "median", "std", "sem"])
    df_copy["anova_id"] = [
        str(x) + "_" + str(y)
        for x, y in zip(df_copy[column_group_1], df_copy[column_group_2])
    ]
    formula = (
        f"{column_for_analysis} ~ C({column_group_1}, Sum)"
        f" * C({column_group_2}, Sum) * C({column_group_1}, Sum)"
        f":C({column_group_2}, Sum)"
    )
    model = ols(formula, data=df_copy).fit()
    # pw = model.t_test_pairwise(x1, method)
    aov_table = anova_lm(model, typ=3)
    aov_table.drop("Intercept", axis=0, inplace=True)
    eta_squared(aov_table)
    omega_squared(aov_table)
    two_way_ci(aov_table, descriptive_stats)
    comp = MultiComparison(df_copy[column_for_analysis], df_copy["anova_id"])
    if post_hoc_test == "bonferroni":
        post_hoc, _, _ = comp.allpairtest(stats.ttest_ind, method="bonf")
        post_hoc_html = post_hoc.as_html()
        post_hoc_df = pd.read_html(StringIO(post_hoc_html), header=0)[0]
    elif post_hoc_test == "tukey":
        post_hoc = comp.tukeyhsd()
        post_hoc_df = pd.DataFrame(
            data=post_hoc._results_table.data[1:],
            columns=post_hoc._results_table.data[0],
        )
    text = serialize_aov(aov_table.map(lambda x: round_sig(x, sig)))
    return TwoWayAnovaData(
        descriptive_stats=descriptive_stats,
        table=aov_table.map(lambda x: round_sig(x, sig)),
        text=text,
        post_hoc=post_hoc_df,
    )


def three_way_anova(
    df: pd.DataFrame,
    column_group_1: str,
    column_group_2: str,
    column_group_3: str,
    column_for_analysis: str,
    test,
):
    """_summary_

    Args:
        df (_type_): _description_
        column_group_1 (_type_): _description_
        column_group_2 (_type_): _description_
        column_group_3 (_type_): _description_
        column_for_analysis (_type_): _description_
        test (_type_): _description_

    Returns:
        _type_: _description_
    """
    df_copy = df.copy()
    pd.options.mode.chained_assignment = None
    df_copy.dropna(subset=[column_for_analysis], inplace=True)
    descriptive_stats = df_copy.groupby(
        [column_group_1, column_group_2, column_group_3]
    )[column_for_analysis].agg(["count", "mean", "median", "std", "sem"])
    df_copy["anova_id"] = [
        "_".join([str(x), str(y), str(z)])
        for x, y, z in zip(
            df_copy[column_group_1], df_copy[column_group_2], df_copy[column_group_3]
        )
    ]
    formula = (
        f"{column_for_analysis} ~ C({column_group_1}, Sum)"
        f" * C({column_group_2}, Sum) * C({column_group_3}, Sum)"
        f" * C({column_group_1}, Sum)"
        f":C({column_group_2}, Sum):C({column_group_3}, Sum)"
    )
    model = ols(formula, data=df_copy).fit()
    # pw = model.t_test_pairwise(x1, method)
    aov_table = anova_lm(model, typ=3)
    eta_squared(aov_table)
    omega_squared(aov_table)
    comp = MultiComparison(df_copy[column_for_analysis], df_copy["anova_id"])
    if test == "bonferroni":
        post_hoc, a1, a2 = comp.allpairtest(stats.ttest_ind, method="bonf")
        post_hoc_html = post_hoc.as_html()
        post_hoc_df = pd.read_html(post_hoc_html, header=0)[0]
    elif test == "tukey":
        post_hoc = comp.tukeyhsd()
        post_hoc_df = pd.DataFrame(
            data=post_hoc._results_table.data[1:],
            columns=post_hoc._results_table.data[0],
        )
    return descriptive_stats, aov_table.round(5), post_hoc_df
