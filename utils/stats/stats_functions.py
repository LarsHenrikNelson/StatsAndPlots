# -*- coding: utf-8 -*-
"""
Created on Mon May 10 13:39:52 2021

Last updated on Sun Oct 17 08:50:00 2021

@author: LarsNelson
"""
#Import need packages
import sys

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
import pandas as pd
from statsmodels.formula.api import ols, rlm
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
from scipy import stats
from scipy.stats import norm, t


def eta_squared(aov):
    aov['eta_sq'] = 'NaN'
    aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
    return aov


def omega_squared(aov):
    mse = aov['sum_sq'][-1]/aov['df'][-1]
    aov['omega_sq'] = 'NaN'
    aov['omega_sq'] = ((aov[:-1]['sum_sq']-(aov[:-1]['df']*mse))
                          /(sum(aov['sum_sq'])+mse))
    return aov


def two_way_ci(aov, group_by_table):
    aov['mse'] = aov['sum_sq']/aov['df']
    
    # group_by_table = df.groupby(
    #     [column_group_1, column_group_2])[column_for_analysis].agg(
    #         ['count','mean'])
    mean_one_two_1 = np.average((group_by_table['mean'][0],
                                 group_by_table['mean'][1]))
    mean_one_two_2 = np.average((group_by_table['mean'][2],
                                 group_by_table['mean'][3]))
    group_1_difference = mean_one_two_1 - mean_one_two_2
    
    mean_two_one_1 = np.average((group_by_table['mean'][0],
                                 group_by_table['mean'][2]))
    mean_two_one_2 = np.average((group_by_table['mean'][1],
                                 group_by_table['mean'][3]))
    group_2_difference = mean_two_one_1 - mean_two_one_2
    
    mean_diff_1 = group_by_table['mean'][0] - group_by_table['mean'][2]
    mean_diff_2 = group_by_table['mean'][1] - group_by_table['mean'][3]
    interaction_diff = mean_diff_1 - mean_diff_2

    group_se = (np.sqrt(aov.loc['Residual', 'mse']
                        * np.sum(1/group_by_table['count']))/2)
    t_crit = t.ppf(0.975, aov.loc['Residual', 'df'])
    ci = group_se*t_crit
    
    upper_one_two = group_1_difference + ci
    lower_one_two = group_1_difference - ci
    
    upper_two_one = group_2_difference + ci
    lower_two_one = group_2_difference - ci
    
    upper_interaction = interaction_diff + (ci * 2)
    lower_interaction = interaction_diff - (ci * 2)
    
    
    aov['mean_diff'] = [group_1_difference, group_2_difference,
                        interaction_diff, np.nan]
    aov['CI_upper'] = [upper_one_two, upper_two_one, upper_interaction,
                       np.nan]
    aov['CI_lower'] = [lower_one_two, lower_two_one, lower_interaction,
                       np.nan]
    return aov


def two_way_anova(df, column_group_1, column_group_2, column_for_analysis,
                  post_hoc_test):
    df_copy = df.copy()
    pd.options.mode.chained_assignment = None
    df_copy[column_for_analysis].dropna(inplace=True)
    descriptive_stats = df_copy.groupby(
        [column_group_1, column_group_2])[column_for_analysis].agg(
            ['count', 'mean', 'median', 'std', 'sem'])
    df_copy['anova_id'] = [str(x) + '_' + str(y) for x, y in zip(
        df_copy[column_group_1], df_copy[column_group_2])]
    formula = f"{column_for_analysis} ~ C({column_group_1}, Sum)" \
    f" * C({column_group_2}, Sum) * C({column_group_1}, Sum)" \
        f":C({column_group_2}, Sum)"
    model = ols(formula, data=df_copy).fit()
    # pw = model.t_test_pairwise(x1, method)
    aov_table = anova_lm(model, typ=3)
    aov_table.drop("Intercept",axis=0, inplace=True)
    eta_squared(aov_table)
    omega_squared(aov_table)
    two_way_ci(aov_table, descriptive_stats)
    comp = MultiComparison(df_copy[column_for_analysis], df_copy['anova_id'])
    if post_hoc_test == 'bonferroni':
        post_hoc, a1, a2 = comp.allpairtest(stats.ttest_ind, method='bonf')
        post_hoc_html = post_hoc.as_html()
        post_hoc_df = pd.read_html(post_hoc_html, header=0)[0]
    elif post_hoc_test == 'tukey':
        post_hoc = comp.tukeyhsd()
        post_hoc_df = pd.DataFrame(data=post_hoc._results_table.data[1:],
                                   columns=post_hoc._results_table.data[0])
    return descriptive_stats, aov_table.round(5), post_hoc_df


# def robust_two_way_anova(df, column_group_1, column_group_2, column_for_analysis, test):
#     pd.options.mode.chained_assignment = None
#     df.dropna(subset=[column_for_analysis], inplace=True)
#     descriptive_stats = df.groupby([column_group_1, column_group_2])[column_for_analysis].agg(['count', 'mean', 'median', 'std', 'sem'])
#     # df['anova_id'] = (df[column_group_1].astype(str) + '_' + df[column_group_2].astype(str))
#     # df['anova_id'] = (['_'.join([str(x), '_', str(y)]) for x, y in zip(df[column_group_1], df[column_group_2])])
#     df['anova_id'] = [str(x) + '_' + str(y) for x, y in zip(df[column_group_1], df[column_group_2])]
#     formula = column_for_analysis + ' ~ ' + 'C(' + column_group_1 + ', Sum) * ' + 'C(' + column_group_2 + ', Sum) * ' + 'C(' + column_group_1 + ', Sum):C(' + column_group_2 + ', Sum)'

#     rlm_model = sm.RLM(y_values, x_constant, M=sm.robust.norms.TrimmedMean())
#     model = rlm_model.fit()
    
#     # model = rlm(formula, data=df, M='HuberT').fit(cov=)
#     # pw = model.t_test_pairwise(x1, method)
#     aov_table = anova_lm(model, typ=3)
#     eta_squared(aov_table)
#     omega_squared(aov_table)
#     comp = MultiComparison(df[column_for_analysis], df['anova_id'])
#     if test == 'bonferroni':
#         post_hoc, a1, a2 = comp.allpairtest(stats.ttest_ind, method='bonf')
#         post_hoc_html = post_hoc.as_html()
#         post_hoc_df = pd.read_html(post_hoc_html, header=0)[0]
#     elif test == 'tukey':
#         post_hoc = comp.tukeyhsd()
#         post_hoc_df = pd.DataFrame(data=post_hoc._results_table.data[1:], columns=post_hoc._results_table.data[0])
#     return descriptive_stats, aov_table.round(5), post_hoc_df


def three_way_anova(df, column_group_1, column_group_2, column_group_3,
                    column_for_analysis, test):
    df_copy = df.copy()
    pd.options.mode.chained_assignment = None
    df_copy.dropna(subset=[column_for_analysis], inplace=True)
    descriptive_stats = df_copy.groupby(
        [column_group_1, column_group_2, column_group_3])[
            column_for_analysis].agg(
                ['count', 'mean', 'median', 'std', 'sem'])
    df_copy['anova_id'] = ['_'.join(
        [str(x), str(y), str(z)]) for x, y, z in zip(
            df_copy[column_group_1], df_copy[column_group_2],
            df_copy[column_group_3])]
    formula = f"{column_for_analysis} ~ C({column_group_1}, Sum)" \
    f" * C({column_group_2}, Sum) * C({column_group_3}, Sum)" \
        f" * C({column_group_1}, Sum)" \
        f":C({column_group_2}, Sum):C({column_group_3}, Sum)"
    model = ols(formula, data=df_copy).fit()
    # pw = model.t_test_pairwise(x1, method)
    aov_table = anova_lm(model, typ=3)
    eta_squared(aov_table)
    omega_squared(aov_table)
    comp = MultiComparison(df_copy[column_for_analysis], df_copy['anova_id'])
    if test == 'bonferroni':
        post_hoc, a1, a2 = comp.allpairtest(stats.ttest_ind, method='bonf')
        post_hoc_html = post_hoc.as_html()
        post_hoc_df = pd.read_html(post_hoc_html, header=0)[0]
    elif test == 'tukey':
        post_hoc = comp.tukeyhsd()
        post_hoc_df = pd.DataFrame(data=post_hoc._results_table.data[1:],
                                   columns=post_hoc._results_table.data[0])
    return descriptive_stats, aov_table.round(5), post_hoc_df



'''
The permutation_test function needs a dataframe input, the column that 
contains your groups,the column contains the data you want to analyze, 
and the number of iterations. The permutation test returns the p_value,
and the plot of the mean differences between the groups. The function
returns the p_value, the replicates, and the plot of the differences between the means.
The plot will normally be centered around zero and needs to be checked for a
normal distribution
'''   
def permutation_test(df, column_for_group, column_for_analysis, iterations):
    groups = df[column_for_group].unique()
    if len(groups) != 2:
        raise Exception('Permutation test will only run with two groups.')
    descriptive_stats = df.groupby([column_for_group])[column_for_analysis].agg(['count', 'mean', 'median', 'std', 'sem'])
    groups = df[column_for_group].unique()
    group_1 = groups[0]
    group_2 = groups[1]
    mean_1 = df.loc[df[column_for_group] == group_1, column_for_analysis].mean()
    mean_2 = df.loc[df[column_for_group] == group_2, column_for_analysis].mean()
    observed_mean = abs(mean_1 - mean_2)
    
    #Return the number of rows containing a specific genotype that is used to split the data
    df_rows = df.loc[df[column_for_group] == group_1, column_for_analysis].dropna()
    rows = len(df_rows)
    
    dataset = np.array(df[column_for_analysis])
    dataset = dataset[~np.isnan(dataset)]
    
    #Permutation test
    np.random.seed(0)
    mean_diff=[]
    for i in range(iterations):
        np.random.shuffle(dataset)
        array_split = np.split(dataset,[0,rows],axis=0)
        array_1_split = array_split[1]
        array_2_split = array_split[2]
        diff = array_1_split.mean()-array_2_split.mean()
        mean_diff.append(diff)
    
    #Two-sided permutation p-value
    per = 0
    for item in mean_diff:
        if abs(item)>= observed_mean:
            per = per+1
    p_value = per/iterations
    print('Permutation p-value: ' + str(p_value))
    return descriptive_stats, p_value, mean_diff

'''

'''
def bootstrap_test(df, column_for_group, column_for_analysis, iterations,
                   base_stat_funct):
    '''
    Boostrap_test function needs a dataframe input, the column that contains your
    groups, the column that contains the data you want to analyze, the iterations,
    and what you want to test. In most cases you will want to compare the mean
    so use np.mean. The function returns the descriptive stats, stats, the
    boostrap replicates, and a histogram of the replicates. This test was
    taken from https://towardsdatascience.com/bootstrapping-vs-permutation-testing-a30237795970.
    There several different ways to run a boostrap test.

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    column_for_group : TYPE
        DESCRIPTION.
    column_for_analysis : TYPE
        DESCRIPTION.
    iterations : TYPE
        DESCRIPTION.
    base_stat_funct : TYPE
        DESCRIPTION.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    descriptive_stats : TYPE
        DESCRIPTION.
    statistics : TYPE
        DESCRIPTION.
    bs_replicates : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.

    '''
    
    groups = df[column_for_group].unique()
    if len(groups) != 2:
        raise Exception('Permutation test will only run with two groups.')
    descriptive_stats = df.groupby(
        [column_for_group])[column_for_analysis].agg([
            'count', 'mean', 'median', 'std', 'sem'])
    group_1 = groups[0]
    group_2 = groups[1]
    sample_1 = np.array(df.loc[df[column_for_group] == group_1,
                               column_for_analysis])
    sample_1 = sample_1[~np.isnan(sample_1)]
    sample_2 = np.array(df.loc[df[column_for_group] == group_2,
                               column_for_analysis])
    sample_2 = sample_2[~np.isnan(sample_2)]
    observed_mean = np.mean(sample_1)-np.mean(sample_2)
    
    bs_replicates = []
    rng = default_rng(42)
    for i in range(iterations):
        sample_1_bs = rng.choice(sample_1, len(sample_1))
        sample_2_bs = rng.choice(sample_2, len(sample_2))
        bs_replicates += [base_stat_funct(sample_1_bs) - base_stat_funct(sample_2_bs)]
    
    bs_mean_diff = np.mean(bs_replicates)
    bs_replicates_shifted = np.array(bs_replicates) - bs_mean_diff - 0
    emp_diff_pctile_rnk = stats.percentileofscore(bs_replicates_shifted,
                                                  observed_mean)
    auc_right = emp_diff_pctile_rnk/100
    auc_left = 1 - emp_diff_pctile_rnk/100
    auc_tail = auc_left if auc_left < auc_right else auc_right
    p_value = auc_tail * 2    
    
    confidence_intervals = BAc_confidence_intervals(bs_replicates,
                                                    bs_mean_diff)
    statistics = pd.DataFrame({'P_value': [p_value],
                               'Mean_diff': [bs_mean_diff],
                               'Lower_CI': [confidence_intervals[0]],
                               'Upper_CI': [confidence_intervals[1]]})
    fig, ax = plt.subplots()
    plt.hist(bs_replicates)
    ax.set_title(column_for_analysis)
    return descriptive_stats, statistics, bs_replicates, fig


def boostrap_two_sample(df, column_for_group, column_for_analysis, iterations,
                        base_stat_funct):
    '''
    This is an alternative version of the boostrap test to compare equality of
    means. This boostrap test uses samples from a shifted mean so the mean of
    each group is equal.

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    column_for_group : TYPE
        DESCRIPTION.
    column_for_analysis : TYPE
        DESCRIPTION.
    iterations : TYPE
        DESCRIPTION.
    base_stat_funct : TYPE
        DESCRIPTION.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    descriptive_stats : TYPE
        DESCRIPTION.
    statistics : TYPE
        DESCRIPTION.
    bs_diff : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.

    '''
    groups = df[column_for_group].unique()
    if len(groups) != 2:
        raise Exception('Permutation test will only run with two groups.')
    descriptive_stats = df.groupby(
        [column_for_group])[column_for_analysis].agg([
            'count', 'mean', 'median', 'std', 'sem'])
    group_1 = groups[0]
    group_2 = groups[1]
    sample_1 = np.array(df.loc[df[column_for_group] == group_1,
                               column_for_analysis])
    sample_1 = sample_1[~np.isnan(sample_1)]
    sample_2 = np.array(df.loc[df[column_for_group] == group_2,
                               column_for_analysis])
    sample_2 = sample_2[~np.isnan(sample_2)]
    observed_mean = np.mean(sample_1)-np.mean(sample_2)
    
    pooled_mean = np.mean(np.concatenate((sample_1, sample_2), axis=None))

    sample_1_shifted = sample_1 - np.mean(sample_1) + pooled_mean
    
    sample_2_shifted = sample_2 - np.mean(sample_2) + pooled_mean

    bs_diff = []
    np.random.seed(0)
    for i in range(iterations):
        sample_1_bs = np.random.choice(sample_1_shifted, len(sample_1))
        sample_2_bs = np.random.choice(sample_2_shifted, len(sample_2))
        bs_diff += [base_stat_funct(sample_1_bs) - base_stat_funct(sample_2_bs)]
    
    # bs_diff = sample_1_bs - sample_2_bs
    
    bs_mean_diff = np.mean(bs_diff)
    
    p_value = np.sum(bs_diff >= observed_mean)/len(bs_diff)
    
    confidence_intervals = BAc_confidence_intervals(bs_diff,
                                                    bs_mean_diff)
    statistics = pd.DataFrame({'P_value': [p_value],
                               'Mean_diff': [bs_mean_diff],
                               'Lower_CI': [confidence_intervals[0]],
                               'Upper_CI': [confidence_intervals[1]]})
    
    fig, ax = plt.subplots()
    plt.hist(bs_diff)
    ax.set_title(column_for_analysis)
    
    return descriptive_stats, statistics, bs_diff, ax

'''
The BAc_confidence_intervals function calculates the bias-corrected and accelerated boostrapped 
confidence intervals. The function needs a boostrapped array of values and
observed statistic which is most commonly the mean of the original
'''     
def BAc_confidence_intervals(bs_replicates, observed_mean):
    bias = norm.ppf((1-stats.percentileofscore(bs_replicates, observed_mean)/100))
    partial_estimates = []
    for i in range(len(bs_replicates)):
        replicates_np = np.array(bs_replicates)
        jackknife_array = np.delete(replicates_np, i)
        partial_estimate = np.mean(jackknife_array)
        partial_estimates += [partial_estimate]
    partial_estimates_array = np.array(partial_estimates)
    partial_mean = np.mean(partial_estimates_array)
    numerator = sum(np.power((partial_mean - partial_estimates_array), 3))
    denominator = sum(np.power((partial_mean - partial_estimates_array), 2))
    acceleration_factor = numerator/np.power((np.sqrt(6*denominator)),3)
    ci_l = (-1.96-bias)/(1-acceleration_factor*(-1.96-bias))-bias
    ci_u = (1.96-bias)/(1-acceleration_factor*(1.96-bias))-bias
    p_ci_l = norm.cdf(ci_l)
    p_ci_u = norm.cdf(ci_u)
    ci_lower = np.percentile(bs_replicates, p_ci_l*100)
    ci_upper = np.percentile(bs_replicates, p_ci_u*100)
    bs_confidence_intervals = [ci_lower, ci_upper]
    return bs_confidence_intervals

'''
This is a general utility function that will calculate the boostrap of a
column of values in a dataframe. The output can be used to calculate the BCa 
confidence intervals using the BCa_confiedence_intervals function.
'''
def boostrap(df, group, column_for_group, column_for_analysis, iterations, base_stat_funct):
    sample_1 = np.array(df.loc[df[column_for_group] == group, column_for_analysis])
    sample_1 = sample_1[~np.isnan(sample_1)]
    np.random.seed(0)
    bs_replicates = []
    for i in range(iterations):
        sample_bs = np.random.choice(sample_1, len(sample_1))
        bs_replicates = base_stat_funct(sample_bs)
    return bs_replicates

'''
The unpaired_t_test function needs a dataframe input, the column that 
contains your groups, and the column contains the data you want to analyze.
Unpaired_t_test only runs a Welch's t-test since most data will either have
unequal sample size or unequal variance. If variance and sample size are the
same then Welch's t-test will return the same rsults 
The unpaired_t_test returns the descriptive statistics, and the statistic
table .
'''
def unpaired_t_test(df, column_for_group, column_for_data):
    groups = df[column_for_group].unique()
    if len(groups) != 2:
        raise Exception('Unpaired_t_test will only run with two groups.')
    descriptive_stats = df.groupby([column_for_group])[column_for_data].agg(['count', 'mean', 'median', 'std', 'sem'])
    groups = df[column_for_group].unique()
    x1 = np.array(df.loc[(df[column_for_group] == groups[0]), column_for_data])
    x1 = x1[~np.isnan(x1)]
    x2 = np.array(df.loc[(df[column_for_group] == groups[1]), column_for_data])
    x2 = x2[~np.isnan(x2)]
    dof = (x1.var()/x1.size + x2.var()/x2.size)**2 / ((x1.var()/x1.size)**2 / (x1.size-1) + (x2.var()/x2.size)**2 / (x2.size-1))
    test_stat, p_value = stats.ttest_ind(x1, x2, equal_var = False)
    stat_table = pd.DataFrame({'Degrees of Freedom': [dof],'Test Statistic': [test_stat], 'P_value': [p_value]})
    return descriptive_stats, stat_table
    
# def pca_analysis(df, list_of_features, list_of_groups, components):
#     #extract features for PCA. All data should have mean=0 and variance=1
#     df1 = df.dropna(subset=list_of_features)
#     x = df1.loc[:,list_of_features].values
#     y = df1.loc[:,list_of_groups].values
    
#     #Stardarize the features
#     # x = StandardScaler().fit_transform(x)
    
#     #Create a covariance matrix
#     cov_data = np.corrcoef(x)
    
#     #Run the PCA analysis
#     pca = PCA(n_components=components)
#     principalComponents = pca.fit_transform(x)
#     principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
#     finalDf = pd.concat([principalDf, df1[groups]], axis=1)
#     explained_variance= pca.explained_variance_ratio_
    
#     if len(list_of_groups) > 1:
#         hue = list_of_groups[0]
#         split = list_of_groups[1]
#     else:
#         hue = list_of_groups[0]
#         split=None
    
#     #plot the PCA
#     sns.relplot(data=finalDf, x="principal component 1", y ='principal component 2', hue=hue, style=split)
#     return explained_variance    
 

   
# def ica_analysis(df, list_of_features, list_of_groups, components, random_state):
#     #extract features for PCA. All data should have mean=0 and variance=1
#     df1 = df.dropna(subset=list_of_features)
#     x = df1.loc[:,list_of_features].values
#     y = df1.loc[:,list_of_groups].values
    
#     #Stardarize the features
#     # x = StandardScaler().fit_transform(x)
    
#     #Create a covariance matrix
#     cov_data = np.corrcoef(x)
    
#     #Run the PCA analysis
#     ICA = FastICA(n_components=components, random_state=random_state)
#     ICAComponents = ICA.fit_transform(x)
#     principalDf = pd.DataFrame(data = ICAComponents, columns = ['principal component 1', 'principal component 2'])
#     finalDf = pd.concat([principalDf, df1[groups]], axis=1)
#     # explained_variance= ICA.explained_variance_ratio_
    
#     if len(list_of_groups) > 1:
#         hue = list_of_groups[0]
#         split = list_of_groups[1]
#     else:
#         hue = list_of_groups[0]
#         split=None
    
#     #plot the PCA
#     sns.relplot(data=finalDf, x="principal component 1", y ='principal component 2', hue=hue, style=split) 
    
    
if __name__ == '__main__':
    bootstrap_test()
    two_way_anova()
    unpaired_t_test()
    BAc_confidence_intervals()
    boostrap()
    permutation_test()
    three_way_anova()
    two_way_ci()
    