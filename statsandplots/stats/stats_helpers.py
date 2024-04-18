from .stats_functions import two_way_anova


def run_batch_aov(columns, data, group, subgroup):
    output = {}
    for i in columns:
        output[i] = two_way_anova(data, group, subgroup, i, "bonferroni")
    return output
