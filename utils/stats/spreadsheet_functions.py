import numpy as np
import pandas as pd
import statsmodels.api as sm


def mEPSC_raw_data(df: dict) -> dict[str, pd.DataFrame]:
    """Sorts and concatenates a dictionary of raw dataframes that
    have a header level 0 with data labels like amplitude and iei
    and header level 1 like cell numbers .

    Args:
        df (dict): Dictionary of raw data

    Returns:
        dictionary: Dictionary containing sorted raw data.
    """
    dict_key = list(df.keys())
    dict_values = [x.dropna(axis=1, how="all") for x in df.values()]
    df1 = pd.concat(dict_values)
    column_names = df1.columns.levels[0].to_list()
    df_dict = {}
    for i in column_names:
        x = []
        for key, value in zip(dict_key, dict_values):
            df_subset = value[i].copy()
            col = df_subset.columns
            ind = pd.Index(key + "_" + str(e) for e in col.tolist())
            df_subset.columns = ind
            x += [df_subset]
        x_concat = pd.concat(x, axis=1)
        df_dict[i] = x_concat
    return df_dict


def return_sorted_dfs(
    df: pd.DataFrame,
    df_td: pd.DataFrame,
    id_columns: list[str],
    group_columns: list[str],
) -> dict[str, pd.DataFrame]:
    """

    Parameters
    ----------
    df : Raw pandas dataframe.
    df_td : Pandas datframe used to create ids and groups. It is recommended
        that you clean your data first as the sorting process is 'blind' in
        that it returns any column that is in the id_columns.
    id_columns : Unique ID use to identify the subject. If there is only one
        column for the id then do not add brackets around the name. The
        id_columns must generate the same id for each excel sheet.
    group_columns : Genotype/sex/treatment groups. If there is only one
        column for the group then do not add brackets around the name.

    Returns
    -------
    column_list : Genotype/sex/treatment groups.
    dataframe_list : Raw data from columns sorted by group_columns.

    """

    df_td_copy = df_td.copy()
    if len(id_columns) > 1:
        df_td_copy["unique_id"] = (
            df_td_copy[id_columns].astype(str).agg("_".join, axis=1)
        )
    else:
        df_td_copy["unique_id"] = df_td_copy[id_columns]
    if len(group_columns) > 1:
        df_td_copy["groups"] = (
            df_td_copy[group_columns].astype(str).agg("_".join, axis=1)
        )
    else:
        df_td_copy["groups"] = df_td_copy[group_columns]
    df_groups = df_td_copy[["unique_id", "groups"]].copy()
    df_t = df.T.reset_index()
    df_t.rename(columns={"index": "unique_id"}, inplace=True)
    final_df = (
        pd.merge(df_groups, df_t, on=["unique_id"]).set_index(["unique_id", "groups"]).T
    )
    final_df.columns = final_df.columns.swaplevel(0, 1)
    column_list = final_df.columns.levels[0].to_list()
    sorted_df = {}
    for i in column_list:
        x = final_df[i].copy()
        sorted_df[i] = x
    return sorted_df


def sort_mepsc_data(
    raw_df_dict: dict,
    id_df: pd.DataFrame,
    id_columns: list[str],
    group_columns: list[str],
) -> dict[str, dict]:
    raw_dict = mEPSC_raw_data(raw_df_dict)
    sorted_dict = {}
    for i in raw_dict:
        sorted_data = return_sorted_dfs(
            raw_dict[i],
            id_df,
            id_columns=id_columns,
            group_columns=group_columns,
        )
        sorted_dict[i] = sorted_data
    return sorted_dict


def mEPSC_reg(df):
    c = df.to_numpy()
    slope = []
    for i in c:
        y = i[~np.isnan(i)]
        x = np.array(range(len(y)))
        x = sm.add_constant(x)
        model_1 = sm.OLS(y, x)
        results_1 = model_1.fit()
        slope_1 = results_1.params
        slope += [slope_1[1]]
    return slope


def find_percent_of_array(df):
    c = df.T.to_numpy()
    front = []
    back = []
    for i in c:
        x = i[~np.isnan(i)]
        f = x[0 : int(len(x) * 0.1)]
        b = x[int(len(x) * 0.9) : len(x)]
        front += [f]
        back += [b]
    return front, back


def return_f_b(iei, amp):
    # iei_t, amp_t = mEPSC_raw_data(df1)

    iei_f, iei_b = find_percent_of_array(iei)
    iei_f_mean = [np.mean(i) for i in iei_f]
    iei_b_mean = [np.mean(i) for i in iei_b]
    # iei_t['IEI_f_mean'] = iei_f_mean
    # iei_t['IEI_b_mean'] = iei_b_mean
    # iei_t.reset_index(inplace=True)
    # iei_final = iei_t[['index','IEI_b_mean', 'IEI_f_mean']].copy()

    amp_f, amp_b = find_percent_of_array(amp)
    amp_f_mean = [np.mean(i) for i in amp_f]
    amp_b_mean = [np.mean(i) for i in amp_b]
    # amp_t['Amp_f_mean'] = amp_f_mean
    # amp_t['Amp_b_mean'] = amp_b_mean
    # amp_t.reset_index(inplace=True)
    # amp_final = amp_t[['index','Amp_b_mean', 'Amp_f_mean']].copy()

    # df_merged_1 = df2.merge(iei_final, left_on=id_column, right_on='index')
    # df_merged_2 = df_merged_1.merge(amp_final, left_on=id_column, right_on='index')
    return iei_f_mean, iei_b_mean, amp_f_mean, amp_b_mean


def raw_mEPSC_reg(df1, df2, id_column):
    iei_t, amp_t = mEPSC_raw_data(df1)

    iei_t = iei_t.transpose()
    amp_t = amp_t.transpose()

    iei_slope = mEPSC_reg(iei_t)
    iei_t["IEI_Slope"] = iei_slope
    iei_t.reset_index(inplace=True)
    iei_final = iei_t[["index", "IEI_Slope"]].copy()

    amp_slope = mEPSC_reg(amp_t)
    amp_t["Amp_Slope"] = amp_slope
    amp_t.reset_index(inplace=True)
    amp_final = amp_t[["index", "Amp_Slope"]].copy()

    df_merged_1 = df2.merge(iei_final, left_on=id_column, right_on="index")
    df_merged_2 = df_merged_1.merge(amp_final, left_on=id_column, right_on="index")
    return df_merged_2, iei_final, amp_final


def membrane_resistance(df, current_list, column_name):
    df1 = df[current_list]
    index_1 = df1.index.values
    c = df1.to_numpy()
    x = np.array(df1.T.index.map(int)) / 1000
    x = x[2:5]
    x = sm.add_constant(x)
    slope = []
    for i in c:
        y = i[2:5]
        model_1 = sm.OLS(y, x)
        results_1 = model_1.fit()
        slope_1 = results_1.params
        slope += [slope_1[1]]
    resistance = pd.DataFrame(data=slope, index=index_1, columns=[column_name])
    return resistance


def create_ap_df(dfs, df_key):

    column_names, df_list = mEPSC_raw_data(dfs)

    (df_key_aps, dataframe_list_aps, final_df_aps) = return_sorted_dfs(
        df_list[0],
        df_key,
        id_columns=["Date", "Epoch"],
        group_columns=["Genotype", "D1_D2"],
    )

    df_ap_concat = pd.concat(dataframe_list_aps, axis=1)

    final = align_aps(df_ap_concat)

    (df_key_aps, dataframe_list_aps, final_df_aps) = return_sorted_dfs(
        final, df_key, id_columns=["Date", "Epoch"], group_columns=["Genotype", "D1_D2"]
    )

    final_ap_list = []
    for i, j in zip(dataframe_list_aps, df_key_aps):
        df_average = pd.DataFrame()
        df_average["Voltage"] = i.mean(axis=1)
        df_average["Velocity"] = df_average["Voltage"].diff() / np.diff(
            np.arange(len(df_average["Voltage"]) + 1) / 10
        )
        df_average.dropna(inplace=True)
        df_average["Genotype"] = j
        final_ap_list += [df_average]

    final_df = pd.concat(final_ap_list, axis=0)
    return final_df


def align_aps(df):
    """
    This function aligns a dataframe of arrays by the index of their max value.
    The function chops off the beginning and end of each array after aligning
    each array to the array that has a

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    arrays : TYPE
        DESCRIPTION.

    """
    pulse_dict = df.to_dict(orient="list")
    ap_max_values = [np.argmax(i) for i in pulse_dict.values()]
    min_ap = min(ap_max_values)
    start_values = [min_ap - i for i in ap_max_values]
    arrays = [
        np.append(i * [j[0]], j) for i, j in zip(start_values, pulse_dict.values())
    ]
    final_df = pd.DataFrame(data=np.transpose(np.array(arrays)), columns=df.columns)
    final_df.dropna(axis=0, how="any", inplace=True)
    return final_df


if __name__ == "__main__":
    return_sorted_dfs()
    return_f_b()
    raw_mEPSC_reg()
    find_percent_of_array()
    mEPSC_reg()
    mEPSC_raw_data()
    membrane_resistance()
    align_aps()
    create_ap_df()
    sort_mepsc_data()
