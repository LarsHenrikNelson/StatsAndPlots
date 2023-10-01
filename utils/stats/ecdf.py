import numpy as np
from numpy.random import default_rng
import pandas as pd


# Could potentially speed this up with numba however,
# numba does not seem to support the new numpy random
# number generator choice method
def rand_samp_column(
    array: np.ndarray, repititions: int, size: int, axis: int
) -> np.ndarray:
    """Randomly sample from from a row or column in a numpy array with
    replacement.

    Args:
        array (np.array): numpy array with rows or columns to be resampled.
        repititions (int): Number of times to resample each column.
        size (int): Number of samples to take with replacement.
        axis (int): Axis to sample from.

    Returns:
        pd.DataFrame: Resampled data output as a single
    """
    # This extracts "size" random values from each cell x times
    # final_arrays = []
    rng = default_rng(42)
    if axis == 1:
        array = array.T
    temp_array = np.zeros(array.shape[0] * size)
    for i in np.arange(0, repititions):
        extracted_df1 = []
        for j in np.arange(0, array.shape[0]):
            # Copy column to numpy array
            x1 = array[j, :]

            # Drop nans
            x2 = x1[~np.isnan(x1)]
            if size is None:
                size_1 = len(x2)
            else:
                size_1 = size

            # Choose samples
            b1 = rng.choice(x2, size=size_1)
            extracted_df1 = np.concatenate([extracted_df1, b1])
        extracted_df1.sort()
        # final_arrays += [extracted_df1]
        temp_array += extracted_df1
    final_array = temp_array / repititions
    return final_array


def sample_from_dfs(
    dfs: dict[str, pd.DataFrame], repititions: int, size: int, axis: int
) -> dict[str, pd.DataFrame]:
    sampled_dfs = {}
    for key, df in dfs.items():
        array = rand_samp_column(df.to_numpy(), repititions, size, axis=1)
        sampled_dfs[key] = pd.DataFrame(array)
    return sampled_dfs


def resampled_cum_prob_df(
    df_dict: dict[str, pd.DataFrame],
    data_id: str,
    group: str,
    repititions: int = 1000,
    size: int = 100,
    axis: int = 1,
) -> pd.DataFrame:
    sampled_dfs = sample_from_dfs(df_dict, repititions, size, axis)
    df_cumsum = []
    for key, df in sampled_dfs.items():
        df.loc[:, data_id] = df.mean(axis=1)
        df.loc[:, "Cumulative Probability"] = (
            1.0 * np.arange(len(df[data_id])) / (len(df[data_id]) - 1)
        )
        df.loc[:, group] = key
        y = df[[data_id, "Cumulative Probability", group]]
        df_cumsum += [y]
    finished_df = pd.concat(df_cumsum)
    return finished_df


def cum_prob_df(dfs, df_keys, data, group):
    """


    Parameters
    ----------
    dfs : TYPE
        DESCRIPTION.
    df_keys : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.
    group : TYPE
        DESCRIPTION.

    Returns
    -------
    finished_df : TYPE
        DESCRIPTION.

    """
    cum_df_list = []
    for df, key in zip(dfs, df_keys):
        df_stacked = df.stack().reset_index(drop=True).sort_values()
        cum_stacked = 1.0 * np.arange(len(df_stacked)) / (len(df_stacked) - 1)
        cum_df = pd.DataFrame(
            {data: df_stacked.to_numpy(), "Cumulative Probability": cum_stacked}
        )
        cum_df[group] = key
        cum_df_list += [cum_df]
    finished_df = pd.concat(cum_df_list)
    return finished_df


if __name__ == "__main__":
    resampled_cum_prob_df()
    sample_from_dfs()
    rand_samp_column()
    cum_prob_df()
