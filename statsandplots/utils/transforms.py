from typing import Literal

from scipy import stats
import numpy as np

from ..stats.circular_stats import periodic_mean, periodic_std, periodic_sem


def sem(a, axis=None):
    if len(a.shape) == 2:
        shape = a.shape[1]
    else:
        shape = a.size
    return np.std(a, axis) / np.sqrt(shape - 1)


def ci(a, axis=None):
    t_critical = stats.t.ppf(1 - 0.05 / 2, len(a) - 1)
    margin_of_error = t_critical * (np.std(a, ddof=1) / np.sqrt(len(a)))
    return margin_of_error


def ci_bca(a):
    res = stats.bootstrap(a, np.mean)
    print(res.confidence_interval)
    return np.array([[res.confidence_interval.high], [res.confidence_interval.low]])


def mad(a, axis=None):
    return np.median(np.abs(a - np.median(a)))


TRANSFORM = Literal["log10", "log2", "ln", "inverse", "ninverse", "sqrt"]
AGGREGATE = Literal["mean", "periodic_mean", "nanmean", "median", "nanmedian"]
ERROR = Literal[
    "sem", "ci", "periodic_std", "periodic_sem", "std", "nanstd", "var", "nanvar", "mad"
]

BACK_TRANSFORM_DICT = {
    "log10": lambda x: 10**x,
    "log2": lambda x: 2**x,
    "ninverse": lambda x: -1 / x,
    "inverse": lambda x: 1 / x,
    "ln": lambda x: np.e**x,
    "sqrt": lambda x: x**2,
}

FUNC_DICT = {
    "sem": sem,
    "ci": ci,
    "ci_bca": ci_bca,
    "mean": np.mean,
    "periodic_mean": periodic_mean,
    "periodic_std": periodic_std,
    "periodic_sem": periodic_sem,
    "nanmean": np.nanmean,
    "nanmedian": np.nanmedian,
    "median": np.median,
    "std": np.std,
    "nanstd": np.nanstd,
    "log10": np.log10,
    "log2": np.log2,
    "ln": np.log,
    "var": np.var,
    "nanvar": np.nanvar,
    "inverse": lambda a, axis=None: 1 / (a + 1e-10),
    "ninverse": lambda a, axis=None: -1 / (a + 1e-10),
    "sqrt": np.sqrt,
    "mad": mad,
    "wrap_pi": lambda a: np.where(a < 0, a + 2 * np.pi, a),
}


def get_func(input):
    if input in FUNC_DICT:
        return FUNC_DICT[input]
    else:
        return lambda a, axis=None: a
