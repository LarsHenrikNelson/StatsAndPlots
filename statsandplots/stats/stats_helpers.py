import math
from typing import TypedDict

import numpy as np
import pandas as pd

__all__ = [
    "round_sig",
]


class BaseData(TypedDict):
    data: pd.DataFrame
    text: str
    descriptive_stats: pd.DataFrame


def round_sig(x, sig=2):
    if np.isnan(x):
        return np.nan
    elif x == 0:
        return 0
    elif x != 0 or not np.isnan(x):
        temp = math.floor(math.log10(abs(x)))
        if np.isnan(temp):
            return round(x, 0)
        else:
            return round(x, sig - int(temp) - 1)
