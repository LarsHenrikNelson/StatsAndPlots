from pathlib import Path
from typing import Union

import pandas as pd


__all__ = ["write_to_txt", "save_aov"]


def serialize_bootstrap(data):
    x = ""
    p_value = "P_vale"
    cib = "Lower_CI"
    cit = "Upper_CI"
    x += f"p = {data[p_value].iloc[0]}, CI[{data[cib].iloc[0]}, {data[cit].iloc}]"
    return x


def write_to_txt(data, filename: Union[str, dict[str, str]] = None):
    if isinstance(data, str):
        temp = data
        data["data"] = temp
    if filename is None:
        filename = Path().cwd() / "stats.txt"
    with open(filename, mode="w", encoding="utf-8") as txtfile:
        for key, value in data.items():
            txtfile.write(key + ": " + value)


def save_aov(data, filename):
    with pd.ExcelWriter(
        filename,
        engine="xlsxwriter",
    ) as writer:
        for outer_key, d in data.items():
            for inner_key, value in d.items():
                value.to_excel(writer, sheet_name=f"{outer_key}_{inner_key}_aov")
