from pathlib import Path
from typing import Union

import pandas as pd


__all__ = ["serialize_aov", "write_to_txt"]


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
        x += f"({groups[i]}: F{int(df.iloc[i]),int(df.iloc[-1])} = {F.iloc[i]}, p = {p.iloc[i]}, η2 = {n2.iloc[i]}, 95% CI[{cib.iloc[i]}, {cit.iloc[i]}];"
        x += " "
    x += "\n"
    return x

def serialize_bootstrap(data):
    x = ""
    x += f"p={data["P_value"].iloc[0]}, CI=[{data["Lower_CI"].iloc[0]}, {data["Upper_CI"].iloc}]"
    return x


def write_to_txt(data, filename: Union[str, dict[str, str]]=None):
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
    
