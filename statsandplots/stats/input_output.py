from pathlib import Path
from typing import Union

import pandas as pd

from .aov import TwoWayAnovaData


__all__ = ["write_to_txt", "save_aov"]


def write_to_txt(
    data: Union[str, TwoWayAnovaData], filename: Union[str, dict[str, str]] = None
):
    if isinstance(data, str):
        data = {}
        temp = {}
        temp["text"] = data
        data["data"] = temp
    if filename is None:
        filename = Path().cwd() / "stats.txt"
    with open(filename, mode="w", encoding="utf-8") as txtfile:
        for outerkey, outervalue in data.items():
            txtfile.write(outerkey + ": " + outervalue["text"])


def save_aov(data, filename):
    with pd.ExcelWriter(
        filename,
        engine="xlsxwriter",
    ) as writer:
        for outer_key, d in data.items():
            for inner_key, value in d.items():
                value.to_excel(writer, sheet_name=f"{outer_key}_{inner_key}_aov")
