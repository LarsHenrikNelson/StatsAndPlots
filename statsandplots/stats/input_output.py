from pathlib import Path
from typing import Union

import pandas as pd

from .stats_helpers import BaseData


__all__ = ["write_to_txt", "save_data"]


SUBSCRIPT_MAP = {
    "0": "₀",
    "1": "₁",
    "2": "₂",
    "3": "₃",
    "4": "₄",
    "5": "₅",
    "6": "₆",
    "7": "₇",
    "8": "₈",
    "9": "₉",
    ".": r"\U+002E",
}


def to_subscript(number):
    return "".join(SUBSCRIPT_MAP[digit] for digit in str(number))


def write_to_txt(data: Union[str, BaseData], filename: Union[str, Path] = None):
    if isinstance(data, str):
        data = {}
        temp = {}
        temp["text"] = data
        data["data"] = temp
    if filename is None:
        filename = Path().cwd() / "stats.txt"
    else:
        filename = Path(filename) / "stats.txt"
    with open(filename, mode="w", encoding="utf-8") as txtfile:
        for outerkey, outervalue in data.items():
            txtfile.write(outerkey + ": " + outervalue["text"])


def save_data(data: Union[str, BaseData], filename: Union[str, Path] = None):
    if filename is None:
        filename = Path().cwd() / "stats.xlsx"
    else:
        filename = Path(filename) / "stats.xlsx"
    with pd.ExcelWriter(
        filename,
        engine="openpyxl",
    ) as writer:
        for outer_key, d in data.items():
            d["table"].to_excel(writer, sheet_name=f"{outer_key}_aov")
            d["descriptive_stats"].to_excel(writer, sheet_name=f"{outer_key}_ds")
