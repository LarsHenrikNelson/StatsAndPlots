import datetime

from . import (
    plotting,  # noqa: F401
    stats,  # noqa: F401
)
from .stats import *  # noqa: F401

now = datetime.datetime.now()

__version__ = f"{now.year}.{now.month}.{now.day}"
