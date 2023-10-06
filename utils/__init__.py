import datetime

__version__ = datetime.datetime.now().strftime("%Y.%m.%d")

from . import (
    plotting,  # noqa: F401
    stats,  # noqa: F401
)
from .stats import *  # noqa: F401
