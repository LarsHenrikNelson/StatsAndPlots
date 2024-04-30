from typing import Literal

import KDEpy
import numpy as np
import numpy.typing as npt


def kde(
    data: npt.ArrayLike,
    kernel: Literal[
        "gaussian",
        "exponential",
        "box",
        "tri",
        "epa",
        "biweight",
        "triweight",
        "tricube",
        "cosine",
    ] = "gaussian",
    bw: Literal["ISJ", "silverman", "scott"] = "ISJ",
    tol: float = 0.01,
):
    data = np.asarray(data)
    power2 = int(np.ceil(np.log2(len(data))))
    width = np.cov(data)
    min_data = data.min() - width * tol
    max_data = data.max() + width * tol
    x = np.linspace(min_data, max_data, num=1 << power2)
    y = KDEpy.FFTKDE(kernel=kernel, bw=bw).fit(data).evaluate(x)
    return x, y
