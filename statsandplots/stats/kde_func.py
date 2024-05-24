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
    tol: float = 3,
):
    data = np.asarray(data)
    kde_obj = KDEpy.FFTKDE(kernel=kernel, bw=bw).fit(data)
    width = np.cov(data) * kde_obj.bw**2
    width = np.sqrt(width)
    min_data = data.min() - width * tol
    max_data = data.max() + width * tol
    power2 = int(np.ceil(np.log2(len(data))))
    x = np.linspace(min_data, max_data, num=(1 << power2))
    y = kde_obj.evaluate(x)
    return x, y
