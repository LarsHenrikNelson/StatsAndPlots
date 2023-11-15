import plotly.graph_objects as go
import numpy as np

from numpy.random import default_rng


from .plot_utils import get_func, transform_func


def _jitter_plot(
    df,
    y,
    unique_groups,
    loc_dict,
    width,
    color_dict,
    marker_dict,
    edgecolor_dict,
    alpha=1,
    seed=42,
    marker_size=2,
    transform=None,
    fig=None,
):
    transform = get_func(transform)
    rng = default_rng(seed)
    jitter_values = rng.random(unique_groups.size)
    # jitter_values *= float(width)
    # jitter_values *= float(jitter)
    # jitter_values -= (jitter * float(width)) / 2.0
    jitter_values *= width
    jitter_values -= width / 2
    for i in unique_groups.unique():
        indexes = np.where(unique_groups == i)[0]
        x = np.array([loc_dict[i]] * indexes.size)
        x += jitter_values[indexes]
        scat = go.Scatter(
            x=x,
            y=transform_func(df[y].iloc[indexes], transform),
            mode="markers",
            marker={
                "symbol": marker_dict[i],
                "color": color_dict[i],
                "line": {"color": edgecolor_dict[i]},
                "opacity": alpha,
                "size": marker_size,
            },
            showlegend=False,
            name=i,
        )
        fig.add_trace(scat)
    # return fig
