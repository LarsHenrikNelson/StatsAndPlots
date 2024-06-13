import plotly.graph_objects as go
import numpy as np

from numpy.random import default_rng


from .plot_utils import get_func

MARKERS = [
    "o",
    "X",
    "^",
    "s",
    "*",
    "d",
    "h",
    "p",
    "<",
    "H",
    "D",
    "v",
    "P",
    ".",
    ">",
    "8",
]


def _jitter_plot(
    df,
    y,
    unique_groups,
    loc_dict,
    width,
    color_dict,
    marker_dict,
    edgecolor_dict,
    fig,
    alpha=1,
    seed=42,
    marker_size=2,
    transform=None,
    unique_id=None,
):
    transform = get_func(transform)
    rng = default_rng(seed)
    jitter_values = rng.random(unique_groups.size)
    jitter_values *= width
    jitter_values -= width / 2
    for i in unique_groups.unique():
        indexes = np.where(unique_groups == i)[0]
        if unique_id is None:
            x = np.array([loc_dict[i]] * indexes.size)
            x += jitter_values[indexes]
            scat = go.Scatter(
                x=x,
                y=transform(df[y].iloc[indexes]),
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
            unique_ids_sub = np.unique(df[unique_id].iloc[indexes])
            for index, ui_group in enumerate(unique_ids_sub):
                sub_indexes = np.where(
                    np.logical_and(df[unique_id] == ui_group, unique_groups == i)
                )[0]
                x = np.array([loc_dict[i]] * sub_indexes.size)
                x += jitter_values[sub_indexes]
                scat = go.Scatter(
                    x=x,
                    y=transform(df[y].iloc[indexes]),
                    mode="markers",
                    marker={
                        "symbol": MARKERS[index],
                        "color": color_dict[i],
                        "line": {"color": edgecolor_dict[i]},
                        "opacity": alpha,
                        "size": marker_size,
                    },
                    showlegend=False,
                    name=i,
                )
                fig.add_trace(scat)
