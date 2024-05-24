This is a simple plotting and stats package written in Python and intend for scientific publications (i.e. plotting data that would be used in t-test or one/two-way ANOVAs). Some of the plots can utilize Matplotlib or Plotly backend. It is currently in development so there may be bugs. See below for install and some simple examples.

Install
```bash
pip install git+https://github.com/LarsHenrikNelson/StatsAndPlots.git
```

<br/>

Import packages
```python
import pandas as pd
from numpy.random import default_rng
from statsandplots.plotting import CategoricalPlot
```
<br/>

Create some data
```python
rng = default_rng()
data = rng.normal(3, 1.0, size=40)
group_1 = ["one" if i < 20 else "two" for i in range(40)]
group_2 = ["blue"] * 10 + ["red"] * 20 + ["blue"] * 10

df = pd.DataFrame({"data": data, "group_1": group_1, "group_2": group_2})
```

<br/>

Plot just one grouping with as many categories as you want. 
```python
# Create plot, you can also just chain this step to any of the ones below
c = CategoricalPlot(
    df=df,
    y="data",
    y_label="your columns",
    group="group_1",
    group_order=[r"one", r"two"],
    group_spacing=1.0,
    title="",
    inplace=False,
)
# Plot example using jitter and mean
# Group spacing distance on the x-axis is where each group will be placed relative to each other
# jitter specifies the proportion (or percentage) of space each group will take up in their allocated space
path = "path/to/you/save/destination"
fig = (
    c.jitter(
        # You need specify each group and each groups color
        color={r"one": "black", r"two": "green"},
        marker="D",
        edgecolor="white",
        alpha=0.8,
        jitter=0.6,
        seed=42,
        transform=None,
        marker_size=10,
    )
    .summary(
        func="mean",
        capsize=0,
        capstyle="round",
        bar_width=0.5,
        err_func="sem",
        linewidth=2,
        transform=None,
        color="black",
    )
    .plot_settings(
        style="default",
        # y_lim=[0, 8],
        y_scale="linear",
        steps=5,
        margins=0.2,
        aspect=1,
        figsize=None,
        labelsize=22,
        linewidth=2,
        ticksize=2,
        ticklabel=20,
        decimals=2,
    )
    .plot(savefig=False, path=path, filetype="png", backend="matplotlib")
)
```

<br/>

Plot with two groupings with as many categories as you want.
```python
# Boxplots can finicky with matplotlib's plot method so you may need to adjust group_spacing and subgroup_spacing
# Plots are plotted in the order that they are added so you generally want to use the boxplot, summary of violin first
path = "path/to/you/save/destination"
fig = (
    CategoricalPlot(
        df=df,
        y="data",
        y_label="your columns",
        group="group_1",
        group_order=[r"one", r"two"],
        subgroup="group_2",
        subgroup_order=["red", "blue"],
        group_spacing=1.2,
        subgroup_spacing=0.8,
        title="",
        inplace=False,
    )
    .boxplot(
        facecolor="blue",
        linecolor="red",
        fliers="",
        box_width=0.9,
        transform=None,
        linewidth=1,
        alpha=0.5,
        line_alpha=0.5,
        show_means=False,
        show_ci=False,
    )
    .jitter(
        # You need specify each group and each groups color
        color={r"one": "black", r"two": "green"},
        marker="D",
        edgecolor="white",
        alpha=0.8,
        jitter=0.5,
        seed=42,
        transform=None,
        unique_id="unique_id",
        marker_size=8,
    )
    .plot_settings(
        style="default",
        # y_lim=[0, 8],
        y_scale="linear",
        steps=5,
        margins=0.05,
        aspect=1,
        figsize=None,
        labelsize=22,
        linewidth=2,
        ticksize=2,
        ticklabel=20,
        decimals=2,
    )
    .plot(savefig=False, path=path, filetype="png", backend="matplotlib")
)
```