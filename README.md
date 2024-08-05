This is a simple plotting and stats package written in Python and intend for scientific publications (i.e. plotting data that would be used in t-test or one/two-way ANOVAs). There is a strong focus on plotting clustered data within groups. This is particularly useful for studies where many neurons are measured per mouse. Some of the plots can utilize Matplotlib or Plotly backend. Data can be transformed (log10, inverse, etc) or aggregated (mean, median, circular mean, etc). Space between groups is easy to specify using width or jitter. It is currently in development so there may be bugs. See below for install some simple examples.

Install
```bash
pip install git+https://github.com/LarsHenrikNelson/StatsAndPlots.git
```

## Example plots
Some example plots with the code used below the plot.

### Jitter with unique id (i.e. animal, data, etc) setting marker type and summary (mean) overlayed.
<img src="examples/jitter-example-1.png">

```python
path = r"path/to/save/data"
fig = (
    CategoricalPlot(
        df=df_lhn.dropna(subset=columns[column]),
        y="Membrane resistance,
        y_label="Membrane resistance",
        group="genotype",
        group_order=[r"Shank3B+/+", r"Shank3B-/-"],
        group_spacing=1.25,
        title="",
        inplace=False,
    )
    .jitter(
        unique_id="date",
        color={r"Shank3B+/+": "orange", r"Shank3B-/-": "orchid"},
        edgecolor="white",
        alpha=0.5,
        jitter=0.8,
        transform=None,
        marker_size=7,
    )
    .summary(
        func="mean",
        capsize=0,
        capstyle="round",
        bar_width=0.8,
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
        margins=0.1,
        aspect=1,
        figsize=None,
        labelsize=22,
        linewidth=2,
        ticksize=2,
        ticklabel=20,
        decimals=2,
    )
    .plot(savefig=True, path=path, filetype="png", backend="matplotlib")
)
```

### Jitteru with unique id (i.e. animal, data, etc). Each unique id is given a column within the group column. With summary overlayed.
<img src="examples/jitteru-example-2.png">

```python

path = r"path/to/save/data"
column = "sttc_5ms"
temp = (
    CategoricalPlot(
        df=df,
        y=column,
        y_label="STTC 5ms",
        group="genotype",
        group_order=[r"WT", r"KO"],
        subgroup="cell_type_short",
        subgroup_order=["PS", "PL", "I"],
        group_spacing=1.0,
        subgroup_spacing=1.4,
        title="",
        inplace=False,
    ).jitteru(
        unique_id="id",
        edgecolor="none",
        width=0.55,
        alpha=0.5,
        color={
            "PS": "green",
            "PL": "grey",
            "I": "royalblue",
            "IB": "purple",
            "IS": "orange",
        },
    )
    .summary(
        func="mean",
        capsize=0,
        capstyle="round",
        bar_width=0.8,
        err_func="ci",
        linewidth=2,
        alpha=0.8,
        color={
            "PS": "green",
            "PL": "grey",
            "I": "royalblue",
        },
    )
)
temp.plot(savefig=True, path=path, filetype="png", backend="matplotlib")
```

### Jitteru using an aggregate func
<img src="examples/jitteru-aggregate-example.png">

```python
path = r"path/to/save/folder"
temp = (
    CategoricalPlot(
        df=df,
        y=column,
        y_label="Waveform width",
        group="genotype",
        group_order=[r"WT", r"KO"],
        subgroup="cell_type_short",
        # subgroup_order=["PS", "PL", "I"],
        subgroup_order=["PS", "PL", "I", "IB", "IS"],
        group_spacing=1.0,
        subgroup_spacing=1.4,
        title="",
        inplace=False,
    )
    .jitteru(
        unique_id="id",
        edgecolor="none",
        marker=".",
        width=0.55,
        marker_size=2,
        alpha=0.8,
        color={
            "PS": "green",
            "PL": "grey",
            "I": "royalblue",
            "IB": "purple",
            "IS": "orange",
        },
    )
    .jitteru(
        unique_id="id",
        edgecolor="black",
        jitter=0.0,
        width=0.55,
        alpha=0.9,
        color="black",
        agg_func="mean",
        marker_size=2,
        marker="o"
    )
)
temp.plot(savefig=True, path=path, filetype="png", backend="matplotlib")
```

### Percent plot where the y-axis represents fraction of samples below and above a cutoff (i.e. p-value cutoff).
<img src="examples/percent-example-1.png">

```python

path = r"path/to/save/folder"
column = "beta_rayleigh_pval"
temp = (
    CategoricalPlot(
        df=df,
        y=column,
        y_label=column,
        group="genotype",
        group_order=[r"WT", r"KO"],
        subgroup="cell_type_short",
        subgroup_order=["PS", "PL", "I", "J", "K"],
        group_spacing=1.0,
        subgroup_spacing=1.0,
        title="",
        inplace=False,
    )
    .percent(
        cutoff=0.05,
        include_bins=[True, True],
        fill=False,
        hatch=True,
        linecolor={
            "PS": "green",
            "PL": "grey",
            "I": "royalblue",
            "J": "orchid",
            "K": "darkorange"
        },
        alpha=0.3,
        bar_width=0.8,
        linewidth=0,
    )
    .plot(savefig=True, path=path, filetype="png", backend="matplotlib")
)
```

### Percent plot with unique id where the y-axis represents fraction of samples below and above a cutoff (i.e. p-value cutoff).
<img src="examples/percent-example-2.png">

```python

path = r"path/to/save/folder"
column = "beta_rayleigh_pval"
temp = (
    CategoricalPlot(
        df=df,
        y=column,
        y_label=column,
        group="genotype",
        group_order=[r"WT", r"KO"],
        subgroup="cell_type_short",
        subgroup_order=["PS", "PL", "I"],
        group_spacing=1.0,
        subgroup_spacing=1.0,
        title="",
        inplace=False,
    )
    .percent(
        cutoff=0.05,
        include_bins=[True, False],
        fill=True,
        hatch=None,
        unique_id="id",
        linecolor={
            "PS": "green",
            "PL": "grey",
            "I": "royalblue",
        },
        facecolor={
            "PS": "green",
            "PL": "grey",
            "I": "royalblue",
        },
        alpha=0.6,
        bar_width=0.8,
        linewidth=1,
    )
    .plot(savefig=True, path=path, filetype="png", backend="matplotlib")
)
```

### Percent plot with unique on top of percent plot.
<img src="examples/percent-example-3.png">

```python

path = r"path/to/save/folder"
column = "beta_rayleigh_pval"
temp = (
    CategoricalPlot(
        df=df,
        y=column,
        y_label=column,
        group="genotype",
        group_order=[r"WT", r"KO"],
        subgroup="cell_type_short",
        subgroup_order=["PS", "PL", "I"],
        group_spacing=1.0,
        subgroup_spacing=1.0,
        title="",
        inplace=False,
    )
    .percent(
        cutoff=0.05,
        include_bins=[True, False],
        fill=True,
        hatch=None,
        linecolor={
            "PS": "green",
            "PL": "grey",
            "I": "royalblue",
        },
        facecolor={
            "PS": "green",
            "PL": "grey",
            "I": "royalblue",
        },
        alpha=0.3,
        bar_width=0.8,
        linewidth=0,
    )
    .percent(
        cutoff=0.05,
        include_bins=[True, False],
        fill=True,
        hatch=None,
        unique_id="id",
        linecolor={
            "PS": "green",
            "PL": "grey",
            "I": "royalblue",
        },
        facecolor={
            "PS": "green",
            "PL": "grey",
            "I": "royalblue",
        },
        alpha=0.6,
        bar_width=0.8,
        linewidth=1,
    )
    .plot(savefig=True, path=path, filetype="png", backend="matplotlib")
)
```

### Summary plot (mean of group).
<img src="examples/summary-example-1.png">

<br/>

### Count plot (similar to a histogram of non-numeric data, splitting categorically similar to other categorical plots)
<img src="examples/count-example.png">

```python
column = "group"
temp = (
    CategoricalPlot(
        df=df_output,
        y=column,
        y_label="Proportion of units",
        group="genotype",
        group_order=["WT", "KO"],
        group_spacing=1.0,
        title="",
        inplace=False,
    ).count(
        bar_width=0.8, facecolor=color_mapping, axis_type="density", linecolor="black"
    )
    .plot()
)
```

## More example code using a random dataset

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