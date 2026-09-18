"""Plot styling and the one figure every notebook draws: a hydrograph."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

# One palette for the whole workshop, so "observed" is the same colour in
# every notebook. Chosen to stay distinguishable in grayscale and for the
# most common forms of colour vision deficiency.
COLORS = {
    "obs": "#333333",
    "physics": "#0072B2",   # blue  — process-based model
    "ml": "#D55E00",        # orange — pure machine learning
    "hybrid": "#009E73",    # green — differentiable hybrid
    "extra": "#CC79A7",     # pink  — fourth series when needed
    "precip": "#56B4E9",
}


def set_style():
    """Apply the workshop's matplotlib defaults. Call once per notebook."""
    plt.rcParams.update({
        "figure.figsize": (10, 4),
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
    })


def hydrograph(time, obs=None, precip=None, title=None, ax=None, log=False, **series):
    """Plot simulated series against observations, with precipitation above.

    ``series`` is passed as keyword arguments so the call reads like the
    thing it draws::

        hydrograph(t, obs=q_obs, precip=p, physics=q_sac, hybrid=q_hybrid)

    Any keyword matching a name in :data:`COLORS` picks up that colour;
    anything else falls back to the default cycle.
    """
    if ax is None:
        if precip is not None:
            fig, (axp, ax) = plt.subplots(
                2, 1, sharex=True, figsize=(11, 5),
                gridspec_kw={"height_ratios": [1, 3], "hspace": 0.08},
            )
            axp.bar(time, precip, width=1.0, color=COLORS["precip"], linewidth=0)
            axp.invert_yaxis()
            axp.set_ylabel("P\n(mm/d)")
            axp.tick_params(labelbottom=False)
            axp.grid(alpha=0.2)
            if title:
                axp.set_title(title)
        else:
            fig, ax = plt.subplots(figsize=(11, 4))
            if title:
                ax.set_title(title)

    if obs is not None:
        ax.plot(time, obs, color=COLORS["obs"], label="Observed", lw=1.8, zorder=1)

    for name, values in series.items():
        if values is None:
            continue
        ax.plot(
            time, values,
            color=COLORS.get(name),
            label=name.replace("_", " ").title(),
            alpha=0.9, zorder=2,
        )

    ax.set_ylabel("Discharge (mm/d)")
    if log:
        ax.set_yscale("log")
    ax.legend(ncol=4, loc="upper right")
    return ax


def scatter_metrics(results: dict, metric: str = "NSE", ax=None):
    """Bar chart comparing one metric across named models."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    names = list(results)
    values = [results[n][metric] for n in names]
    ax.bar(names, values, color=[COLORS.get(n, "#888888") for n in names])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel(metric)
    return ax
