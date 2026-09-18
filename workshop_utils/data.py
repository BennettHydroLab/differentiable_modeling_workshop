"""Loading and shaping minicamels data for the workshop.

`minicamels` already gives us clean xarray Datasets. What this module adds is
the workshop's *conventions*: which basins we use as running examples, how we
split time, and how we hand arrays to PyTorch.
"""

from __future__ import annotations

import numpy as np

# Three running examples, chosen from the minicamels attribute table to span
# very different rainfall-runoff behaviour. Using named constants instead of
# bare gauge IDs keeps the notebooks readable.
BASIN_TEMPERATE = "02016000"  # Cowpasture River, VA — humid, snow-free, well behaved
BASIN_SNOWY = "13313000"      # Johnson Creek, ID — 74% of precip falls as snow
BASIN_ARID = "06353000"       # Cedar Creek, ND — runoff ratio 0.04, flashy and hard

BASIN_LABELS = {
    BASIN_TEMPERATE: "Cowpasture River, VA (humid temperate)",
    BASIN_SNOWY: "Johnson Creek, ID (snow dominated)",
    BASIN_ARID: "Cedar Creek, ND (semi-arid)",
}

FORCING_VARS = ["prcp", "tmax", "tmin", "srad", "vp"]
TARGET_VAR = "qobs"

_CACHE: dict = {}


def _dataset():
    """A single shared MiniCamels handle, so repeated calls do not refetch."""
    if "ds" not in _CACHE:
        from minicamels import MiniCamels

        _CACHE["ds"] = MiniCamels()
    return _CACHE["ds"]


def load_basin(basin_id: str = BASIN_TEMPERATE):
    """Load one basin's daily timeseries as an xarray Dataset.

    Adds a ``tmean`` variable, since most conceptual models want a single
    daily temperature rather than the min/max pair CAMELS distributes.
    """
    key = f"ts::{basin_id}"
    if key not in _CACHE:
        ts = _dataset().load_basin(basin_id)
        ts["tmean"] = (ts["tmax"] + ts["tmin"]) / 2.0
        ts["tmean"].attrs.update(units="degC", long_name="mean daily air temperature")
        _CACHE[key] = ts
    return _CACHE[key]


def attributes(basin_id: str | None = None):
    """Static catchment attributes: the whole table, or one basin's row."""
    attrs = _dataset().attributes()
    return attrs if basin_id is None else attrs.loc[basin_id]


def split_by_water_year(ds, train, test, spinup_days: int = 365):
    """Split a basin timeseries into train/test by water year.

    ``train`` and ``test`` are ``(first_wy, last_wy)`` tuples, inclusive. A
    water year ends 30 September, so WY2000 runs 1999-10-01 to 2000-09-30.

    ``spinup_days`` of data is prepended to each period so model states
    (soil moisture, snowpack) start from something physically sensible rather
    than from whatever we guessed. Those days are returned but should be
    excluded when you compute the loss — see ``spinup`` in the result.
    """
    def _slice(bounds):
        first, last = bounds
        start = np.datetime64(f"{first - 1}-10-01") - np.timedelta64(spinup_days, "D")
        end = np.datetime64(f"{last}-09-30")
        return ds.sel(time=slice(start, end))

    return {
        "train": _slice(train),
        "test": _slice(test),
        "spinup": spinup_days,
    }


def to_tensors(ds, variables=None, target=TARGET_VAR, device="cpu", dtype=None):
    """Convert an xarray Dataset to the ``(forcings, target)`` tensors we train on.

    Returns ``forcings`` of shape ``(time, n_variables)`` and ``target`` of
    shape ``(time,)``. Observation gaps are left as NaN — the metrics in
    :mod:`workshop_utils.metrics` mask them, so do not fill them here.
    """
    import torch

    dtype = dtype or torch.float32
    variables = variables or FORCING_VARS

    forcings = np.stack([ds[v].values for v in variables], axis=-1)
    y = ds[target].values

    return (
        torch.tensor(forcings, dtype=dtype, device=device),
        torch.tensor(y, dtype=dtype, device=device),
    )


def standardizer(x):
    """Return ``(normalize, denormalize)`` closures fitted to ``x``.

    Neural networks train badly on raw hydrologic units — precipitation in
    mm/day and radiation in W/m² differ by two orders of magnitude. Fit this
    on the *training* period only and reuse it on the test period, or you
    have leaked information across the split.
    """
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (lambda v: (v - mean) / std), (lambda v: v * std + mean)
