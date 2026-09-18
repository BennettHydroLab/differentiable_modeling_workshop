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

        # CAMELS ships no PET timeseries, only a long-term mean. We compute a
        # Hamon PET from temperature and then rescale it so its long-term mean
        # matches the CAMELS-reported `mean_pet` for this basin. Raw Hamon
        # underestimates that target by a factor that ranges from about 1.2 to
        # 2.2 across the 50 basins (it is worst in cold catchments), so a single
        # global correction would not do; the per-basin rescale is exact in the
        # mean and keeps the seasonal shape Hamon gives us.
        pet_raw = potential_et(ts["tmean"].values)
        target = float(attributes(basin_id)["mean_pet"])
        scale = target / float(pet_raw.mean())

        ts["pet"] = (("time",), pet_raw * scale)
        ts["pet"].attrs.update(
            units="mm/day",
            long_name="potential evapotranspiration",
            method=f"Hamon, rescaled by {scale:.2f} to match CAMELS mean_pet",
        )
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


def potential_et(tmean, method: str = "hamon"):
    """Potential evapotranspiration (mm/day) from mean daily temperature.

    minicamels distributes no PET timeseries, and every conceptual model in
    this workshop needs one. Rather than have each notebook invent its own,
    everything uses this function, so results stay comparable.

    This is Hamon's temperature-only method. It is crude — it knows nothing
    about wind, humidity or vegetation — but it needs only what CAMELS gives
    us, and PET error is largely absorbed by the calibrated parameters. Works
    on numpy arrays and on torch tensors, and is differentiable in the latter
    case.
    """
    if method != "hamon":
        raise ValueError(f"unknown PET method: {method!r}")

    xp = _backend(tmean)

    # Saturation vapour pressure (hPa), Tetens' formula.
    es = 6.108 * xp.exp(17.27 * tmean / (tmean + 237.3))

    # Hamon: PET proportional to saturated vapour density. The 29.8 lumps the
    # unit conversion and the standard daylight-hours coefficient together.
    pet = 29.8 * es / (tmean + 273.3)

    # No appreciable ET below freezing. A hard mask is fine here because PET
    # is an input, not something we differentiate a parameter through.
    return xp.where(tmean > -5.0, pet, xp.zeros_like(pet))


def _backend(x):
    """Array module matching ``x`` — mirrors the helper in metrics.py."""
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return torch
    except ImportError:
        pass
    return np
