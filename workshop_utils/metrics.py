"""Evaluation metrics for hydrologic simulation.

Every function accepts numpy arrays or torch tensors and returns the same
flavour it was given, so the identical call works inside a training loop
(differentiable, on tensors) and in an evaluation cell (on numpy).
"""

from __future__ import annotations

import numpy as np

try:  # torch is optional so this module stays importable in a slim kernel
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


def _is_torch(x) -> bool:
    return _HAS_TORCH and isinstance(x, torch.Tensor)


def _backend(x):
    """Return the array module that matches ``x``."""
    return torch if _is_torch(x) else np


def _drop_nan(sim, obs):
    """Drop time steps where the observation is missing.

    Streamflow records have gaps; scoring against them silently poisons the
    mean. We mask on ``obs`` only, so a NaN appearing in ``sim`` still shows
    up as a NaN score rather than being quietly hidden.
    """
    xp = _backend(obs)
    mask = ~xp.isnan(obs)
    return sim[mask], obs[mask]


def nse(sim, obs):
    """Nash-Sutcliffe efficiency.

    1.0 is perfect; 0.0 means the simulation is no better than predicting the
    mean of the observations. Because the denominator is the observed
    variance, NSE is dominated by the ability to match high flows.
    """
    sim, obs = _drop_nan(sim, obs)
    xp = _backend(obs)
    denom = ((obs - obs.mean()) ** 2).sum()
    return 1.0 - ((sim - obs) ** 2).sum() / denom


def kge(sim, obs):
    """Kling-Gupta efficiency (Gupta et al., 2009).

    Decomposes skill into correlation, variability ratio and bias ratio,
    which is why it is usually preferred over NSE for hydrologic calibration:
    a model can no longer buy a good score purely by damping its variance.
    """
    sim, obs = _drop_nan(sim, obs)
    xp = _backend(obs)

    sim_m, obs_m = sim.mean(), obs.mean()
    sim_s, obs_s = sim.std(), obs.std()

    cov = ((sim - sim_m) * (obs - obs_m)).mean()
    r = cov / (sim_s * obs_s)
    alpha = sim_s / obs_s  # variability ratio
    beta = sim_m / obs_m  # bias ratio

    return 1.0 - xp.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def pbias(sim, obs):
    """Percent bias. Positive means the model runs wet."""
    sim, obs = _drop_nan(sim, obs)
    return 100.0 * (sim - obs).sum() / obs.sum()


def rmse(sim, obs):
    """Root mean squared error, in the units of the input."""
    sim, obs = _drop_nan(sim, obs)
    xp = _backend(obs)
    return xp.sqrt(((sim - obs) ** 2).mean())


def log_nse(sim, obs, eps: float = 1e-2):
    """NSE computed on log-transformed flows, to weight low flows.

    ``eps`` keeps zero-flow days finite; it is in mm/day, so 1e-2 sits below
    the precision of most gauge records.
    """
    sim, obs = _drop_nan(sim, obs)
    xp = _backend(obs)
    return nse(xp.log(sim + eps), xp.log(obs + eps))


def summary(sim, obs) -> dict:
    """All metrics at once, as plain Python floats, for printing in a table."""
    out = {}
    for name, fn in [
        ("NSE", nse),
        ("KGE", kge),
        ("logNSE", log_nse),
        ("PBIAS", pbias),
        ("RMSE", rmse),
    ]:
        value = fn(sim, obs)
        out[name] = float(value.detach()) if _is_torch(value) else float(value)
    return out
