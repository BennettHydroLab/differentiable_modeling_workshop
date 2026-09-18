"""Shared helpers for the differentiable hydrologic modeling workshop.

Kept deliberately small. Anything that teaches a concept belongs in a
notebook where participants can see it; this module only holds the plumbing
we do not want to retype in every notebook.
"""

from workshop_utils.metrics import kge, log_nse, nse, pbias, rmse, summary
from workshop_utils.data import (
    BASIN_SNOWY,
    BASIN_ARID,
    BASIN_TEMPERATE,
    load_basin,
    split_by_water_year,
    to_tensors,
)
from workshop_utils.plotting import (
    hydrograph,
    set_style,
    COLORS,
)

__all__ = [
    "nse", "kge", "log_nse", "pbias", "rmse", "summary",
    "load_basin", "split_by_water_year", "to_tensors",
    "BASIN_SNOWY", "BASIN_ARID", "BASIN_TEMPERATE",
    "hydrograph", "set_style", "COLORS",
]

__version__ = "0.1.0"
