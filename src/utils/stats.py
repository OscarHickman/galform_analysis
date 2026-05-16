"""Small statistics helpers that ignore non-positive values when computing spreads.

These helpers are conservative for correlation function estimators: when a bin
contains zero or negative ξ values (common at small separations or noisy runs),
we exclude them from std/percentile/spread calculations to avoid inflating the
reported uncertainty.
"""
from __future__ import annotations

import numpy as np


def _positive_masked(arr: np.ndarray, axis: int = 0) -> np.ma.MaskedArray:
    """Return a masked array with non-positive entries masked along axis=0.

    The input `arr` is expected to have runs along axis=0 (shape: [n_runs, n_bins]).
    """
    a = np.asarray(arr)
    mask = a <= 0
    return np.ma.masked_array(a, mask=mask)


def positive_count(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """Count strictly positive samples along `axis`.

    Returns integer array with counts of values > 0 along the requested axis.
    """
    ma = _positive_masked(arr, axis=axis)
    return np.sum(~ma.mask, axis=axis)


def positive_std(arr: np.ndarray, axis: int = 0, ddof: int = 0) -> np.ndarray:
    """Compute std over strictly positive samples; NaN where count == 0."""
    ma = _positive_masked(arr, axis=axis)
    std = ma.std(axis=axis, ddof=ddof)
    # Convert masked invalids to np.nan
    std = np.where(std.mask, np.nan, std.data) if isinstance(std, np.ma.MaskedArray) else std
    return std


def positive_percentile(arr: np.ndarray, q: float, axis: int = 0) -> np.ndarray:
    """Compute percentile `q` over strictly positive samples; NaN if none."""
    a = np.asarray(arr)
    # Select only positive values along axis by reshaping/fancy indexing
    # We'll compute percentiles per column when axis=0 (common case)
    if a.size == 0:
        return np.array([])
    if axis != 0:
        # Delegate to numpy after moving axis 0 to requested axis
        a = np.moveaxis(a, axis, 0)
    # a has runs on axis 0
    out = []
    for i in range(a.shape[1]):
        col = a[:, i]
        pos = col[col > 0]
        if pos.size == 0:
            out.append(np.nan)
        else:
            out.append(float(np.nanpercentile(pos, q)))
    return np.array(out)


def positive_se(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """Standard error: positive-only std / sqrt(n_positive). Returns NaN when n_positive==0."""
    std = positive_std(arr, axis=axis)
    npos = positive_count(arr, axis=axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        se = std / np.sqrt(npos)
    se = np.where(npos > 0, se, np.nan)
    return se
