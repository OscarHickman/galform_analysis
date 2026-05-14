"""Shared helpers for the τ₀ paper notebooks.

The dynamical-friction-timescale runs share a common shape:
  - Three lc16.newmg variants: Default, 0·τ₀, ∞·τ₀ (parameter ``tau0``)
  - Two snapshots available: iz207 (z=0.5) and iz271 (z=0)
  - 16 subvolumes (out of 1024 total in the L800 box)

Each notebook needs to do the same boilerplate: walk runs × ivols, compute a
per-ivol summary, then stack to get a mean ± SEM ± bootstrap range. This
module provides:

  - RUNS: ordered dict label → path
  - per-ivol loaders for the fields the analyses need
  - per-ivol summary helpers (SHMR, SMF cen/sat split, HOD)
  - stack_per_ivol(): mean / SEM / bootstrap percentiles across ivols

The L800 box has side length BOX_SIZE_MPC_H = 542.16 Mpc/h.
Per-subvolume volume is ``Parameters/volume`` from the HDF5 file.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from galform_analysis.io.loaders import (
    open_galaxies_hdf5,
    get_output_group,
    _get_first_array,
    _get_redshift_from_file,
    _get_redshift_from_zsnap,
)

# ──────────────────────────────────────────────────────────────────────
# Run / snapshot configuration
# ──────────────────────────────────────────────────────────────────────

TAU0_INVESTIGATION_ROOT = Path(
    "/cosma5/data/durham/dc-hick2/Tau0_Investigation"
)

# Ordered so plots use a stable colour cycle: Default first, then extremes.
RUNS: "OrderedDict[str, Path]" = OrderedDict(
    [
        (
            "Default",
            TAU0_INVESTIGATION_ROOT / "Galform_Out_Default" / "L800" / "lc16.newmg",
        ),
        (
            "tau0=0",
            TAU0_INVESTIGATION_ROOT / "Galform_Out_0tau0" / "L800" / "lc16.newmg",
        ),
        (
            "tau0=inf",
            TAU0_INVESTIGATION_ROOT / "Galform_Out_1e6tau0" / "L800" / "lc16.newmg",
        ),
    ]
)

# Display labels for legends/titles (LaTeX-safe).
RUN_LABELS: Dict[str, str] = {
    "Default": r"$\tau_0$ default",
    "tau0=0": r"$\tau_0 = 0$ (instant merge)",
    "tau0=inf": r"$\tau_0 \to \infty$ (no merge)",
}

# Consistent colour scheme across all paper figures.
RUN_COLORS: Dict[str, str] = {
    "Default": "k",
    "tau0=0": "C3",  # red — instant merging
    "tau0=inf": "C0",  # blue — never merging
}

RUN_MARKERS: Dict[str, str] = {
    "Default": "o",
    "tau0=0": "s",
    "tau0=inf": "^",
}

# Snapshots: only iz207 and iz271 are run for the τ₀ campaign.
SNAPSHOTS: "OrderedDict[str, Tuple[int, float]]" = OrderedDict(
    [
        ("iz271", (271, 0.0)),
        ("iz207", (207, 0.5)),
    ]
)

DEFAULT_IVOLS: Tuple[int, ...] = tuple(range(16))

# L800 box geometry. Side length in comoving Mpc/h.
BOX_SIZE_MPC_H = 542.16

# FOF-central definition: a central whose own subhalo dominates the FOF group.
FOF_CENTRAL_RATIO_THRESHOLD = 0.5


# ──────────────────────────────────────────────────────────────────────
# Per-ivol field loader
# ──────────────────────────────────────────────────────────────────────


def load_galaxy_fields(
    iz_path: Path,
    ivol: int,
) -> Optional[Dict[str, Any]]:
    """Load the fields all τ₀ analyses need from one (run, snapshot, ivol).

    Returns dict with:
        mstar, mhalo (own subhalo), mhhalo (host FOF), is_central (0/1),
        z, V_ivol, n_subvol_total
    or None if the file is missing/unreadable.

    All masses in Msun/h.
    """
    f = open_galaxies_hdf5(str(iz_path), ivol=ivol)
    if f is None:
        return None

    try:
        g = get_output_group(f)
        if g is None:
            return None

        m_disk = _get_first_array(g, ["mstars_disk"])
        m_bulge = _get_first_array(g, ["mstars_bulge"])
        if m_disk.size and m_bulge.size:
            mstar = m_disk + m_bulge
        else:
            mstar = _get_first_array(
                g, ["mstars", "StellarMass", "Mstar", "mstars_allburst"]
            )

        mhalo = _get_first_array(g, ["mhalo", "mchalo"])
        mhhalo = _get_first_array(g, ["mhhalo", "mhalo_host"])
        is_central = (
            np.asarray(g["is_central"]).astype(np.int8)
            if "is_central" in g
            else None
        )

        arrays = {
            "mstar": np.ravel(mstar),
            "mhalo": np.ravel(mhalo),
            "mhhalo": np.ravel(mhhalo),
            "is_central": is_central,
        }
        sizes = [v.size for v in arrays.values() if v is not None and v.size]
        if not sizes:
            return None
        n = min(sizes)
        for k, v in list(arrays.items()):
            if v is None:
                continue
            arrays[k] = v[:n]

        z = _get_redshift_from_file(f) or _get_redshift_from_zsnap(
            str(iz_path), ivol
        )

        V_ivol = None
        n_subvol = None
        if "Parameters" in f:
            params = f["Parameters"]
            if "volume" in params:
                V_ivol = float(np.array(params["volume"]))
            if "n_subvolumes" in params:
                n_subvol = int(np.array(params["n_subvolumes"]))

        # Tree halo masses (one per FOF group at this snapshot, including
        # halos that host no qualifying galaxies — needed for HOD denominators).
        tree_mphalo = None
        if "Trees" in f and "mphalo" in f["Trees"]:
            tree_mphalo = np.asarray(f["Trees"]["mphalo"], dtype=np.float64)

        return {
            **arrays,
            "z": z,
            "V_ivol": V_ivol,
            "n_subvol_total": n_subvol,
            "tree_mphalo": tree_mphalo,
            "ivol": ivol,
            "iz": iz_path.name,
        }
    finally:
        try:
            f.close()
        except Exception:
            pass


def fof_central_mask(data: Dict[str, Any]) -> np.ndarray:
    """Boolean mask: galaxy is the central of the dominant subhalo of its FOF group."""
    is_cen = data.get("is_central")
    mhalo = data["mhalo"]
    mhhalo = data["mhhalo"]
    if is_cen is None:
        return np.zeros_like(mhalo, dtype=bool)
    return (is_cen == 1) & (
        mhalo / np.maximum(mhhalo, 1e-30) > FOF_CENTRAL_RATIO_THRESHOLD
    )


# ──────────────────────────────────────────────────────────────────────
# Per-ivol summary helpers
# ──────────────────────────────────────────────────────────────────────


def central_shmr_per_ivol(
    data: Dict[str, Any],
    halo_bins: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Median + IQR central log10(M*) in bins of log10(Mhhalo).

    Uses FOF centrals only. Returns NaN-filled arrays for empty bins.
    """
    mask = fof_central_mask(data) & (data["mstar"] > 0) & (data["mhhalo"] > 0)
    log_mh = np.log10(data["mhhalo"][mask])
    log_ms = np.log10(data["mstar"][mask])

    n_bins = len(halo_bins) - 1
    median = np.full(n_bins, np.nan)
    p16 = np.full(n_bins, np.nan)
    p84 = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=np.int64)

    if log_mh.size:
        idx = np.digitize(log_mh, halo_bins) - 1
        for i in range(n_bins):
            sel = idx == i
            if not np.any(sel):
                continue
            ms_i = log_ms[sel]
            counts[i] = ms_i.size
            median[i] = np.median(ms_i)
            p16[i] = np.percentile(ms_i, 16)
            p84[i] = np.percentile(ms_i, 84)

    return {
        "centers": 0.5 * (halo_bins[1:] + halo_bins[:-1]),
        "median": median,
        "p16": p16,
        "p84": p84,
        "counts": counts,
    }


def smf_split_per_ivol(
    data: Dict[str, Any],
    mstar_bins: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Stellar mass function split into total / central / satellite.

    Returns φ in [Mpc/h]^{-3} dex^{-1} for each component. Uses the GALFORM
    is_central flag (1 → central, 0 → satellite) — note this is *all*
    centrals (any subhalo), not just FOF centrals.
    """
    V_ivol = data.get("V_ivol")
    if V_ivol is None or V_ivol <= 0:
        return {
            "centers": 0.5 * (mstar_bins[1:] + mstar_bins[:-1]),
            "phi_total": np.full(len(mstar_bins) - 1, np.nan),
            "phi_cen": np.full(len(mstar_bins) - 1, np.nan),
            "phi_sat": np.full(len(mstar_bins) - 1, np.nan),
            "counts_total": np.zeros(len(mstar_bins) - 1, dtype=np.int64),
            "counts_cen": np.zeros(len(mstar_bins) - 1, dtype=np.int64),
            "counts_sat": np.zeros(len(mstar_bins) - 1, dtype=np.int64),
        }

    mstar = data["mstar"]
    is_cen = data.get("is_central")

    valid = (mstar > 0) & np.isfinite(mstar)
    log_ms = np.log10(mstar[valid])
    is_cen_v = is_cen[valid] if is_cen is not None else None

    dlog = np.diff(mstar_bins)
    counts_total, _ = np.histogram(log_ms, bins=mstar_bins)
    if is_cen_v is not None:
        counts_cen, _ = np.histogram(log_ms[is_cen_v == 1], bins=mstar_bins)
        counts_sat, _ = np.histogram(log_ms[is_cen_v == 0], bins=mstar_bins)
    else:
        counts_cen = np.zeros_like(counts_total)
        counts_sat = np.zeros_like(counts_total)

    return {
        "centers": 0.5 * (mstar_bins[1:] + mstar_bins[:-1]),
        "phi_total": counts_total / (dlog * V_ivol),
        "phi_cen": counts_cen / (dlog * V_ivol),
        "phi_sat": counts_sat / (dlog * V_ivol),
        "counts_total": counts_total,
        "counts_cen": counts_cen,
        "counts_sat": counts_sat,
    }


def hod_per_ivol(
    data: Dict[str, Any],
    halo_bins: np.ndarray,
    mstar_min: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """⟨N_total⟩, ⟨N_cen⟩, ⟨N_sat⟩ vs log10(Mhalo).

    Two-histogram method: numerator counts qualifying galaxies binned by
    host-halo mass; denominator counts FOF groups from Trees/mphalo (so
    halos with zero qualifying galaxies are included).

    Centrals here are *FOF centrals* (is_central==1 ∧ mhalo/mhhalo > 0.5).
    """
    n_bins = len(halo_bins) - 1
    centers = 0.5 * (halo_bins[1:] + halo_bins[:-1])
    tree_mphalo = data.get("tree_mphalo")
    if tree_mphalo is None or tree_mphalo.size == 0:
        empty = np.zeros(n_bins)
        return {
            "centers": centers,
            "n_total": empty.copy(),
            "n_cen": empty.copy(),
            "n_sat": empty.copy(),
            "halo_counts": np.zeros(n_bins, dtype=np.int64),
        }

    counts_halos, _ = np.histogram(np.log10(tree_mphalo), bins=halo_bins)

    mstar = data["mstar"]
    mhhalo = data["mhhalo"]
    sel = (mhhalo > 0) & np.isfinite(mhhalo) & (mstar > 0) & np.isfinite(mstar)
    if mstar_min is not None:
        sel &= mstar >= mstar_min

    log_mhhalo = np.log10(mhhalo[sel])
    counts_total, _ = np.histogram(log_mhhalo, bins=halo_bins)

    is_cen = data.get("is_central")
    mhalo = data["mhalo"]
    if is_cen is not None:
        fof_cen = (
            (is_cen[sel] == 1)
            & (mhalo[sel] / np.maximum(mhhalo[sel], 1e-30) > FOF_CENTRAL_RATIO_THRESHOLD)
        )
        counts_cen, _ = np.histogram(log_mhhalo[fof_cen], bins=halo_bins)
    else:
        counts_cen = np.zeros_like(counts_total)

    counts_sat = counts_total - counts_cen

    n_total = np.divide(
        counts_total.astype(float),
        counts_halos,
        out=np.zeros(n_bins),
        where=counts_halos > 0,
    )
    n_cen = np.divide(
        counts_cen.astype(float),
        counts_halos,
        out=np.zeros(n_bins),
        where=counts_halos > 0,
    )
    n_sat = np.divide(
        counts_sat.astype(float),
        counts_halos,
        out=np.zeros(n_bins),
        where=counts_halos > 0,
    )

    return {
        "centers": centers,
        "n_total": n_total,
        "n_cen": n_cen,
        "n_sat": n_sat,
        "halo_counts": counts_halos,
    }


# ──────────────────────────────────────────────────────────────────────
# Stacking across ivols
# ──────────────────────────────────────────────────────────────────────


def collect_per_ivol(
    run_path: Path,
    snapshot: str,
    ivols: Tuple[int, ...],
    summarise: Callable[[Dict[str, Any]], Dict[str, np.ndarray]],
) -> List[Dict[str, np.ndarray]]:
    """Walk ivols for one (run, snapshot), apply ``summarise`` per ivol, drop None."""
    out: List[Dict[str, np.ndarray]] = []
    iz_path = run_path / snapshot
    for iv in ivols:
        data = load_galaxy_fields(iz_path, iv)
        if data is None:
            continue
        try:
            summary = summarise(data)
        except Exception:
            continue
        if summary is None:
            continue
        out.append(summary)
    return out


def _stack_field(
    summaries: List[Dict[str, np.ndarray]],
    key: str,
) -> Optional[np.ndarray]:
    if not summaries:
        return None
    return np.vstack([s[key] for s in summaries])


def stack_per_ivol(
    summaries: List[Dict[str, np.ndarray]],
    keys: Tuple[str, ...],
    nboot: int = 500,
    seed: int = 17,
) -> Dict[str, Any]:
    """Aggregate a list of per-ivol summary dicts.

    For each key in ``keys`` returns:
        mean, sem, boot_lo (16th), boot_hi (84th), boot_med (median across boots)

    The bootstrap resamples ivols with replacement, recomputes the per-bin
    mean across the bootstrap sample, and takes the percentile across the
    bootstrap distribution. SEM is std / sqrt(n).
    """
    if not summaries:
        return {"n_used": 0}

    rng = np.random.default_rng(seed)
    n = len(summaries)
    out: Dict[str, Any] = {
        "n_used": n,
        "centers": summaries[0].get("centers"),
    }

    for key in keys:
        stack = _stack_field(summaries, key)
        if stack is None:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(invalid="ignore"):
                mean = np.nanmean(stack, axis=0)
                std = np.nanstd(stack, axis=0)
            sem = std / np.sqrt(np.maximum(n, 1))

            # Bootstrap over ivols
            idx = np.arange(n)
            boots = np.empty((nboot, stack.shape[1]), dtype=float)
            for b in range(nboot):
                pick = rng.choice(idx, size=n, replace=True)
                with np.errstate(invalid="ignore"):
                    boots[b] = np.nanmean(stack[pick], axis=0)
            boot_lo = np.nanpercentile(boots, 16, axis=0)
            boot_hi = np.nanpercentile(boots, 84, axis=0)
            boot_med = np.nanmedian(boots, axis=0)

        out[key] = {
            "stack": stack,
            "mean": mean,
            "sem": sem,
            "boot_lo": boot_lo,
            "boot_hi": boot_hi,
            "boot_med": boot_med,
        }
    return out


def run_for_all_runs(
    snapshot: str,
    ivols: Tuple[int, ...],
    summarise: Callable[[Dict[str, Any]], Dict[str, np.ndarray]],
    keys: Tuple[str, ...],
    nboot: int = 500,
    seed: int = 17,
) -> Dict[str, Dict[str, Any]]:
    """Convenience: compute per-ivol summaries for every run at one snapshot, then stack.

    Returns dict run_label → stacked dict (output of stack_per_ivol).
    """
    out: Dict[str, Dict[str, Any]] = {}
    for label, run_path in RUNS.items():
        summaries = collect_per_ivol(run_path, snapshot, ivols, summarise)
        out[label] = stack_per_ivol(
            summaries, keys=keys, nboot=nboot, seed=seed
        )
    return out


# ──────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────


def style_for(label: str) -> Dict[str, Any]:
    """Return color/marker kwargs for a run label."""
    return {
        "color": RUN_COLORS.get(label, "C7"),
        "marker": RUN_MARKERS.get(label, "o"),
        "label": RUN_LABELS.get(label, label),
    }


def safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full_like(num, np.nan, dtype=float)
    m = np.isfinite(num) & np.isfinite(den) & (den != 0)
    out[m] = num[m] / den[m]
    return out
