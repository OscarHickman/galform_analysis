"""Halo Mass Function (HMF) computation utilities."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl

from galform_analysis.config import DEFAULT_HALO_MASS_BINS, get_base_dir
from galform_analysis.readers.loaders import close_snapshot, read_snapshot_data

from ._common import _avg_phi_over_snapshots


def hmf_given_redshift_and_subvolume(
    iz_path: str,
    ivol: int,
    bins: np.ndarray = None,
    halo_mass_lower_limit: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Compute halo mass function for a single subvolume.

    Args:
        iz_path: Path to snapshot directory (e.g. '/path/to/iz155').
        ivol: Subvolume index.
        bins: log10(M_halo [M_sun/h]) bin edges. Defaults to DEFAULT_HALO_MASS_BINS.
        halo_mass_lower_limit: Optional lower mass cut (M_sun/h) before binning.

    Returns:
        dict with keys: iz, ivol, z, centers, phi [Mpc^-3 dex^-1], counts, V_ivol.
        None if data is invalid or missing.
    """
    if bins is None:
        bins = DEFAULT_HALO_MASS_BINS

    try:
        d = read_snapshot_data(iz_path, ivol=ivol)
    except Exception:
        return None

    V_ivol = d.get("V_ivol")
    mhalo = d.get("mhalo")
    z = d.get("z")
    close_snapshot(d)

    if V_ivol is None or V_ivol <= 0 or mhalo is None:
        return None

    mask = (mhalo > 0) & np.isfinite(mhalo)
    if halo_mass_lower_limit is not None:
        mask &= mhalo >= halo_mass_lower_limit

    mhalo = mhalo[mask]
    if mhalo.size == 0:
        return None

    logM = np.log10(mhalo)
    counts, edges = np.histogram(logM, bins=bins)
    dlogM = np.diff(edges)
    phi = counts / (dlogM * V_ivol)
    centers = 0.5 * (edges[1:] + edges[:-1])

    return {
        "iz": Path(iz_path).name,
        "ivol": ivol,
        "z": z,
        "centers": centers,
        "phi": phi,
        "counts": counts,
        "V_ivol": V_ivol,
    }


def hmfs_given_redshifts_and_subvolume(
    ivol: int,
    iz_nums: List[int],
    base_dir: Optional[str] = None,
    halo_mass_lower_limit: Optional[float] = None,
) -> Optional[pl.DataFrame]:
    """HMF for one subvolume across multiple snapshots, returned as a long-form
    DataFrame.

    Args:
        ivol: Subvolume index.
        iz_nums: List of snapshot numbers.
        base_dir: Base directory; defaults to configured base dir.
        halo_mass_lower_limit: Optional lower mass cut (M_sun/h) before binning.

    Returns:
        polars DataFrame with columns: iz, iz_num, z, log_M, phi, counts.
        None if no snapshot produced valid data.
    """
    if base_dir is None:
        base_dir = str(get_base_dir())

    rows = []
    for iz_num in iz_nums:
        iz_path = os.path.join(base_dir, f"iz{iz_num}")
        res = hmf_given_redshift_and_subvolume(
            iz_path, ivol, halo_mass_lower_limit=halo_mass_lower_limit
        )
        if res is None:
            continue
        for i, (center, phi_val) in enumerate(zip(res["centers"], res["phi"])):
            rows.append(
                {
                    "iz": f"iz{iz_num}",
                    "iz_num": iz_num,
                    "z": res["z"],
                    "log_M": center,
                    "phi": phi_val,
                    "counts": res["counts"][i],
                }
            )

    if not rows:
        return None
    return pl.DataFrame(rows)


def avg_hmf_given_redshift_and_subvolumes(
    iz_num: int,
    ivols: List[int],
    bins: np.ndarray = None,
    base_dir: Optional[str] = None,
    halo_mass_lower_limit: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """HMF from combined halos across multiple subvolumes for one snapshot.

    Pools all halos from the given subvolumes before binning, normalising by
    n_used * V_ivol (each subvolume is an independent realisation of the same
    full box).

    Args:
        iz_num: Snapshot number (e.g. 207 for 'iz207').
        ivols: Subvolume indices to combine.
        bins: log10(M_halo) bin edges. Defaults to DEFAULT_HALO_MASS_BINS.
        base_dir: Base directory; defaults to configured base dir.
        halo_mass_lower_limit: Optional lower mass cut (M_sun/h).

    Returns:
        dict with keys:
            iz, z, centers, phi, counts, V_total, V_ivol, n_used, n_requested.
        None if no subvolume produced valid data.
    """
    if bins is None:
        bins = DEFAULT_HALO_MASS_BINS
    if base_dir is None:
        base_dir = str(get_base_dir())

    iz_path = os.path.join(base_dir, f"iz{iz_num}")
    if not os.path.isdir(iz_path):
        return None

    all_logM = []
    V_ivol = None
    z = None
    n_used = 0

    for iv in ivols:
        try:
            d = read_snapshot_data(iz_path, ivol=iv)
        except Exception:
            continue

        V_current = d.get("V_ivol")
        mhalo = d.get("mhalo")
        if z is None:
            z = d.get("z")
        if V_ivol is None:
            V_ivol = V_current
        close_snapshot(d)

        if V_current is None or V_current <= 0 or mhalo is None:
            continue

        mask = (mhalo > 0) & np.isfinite(mhalo)
        if halo_mass_lower_limit is not None:
            mask &= mhalo >= halo_mass_lower_limit

        mhalo_filtered = mhalo[mask]
        if mhalo_filtered.size == 0:
            continue

        all_logM.append(np.log10(mhalo_filtered))
        n_used += 1

    if n_used == 0 or V_ivol is None or V_ivol <= 0:
        return None

    all_logM = np.concatenate(all_logM)
    counts, edges = np.histogram(all_logM, bins=bins)
    dlogM = np.diff(edges)
    phi = counts / (dlogM * n_used * V_ivol)
    centers = 0.5 * (edges[1:] + edges[:-1])

    return {
        "iz": f"iz{iz_num}",
        "z": z,
        "centers": centers,
        "phi": phi,
        "counts": counts,
        "V_total": V_ivol,
        "V_ivol": V_ivol,
        "n_used": n_used,
        "n_requested": len(ivols),
    }


def avg_hmf_given_redshifts_and_subvolume(
    ivol: int,
    iz_nums: List[int],
    bins: np.ndarray = None,
    base_dir: Optional[str] = None,
    halo_mass_lower_limit: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Average HMF for one subvolume across multiple snapshots.

    Args:
        ivol: Subvolume index.
        iz_nums: List of snapshot numbers (e.g. [82, 100, 120, 155]).
        bins: log10(M_halo) bin edges. Defaults to DEFAULT_HALO_MASS_BINS.
        base_dir: Base directory; defaults to configured base dir.
        halo_mass_lower_limit: Optional lower mass cut (M_sun/h).

    Returns:
        dict with keys:
            ivol, iz_list, z_list, centers, phi, phi_std, n_used, n_requested.
        None if no snapshot produced valid data.
    """
    if bins is None:
        bins = DEFAULT_HALO_MASS_BINS
    if base_dir is None:
        base_dir = str(get_base_dir())
    return _avg_phi_over_snapshots(
        hmf_given_redshift_and_subvolume,
        ivol,
        iz_nums,
        bins,
        base_dir,
        halo_mass_lower_limit=halo_mass_lower_limit,
    )


def compute_hmf_from_aggregated(
    agg_data: Optional[Dict[str, Any]],
    bins: np.ndarray = None,
    halo_mass_lower_limit: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Compute HMF from pre-aggregated halo masses.

    Args:
        agg_data: dict with keys 'mhalo' (array), 'volume' (float),
            'iz' (str), 'z' (float).
        bins: log10(M_halo) bin edges. Defaults to DEFAULT_HALO_MASS_BINS.
        halo_mass_lower_limit: Optional lower mass cut (M_sun/h).

    Returns:
        dict with keys: iz, z, centers, phi, counts. None if insufficient data.
    """
    if bins is None:
        bins = DEFAULT_HALO_MASS_BINS

    if agg_data is None or "mhalo" not in agg_data or agg_data.get("volume", 0) <= 0:
        return None

    mhalo = agg_data["mhalo"]
    mask = (mhalo > 0) & np.isfinite(mhalo)
    if halo_mass_lower_limit is not None:
        mask &= mhalo >= halo_mass_lower_limit

    mhalo = mhalo[mask]
    if len(mhalo) == 0:
        return None

    logM = np.log10(mhalo)
    counts, edges = np.histogram(logM, bins=bins)
    dlogM = np.diff(edges)
    phi = counts / (dlogM * agg_data["volume"])

    return {
        "iz": agg_data["iz"],
        "z": agg_data["z"],
        "centers": 0.5 * (edges[1:] + edges[:-1]),
        "phi": phi,
        "counts": counts,
    }
