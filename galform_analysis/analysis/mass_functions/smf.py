"""Stellar Mass Function (SMF) computation utilities."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl

from galform_analysis.config import DEFAULT_STELLAR_MASS_BINS, get_base_dir
from galform_analysis.readers.loaders import close_snapshot, read_snapshot_data

from ._common import _avg_phi_over_snapshots


def smf_given_redshift_and_subvolume(
    iz_path: str, ivol: int, bins: np.ndarray = None
) -> Optional[Dict[str, Any]]:
    """Compute stellar mass function for a single subvolume.

    Args:
        iz_path: Path to snapshot directory (e.g. '/path/to/iz155').
        ivol: Subvolume index.
        bins: log10(M_star [M_sun/h]) bin edges. Defaults to DEFAULT_STELLAR_MASS_BINS.

    Returns:
        dict with keys: iz, ivol, z, centers, phi [Mpc^-3 dex^-1], counts, V_ivol.
        None if data is invalid or missing.
    """
    if bins is None:
        bins = DEFAULT_STELLAR_MASS_BINS

    try:
        d = read_snapshot_data(iz_path, ivol=ivol)
    except Exception:
        return None

    V_ivol = d.get("V_ivol")
    mstar = d.get("mstar")
    z = d.get("z")
    close_snapshot(d)

    if V_ivol is None or V_ivol <= 0 or mstar is None:
        return None

    mstar = mstar[(mstar > 0) & np.isfinite(mstar)]
    if mstar.size == 0:
        return None

    logM = np.log10(mstar)
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


def smfs_given_redshifts_and_subvolume(
    ivol: int,
    iz_nums: List[int],
    base_dir: Optional[str] = None,
) -> Optional[pl.DataFrame]:
    """SMF for one subvolume across multiple snapshots, returned as a long-form DataFrame.

    Args:
        ivol: Subvolume index.
        iz_nums: List of snapshot numbers (e.g. [82, 155, 207]).
        base_dir: Base directory; defaults to configured base dir.

    Returns:
        polars DataFrame with columns: iz, iz_num, z, log_M, phi, counts.
        None if no snapshot produced valid data.
    """
    if base_dir is None:
        base_dir = str(get_base_dir())

    rows = []
    for iz_num in iz_nums:
        iz_path = os.path.join(base_dir, f"iz{iz_num}")
        res = smf_given_redshift_and_subvolume(iz_path, ivol)
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


def avg_smf_given_redshift_and_subvolumes(
    iz_num: int,
    ivols: List[int],
    bins: np.ndarray = None,
    base_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Average SMF over a list of subvolumes for one snapshot.

    Args:
        iz_num: Snapshot number (e.g. 207 for 'iz207').
        ivols: Subvolume indices to average over.
        bins: log10(M_star) bin edges. Defaults to DEFAULT_STELLAR_MASS_BINS.
        base_dir: Base directory; defaults to configured base dir.

    Returns:
        dict with keys: iz, z, centers, phi, phi_std, n_used, n_requested.
        None if no subvolume produced valid data.
    """
    if bins is None:
        bins = DEFAULT_STELLAR_MASS_BINS
    if base_dir is None:
        base_dir = str(get_base_dir())

    iz_path = os.path.join(base_dir, f"iz{iz_num}")
    if not os.path.isdir(iz_path):
        return None

    per_phi = []
    z = None
    centers_ref = None

    for iv in ivols:
        res = smf_given_redshift_and_subvolume(iz_path, iv, bins=bins)
        if res is None:
            continue
        if centers_ref is None:
            centers_ref = res["centers"]
        if z is None:
            z = res["z"]
        per_phi.append(res["phi"])

    if not per_phi:
        return None

    per_phi = np.array(per_phi)
    centers = centers_ref if centers_ref is not None else 0.5 * (bins[1:] + bins[:-1])

    return {
        "iz": f"iz{iz_num}",
        "z": z,
        "centers": centers,
        "phi": per_phi.mean(axis=0),
        "phi_std": per_phi.std(axis=0),
        "n_used": per_phi.shape[0],
        "n_requested": len(ivols),
    }


def avg_smf_given_redshifts_and_subvolume(
    ivol: int,
    iz_nums: List[int],
    bins: np.ndarray = None,
    base_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Average SMF for one subvolume across multiple snapshots.

    Args:
        ivol: Subvolume index.
        iz_nums: List of snapshot numbers (e.g. [82, 100, 120, 155]).
        bins: log10(M_star) bin edges. Defaults to DEFAULT_STELLAR_MASS_BINS.
        base_dir: Base directory; defaults to configured base dir.

    Returns:
        dict with keys: ivol, iz_list, z_list, centers, phi, phi_std, n_used, n_requested.
        None if no snapshot produced valid data.
    """
    if bins is None:
        bins = DEFAULT_STELLAR_MASS_BINS
    if base_dir is None:
        base_dir = str(get_base_dir())
    return _avg_phi_over_snapshots(
        smf_given_redshift_and_subvolume, ivol, iz_nums, bins, base_dir
    )


def compute_smf_from_aggregated(
    agg_data: Optional[Dict[str, Any]], bins: np.ndarray = None
) -> Optional[Dict[str, Any]]:
    """Compute SMF from pre-aggregated stellar masses.

    Args:
        agg_data: dict with keys 'mstar' (array), 'volume' (float), 'iz' (str), 'z' (float).
        bins: log10(M_star) bin edges. Defaults to DEFAULT_STELLAR_MASS_BINS.

    Returns:
        dict with keys: iz, z, centers, phi, counts. None if insufficient data.
    """
    if bins is None:
        bins = DEFAULT_STELLAR_MASS_BINS

    if agg_data is None or "mstar" not in agg_data or agg_data.get("volume", 0) <= 0:
        return None

    mstar = agg_data["mstar"]
    mstar = mstar[(mstar > 0) & np.isfinite(mstar)]
    if len(mstar) == 0:
        return None

    logM = np.log10(mstar)
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
