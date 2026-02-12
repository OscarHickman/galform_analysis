"""Halo Occupation Distribution (HOD) computation utilities.

HOD is computed as \langle N_gal | M_halo \rangle using host-halo IDs and masses
stored in GALFORM galaxies.hdf5 outputs. The halo IDs (e.g. ``ihhalo``) are
indices into the merger trees, so this provides an HOD based on the merger tree
hierarchy rather than spatial tiling.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from ...config import DEFAULT_HALO_MASS_BINS, get_base_dir
from ...io.loaders import (
    open_galaxies_hdf5,
    get_output_group,
    _get_first_array,
    _get_redshift_from_file,
    _get_redshift_from_zsnap,
)


def _normalize_arrays(arrays: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], int]:
    """Ensure arrays are 1D and trimmed to a common length."""
    arrays = {k: np.ravel(v) for k, v in arrays.items() if v is not None}
    if not arrays:
        return {}, 0
    lengths = [len(v) for v in arrays.values()]
    n = min(lengths)
    arrays = {k: v[:n] for k, v in arrays.items()}
    return arrays, n


def _load_hod_galaxy_arrays(
    iz_path: str,
    ivol: int,
    galaxy_stellar_mass_min: Optional[float] = None,
    halo_id_field: Optional[str] = None,
    halo_mass_field: Optional[str] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Load galaxy arrays required for HOD calculations.

    Args:
        iz_path: Snapshot directory path (e.g. /.../iz207)
        ivol: Subvolume index
        galaxy_stellar_mass_min: Optional stellar mass cut (Msun/h) for galaxy selection
        halo_id_field: Optional override for halo ID field (default searches ihhalo, SubhaloID, index)
        halo_mass_field: Optional override for halo mass field (default searches mhhalo, mhalo, mchalo)

    Returns:
        (arrays, metadata) where arrays include:
            - 'halo_id'
            - 'halo_mass'
            - 'is_central' (if present)
            - 'mstar' (if present)
    """
    f = open_galaxies_hdf5(iz_path, ivol=ivol)
    if f is None:
        raise FileNotFoundError(f"Missing or unreadable galaxies.hdf5 at {iz_path}/ivol{ivol}")

    try:
        g = get_output_group(f)
        if g is None:
            raise RuntimeError("No OutputNNN group found in HDF5 file")

        arrays: Dict[str, np.ndarray] = {}

        # Halo ID field (merger tree index)
        id_candidates = [halo_id_field] if halo_id_field else []
        id_candidates += ["ihhalo", "SubhaloID", "SubhaloIndex", "index"]
        halo_id = None
        used_id_field = None
        for key in id_candidates:
            if key and key in g:
                halo_id = np.asarray(g[key])
                used_id_field = key
                break
        if halo_id is None:
            raise KeyError("Could not find a halo ID field (ihhalo/SubhaloID/SubhaloIndex/index)")
        arrays["halo_id"] = halo_id

        # Halo mass field
        mass_candidates = [halo_mass_field] if halo_mass_field else []
        mass_candidates += ["mhhalo", "mhalo", "mchalo", "Mhalo", "M_Halo"]
        halo_mass = _get_first_array(g, [k for k in mass_candidates if k])
        if halo_mass.size == 0:
            raise KeyError("Could not find a halo mass field (mhhalo/mhalo/mchalo)")
        arrays["halo_mass"] = halo_mass
        used_mass_field = None
        for k in mass_candidates:
            if k and k in g:
                used_mass_field = k
                break

        # Central flag (optional)
        if "is_central" in g:
            arrays["is_central"] = np.asarray(g["is_central"])

        # Stellar mass for galaxy selection (optional)
        m_disk = _get_first_array(g, ["mstars_disk"])
        m_bulge = _get_first_array(g, ["mstars_bulge"])
        if m_disk.size and m_bulge.size:
            arrays["mstar"] = m_disk + m_bulge
        else:
            arrays["mstar"] = _get_first_array(
                g, ["mstars", "StellarMass", "Mstar", "mstars_allburst"]
            )

        arrays, n = _normalize_arrays(arrays)
        if n == 0:
            raise RuntimeError("No galaxy data found for HOD calculation")

        # Apply mask
        mask = np.isfinite(arrays["halo_mass"]) & (arrays["halo_mass"] > 0)
        mask &= np.isfinite(arrays["halo_id"])

        if galaxy_stellar_mass_min is not None:
            if "mstar" not in arrays or arrays["mstar"].size == 0:
                raise KeyError("mstar field not found - cannot apply stellar mass cut")
            mask &= arrays["mstar"] >= galaxy_stellar_mass_min

        arrays = {k: v[mask] for k, v in arrays.items()}

        meta: Dict[str, Any] = {
            "iz": Path(iz_path).name,
            "ivol": ivol,
            "z": _get_redshift_from_file(f) or _get_redshift_from_zsnap(iz_path, ivol),
            "halo_id_field": used_id_field,
            "halo_mass_field": used_mass_field,
        }

        return arrays, meta
    finally:
        try:
            f.close()
        except Exception:
            pass


def _compute_per_halo_occupation(
    halo_id: np.ndarray,
    halo_mass: np.ndarray,
    is_central: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Aggregate per-halo occupation statistics.

    Returns:
        halo_mass_per_halo, n_gal_per_halo, n_cen_per_halo (or None)
    """
    # Ensure arrays are 1D
    halo_id = np.ravel(halo_id)
    halo_mass = np.ravel(halo_mass)
    if is_central is not None:
        is_central = np.ravel(is_central).astype(int, copy=False)

    unique_ids, inv = np.unique(halo_id, return_inverse=True)
    n_halos = unique_ids.size
    if n_halos == 0:
        return np.array([]), np.array([]), None

    # Count galaxies per halo
    n_gal_per_halo = np.bincount(inv, minlength=n_halos).astype(np.int64, copy=False)

    # Count centrals per halo (optional)
    n_cen_per_halo = None
    if is_central is not None:
        n_cen_per_halo = np.bincount(inv, weights=is_central, minlength=n_halos)

    # Determine halo mass per halo (use max mass for robustness)
    order = np.argsort(inv)
    inv_sorted = inv[order]
    mass_sorted = halo_mass[order]
    start_idx = np.r_[0, np.flatnonzero(np.diff(inv_sorted)) + 1]
    halo_mass_per_halo = np.maximum.reduceat(mass_sorted, start_idx)

    return halo_mass_per_halo, n_gal_per_halo, n_cen_per_halo


def _compute_hod_from_per_halo(
    halo_mass: np.ndarray,
    n_gal: np.ndarray,
    n_cen: Optional[np.ndarray],
    bins: np.ndarray,
    halo_mass_lower_limit: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Compute binned HOD statistics from per-halo arrays."""
    if halo_mass_lower_limit is not None:
        mask = halo_mass >= halo_mass_lower_limit
        halo_mass = halo_mass[mask]
        n_gal = n_gal[mask]
        if n_cen is not None:
            n_cen = n_cen[mask]

    if halo_mass.size == 0:
        return {
            "centers": 0.5 * (bins[1:] + bins[:-1]),
            "mean_occupation": np.zeros(len(bins) - 1),
            "mean_central": None,
            "mean_satellite": None,
            "counts_halos": np.zeros(len(bins) - 1, dtype=int),
            "counts_galaxies": np.zeros(len(bins) - 1, dtype=float),
        }

    logM = np.log10(halo_mass)
    counts_halos, edges = np.histogram(logM, bins=bins)
    counts_galaxies, _ = np.histogram(logM, bins=bins, weights=n_gal)

    mean_occupation = np.divide(
        counts_galaxies,
        counts_halos,
        out=np.zeros_like(counts_galaxies, dtype=float),
        where=counts_halos > 0,
    )

    mean_central = None
    mean_satellite = None
    if n_cen is not None:
        counts_cen, _ = np.histogram(logM, bins=bins, weights=n_cen)
        mean_central = np.divide(
            counts_cen,
            counts_halos,
            out=np.zeros_like(counts_cen, dtype=float),
            where=counts_halos > 0,
        )
        mean_satellite = mean_occupation - mean_central

    centers = 0.5 * (edges[1:] + edges[:-1])
    return {
        "centers": centers,
        "mean_occupation": mean_occupation,
        "mean_central": mean_central,
        "mean_satellite": mean_satellite,
        "counts_halos": counts_halos,
        "counts_galaxies": counts_galaxies,
    }


def hod_given_redshift_and_subvolume(
    iz_path: str,
    ivol: int,
    bins: np.ndarray = None,
    galaxy_stellar_mass_min: Optional[float] = None,
    halo_mass_lower_limit: Optional[float] = None,
    halo_id_field: Optional[str] = None,
    halo_mass_field: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Compute HOD for a single subvolume.

    Args:
        iz_path: Path to snapshot directory
        ivol: Subvolume index
        bins: log10(M_halo) bin edges; defaults to DEFAULT_HALO_MASS_BINS
        galaxy_stellar_mass_min: Optional stellar mass cut for galaxy selection (Msun/h)
        halo_mass_lower_limit: Optional halo mass lower limit (Msun/h)
        halo_id_field: Optional halo ID field override
        halo_mass_field: Optional halo mass field override

    Returns:
        Dictionary with keys:
            - 'iz', 'ivol', 'z'
            - 'centers'
            - 'mean_occupation'
            - 'mean_central' (optional)
            - 'mean_satellite' (optional)
            - 'counts_halos'
            - 'counts_galaxies'
            - 'n_halos'
            - 'n_galaxies'
            - 'halo_id_field', 'halo_mass_field'
        Returns None if data invalid or missing.
    """
    if bins is None:
        bins = DEFAULT_HALO_MASS_BINS

    try:
        arrays, meta = _load_hod_galaxy_arrays(
            iz_path,
            ivol,
            galaxy_stellar_mass_min=galaxy_stellar_mass_min,
            halo_id_field=halo_id_field,
            halo_mass_field=halo_mass_field,
        )
    except Exception:
        return None

    halo_mass, n_gal, n_cen = _compute_per_halo_occupation(
        arrays["halo_id"],
        arrays["halo_mass"],
        arrays.get("is_central"),
    )

    if halo_mass.size == 0:
        return None

    hod_stats = _compute_hod_from_per_halo(
        halo_mass=halo_mass,
        n_gal=n_gal,
        n_cen=n_cen,
        bins=bins,
        halo_mass_lower_limit=halo_mass_lower_limit,
    )

    return {
        "iz": meta["iz"],
        "ivol": meta["ivol"],
        "z": meta["z"],
        "centers": hod_stats["centers"],
        "mean_occupation": hod_stats["mean_occupation"],
        "mean_central": hod_stats["mean_central"],
        "mean_satellite": hod_stats["mean_satellite"],
        "counts_halos": hod_stats["counts_halos"],
        "counts_galaxies": hod_stats["counts_galaxies"],
        "n_halos": int(halo_mass.size),
        "n_galaxies": int(np.sum(n_gal)),
        "halo_id_field": meta.get("halo_id_field"),
        "halo_mass_field": meta.get("halo_mass_field"),
        "selection": {
            "galaxy_stellar_mass_min": galaxy_stellar_mass_min,
            "halo_mass_lower_limit": halo_mass_lower_limit,
        },
    }


def hods_given_redshifts_and_subvolume(
    ivol: int,
    iz_nums: List[int],
    base_dir: Optional[str] = None,
    galaxy_stellar_mass_min: Optional[float] = None,
    halo_mass_lower_limit: Optional[float] = None,
) -> Optional[Tuple[pd.DataFrame, List[Dict[str, Any]]]]:
    """Compute HODs for a single subvolume across multiple snapshots.

    Returns a DataFrame of binned HOD values per redshift.
    """
    if base_dir is None:
        base_dir = str(get_base_dir())

    results_by_z = []
    for iz_num in iz_nums:
        iz_path = os.path.join(base_dir, f"iz{iz_num}")
        if not os.path.isdir(iz_path):
            continue

        result = hod_given_redshift_and_subvolume(
            iz_path,
            ivol,
            bins=None,
            galaxy_stellar_mass_min=galaxy_stellar_mass_min,
            halo_mass_lower_limit=halo_mass_lower_limit,
        )
        if result is not None:
            results_by_z.append({
                "iz": f"iz{iz_num}",
                "iz_num": iz_num,
                "z": result["z"],
                "centers": result["centers"],
                "mean_occupation": result["mean_occupation"],
                "mean_central": result.get("mean_central"),
                "mean_satellite": result.get("mean_satellite"),
                "counts_halos": result["counts_halos"],
            })

    if not results_by_z:
        return None

    # Build DataFrame (one row per mass bin per redshift)
    df_rows = []
    for res in results_by_z:
        for i, center in enumerate(res["centers"]):
            row = {
                "iz": res["iz"],
                "iz_num": res["iz_num"],
                "z": res["z"],
                "log_M": center,
                "mean_occupation": res["mean_occupation"][i],
                "counts_halos": res["counts_halos"][i],
            }
            if res.get("mean_central") is not None:
                row["mean_central"] = res["mean_central"][i]
            if res.get("mean_satellite") is not None:
                row["mean_satellite"] = res["mean_satellite"][i]
            df_rows.append(row)

    return pd.DataFrame(df_rows), results_by_z


def avg_hod_given_redshift_and_subvolumes(
    iz_num: int,
    ivols: List[int],
    bins: np.ndarray = None,
    base_dir: Optional[str] = None,
    galaxy_stellar_mass_min: Optional[float] = None,
    halo_mass_lower_limit: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Compute HOD by combining halos from multiple subvolumes.

    This concatenates halo samples across subvolumes to improve statistics.
    """
    if bins is None:
        bins = DEFAULT_HALO_MASS_BINS
    if base_dir is None:
        base_dir = str(get_base_dir())

    iz_path = os.path.join(base_dir, f"iz{iz_num}")
    if not os.path.isdir(iz_path):
        return None

    all_halo_mass = []
    all_n_gal = []
    all_n_cen = []
    z_val = None
    n_used = 0

    for iv in ivols:
        try:
            arrays, meta = _load_hod_galaxy_arrays(
                iz_path,
                iv,
                galaxy_stellar_mass_min=galaxy_stellar_mass_min,
            )
        except Exception:
            continue

        if z_val is None:
            z_val = meta.get("z")

        halo_mass, n_gal, n_cen = _compute_per_halo_occupation(
            arrays["halo_id"],
            arrays["halo_mass"],
            arrays.get("is_central"),
        )

        if halo_mass.size == 0:
            continue

        all_halo_mass.append(halo_mass)
        all_n_gal.append(n_gal)
        if n_cen is not None:
            all_n_cen.append(n_cen)
        n_used += 1

    if n_used == 0:
        return None

    halo_mass = np.concatenate(all_halo_mass)
    n_gal = np.concatenate(all_n_gal)
    n_cen = np.concatenate(all_n_cen) if all_n_cen else None

    hod_stats = _compute_hod_from_per_halo(
        halo_mass=halo_mass,
        n_gal=n_gal,
        n_cen=n_cen,
        bins=bins,
        halo_mass_lower_limit=halo_mass_lower_limit,
    )

    return {
        "iz": f"iz{iz_num}",
        "z": z_val,
        "centers": hod_stats["centers"],
        "mean_occupation": hod_stats["mean_occupation"],
        "mean_central": hod_stats["mean_central"],
        "mean_satellite": hod_stats["mean_satellite"],
        "counts_halos": hod_stats["counts_halos"],
        "counts_galaxies": hod_stats["counts_galaxies"],
        "n_halos": int(halo_mass.size),
        "n_galaxies": int(np.sum(n_gal)),
        "n_used": n_used,
        "n_requested": len(ivols),
        "selection": {
            "galaxy_stellar_mass_min": galaxy_stellar_mass_min,
            "halo_mass_lower_limit": halo_mass_lower_limit,
        },
    }


def avg_hod_given_redshifts_and_subvolume(
    ivol: int,
    iz_nums: List[int],
    bins: np.ndarray = None,
    base_dir: Optional[str] = None,
    galaxy_stellar_mass_min: Optional[float] = None,
    halo_mass_lower_limit: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Average HOD for a single subvolume across multiple snapshots."""
    if bins is None:
        bins = DEFAULT_HALO_MASS_BINS
    if base_dir is None:
        base_dir = str(get_base_dir())

    per_mean = []
    per_cen = []
    per_sat = []
    iz_list = []
    z_list = []
    centers_ref = None

    for iz_num in iz_nums:
        iz_path = os.path.join(base_dir, f"iz{iz_num}")
        if not os.path.isdir(iz_path):
            continue

        res = hod_given_redshift_and_subvolume(
            iz_path,
            ivol,
            bins=bins,
            galaxy_stellar_mass_min=galaxy_stellar_mass_min,
            halo_mass_lower_limit=halo_mass_lower_limit,
        )
        if res is None:
            continue

        if centers_ref is None:
            centers_ref = res["centers"]

        per_mean.append(res["mean_occupation"])
        if res.get("mean_central") is not None:
            per_cen.append(res["mean_central"])
        if res.get("mean_satellite") is not None:
            per_sat.append(res["mean_satellite"])
        iz_list.append(f"iz{iz_num}")
        z_list.append(res["z"])

    if not per_mean:
        return None

    per_mean = np.array(per_mean)
    mean_central = np.array(per_cen).mean(axis=0) if per_cen else None
    mean_satellite = np.array(per_sat).mean(axis=0) if per_sat else None

    centers = centers_ref if centers_ref is not None else 0.5 * (bins[1:] + bins[:-1])

    return {
        "ivol": ivol,
        "iz_list": iz_list,
        "z_list": z_list,
        "centers": centers,
        "mean_occupation": per_mean.mean(axis=0),
        "mean_occupation_std": per_mean.std(axis=0),
        "mean_central": mean_central,
        "mean_satellite": mean_satellite,
        "n_used": per_mean.shape[0],
        "n_requested": len(iz_nums),
    }
