"""Halo Occupation Distribution (HOD) computation utilities.

HOD is computed as ⟨N_gal | M_halo⟩ using a two-histogram approach:
  - **Numerator**: galaxy counts per host-halo mass bin (using ``mhhalo``).
  - **Denominator**: halo counts per mass bin from the merger-tree catalog
    (``Trees/mphalo``), which provides one mass per FOF group at the current
    snapshot, including empty halos with no qualifying galaxies.

This avoids the need for a per-galaxy tree index (which GALFORM does not
store directly) and correctly counts halos that contain zero galaxies above
the stellar-mass threshold.

Central/satellite decomposition:
  - A galaxy is flagged as **FOF central** if it is the central
    (``is_central==1``) of the main (most massive) subhalo within its FOF
    group, identified by ``mhalo / mhhalo > FOF_CENTRAL_RATIO_THRETSHOLD``.
  - Satellites = total - centrals.
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

# A galaxy with mhalo/mhhalo above this threshold is considered the
# central of the main subhalo (i.e. the FOF-group central).
FOF_CENTRAL_RATIO_THRESHOLD = 0.5


def _load_hod_data(
    iz_path: str,
    ivol: int,
    galaxy_stellar_mass_min: Optional[float] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, Any]]:
    """Load galaxy arrays and the tree halo-mass catalog for HOD.

    Returns:
        (galaxy_arrays, tree_halo_masses, metadata)

        *galaxy_arrays* contains per-galaxy 1-D arrays:
            mhhalo, mhalo, is_central, mstar, galaxy_selection_mask

        *tree_halo_masses* is a 1-D array with one entry per merger tree
        (= one per FOF group) giving the tree's halo mass at this snapshot.

        *metadata* has 'iz', 'ivol', 'z'.
    """
    f = open_galaxies_hdf5(iz_path, ivol=ivol)
    if f is None:
        raise FileNotFoundError(
            f"Missing or unreadable galaxies.hdf5 at {iz_path}/ivol{ivol}"
        )

    try:
        g = get_output_group(f)
        if g is None:
            raise RuntimeError("No OutputNNN group found in HDF5 file")

        # ---- Tree halo masses (one per FOF group) ----
        if "Trees" in f and "mphalo" in f["Trees"]:
            tree_mphalo = np.asarray(f["Trees"]["mphalo"], dtype=np.float64)
        else:
            raise KeyError("Trees/mphalo not found — cannot build halo catalog")

        # ---- Per-galaxy arrays ----
        # Host-halo mass (mass of the FOF group this galaxy lives in)
        mhhalo = _get_first_array(g, ["mhhalo", "mhalo", "mchalo"])
        if mhhalo.size == 0:
            raise KeyError("No host-halo mass field found (mhhalo/mhalo/mchalo)")

        # Own subhalo mass (used for FOF-central identification)
        mhalo = _get_first_array(g, ["mhalo", "mchalo"])

        # Central flag
        is_central = (
            np.asarray(g["is_central"]) if "is_central" in g else None
        )

        # Stellar mass
        m_disk = _get_first_array(g, ["mstars_disk"])
        m_bulge = _get_first_array(g, ["mstars_bulge"])
        if m_disk.size and m_bulge.size:
            mstar = m_disk + m_bulge
        else:
            mstar = _get_first_array(
                g, ["mstars", "StellarMass", "Mstar", "mstars_allburst"]
            )

        # ---- Align lengths & flatten ----
        all_arrs: Dict[str, np.ndarray] = {
            "mhhalo": mhhalo,
            "mhalo": mhalo,
        }
        if is_central is not None:
            all_arrs["is_central"] = is_central
        if mstar.size:
            all_arrs["mstar"] = mstar

        all_arrs = {
            k: np.ravel(v) for k, v in all_arrs.items() if v is not None
        }
        n = min(len(v) for v in all_arrs.values()) if all_arrs else 0
        if n == 0:
            raise RuntimeError("No galaxy data found for HOD calculation")
        all_arrs = {k: v[:n] for k, v in all_arrs.items()}

        # Physical validity
        valid = np.isfinite(all_arrs["mhhalo"]) & (all_arrs["mhhalo"] > 0)
        all_arrs = {k: v[valid] for k, v in all_arrs.items()}

        # Galaxy selection mask (stellar-mass cut)
        galaxy_selection_mask = None
        if galaxy_stellar_mass_min is not None:
            if "mstar" not in all_arrs or all_arrs["mstar"].size == 0:
                raise KeyError(
                    "mstar not found — cannot apply stellar mass cut"
                )
            galaxy_selection_mask = all_arrs["mstar"] >= galaxy_stellar_mass_min
        all_arrs["galaxy_selection_mask"] = galaxy_selection_mask

        meta: Dict[str, Any] = {
            "iz": Path(iz_path).name,
            "ivol": ivol,
            "z": (
                _get_redshift_from_file(f)
                or _get_redshift_from_zsnap(iz_path, ivol)
            ),
        }

        return all_arrs, tree_mphalo, meta

    finally:
        try:
            f.close()
        except Exception:
            pass


def _compute_hod_two_histogram(
    galaxy_mhhalo: np.ndarray,
    tree_mphalo: np.ndarray,
    bins: np.ndarray,
    is_central: Optional[np.ndarray] = None,
    galaxy_mhalo: Optional[np.ndarray] = None,
    galaxy_selection_mask: Optional[np.ndarray] = None,
    halo_mass_lower_limit: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute HOD using the two-histogram method.

    Parameters
    ----------
    galaxy_mhhalo : array
        Host-halo mass for each galaxy (Msun/h).
    tree_mphalo : array
        Halo mass for each merger tree / FOF group (Msun/h).
    bins : array
        log10(M) bin edges.
    is_central : array, optional
        1 for central, 0 for satellite (per galaxy).
    galaxy_mhalo : array, optional
        Own subhalo mass per galaxy; used with *is_central* to identify
        FOF-group centrals (mhalo/mhhalo > FOF_CENTRAL_RATIO_THRESHOLD).
    galaxy_selection_mask : bool array, optional
        True for galaxies passing the stellar-mass cut.
    halo_mass_lower_limit : float, optional
        Exclude halos below this mass (Msun/h) from both histograms.
    """
    # Apply halo mass lower limit
    if halo_mass_lower_limit is not None:
        tree_mphalo = tree_mphalo[tree_mphalo >= halo_mass_lower_limit]
        gal_keep = galaxy_mhhalo >= halo_mass_lower_limit
        galaxy_mhhalo = galaxy_mhhalo[gal_keep]
        if is_central is not None:
            is_central = is_central[gal_keep]
        if galaxy_mhalo is not None:
            galaxy_mhalo = galaxy_mhalo[gal_keep]
        if galaxy_selection_mask is not None:
            galaxy_selection_mask = galaxy_selection_mask[gal_keep]

    centers = 0.5 * (bins[1:] + bins[:-1])
    n_bins = len(bins) - 1

    if tree_mphalo.size == 0:
        empty = np.zeros(n_bins)
        return {
            "centers": centers,
            "mean_occupation": empty.copy(),
            "mean_central": None,
            "mean_satellite": None,
            "counts_halos": np.zeros(n_bins, dtype=int),
            "counts_galaxies": np.zeros(n_bins, dtype=int),
        }

    # Denominator: halo counts from tree catalog
    log_tree = np.log10(tree_mphalo)
    counts_halos, _ = np.histogram(log_tree, bins=bins)

    # Select qualifying galaxies
    if galaxy_selection_mask is not None:
        sel = galaxy_selection_mask
        gal_mhh = galaxy_mhhalo[sel]
        gal_is_cen = is_central[sel] if is_central is not None else None
        gal_mh = galaxy_mhalo[sel] if galaxy_mhalo is not None else None
    else:
        gal_mhh = galaxy_mhhalo
        gal_is_cen = is_central
        gal_mh = galaxy_mhalo

    log_gal_mhh = np.log10(gal_mhh)

    # Numerator: galaxy counts (all qualifying galaxies, binned by mhhalo)
    counts_galaxies, _ = np.histogram(log_gal_mhh, bins=bins)

    mean_occupation = np.divide(
        counts_galaxies.astype(float),
        counts_halos,
        out=np.zeros(n_bins),
        where=counts_halos > 0,
    )

    # Central / satellite decomposition
    mean_central = None
    mean_satellite = None
    if gal_is_cen is not None and gal_mh is not None:
        # FOF central = is_central==1 AND mhalo/mhhalo > threshold
        fof_cen_mask = (gal_is_cen == 1) & (
            gal_mh / (gal_mhh + 1e-30) > FOF_CENTRAL_RATIO_THRESHOLD
        )
        counts_cen, _ = np.histogram(
            log_gal_mhh[fof_cen_mask], bins=bins
        )
        mean_central = np.divide(
            counts_cen.astype(float),
            counts_halos,
            out=np.zeros(n_bins),
            where=counts_halos > 0,
        )
        mean_satellite = mean_occupation - mean_central

    return {
        "centers": centers,
        "mean_occupation": mean_occupation,
        "mean_central": mean_central,
        "mean_satellite": mean_satellite,
        "counts_halos": counts_halos,
        "counts_galaxies": counts_galaxies,
    }


# ──────────────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────────────


def hod_given_redshift_and_subvolume(
    iz_path: str,
    ivol: int,
    bins: np.ndarray = None,
    galaxy_stellar_mass_min: Optional[float] = None,
    halo_mass_lower_limit: Optional[float] = None,
    halo_id_field: Optional[str] = None,
    halo_mass_field: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Compute HOD for a single subvolume using the two-histogram method.

    The denominator uses the tree halo catalog (``Trees/mphalo``), which
    counts **all** FOF groups including those with zero qualifying galaxies.
    The numerator counts qualifying galaxies binned by their host-halo mass
    (``mhhalo``).

    Args:
        iz_path: Path to snapshot directory.
        ivol: Subvolume index.
        bins: log10(M_halo) bin edges; defaults to DEFAULT_HALO_MASS_BINS.
        galaxy_stellar_mass_min: Optional stellar mass cut (Msun/h).
        halo_mass_lower_limit: Optional halo mass lower limit (Msun/h).
        halo_id_field: Unused (kept for API compatibility).
        halo_mass_field: Unused (kept for API compatibility).

    Returns:
        Dictionary with HOD results, or None on failure.
    """
    if bins is None:
        bins = DEFAULT_HALO_MASS_BINS

    try:
        arrays, tree_mphalo, meta = _load_hod_data(
            iz_path,
            ivol,
            galaxy_stellar_mass_min=galaxy_stellar_mass_min,
        )
    except Exception:
        return None

    hod_stats = _compute_hod_two_histogram(
        galaxy_mhhalo=arrays["mhhalo"],
        tree_mphalo=tree_mphalo,
        bins=bins,
        is_central=arrays.get("is_central"),
        galaxy_mhalo=arrays.get("mhalo"),
        galaxy_selection_mask=arrays.get("galaxy_selection_mask"),
        halo_mass_lower_limit=halo_mass_lower_limit,
    )

    # Count qualifying galaxies for metadata
    sel = arrays.get("galaxy_selection_mask")
    n_galaxies = (
        int(sel.sum()) if sel is not None else int(arrays["mhhalo"].size)
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
        "n_halos": int(tree_mphalo.size),
        "n_galaxies": n_galaxies,
        "halo_id_field": None,
        "halo_mass_field": "mhhalo",
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

    Returns (DataFrame, list-of-result-dicts) or None.
    """
    if base_dir is None:
        base_dir = str(get_base_dir())

    results_by_z: List[Dict[str, Any]] = []
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
    """Compute HOD by combining data from multiple subvolumes.

    Galaxy and tree-halo arrays from each subvolume are concatenated before
    computing the two-histogram HOD, giving a single combined result with
    improved statistics.
    """
    if bins is None:
        bins = DEFAULT_HALO_MASS_BINS
    if base_dir is None:
        base_dir = str(get_base_dir())

    iz_path = os.path.join(base_dir, f"iz{iz_num}")
    if not os.path.isdir(iz_path):
        return None

    all_mhhalo: List[np.ndarray] = []
    all_mhalo: List[Optional[np.ndarray]] = []
    all_is_central: List[Optional[np.ndarray]] = []
    all_sel_mask: List[Optional[np.ndarray]] = []
    all_tree_mphalo: List[np.ndarray] = []
    z_val = None
    n_used = 0

    for iv in ivols:
        try:
            arrays, tree_mphalo, meta = _load_hod_data(
                iz_path,
                iv,
                galaxy_stellar_mass_min=galaxy_stellar_mass_min,
            )
        except Exception:
            continue

        if z_val is None:
            z_val = meta.get("z")

        all_mhhalo.append(arrays["mhhalo"])
        all_mhalo.append(arrays.get("mhalo"))
        all_is_central.append(arrays.get("is_central"))
        all_sel_mask.append(arrays.get("galaxy_selection_mask"))
        all_tree_mphalo.append(tree_mphalo)
        n_used += 1

    if n_used == 0:
        return None

    mhhalo = np.concatenate(all_mhhalo)
    mhalo = (
        np.concatenate(all_mhalo)
        if all(a is not None for a in all_mhalo)
        else None
    )
    is_central = (
        np.concatenate(all_is_central)
        if all(a is not None for a in all_is_central)
        else None
    )
    sel_mask = (
        np.concatenate(all_sel_mask)
        if all(a is not None for a in all_sel_mask)
        else None
    )
    tree_mphalo = np.concatenate(all_tree_mphalo)

    hod_stats = _compute_hod_two_histogram(
        galaxy_mhhalo=mhhalo,
        tree_mphalo=tree_mphalo,
        bins=bins,
        is_central=is_central,
        galaxy_mhalo=mhalo,
        galaxy_selection_mask=sel_mask,
        halo_mass_lower_limit=halo_mass_lower_limit,
    )

    n_galaxies = (
        int(sel_mask.sum()) if sel_mask is not None else int(mhhalo.size)
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
        "n_halos": int(tree_mphalo.size),
        "n_galaxies": n_galaxies,
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
    """Average HOD for a single subvolume across multiple snapshots.

    Each snapshot is computed independently; results are averaged
    bin-by-bin with standard deviation.
    """
    if bins is None:
        bins = DEFAULT_HALO_MASS_BINS
    if base_dir is None:
        base_dir = str(get_base_dir())

    per_mean: List[np.ndarray] = []
    per_cen: List[np.ndarray] = []
    per_sat: List[np.ndarray] = []
    iz_list: List[str] = []
    z_list: List[float] = []
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

    per_mean_arr = np.array(per_mean)
    mean_central = np.array(per_cen).mean(axis=0) if per_cen else None
    mean_satellite = np.array(per_sat).mean(axis=0) if per_sat else None

    centers = (
        centers_ref
        if centers_ref is not None
        else 0.5 * (bins[1:] + bins[:-1])
    )

    return {
        "ivol": ivol,
        "iz_list": iz_list,
        "z_list": z_list,
        "centers": centers,
        "mean_occupation": per_mean_arr.mean(axis=0),
        "mean_occupation_std": per_mean_arr.std(axis=0),
        "mean_central": mean_central,
        "mean_satellite": mean_satellite,
        "n_used": per_mean_arr.shape[0],
        "n_requested": len(iz_nums),
    }
