"""Convenience wrappers for computing linear matter xi at snapshot redshifts.

These functions read the redshift from an iz snapshot directory and delegate
to compute_matter_xi for the CAMB-based linear matter correlation function.
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl

from galform_analysis.config import SimulationConfig, get_base_dir
from galform_analysis.readers.loaders import close_snapshot, read_snapshot_data

from .matter_xi import compute_matter_xi


def matter_xi_at_snapshot(
    iz_path: str,
    sim: SimulationConfig,
    rbins: Optional[np.ndarray] = None,
    ns: float = 0.961,
) -> Optional[pl.DataFrame]:
    """Compute linear matter xi_m(r) at the redshift of an iz snapshot.

    Args:
        iz_path: Path to snapshot directory (e.g. .../iz155).
        sim: SimulationConfig supplying cosmological parameters.
        rbins: Radial bin edges in Mpc/h. Defaults to DEFAULT_RBINS.
        ns: Scalar spectral index (0.961 for L800/WMAP, 1.0 for Millennium).

    Returns:
        DataFrame with columns ['r', 'xi'], or None on failure.
    """
    try:
        data = read_snapshot_data(iz_path, ivol=0)
        z = data.get("z") or 0.0
        close_snapshot(data)
        return compute_matter_xi(sim, z=float(z), rbins=rbins, ns=ns)
    except Exception:
        return None


def matter_xi_at_snapshots(
    iz_nums: List[int],
    sim: SimulationConfig,
    rbins: Optional[np.ndarray] = None,
    ns: float = 0.961,
    base_dir: Optional[str] = None,
) -> List[Optional[pl.DataFrame]]:
    """Compute linear matter xi_m(r) at the redshifts of multiple snapshots.

    Args:
        iz_nums: List of numeric snapshot indices (e.g. [82, 120, 155, 207]).
        sim: SimulationConfig supplying cosmological parameters.
        rbins: Radial bin edges in Mpc/h. Defaults to DEFAULT_RBINS.
        ns: Scalar spectral index.
        base_dir: Base directory; defaults to get_base_dir().

    Returns:
        List of DataFrames (one per snapshot), None entries where z unavailable.
    """
    if base_dir is None:
        base_dir = str(get_base_dir())

    results = []
    for iz_num in iz_nums:
        iz_path = os.path.join(base_dir, f"iz{iz_num}")
        res = matter_xi_at_snapshot(iz_path, sim=sim, rbins=rbins, ns=ns)
        if res is not None:
            res.attrs["iz"] = f"iz{iz_num}"
        results.append(res)
    return results


def dm_correlation_given_redshift_and_subvolume(
    iz_path: str,
    ivol: int,
    rbins: Optional[np.ndarray] = None,
    nthreads: int = 4,
    mhhalo_min: Optional[float] = None,
):
    """Dark matter halo 2PCF using the galaxies.hdf5 subvolume."""
    from .correlation import halo_correlation_given_redshift_and_subvolume

    return halo_correlation_given_redshift_and_subvolume(
        iz_path=iz_path,
        ivol=ivol,
        rbins=rbins,
        nthreads=nthreads,
        mhhalo_min=mhhalo_min,
    )


def dm_correlations_given_redshifts_and_subvolume(
    iz_nums: List[int],
    ivol: int,
    rbins: Optional[np.ndarray] = None,
    nthreads: int = 4,
    mhhalo_min: Optional[float] = None,
) -> List[Optional[pl.DataFrame]]:
    """Compute DM 2PCF for a list of snapshots in one subvolume."""
    results = []
    for iz_num in iz_nums:
        iz_path = os.path.join(str(get_base_dir()), f"iz{iz_num}")
        res = dm_correlation_given_redshift_and_subvolume(
            iz_path=iz_path,
            ivol=ivol,
            rbins=rbins,
            nthreads=nthreads,
            mhhalo_min=mhhalo_min,
        )
        if res is not None:
            res.attrs["iz"] = f"iz{iz_num}"
        results.append(res)
    return results


def avg_dm_correlation_given_subvolume_and_redshifts(
    iz_nums: List[int],
    ivol: int,
    rbins: Optional[np.ndarray] = None,
    nthreads: int = 4,
    mhhalo_min: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Average DM 2PCF across multiple redshifts for one subvolume."""
    results = dm_correlations_given_redshifts_and_subvolume(
        iz_nums=iz_nums,
        ivol=ivol,
        rbins=rbins,
        nthreads=nthreads,
        mhhalo_min=mhhalo_min,
    )
    valid = [res for res in results if res is not None]
    if len(valid) == 0:
        return None

    r = valid[0]["r"]
    xi_stack = np.vstack([res["xi"] for res in valid])
    xi_mean = np.nanmean(xi_stack, axis=0)
    xi_std = np.nanstd(xi_stack, axis=0)

    return {
        "r": r,
        "xi_mean": xi_mean,
        "xi_std": xi_std,
        "rbins": valid[0].attrs.get("rbins"),
        "ngal_list": [res.attrs.get("ngal") for res in valid],
        "z_list": [res.attrs.get("z") for res in valid],
        "iz_list": [res.attrs.get("iz") for res in valid],
    }
