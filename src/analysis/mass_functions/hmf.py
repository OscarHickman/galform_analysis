"""Halo Mass Function (HMF) computation utilities."""

import os
import numpy as np
import polars as pl
from pathlib import Path
from typing import Optional, Dict, Any, List

from readers.loaders import read_snapshot_data, close_snapshot
from config import DEFAULT_HALO_MASS_BINS, get_base_dir


def hmf_given_redshift_and_subvolume(iz_path: str,
                          ivol: int,
                          bins: np.ndarray = None,
                          halo_mass_lower_limit: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Compute halo mass function for a single subvolume.

    Args:
        iz_path: Path to snapshot directory (e.g. '/path/to/iz155').
        ivol: Subvolume index (integer extracted from 'ivolXXX').
          bins: log10(M) bin edges in log10(M_sun/h). Defaults to DEFAULT_HALO_MASS_BINS.
              Stored mhalo values are in M_sun/h.
        halo_mass_lower_limit: Optional lower bound in M_sun/h
            to exclude halos with mass below this threshold before binning.

    Returns:
        Dictionary with keys:
            - 'iz': snapshot folder name
            - 'ivol': subvolume index
            - 'z': redshift (from file)
            - 'centers': bin centers log10(M_sun/h)
            - 'phi': number density [Mpc^-3 dex^-1]
            - 'counts': raw counts per bin
            - 'V_ivol': comoving volume (if present)
        Returns None if data invalid or missing.
    """
    if bins is None:
        bins = DEFAULT_HALO_MASS_BINS

    try:
        d = read_snapshot_data(iz_path, ivol=ivol)
    except Exception:
        return None

    V_ivol = d.get('V_ivol')
    mhalo = d.get('mhalo')
    z = d.get('z')
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
        'iz': Path(iz_path).name,
        'ivol': ivol,
        'z': z,
        'centers': centers,
        'phi': phi,
        'counts': counts,
        'V_ivol': V_ivol,
    }

def hmfs_given_redshifts_and_subvolume(ivol: int,
                             iz_nums: List[int],
                             base_dir: Optional[str] = None,
                             halo_mass_lower_limit: Optional[float] = None) -> None:
    """Compute HMFs for a single subvolume across multiple snapshots (redshifts).

    Args:
        ivol: Subvolume index.
        iz_nums: List of snapshot numbers to process.
        base_dir: Base directory containing snapshot folders (expects Path-like).
        halo_mass_lower_limit: Optional lower mass cut applied before binning.
    """

    results_by_z = []
    for iz_num in iz_nums:
        iz_path = str(base_dir / f'iz{iz_num}')
        result = hmf_given_redshift_and_subvolume(
            iz_path, ivol, bins=None, halo_mass_lower_limit=halo_mass_lower_limit)
        if result is not None:
            results_by_z.append({
                'iz': f'iz{iz_num}',
                'iz_num': iz_num,
                'z': result['z'],
                'centers': result['centers'],
                'phi': result['phi'],
                'counts': result['counts']
            })

    if not results_by_z:
        return None

    # Build DataFrame (one row per mass bin per redshift)
    df_rows = []
    for res in results_by_z:
        for i, (center, phi_val) in enumerate(zip(res['centers'], res['phi'])):
            df_rows.append({
                'iz': res['iz'],
                'iz_num': res['iz_num'],
                'z': res['z'],
                'log_M': center,
                'phi': phi_val,
                'counts': res['counts'][i]
            })
    
    return pl.DataFrame(df_rows), results_by_z


def avg_hmf_given_redshift_and_subvolumes(iz_num: int,
                                ivols: List[int],
                                bins: np.ndarray = None,
                                base_dir: Optional[str] = None,
                                halo_mass_lower_limit: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Compute HMF by combining halos from multiple subvolumes as one larger volume.

    This treats multiple subvolumes as a single larger volume, combining all halos
    before computing the HMF. This is the correct approach for subvolume convergence
    analysis, as opposed to computing separate HMFs and averaging them.

    Args:
        iz_num: Numeric snapshot identifier (e.g. 207 for 'iz207').
        ivols: List of subvolume indices to include.
        bins: Optional log10(M) bin edges (defaults to DEFAULT_HALO_MASS_BINS).
        base_dir: Optional base directory; defaults to configured base dir.
        halo_mass_lower_limit: Optional lower mass cut applied before binning.

    Returns:
        Dictionary with keys:
            - 'iz': snapshot name (e.g. 'iz207')
            - 'z': redshift (from first successful subvolume)
            - 'centers': bin centers
            - 'phi': number density [Mpc^-3 dex^-1] from combined volume
            - 'counts': total counts across all subvolumes
            - 'V_total': total comoving volume (sum of all subvolumes)
            - 'n_used': number of successful subvolumes
            - 'n_requested': length of ivols list
        Returns None if no subvolume produced valid data.
    """
    if bins is None:
        bins = DEFAULT_HALO_MASS_BINS
    if base_dir is None:
        base_dir = str(get_base_dir())

    iz_path = os.path.join(base_dir, f'iz{iz_num}')
    if not os.path.isdir(iz_path):
        return None

    # Collect all halos from all subvolumes
    # Subvolumes are independent samples of the SAME full box.
    # Combining N subvolumes gives N× more halos (better statistics),
    # but we normalize by the SINGLE box volume V, not N×V.
    # When N=1024, we have maximum statistics. When N<1024, we have fewer
    # realizations but the expectation value should be the same.
    all_logM = []
    V_ivol = None
    z = None
    n_used = 0

    for iv in ivols:
        try:
            d = read_snapshot_data(iz_path, ivol=iv)
        except Exception:
            continue

        V_current = d.get('V_ivol')
        mhalo = d.get('mhalo')
        if z is None:
            z = d.get('z')
        if V_ivol is None:
            V_ivol = V_current
        close_snapshot(d)

        if V_current is None or V_current <= 0 or mhalo is None:
            continue

        # Apply mass cut and filtering
        mask = (mhalo > 0) & np.isfinite(mhalo)
        if halo_mass_lower_limit is not None:
            mask &= mhalo >= halo_mass_lower_limit

        mhalo_filtered = mhalo[mask]
        if mhalo_filtered.size == 0:
            continue

        # Add to combined dataset
        all_logM.append(np.log10(mhalo_filtered))
        n_used += 1

    if n_used == 0 or V_ivol is None or V_ivol <= 0:
        return None

    # Combine all log masses from all realizations
    all_logM = np.concatenate(all_logM)

    # Compute HMF on the combined dataset
    # Each subvolume samples 1/n_total of the galaxy population (where n_total=1024)
    # When combining n_used subvolumes, we have n_used/n_total of the full statistics.
    # Normalize by n_used * V_ivol to get the correct HMF estimate (should be same regardless of n_used).
    counts, edges = np.histogram(all_logM, bins=bins)
    dlogM = np.diff(edges)
    phi = counts / (dlogM * n_used * V_ivol)
    centers = 0.5 * (edges[1:] + edges[:-1])

    return {
        'iz': f'iz{iz_num}',
        'z': z,
        'centers': centers,
        'phi': phi,
        'counts': counts,
        'V_total': V_ivol,  # Single box volume (all subvolumes sample the same volume)
        'V_ivol': V_ivol,
        'n_used': n_used,
        'n_requested': len(ivols),
    }


def avg_hmf_given_redshifts_and_subvolume(ivol: int,
                                 iz_nums: List[int],
                                 bins: np.ndarray = None,
                                 base_dir: Optional[str] = None,
                                 halo_mass_lower_limit: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Average HMF for a single subvolume across multiple snapshots (redshifts).

    Calls ``hmf_given_redshift_and_subvolume`` for the same subvolume at different redshifts
    and averages the resulting ``phi`` arrays.

    Args:
        ivol: Subvolume index to use across all snapshots.
        iz_nums: List of numeric snapshot identifiers (e.g. [82, 100, 120, 155]).
        bins: Optional log10(M) bin edges (defaults to DEFAULT_HALO_MASS_BINS).
        base_dir: Optional base directory; defaults to configured base dir.
        halo_mass_lower_limit: Optional lower mass cut applied before binning.

    Returns:
        Dictionary with keys:
            - 'ivol': the subvolume index used
            - 'iz_list': list of snapshot names that contributed data
            - 'z_list': list of redshifts (parallel to iz_list)
            - 'centers': bin centers
            - 'phi': mean number density across provided snapshots
            - 'phi_std': standard deviation across provided snapshots
            - 'n_used': number of successful snapshots
            - 'n_requested': length of iz_nums list
        Returns None if no snapshot produced valid data.
    """
    if bins is None:
        bins = DEFAULT_HALO_MASS_BINS
    if base_dir is None:
        base_dir = str(get_base_dir())

    per_phi = []
    iz_list = []
    z_list = []
    centers_ref = None

    for iz_num in iz_nums:
        iz_path = os.path.join(base_dir, f'iz{iz_num}')
        if not os.path.isdir(iz_path):
            continue

        res = hmf_given_redshift_and_subvolume(
            iz_path, ivol, bins=bins, halo_mass_lower_limit=halo_mass_lower_limit)
        if res is None:
            continue

        if centers_ref is None:
            centers_ref = res['centers']

        per_phi.append(res['phi'])
        iz_list.append(f'iz{iz_num}')
        z_list.append(res['z'])

    if not per_phi:
        return None

    per_phi = np.array(per_phi)
    centers = centers_ref if centers_ref is not None else 0.5 * (bins[1:] + bins[:-1])

    return {
        'ivol': ivol,
        'iz_list': iz_list,
        'z_list': z_list,
        'centers': centers,
        'phi': per_phi.mean(axis=0),
        'phi_std': per_phi.std(axis=0),
        'n_used': per_phi.shape[0],
        'n_requested': len(iz_nums),
    }


def compute_hmf_from_aggregated(agg_data: Optional[Dict[str, Any]], 
                               bins: np.ndarray = None,
                               halo_mass_lower_limit: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Compute halo mass function from pre-aggregated data.
    
    This is useful when you've already collected all halo masses
    and just need to bin them.
    
    Args:
        agg_data: Dictionary with keys 'mhalo' (array), 'volume' (float),
                 'iz' (str), 'z' (float)
        bins: Mass bins in log10(M_sun), defaults to DEFAULT_HALO_MASS_BINS
        halo_mass_lower_limit: Optional lower mass cut applied before binning.
        
    Returns:
        Dictionary with keys: 'iz', 'z', 'centers', 'phi', 'counts'
        Returns None if insufficient data
    """
    if bins is None:
        bins = DEFAULT_HALO_MASS_BINS
        
    if agg_data is None or 'mhalo' not in agg_data or agg_data.get('volume', 0) <= 0:
        return None
    
    mhalo = agg_data['mhalo']
    mask = (mhalo > 0) & np.isfinite(mhalo)
    if halo_mass_lower_limit is not None:
        mask &= mhalo >= halo_mass_lower_limit
    
    mhalo = mhalo[mask]
    if len(mhalo) == 0:
        return None
        
    logM = np.log10(mhalo)
    counts, edges = np.histogram(logM, bins=bins)
    dlogM = np.diff(edges)
    phi = counts / (dlogM * agg_data['volume'])
    
    return {
        'iz': agg_data['iz'],
        'z': agg_data['z'],
        'centers': 0.5 * (edges[1:] + edges[:-1]),
        'phi': phi,
        'counts': counts
    }
