"""Halo Mass Function (HMF) computation utilities."""

import os
import glob
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List

from ..io.loaders import read_snapshot_data, close_snapshot
from ..config import DEFAULT_HALO_MASS_BINS


def compute_hmf_avg_by_snapshot(iz_path: str,
                                bins: np.ndarray = None,
                                ivol_sample: Optional[int] = None,
                                ivol_list: Optional[List[int]] = None,
                                random_seed: int = 42) -> Optional[Dict[str, Any]]:
    """Compute halo mass function by averaging across subvolumes.
    
    This function iterates through all subvolumes in a snapshot, computes
    the halo mass function for each, and returns the average with
    standard deviation.
    
    Args:
        iz_path: Path to snapshot directory (e.g., '/path/to/iz155')
        bins: Mass bins in log10(M_sun), defaults to DEFAULT_HALO_MASS_BINS
        ivol_sample: If set, randomly sample this many subvolumes instead of using all
        ivol_list: Explicit list of ivol indices to use (overrides ivol_sample if provided)
        random_seed: Random seed for sampling (default: 42)
        
    Returns:
        Dictionary with keys:
            - 'iz': Snapshot name (e.g., 'iz155')
            - 'z': Redshift
            - 'centers': Bin centers in log10(M_sun)
            - 'phi': Mean number density [Mpc^-3 dex^-1]
            - 'phi_std': Standard deviation of phi
            - 'n_used': Number of subvolumes successfully processed
            - 'n_total': Total number of subvolumes attempted
        Returns None if no valid data found
    """
    if bins is None:
        bins = DEFAULT_HALO_MASS_BINS
        
    ivol_paths_all = sorted(glob.glob(os.path.join(iz_path, 'ivol*')))
    ivol_paths = ivol_paths_all
    if not ivol_paths:
        return None

    # Explicit selection overrides sampling
    if ivol_list:
        # Filter paths by requested indices (silently ignore missing)
        index_map = {int(Path(p).name.replace('ivol', '')): p for p in ivol_paths_all}
        ivol_paths = [index_map[i] for i in ivol_list if i in index_map]
    elif ivol_sample is not None and ivol_sample > 0:
        if ivol_sample < len(ivol_paths):
            rng = np.random.default_rng(random_seed)
            sel_idx = rng.choice(len(ivol_paths), size=ivol_sample, replace=False)
            ivol_paths = [ivol_paths[i] for i in sel_idx]

    per_phi = []
    z = None
    n_total = len(ivol_paths)
    
    for ivp in ivol_paths:
        iv = int(Path(ivp).name.replace('ivol', ''))
        try:
            d = read_snapshot_data(iz_path, ivol=iv)
        except Exception:
            continue
            
        V_ivol = d.get('V_ivol')
        mhalo = d.get('mhalo')
        if z is None:
            z = d.get('z')
        close_snapshot(d)
        
        if V_ivol is None or V_ivol <= 0 or mhalo is None:
            continue
            
        mhalo = mhalo[(mhalo > 0) & np.isfinite(mhalo)]
        if mhalo.size == 0:
            continue
            
        logM = np.log10(mhalo)
        counts, edges = np.histogram(logM, bins=bins)
        dlogM = np.diff(edges)
        phi = counts / (dlogM * V_ivol)
        per_phi.append(phi)

    if not per_phi:
        return None

    per_phi = np.array(per_phi)
    centers = 0.5 * (bins[1:] + bins[:-1])
    
    return {
        'iz': Path(iz_path).name,
        'z': z,
        'centers': centers,
        'phi': per_phi.mean(axis=0),
        'phi_std': per_phi.std(axis=0),
        'n_used': per_phi.shape[0],
        'n_total': n_total,
    }


def compute_hmf_from_aggregated(agg_data: Optional[Dict[str, Any]], 
                               bins: np.ndarray = None) -> Optional[Dict[str, Any]]:
    """Compute halo mass function from pre-aggregated data.
    
    This is useful when you've already collected all halo masses
    and just need to bin them.
    
    Args:
        agg_data: Dictionary with keys 'mhalo' (array), 'volume' (float),
                 'iz' (str), 'z' (float)
        bins: Mass bins in log10(M_sun), defaults to DEFAULT_HALO_MASS_BINS
        
    Returns:
        Dictionary with keys: 'iz', 'z', 'centers', 'phi', 'counts'
        Returns None if insufficient data
    """
    if bins is None:
        bins = DEFAULT_HALO_MASS_BINS
        
    if agg_data is None or 'mhalo' not in agg_data or agg_data.get('volume', 0) <= 0:
        return None
    
    mhalo = agg_data['mhalo']
    mhalo = mhalo[(mhalo > 0) & np.isfinite(mhalo)]
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
