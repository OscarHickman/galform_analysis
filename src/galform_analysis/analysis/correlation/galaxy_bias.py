"""
Galaxy bias calculations from galaxy and DM halo correlation functions.

This module computes galaxy bias b(r) = sqrt(xi_gal / xi_dm) from the 
two-point correlation functions of galaxies and dark matter halos.
"""

from typing import Dict, Optional, Any
import numpy as np

from .correlation import correlation_given_redshift_and_subvolume


def compute_galaxy_bias(
    iz_path: str,
    ivol: int,
    rbins: Optional[np.ndarray] = None,
    nthreads: int = 4,
    mhalo_min: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Compute galaxy bias by comparing galaxy and DM halo correlation functions.
    
    Uses central galaxies as proxies for DM halo positions. The bias is computed as:
        b(r) = sqrt(xi_galaxies(r) / xi_halos(r))
    
    Args:
        iz_path: Path to snapshot directory
        ivol: Subvolume number
        rbins: Radial bin edges (Mpc/h). Defaults to DEFAULT_RBINS
        nthreads: Number of OpenMP threads for Corrfunc
        mhalo_min: Minimum halo mass cut in Msun. Applied to both galaxies and halos.
    
    Returns:
        dict with keys:
            - 'r': radial bin centers (Mpc/h)
            - 'bias': galaxy bias b(r)
            - 'xi_gal': galaxy correlation function
            - 'xi_dm': DM halo correlation function
            - 'ngal': number of galaxies
            - 'nhalo': number of halos (centrals)
            - 'z': redshift
            - 'ivol': subvolume number
            - 'boxsize': box size (Mpc/h)
        Returns None if computation fails.
    """
    # Compute galaxy correlation (all galaxies)
    gal_result = correlation_given_redshift_and_subvolume(
        iz_path=iz_path,
        ivol=ivol,
        rbins=rbins,
        nthreads=nthreads,
        centrals_only=False,
        mhalo_min=mhalo_min,
    )
    
    # Compute DM halo correlation (centrals only = halo centers)
    dm_result = correlation_given_redshift_and_subvolume(
        iz_path=iz_path,
        ivol=ivol,
        rbins=rbins,
        nthreads=nthreads,
        centrals_only=True,  # Each central represents a halo
        mhalo_min=mhalo_min,
    )
    
    if gal_result is None or dm_result is None:
        print(f"Warning: Could not compute bias for {iz_path}/ivol{ivol}")
        return None
    
    # Extract correlation functions
    r = gal_result['r']
    xi_gal = gal_result['xi']
    xi_dm = dm_result['xi']
    
    # Compute bias where both correlations are positive
    mask = (xi_gal > 0) & (xi_dm > 0)
    bias = np.full_like(xi_gal, np.nan)
    bias[mask] = np.sqrt(xi_gal[mask] / xi_dm[mask])
    
    return {
        'r': r,
        'bias': bias,
        'xi_gal': xi_gal,
        'xi_dm': xi_dm,
        'ngal': gal_result['ngal'],
        'nhalo': dm_result['ngal'],  # ngal in dm_result is actually nhalo
        'z': gal_result['z'],
        'ivol': ivol,
        'boxsize': gal_result['boxsize'],
        'rbins': gal_result['rbins'],
    }


def avg_galaxy_bias_over_subvolumes(
    iz_path: str,
    ivols: list,
    rbins: Optional[np.ndarray] = None,
    nthreads: int = 4,
    mhalo_min: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Compute average galaxy bias over multiple subvolumes.
    
    Args:
        iz_path: Path to snapshot directory
        ivols: List of subvolume numbers
        rbins: Radial bin edges (Mpc/h)
        nthreads: Number of OpenMP threads
        mhalo_min: Minimum halo mass cut in Msun
    
    Returns:
        dict with mean and std of bias over subvolumes, or None if all fail
    """
    results = []
    
    for ivol in ivols:
        res = compute_galaxy_bias(
            iz_path=iz_path,
            ivol=ivol,
            rbins=rbins,
            nthreads=nthreads,
            mhalo_min=mhalo_min,
        )
        if res is not None:
            results.append(res)
    
    if not results:
        print(f"Warning: No successful bias calculations for {iz_path}")
        return None
    
    # Stack bias values and compute mean/std
    bias_stack = np.array([r['bias'] for r in results])
    r_mean = results[0]['r']
    bias_mean = np.nanmean(bias_stack, axis=0)
    bias_std = np.nanstd(bias_stack, axis=0)
    
    # Also average the correlation functions
    xi_gal_stack = np.array([r['xi_gal'] for r in results])
    xi_dm_stack = np.array([r['xi_dm'] for r in results])
    xi_gal_mean = np.nanmean(xi_gal_stack, axis=0)
    xi_dm_mean = np.nanmean(xi_dm_stack, axis=0)
    
    return {
        'r': r_mean,
        'bias': bias_mean,
        'bias_std': bias_std,
        'xi_gal': xi_gal_mean,
        'xi_dm': xi_dm_mean,
        'z': results[0]['z'],
        'n_used': len(results),
        'n_requested': len(ivols),
        'rbins': results[0]['rbins'],
    }
