"""
Galaxy bias calculations from galaxy and DM halo correlation functions.

This module computes galaxy bias b(r) = sqrt(xi_gal / xi_dm) from the 
two-point correlation functions of galaxies and dark matter halos.

Returns data as pandas DataFrames with metadata in df.attrs
"""

from typing import Optional, Sequence
import numpy as np

import polars as pl

from .correlation import correlation_given_redshift_and_subvolume, halo_correlation_given_redshift_and_subvolume


def compute_galaxy_bias(
    iz_path: str,
    ivol: int,
    rbins: Optional[np.ndarray] = None,
    nthreads: int = 4,
    centrals_only: bool = True,
    mhalo_min: Optional[float] = None,
    mhhalo_min: Optional[float] = None,
) -> Optional[pl.DataFrame]:
    """Compute galaxy bias by comparing galaxy and dark matter halo correlation functions.

    Both samples use central galaxies only (is_central == 1):
    - Galaxies: All central galaxies, uses mhalo (subhalo mass) for filtering.
    - Dark Matter halos: Same central galaxies representing (sub)halo centers,
                         uses mhhalo (host halo mass) for filtering.
    
    Both samples have the same positions (from central galaxies) and same count,
    but may use different mass thresholds. This is the standard approach for galaxy bias.
    The centrals_only parameter is kept for API compatibility but is always enforced as True.
    Bias is computed as: b(r) = sqrt(xi_gal(r) / xi_dm(r))

    Args:
        iz_path: Path to snapshot directory
        ivol: Subvolume number
        rbins: Radial bin edges (Mpc/h). Defaults to DEFAULT_RBINS
        nthreads: Number of OpenMP threads for Corrfunc
        centrals_only: Kept for API compatibility, always enforced as True
        mhalo_min: Minimum subhalo mass cut in Msun (applied to galaxies). 
                   If None, no mass cut applied to galaxies.
        mhhalo_min: Minimum host halo mass cut in Msun (applied to DM halos).
                    If None, uses mhalo_min value (backward compatibility).

    Returns:
        DataFrame with columns ['r', 'bias', 'xi_gal', 'xi_dm'] and metadata in df.attrs.
        Returns None if computation fails.
    """
    # Backward compatibility: if mhhalo_min not specified, use mhalo_min for both
    if mhhalo_min is None:
        mhhalo_min = mhalo_min

    # Compute galaxy correlation (always central galaxies only)
    gal_result = correlation_given_redshift_and_subvolume(
        iz_path=iz_path,
        ivol=ivol,
        rbins=rbins,
        nthreads=nthreads,
        centrals_only=True,  # Always True
        mhalo_min=mhalo_min,
    )

    # Compute DM halo correlation using proper halo mass filtering
    dm_result = halo_correlation_given_redshift_and_subvolume(
        iz_path=iz_path,
        ivol=ivol,
        rbins=rbins,
        nthreads=nthreads,
        mhhalo_min=mhhalo_min,
    )
    
    if gal_result is None or dm_result is None:
        return None
    
    # Extract correlation functions
    r = gal_result['r'].to_numpy()
    xi_gal = gal_result['xi'].to_numpy()
    xi_dm = dm_result['xi'].to_numpy()
    
    # Compute bias where both correlations are positive
    mask = (xi_gal > 0) & (xi_dm > 0)
    bias = np.full_like(xi_gal, np.nan)
    bias[mask] = np.sqrt(xi_gal[mask] / xi_dm[mask])
    
    # Metadata (scalar values)
    metadata = {
        'ngal': gal_result.attrs.get('ngal'),
        'nhalo': dm_result.attrs.get('nhalo'),
        'z': gal_result.attrs.get('z'),
        'ivol': ivol,
        'boxsize': gal_result.attrs.get('boxsize'),
        'rbins': gal_result.attrs.get('rbins'),
    }

    df = pl.DataFrame({'r': r, 'bias': bias, 'xi_gal': xi_gal, 'xi_dm': xi_dm})
    df.attrs = metadata
    return df


def avg_galaxy_bias_over_subvolumes(
    iz_path: str,
    ivols: Sequence[int],
    rbins: Optional[np.ndarray] = None,
    nthreads: int = 4,
    centrals_only: bool = True,
    mhalo_min: Optional[float] = None,
    mhhalo_min: Optional[float] = None,
) -> Optional[pl.DataFrame]:
    """Compute average galaxy bias over multiple subvolumes.

    Always uses central galaxies only for both galaxy and halo samples.
    The centrals_only parameter is kept for API compatibility but is always enforced as True.

    Args:
        iz_path: Path to snapshot directory
        ivols: List of subvolume numbers
        rbins: Radial bin edges (Mpc/h)
        nthreads: Number of OpenMP threads
        centrals_only: Kept for API compatibility, always enforced as True
        mhalo_min: Minimum subhalo mass cut in Msun (applied to galaxies)
        mhhalo_min: Minimum host halo mass cut in Msun (applied to DM halos).
                    If None, uses mhalo_min value for backward compatibility.
    Returns:
        DataFrame with columns ['r', 'bias', 'bias_std', 'xi_gal', 'xi_dm'] and metadata
        in df.attrs, or None if all computations fail.
    """
    results = []

    for ivol in ivols:
        res = compute_galaxy_bias(
            iz_path=iz_path,
            ivol=ivol,
            rbins=rbins,
            nthreads=nthreads,
            centrals_only=True,  # Always True
            mhalo_min=mhalo_min,
            mhhalo_min=mhhalo_min,
        )
        if res is not None:
            results.append(res)

    if not results:
        return None

    # Reference r
    r_mean = results[0]['r'].to_numpy()

    bias_stack = np.vstack([res['bias'].to_numpy() for res in results])
    xi_gal_stack = np.vstack([res['xi_gal'].to_numpy() for res in results])
    xi_dm_stack = np.vstack([res['xi_dm'].to_numpy() for res in results])

    bias_mean = np.nanmean(bias_stack, axis=0)
    bias_std = np.nanstd(bias_stack, axis=0)
    xi_gal_mean = np.nanmean(xi_gal_stack, axis=0)
    xi_dm_mean = np.nanmean(xi_dm_stack, axis=0)

    metadata = {
        'z': results[0].attrs.get('z'),
        'n_used': len(results),
        'n_requested': len(ivols),
        'rbins': results[0].attrs.get('rbins'),
    }

    df = pl.DataFrame({
        'r': r_mean,
        'bias': bias_mean,
        'bias_std': bias_std,
        'xi_gal': xi_gal_mean,
        'xi_dm': xi_dm_mean,
    })
    df.attrs = metadata
    return df
