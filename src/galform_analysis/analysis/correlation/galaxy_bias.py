"""
Galaxy bias calculations from galaxy and DM halo correlation functions.

This module computes galaxy bias b(r) = sqrt(xi_gal / xi_dm) from the 
two-point correlation functions of galaxies and dark matter halos.

Returns data as pandas DataFrames with metadata in df.attrs
"""

from typing import Dict, Optional, Sequence, Union
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from .correlation import correlation_given_redshift_and_subvolume
from .dm_correlation import dm_correlation_from_tree_file


def compute_galaxy_bias(
    iz_path: str,
    ivol: int,
    tree_file: str,
    rbins: Optional[np.ndarray] = None,
    nthreads: int = 4,
    mhalo_min: Optional[float] = None,
    dm_snapshot_idx: Optional[int] = None,
    dm_file_format: str = 'auto',
) -> Optional[pd.DataFrame]:
    """Compute galaxy bias by comparing galaxy and DM halo correlation functions.

    Galaxy 2PCF is computed from galaxies.hdf5; DM 2PCF is computed from
    merger tree halo positions in ``tree_file``. Bias is:
        b(r) = sqrt(xi_galaxies(r) / xi_halos(r))

    Args:
        iz_path: Path to snapshot directory
        ivol: Subvolume number
        tree_file: Path to merger tree file containing DM halo positions
        rbins: Radial bin edges (Mpc/h). Defaults to DEFAULT_RBINS
        nthreads: Number of OpenMP threads for Corrfunc
        mhalo_min: Minimum halo mass cut in Msun. Applied to both galaxies and halos.
        dm_snapshot_idx: Optional snapshot index for tree files with multiple snapshots
        dm_file_format: 'hdf5', 'binary', or 'auto'

    Returns:
        DataFrame with columns ['r', 'bias', 'xi_gal', 'xi_dm'] and metadata in df.attrs.
        Returns None if computation fails.
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required for compute_galaxy_bias. Install with: pip install pandas")

    # Compute galaxy correlation (all galaxies)
    gal_result = correlation_given_redshift_and_subvolume(
        iz_path=iz_path,
        ivol=ivol,
        rbins=rbins,
        nthreads=nthreads,
        centrals_only=False,
        mhalo_min=mhalo_min,
    )

    # Compute DM halo correlation from merger trees
    dm_result = dm_correlation_from_tree_file(
        tree_file=tree_file,
        rbins=rbins,
        nthreads=nthreads,
        snapshot_idx=dm_snapshot_idx,
        mhalo_min=mhalo_min,
        file_format=dm_file_format,
        boxsize_override=gal_result.attrs.get('boxsize'),
    )
    
    if gal_result is None or dm_result is None:
        return None
    
    # Extract correlation functions
    r = gal_result['r'].to_numpy()
    xi_gal = gal_result['xi'].to_numpy()

    r_dm = np.asarray(dm_result['r'])
    xi_dm_raw = np.asarray(dm_result['xi'])
    if r_dm.shape != r.shape:
        # Interpolate DM xi onto galaxy r grid
        valid = np.isfinite(r_dm) & np.isfinite(xi_dm_raw)
        if valid.sum() < 2:
            return None
        xi_dm = np.interp(r, r_dm[valid], xi_dm_raw[valid], left=np.nan, right=np.nan)
    else:
        xi_dm = xi_dm_raw
    
    # Compute bias where both correlations are positive
    mask = (xi_gal > 0) & (xi_dm > 0)
    bias = np.full_like(xi_gal, np.nan)
    bias[mask] = np.sqrt(xi_gal[mask] / xi_dm[mask])
    
    # Metadata (scalar values)
    metadata = {
        'ngal': gal_result.attrs.get('ngal'),
        'nhalo': dm_result.get('nhalo'),
        'z': gal_result.attrs.get('z'),
        'ivol': ivol,
        'boxsize': gal_result.attrs.get('boxsize'),
        'rbins': gal_result.attrs.get('rbins'),
        'dm_z': dm_result.get('z'),
        'dm_boxsize': dm_result.get('boxsize'),
        'tree_file': tree_file,
    }

    df = pd.DataFrame({'r': r, 'bias': bias, 'xi_gal': xi_gal, 'xi_dm': xi_dm})
    df.attrs.update(metadata)
    return df


def avg_galaxy_bias_over_subvolumes(
    iz_path: str,
    ivols: Sequence[int],
    tree_files: Union[Sequence[str], Dict[int, str]],
    rbins: Optional[np.ndarray] = None,
    nthreads: int = 4,
    mhalo_min: Optional[float] = None,
    dm_snapshot_idx: Optional[int] = None,
    dm_file_format: str = 'auto',
) -> Optional[pd.DataFrame]:
    """Compute average galaxy bias over multiple subvolumes.

    Args:
        iz_path: Path to snapshot directory
        ivols: List of subvolume numbers
        tree_files: Sequence of tree files aligned with ivols, or dict mapping ivol->path
        rbins: Radial bin edges (Mpc/h)
        nthreads: Number of OpenMP threads
        mhalo_min: Minimum halo mass cut in Msun
    Returns:
        DataFrame with columns ['r', 'bias', 'bias_std', 'xi_gal', 'xi_dm'] and metadata
        in df.attrs, or None if all computations fail.
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required for avg_galaxy_bias_over_subvolumes. Install with: pip install pandas")

    results = []

    ivol_list = list(ivols)
    for ivol in ivol_list:
        if isinstance(tree_files, dict):
            tree_file = tree_files.get(ivol)
        else:
            idx = ivol_list.index(ivol)
            tree_file = tree_files[idx] if idx < len(tree_files) else None
        if tree_file is None:
            continue
        res = compute_galaxy_bias(
            iz_path=iz_path,
            ivol=ivol,
            tree_file=tree_file,
            rbins=rbins,
            nthreads=nthreads,
            mhalo_min=mhalo_min,
            dm_snapshot_idx=dm_snapshot_idx,
            dm_file_format=dm_file_format,
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

    df = pd.DataFrame({
        'r': r_mean,
        'bias': bias_mean,
        'bias_std': bias_std,
        'xi_gal': xi_gal_mean,
        'xi_dm': xi_dm_mean,
    })
    df.attrs.update(metadata)
    return df
