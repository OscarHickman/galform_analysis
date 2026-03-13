"""Compatibility wrapper for legacy module name.

This module name is historical. The implementation is a group-sampling correction,
not a mass-weighted correlation function. New code should import from
``group_sampling_correlation``.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from ...config import DEFAULT_RBINS, get_base_dir
from .group_sampling_correlation import (
    compute_group_sampling_corrected_xi,
    compute_notebook_style_correlations_for_nvolumes,
    compute_notebook_style_standard_xi,
    load_notebook_style_galaxies,
    xi_unbiased_from_group_downsample_marked,
)


def compute_weighted_xi_corrfunc(
    positions: np.ndarray,
    groupids: np.ndarray,
    boxsize: float,
    rbins: Optional[np.ndarray] = None,
    sampling_fraction: float = 1.0,
    nthreads: int = 4,
) -> pd.DataFrame:
    """Backward-compatible API: corrected xi from positions + group IDs."""
    if rbins is None:
        rbins = DEFAULT_RBINS
    xi = xi_unbiased_from_group_downsample_marked(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        groupids,
        f_group=sampling_fraction,
        rbins=rbins,
        period=boxsize,
        num_threads=nthreads,
    )
    r = 0.5 * (rbins[:-1] + rbins[1:])
    df = pd.DataFrame({'r': r, 'xi': xi})
    df.attrs.update({'rbins': rbins, 'sampling_fraction': sampling_fraction, 'ngal': positions.shape[0]})
    return df


def weighted_correlation_given_redshift_and_subvolume(
    iz_path: str,
    ivol: int,
    rbins: Optional[np.ndarray] = None,
    nthreads: int = 4,
    mhalo_min: float = 1e11,
    mstar_min_log10: float = 9.0,
    sampling_fraction: float = 1.0,
    boxsize: float = 542.16,
) -> Optional[pd.DataFrame]:
    """Backward-compatible API for a single subvolume.

    Uses notebook-style preprocessing and group-sampling correction.
    """
    if rbins is None:
        rbins = DEFAULT_RBINS
    iz_name = Path(iz_path).name
    if not iz_name.startswith('iz'):
        return None
    iz_num = int(iz_name[2:])
    base_dir = str(Path(iz_path).parent)

    gals = load_notebook_style_galaxies(
        base_dir=base_dir,
        iz_num=iz_num,
        ivols=np.array([ivol]),
        boxsize=boxsize,
        mhalo_min=mhalo_min,
    )
    xi = compute_group_sampling_corrected_xi(
        gals,
        rbins=rbins,
        sampling_fraction=sampling_fraction,
        boxsize=boxsize,
        mstar_min_log10=mstar_min_log10,
        num_threads=nthreads,
    )
    r = 0.5 * (rbins[:-1] + rbins[1:])
    df = pd.DataFrame({'r': r, 'xi': xi})
    df.attrs.update({'ivol': ivol, 'iz': iz_name, 'sampling_fraction': sampling_fraction})
    return df


def avg_weighted_correlation_given_redshift_and_subvolumes(
    iz_num: int,
    ivols: List[int],
    rbins: Optional[np.ndarray] = None,
    nthreads: int = 4,
    base_dir: Optional[str] = None,
    mhalo_min: float = 1e11,
    mstar_min_log10: float = 9.0,
    sampling_fraction: Optional[float] = None,
    boxsize: float = 542.16,
    n_total_subvolumes: int = 1024,
) -> Optional[pd.DataFrame]:
    """Backward-compatible API for combining selected subvolumes."""
    if rbins is None:
        rbins = DEFAULT_RBINS
    if base_dir is None:
        base_dir = str(get_base_dir())

    gals = load_notebook_style_galaxies(
        base_dir=base_dir,
        iz_num=iz_num,
        ivols=np.array(ivols, dtype=int),
        boxsize=boxsize,
        mhalo_min=mhalo_min,
    )
    f = sampling_fraction if sampling_fraction is not None else (len(ivols) / float(n_total_subvolumes))
    xi = compute_group_sampling_corrected_xi(
        gals,
        rbins=rbins,
        sampling_fraction=f,
        boxsize=boxsize,
        mstar_min_log10=mstar_min_log10,
        num_threads=nthreads,
    )
    r = 0.5 * (rbins[:-1] + rbins[1:])
    df = pd.DataFrame({'r': r, 'xi': xi})
    df.attrs.update({'iz': f'iz{iz_num}', 'n_ivols': len(ivols), 'sampling_fraction': f})
    return df


def weighted_correlations_given_redshifts_and_subvolume(
    iz_nums: List[int],
    ivol: int,
    rbins: Optional[np.ndarray] = None,
    nthreads: int = 4,
    base_dir: Optional[str] = None,
    mhalo_min: float = 1e11,
    mstar_min_log10: float = 9.0,
    sampling_fraction: float = 1.0,
    boxsize: float = 542.16,
) -> List[pd.DataFrame]:
    """Backward-compatible per-redshift list API."""
    if base_dir is None:
        base_dir = str(get_base_dir())
    out: List[pd.DataFrame] = []
    for iz_num in iz_nums:
        iz_path = str(Path(base_dir) / f'iz{iz_num}')
        res = weighted_correlation_given_redshift_and_subvolume(
            iz_path=iz_path,
            ivol=ivol,
            rbins=rbins,
            nthreads=nthreads,
            mhalo_min=mhalo_min,
            mstar_min_log10=mstar_min_log10,
            sampling_fraction=sampling_fraction,
            boxsize=boxsize,
        )
        if res is not None:
            out.append(res)
    return out


def avg_weighted_correlation_given_subvolume_and_redshifts(
    iz_nums: List[int],
    ivol: int,
    rbins: Optional[np.ndarray] = None,
    nthreads: int = 4,
    base_dir: Optional[str] = None,
    mhalo_min: float = 1e11,
    mstar_min_log10: float = 9.0,
    sampling_fraction: float = 1.0,
    boxsize: float = 542.16,
) -> Optional[pd.DataFrame]:
    """Backward-compatible mean-over-redshifts API."""
    results = weighted_correlations_given_redshifts_and_subvolume(
        iz_nums=iz_nums,
        ivol=ivol,
        rbins=rbins,
        nthreads=nthreads,
        base_dir=base_dir,
        mhalo_min=mhalo_min,
        mstar_min_log10=mstar_min_log10,
        sampling_fraction=sampling_fraction,
        boxsize=boxsize,
    )
    if not results:
        return None

    xi_arr = np.vstack([r['xi'].to_numpy() for r in results])
    r = results[0]['r'].to_numpy()
    df = pd.DataFrame({'r': r, 'xi': xi_arr.mean(axis=0), 'xi_std': xi_arr.std(axis=0)})
    df.attrs.update({'ivol': ivol, 'n_used': len(results), 'used_iz': [r.attrs.get('iz') for r in results]})
    return df


__all__ = [
    'load_notebook_style_galaxies',
    'compute_notebook_style_standard_xi',
    'compute_group_sampling_corrected_xi',
    'compute_notebook_style_correlations_for_nvolumes',
    'xi_unbiased_from_group_downsample_marked',
    'compute_weighted_xi_corrfunc',
    'weighted_correlation_given_redshift_and_subvolume',
    'avg_weighted_correlation_given_redshift_and_subvolumes',
    'weighted_correlations_given_redshifts_and_subvolume',
    'avg_weighted_correlation_given_subvolume_and_redshifts',
]
