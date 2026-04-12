"""Halo-ID-based correction utilities for halo-sampled GALFORM catalogues.

This module mirrors the public API of ``group_sampling_correlation`` but uses
the catalogue halo/tree identifier directly to separate same-halo and
different-halo galaxy pairs.

When a true halo identifier is available in the GALFORM output (for example
``DHaloID`` in a z=0 run), the correction is exact under the halo-sampling
model. For non-z=0 runs, the code falls back to ``TreeID`` when no valid
``DHaloID`` values are present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from astropy.cosmology import LambdaCDM
from halotools.empirical_models import halo_mass_to_halo_radius
from halotools.mock_observables import marked_tpcf, npairs_3d, tpcf

_NOTEBOOK_COSMO = LambdaCDM(67.777, 0.307, 0.693)


def _select_halo_identifier(galaxies_group: h5py.Group) -> Tuple[np.ndarray, str]:
    """Return the preferred halo-like identifier available in one output group."""
    def _has_any_valid_identifier(values: np.ndarray) -> bool:
        if values.size == 0:
            return False
        if values.dtype.kind in 'iu':
            return np.any(values >= 0)
        return np.any(np.isfinite(values))

    def _is_informative_identifier(values: np.ndarray) -> bool:
        """Require at least two distinct valid IDs to avoid degenerate halo grouping."""
        if values.size == 0:
            return False

        if values.dtype.kind in 'iu':
            valid = values[values >= 0]
        else:
            valid = values[np.isfinite(values)]

        if valid.size == 0:
            return False
        return np.unique(valid).size > 1

    # Prefer explicit halo index fields when present in GALFORM outputs,
    # but skip degenerate fields that collapse all galaxies into one ID.
    for key in ('ihhalo', 'ihalof', 'DHaloID', 'TreeID', 'SubhaloID'):
        if key not in galaxies_group:
            continue
        values = np.asarray(galaxies_group[key])
        if _is_informative_identifier(values):
            return values, key

    # Fallback for degenerate-but-valid IDs.
    for key in ('ihhalo', 'ihalof', 'TreeID', 'SubhaloID', 'DHaloID'):
        if key in galaxies_group:
            values = np.asarray(galaxies_group[key])
            if _has_any_valid_identifier(values):
                return values, key

    # Last fallback when everything exists but no key carries valid values.
    for key in ('ihhalo', 'ihalof', 'TreeID', 'SubhaloID', 'DHaloID'):
        if key in galaxies_group:
            return np.asarray(galaxies_group[key]), key

    raise KeyError('Expected one of DHaloID, TreeID or SubhaloID in GALFORM output')


def _select_output_group_for_halo_correction(f: h5py.File) -> h5py.Group:
    """Pick the highest-numbered OutputNNN group.

    This matches the default loader behavior used elsewhere in the codebase.
    """
    outs = sorted([k for k in f.keys() if k.startswith('Output') and k[6:].isdigit()])
    if not outs:
        raise KeyError('No OutputNNN group found in GALFORM output')
    return f[outs[-1]]


def _read_single_ivol_galaxies(gal_file: Path, mhalo_min: float = 1e11) -> pd.DataFrame:
    """Read one GALFORM subvolume using a true halo/tree identifier when available."""
    with h5py.File(gal_file, 'r') as f:
        g = _select_output_group_for_halo_correction(f)
        logh = np.log10(0.7)

        mhhalo = np.asarray(g['mhhalo'])
        halo_id, halo_id_source = _select_halo_identifier(g)
        mask = mhhalo > mhalo_min
        if halo_id.dtype.kind in 'iu':
            mask &= halo_id >= 0

        gal = pd.DataFrame(
            {
                'halo_id': halo_id[mask].astype(np.int64, copy=False),
                'is_central': np.asarray(g['is_central'])[mask],
                'xgal': np.asarray(g['xgal'])[mask],
                'ygal': np.asarray(g['ygal'])[mask],
                'zgal': np.asarray(g['zgal'])[mask],
                'mstar': (
                    np.log10(np.asarray(g['mstars_bulge']) + np.asarray(g['mstars_disk']) + 1e-3)
                    - logh
                )[mask],
                'mhalo': (np.log10(mhhalo + 1e-3) - logh)[mask],
                'halo_id_source': halo_id_source,
            }
        )
    return gal


def load_halo_sampled_galaxies(
    base_dir: str,
    iz_num: int,
    ivols: np.ndarray,
    boxsize: float = 542.16,
    mhalo_min: float = 1e11,
    centrals_only: bool = False,
) -> pd.DataFrame:
    """Load galaxies using halo/tree IDs instead of a synthetic group proxy."""
    gal_chunks: List[pd.DataFrame] = []
    iz_path = Path(base_dir) / f'iz{iz_num}'

    for ivol in ivols:
        gal_file = iz_path / f'ivol{int(ivol)}' / 'galaxies.hdf5'
        if not gal_file.is_file():
            continue
        gal_chunks.append(_read_single_ivol_galaxies(gal_file, mhalo_min=mhalo_min))

    if not gal_chunks:
        return pd.DataFrame(
            columns=[
                'halo_id',
                'halo_index',
                'halo_id_source',
                'is_central',
                'xgal',
                'ygal',
                'zgal',
                'mstar',
                'mhalo',
                'dr',
                'rhalo',
                'dr_norm',
            ]
        )

    gals = pd.concat(gal_chunks, ignore_index=True)

    unique_ids, halo_index = np.unique(gals['halo_id'].values, return_inverse=True)
    gals['halo_index'] = halo_index.astype(np.int64)

    cen = gals[gals['is_central'] == 1]
    pos_cen = np.zeros((len(unique_ids), 3), dtype=np.float64)
    if not cen.empty:
        pos_cen[cen['halo_index'].values, 0] = cen['xgal'].values
        pos_cen[cen['halo_index'].values, 1] = cen['ygal'].values
        pos_cen[cen['halo_index'].values, 2] = cen['zgal'].values

    dx = gals['xgal'].values - pos_cen[gals['halo_index'].values, 0]
    dy = gals['ygal'].values - pos_cen[gals['halo_index'].values, 1]
    dz = gals['zgal'].values - pos_cen[gals['halo_index'].values, 2]

    dx = np.where(dx > boxsize / 2.0, dx - boxsize, dx)
    dx = np.where(dx < -boxsize / 2.0, dx + boxsize, dx)
    dy = np.where(dy > boxsize / 2.0, dy - boxsize, dy)
    dy = np.where(dy < -boxsize / 2.0, dy + boxsize, dy)
    dz = np.where(dz > boxsize / 2.0, dz - boxsize, dz)
    dz = np.where(dz < -boxsize / 2.0, dz + boxsize, dz)

    dr = np.sqrt(dx * dx + dy * dy + dz * dz)
    rhalo = halo_mass_to_halo_radius(10 ** gals['mhalo'].values, _NOTEBOOK_COSMO, 0, '200c')
    dr_norm = dr / rhalo

    gals = gals.assign(dr=dr, rhalo=rhalo, dr_norm=dr_norm)
    gals = gals[gals['dr_norm'] < 1]
    if centrals_only:
        gals = gals[gals['is_central'] == 1]
    return gals


def compute_notebook_style_standard_xi(
    gals: pd.DataFrame,
    rbins: np.ndarray,
    boxsize: float = 542.16,
    mstar_min_log10: Optional[float] = None,
    num_threads: int = 4,
) -> np.ndarray:
    """Standard 2PCF in notebook style (halotools Landy-Szalay)."""
    if mstar_min_log10 is None:
        mask = np.ones(len(gals), dtype=bool)
    else:
        mask = gals['mstar'].values > mstar_min_log10
    sample = gals.loc[mask, ['xgal', 'ygal', 'zgal']].values
    if sample.shape[0] < 2:
        return np.full(len(rbins) - 1, np.nan)
    return tpcf(sample, rbins, period=boxsize, estimator='Landy-Szalay', num_threads=num_threads)


def xi_unbiased_from_halo_downsample(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    halo_id: np.ndarray,
    sampling_fraction: float,
    rbins: np.ndarray,
    period: float,
    num_threads: int = 1,
) -> np.ndarray:
    """Recover the full-box xi from a halo-sampled catalogue using exact same-halo counts."""
    if not 0.0 < sampling_fraction <= 1.0:
        raise ValueError('sampling_fraction must lie in the interval (0, 1]')

    sample = np.vstack((x, y, z)).T
    if sample.shape[0] < 2:
        return np.full(len(rbins) - 1, np.nan)

    dd_all_obs = np.diff(npairs_3d(sample, sample, rbins, period=period))

    _, halo_compact = np.unique(halo_id, return_inverse=True)
    marks = np.vstack((halo_compact.astype(float), np.ones_like(halo_compact, dtype=float))).T
    match_fraction = marked_tpcf(
        sample,
        rbins,
        marks1=marks,
        period=period,
        normalize_by='number_counts',
        weight_func_id=3,
        num_threads=num_threads,
    )

    dd_1h_obs = match_fraction * dd_all_obs
    dd_2h_obs = dd_all_obs - dd_1h_obs
    inv_f = 1.0 / sampling_fraction
    dd_full_hat = inv_f * dd_1h_obs + (inv_f * inv_f) * dd_2h_obs

    n_obs = sample.shape[0]
    n_full = n_obs * inv_f
    shell_vol = np.diff((4.0 * np.pi / 3.0) * rbins ** 3)
    rr = shell_vol / (period ** 3)

    dd = dd_full_hat / (n_full ** 2)
    return dd / rr - 1.0


def compute_halo_sampling_corrected_xi(
    gals: pd.DataFrame,
    rbins: np.ndarray,
    sampling_fraction: float,
    boxsize: float = 542.16,
    mstar_min_log10: Optional[float] = None,
    num_threads: int = 16,
) -> np.ndarray:
    """Halo-ID-corrected 2PCF using exact same-halo and different-halo separation."""
    if mstar_min_log10 is None:
        mask = np.ones(len(gals), dtype=bool)
    else:
        mask = gals['mstar'].values > mstar_min_log10
    if np.count_nonzero(mask) < 2:
        return np.full(len(rbins) - 1, np.nan)

    xyz = gals.loc[mask, ['xgal', 'ygal', 'zgal']].values.T
    halo_id = gals.loc[mask, 'halo_id'].values
    return xi_unbiased_from_halo_downsample(
        *xyz,
        halo_id,
        sampling_fraction=sampling_fraction,
        rbins=rbins,
        period=boxsize,
        num_threads=num_threads,
    )


def compute_halo_sampling_correlations_for_nvolumes(
    base_dir: str,
    iz_num: int,
    nvolumes_list: List[int],
    rbins: np.ndarray,
    boxsize: float = 542.16,
    mstar_min_log10: Optional[float] = None,
    mhalo_min: float = 1e11,
    centrals_only: bool = False,
    n_total_subvolumes: int = 1024,
    num_threads: int = 16,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Compute standard and halo-ID-corrected xi for each requested N_subvol."""
    results: Dict[int, Dict[str, np.ndarray]] = {}
    for nvol in nvolumes_list:
        gals = load_halo_sampled_galaxies(
            base_dir=base_dir,
            iz_num=iz_num,
            ivols=np.arange(nvol),
            boxsize=boxsize,
            mhalo_min=mhalo_min,
            centrals_only=centrals_only,
        )

        xi_standard = compute_notebook_style_standard_xi(
            gals,
            rbins=rbins,
            boxsize=boxsize,
            mstar_min_log10=mstar_min_log10,
            num_threads=num_threads,
        )
        xi_corrected = compute_halo_sampling_corrected_xi(
            gals,
            rbins=rbins,
            sampling_fraction=nvol / float(n_total_subvolumes),
            boxsize=boxsize,
            mstar_min_log10=mstar_min_log10,
            num_threads=num_threads,
        )

        if mstar_min_log10 is None:
            ngal = len(gals)
        else:
            ngal = int(np.count_nonzero(gals['mstar'].values > mstar_min_log10))

        results[nvol] = {
            'xi_standard': xi_standard,
            'xi_corrected': xi_corrected,
            'ngal': int(ngal),
            'nhalo': int(gals['halo_id'].nunique()),
        }

    return results