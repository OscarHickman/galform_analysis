"""
Notebook-style correlation utilities for subvolume downsampling studies.

This module mirrors the logic in ``kai/kai_corrfunc.ipynb``:
- build galaxy samples across selected subvolumes
- compute standard 2PCF with halotools ``tpcf``
- compute group-sampling-corrected 2PCF using marked pair fractions

Note: this is a group-sampling correction, not mass weighting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import pandas as pd
from astropy.cosmology import LambdaCDM
from halotools.empirical_models import halo_mass_to_halo_radius
from halotools.mock_observables import marked_tpcf, npairs_3d, tpcf

from ...io.loaders import get_output_group


_NOTEBOOK_COSMO = LambdaCDM(67.777, 0.307, 0.693)


def _is_informative_identifier(values: np.ndarray) -> bool:
    """Return True when an ID array can separate at least two groups."""
    if values.size == 0:
        return False

    if values.dtype.kind in 'iu':
        valid = values[values >= 0]
    else:
        valid = values[np.isfinite(values)]

    if valid.size == 0:
        return False
    return np.unique(valid).size > 1


def _select_group_identifier(galaxies_group: h5py.Group, mhhalo: np.ndarray) -> np.ndarray:
    """Choose a robust group identifier for same-halo pair bookkeeping.

    Preference order:
    1) notebook-style synthetic key ``mhhalo*vhhalo`` (when available and informative),
    2) explicit halo/tree identifiers present in GALFORM outputs.
    """
    if 'vhhalo' in galaxies_group:
        synthetic_id = (mhhalo * np.asarray(galaxies_group['vhhalo'])).astype(np.int64)
        if _is_informative_identifier(synthetic_id):
            return synthetic_id

    for key in ('ihhalo', 'ihalof', 'DHaloID', 'TreeID', 'SubhaloID'):
        if key not in galaxies_group:
            continue
        values = np.asarray(galaxies_group[key])
        if _is_informative_identifier(values):
            return values

    # Last-chance fallback to preserve behavior even for degenerate IDs.
    if 'vhhalo' in galaxies_group:
        return (mhhalo * np.asarray(galaxies_group['vhhalo'])).astype(np.int64)
    for key in ('ihhalo', 'ihalof', 'DHaloID', 'TreeID', 'SubhaloID'):
        if key in galaxies_group:
            return np.asarray(galaxies_group[key])

    raise KeyError('Expected one of vhhalo, ihhalo, ihalof, DHaloID, TreeID or SubhaloID in GALFORM output')


def _read_single_ivol_galaxies(gal_file: Path, mhalo_min: float = 1e11) -> pd.DataFrame:
    """Read one GALFORM subvolume in the same style as the notebook helper."""
    with h5py.File(gal_file, 'r') as f:
        g = get_output_group(f)
        if g is None:
            raise KeyError('No OutputNNN group found in GALFORM output')

        logh = np.log10(0.7)

        if 'mhhalo' in g:
            mhhalo = np.asarray(g['mhhalo'])
        elif 'mhalo' in g:
            mhhalo = np.asarray(g['mhalo'])
        else:
            raise KeyError('Expected mhhalo or mhalo in GALFORM output')

        if 'mstars_bulge' in g and 'mstars_disk' in g:
            mstar_raw = np.asarray(g['mstars_bulge']) + np.asarray(g['mstars_disk'])
        elif 'mstars' in g:
            mstar_raw = np.asarray(g['mstars'])
        else:
            raise KeyError('Expected mstars_bulge+mstars_disk or mstars in GALFORM output')

        igrp_raw = _select_group_identifier(g, mhhalo)
        mask = mhhalo > mhalo_min
        if igrp_raw.dtype.kind in 'iu':
            mask &= igrp_raw >= 0
        else:
            mask &= np.isfinite(igrp_raw)

        if np.count_nonzero(mask) == 0:
            return pd.DataFrame(columns=['igrp', 'is_central', 'xgal', 'ygal', 'zgal', 'mstar', 'mhalo'])

        igrp = np.asarray(igrp_raw)[mask]
        if igrp.dtype.kind not in 'iu':
            igrp = np.rint(igrp).astype(np.int64)
        else:
            igrp = igrp.astype(np.int64, copy=False)

        gal = pd.DataFrame(
            {
                'igrp': igrp,
                'is_central': np.asarray(g['is_central'])[mask],
                'xgal': np.asarray(g['xgal'])[mask],
                'ygal': np.asarray(g['ygal'])[mask],
                'zgal': np.asarray(g['zgal'])[mask],
                'mstar': (
                    np.log10(np.clip(mstar_raw, 0.0, None) + 1e-3)
                    - logh
                )[mask],
                'mhalo': (np.log10(mhhalo + 1e-3) - logh)[mask],
            }
        )
    return gal


def load_notebook_style_galaxies(
    base_dir: str,
    iz_num: int,
    ivols: np.ndarray,
    boxsize: float = 542.16,
    mhalo_min: float = 1e11,
    centrals_only: bool = False,
) -> pd.DataFrame:
    """Load and post-process galaxies exactly as in ``kai_corrfunc.ipynb``."""
    gal_chunks: List[pd.DataFrame] = []
    iz_path = Path(base_dir) / f'iz{iz_num}'

    for ivol in ivols:
        gal_file = iz_path / f'ivol{int(ivol)}' / 'galaxies.hdf5'
        if not gal_file.is_file():
            continue
        gal_chunks.append(_read_single_ivol_galaxies(gal_file, mhalo_min=mhalo_min))

    if not gal_chunks:
        return pd.DataFrame(
            columns=['igrp', 'is_central', 'xgal', 'ygal', 'zgal', 'mstar', 'mhalo', 'dr', 'rhalo', 'dr_norm']
        )

    gals = pd.concat(gal_chunks, ignore_index=True)

    # Compact group IDs to contiguous [0, ngrp)
    gals['igrp'] = pd.factorize(gals['igrp'].values, sort=False)[0].astype(np.int64)
    ngrp = int(gals['igrp'].max()) + 1

    # Reproduce notebook's radial-within-halo filtering.
    cen = gals[gals['is_central'] == 1]
    pos_cen = np.zeros((ngrp, 3), dtype=np.float64)
    if not cen.empty:
        pos_cen[cen['igrp'].values, 0] = cen['xgal'].values
        pos_cen[cen['igrp'].values, 1] = cen['ygal'].values
        pos_cen[cen['igrp'].values, 2] = cen['zgal'].values

    # Some filtered groups can lose their flagged central; fall back to max-mstar member.
    missing = np.ones(ngrp, dtype=bool)
    if not cen.empty:
        missing[np.unique(cen['igrp'].values)] = False
    if np.any(missing):
        idx_max = gals.groupby('igrp')['mstar'].idxmax()
        fallback = gals.loc[idx_max, ['igrp', 'xgal', 'ygal', 'zgal']]
        pos_cen[fallback['igrp'].values, 0] = fallback['xgal'].values
        pos_cen[fallback['igrp'].values, 1] = fallback['ygal'].values
        pos_cen[fallback['igrp'].values, 2] = fallback['zgal'].values

    dx = gals['xgal'].values - pos_cen[gals['igrp'].values, 0]
    dy = gals['ygal'].values - pos_cen[gals['igrp'].values, 1]
    dz = gals['zgal'].values - pos_cen[gals['igrp'].values, 2]

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
    valid = np.isfinite(gals['dr_norm']) & np.isfinite(gals['rhalo']) & (gals['rhalo'] > 0) & (gals['dr_norm'] >= 0)
    gals = gals[valid & (gals['dr_norm'] < 1)]
    if centrals_only:
        gals = gals[gals['is_central'] == 1]
    return gals


def compute_notebook_style_standard_xi(
    gals: pd.DataFrame,
    rbins: np.ndarray,
    boxsize: float = 542.16,
    mstar_min_log10: float = 9.0,
    num_threads: int = 4,
) -> np.ndarray:
    """Standard 2PCF in notebook style (halotools Landy-Szalay)."""
    mask = gals['mstar'].values > mstar_min_log10
    sample = gals.loc[mask, ['xgal', 'ygal', 'zgal']].values
    if sample.shape[0] < 2:
        return np.full(len(rbins) - 1, np.nan)
    return tpcf(sample, rbins, period=boxsize, estimator='Landy-Szalay', num_threads=num_threads)


def xi_unbiased_from_group_downsample_marked(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    gid: np.ndarray,
    f_group: float,
    rbins: np.ndarray,
    period: float,
    num_threads: int = 1,
) -> np.ndarray:
    """Notebook-equivalent group-sampling correction estimator."""
    sample = np.vstack((x, y, z)).T

    # Keep same counting convention as notebook implementation.
    DD_all_obs = np.diff(npairs_3d(sample, sample, rbins, period=period))

    _, gid_compact = np.unique(gid, return_inverse=True)
    marks = np.vstack((gid_compact.astype(float), np.ones_like(gid_compact, dtype=float))).T

    M_eq = marked_tpcf(
        sample,
        rbins,
        marks1=marks,
        period=period,
        normalize_by='number_counts',
        weight_func_id=3,
        num_threads=num_threads,
    )
    DD_same_obs = M_eq * DD_all_obs

    inv_f = 1.0 / f_group
    inv_f2 = inv_f * inv_f
    DD_full_hat = inv_f2 * DD_all_obs + (inv_f - inv_f2) * DD_same_obs

    vbox = period ** 3
    n_obs = sample.shape[0]
    n_full = n_obs / f_group
    shell_vol = np.diff((4.0 * np.pi / 3.0) * rbins ** 3)

    # Equivalent to notebook LS-with-analytic-randoms formulation.
    DD = DD_full_hat / (n_full ** 2)
    RR = shell_vol / vbox
    xi_full = DD / RR - 1.0
    return xi_full


def compute_group_sampling_corrected_xi(
    gals: pd.DataFrame,
    rbins: np.ndarray,
    sampling_fraction: float,
    boxsize: float = 542.16,
    mstar_min_log10: float = 9.0,
    num_threads: int = 4,
) -> np.ndarray:
    """Group-sampling-corrected 2PCF in notebook style."""
    mask = gals['mstar'].values > mstar_min_log10
    if np.count_nonzero(mask) < 2:
        return np.full(len(rbins) - 1, np.nan)

    xyz = gals.loc[mask, ['xgal', 'ygal', 'zgal']].values.T
    gid = gals.loc[mask, 'igrp'].values
    return xi_unbiased_from_group_downsample_marked(
        *xyz,
        gid,
        f_group=sampling_fraction,
        rbins=rbins,
        period=boxsize,
        num_threads=num_threads,
    )


def compute_notebook_style_correlations_for_nvolumes(
    base_dir: str,
    iz_num: int,
    nvolumes_list: List[int],
    rbins: np.ndarray,
    boxsize: float = 542.16,
    mstar_min_log10: float = 9.0,
    mhalo_min: float = 1e11,
    centrals_only: bool = False,
    n_total_subvolumes: int = 1024,
    num_threads: int = 4,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Compute standard and corrected xi for each requested N_subvol."""
    results: Dict[int, Dict[str, np.ndarray]] = {}
    for nvol in nvolumes_list:
        gals = load_notebook_style_galaxies(
            base_dir=base_dir,
            iz_num=iz_num,
            ivols=np.arange(nvol),
            boxsize=boxsize,
            mhalo_min=mhalo_min,
            centrals_only=centrals_only,
        )

        xi_standard = compute_notebook_style_standard_xi(
            gals, rbins=rbins, boxsize=boxsize, mstar_min_log10=mstar_min_log10, num_threads=num_threads
        )
        xi_corrected = compute_group_sampling_corrected_xi(
            gals,
            rbins=rbins,
            sampling_fraction=nvol / float(n_total_subvolumes),
            boxsize=boxsize,
            mstar_min_log10=mstar_min_log10,
            num_threads=num_threads,
        )

        results[nvol] = {
            'xi_standard': xi_standard,
            'xi_corrected': xi_corrected,
            'ngal': int(np.count_nonzero(gals['mstar'].values > mstar_min_log10)),
        }

    return results
