"""
Mass-weighted two-point correlation function calculations for galaxies from GALFORM.

This module computes the 2PCF with bias correction for incomplete group/halo sampling,
mirroring the API of correlation.py but accounting for galaxy weighting based on
halo/group membership. Uses marked pair counting to correct for downsampling bias.

The key difference from regular correlation.py:
- Corrects for the fact that we may only sample a fraction of groups
- Uses marked TPCF to identify same-group pairs
- Applies unbiased estimator to account for sampling fraction

Reference: xi_unbiased_from_group_downsample_marked in P029-GALFORM_dynfric.ipynb
"""

import os
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from halotools.mock_observables import npairs_3d, marked_tpcf

from ...config import DEFAULT_RBINS, get_base_dir
from ...io.loaders import open_galaxies_hdf5, get_output_group


def _load_positions_and_groupids_from_hdf5(
    iz_path: str,
    ivol: int,
    centrals_only: bool = True,
    mhalo_min: Optional[float] = None,
    mstar_min_log10: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    """Load galaxy positions (x,y,z), group IDs, and redshift from HDF5 subvolume.

    Opens the HDF5 file exactly once and reads only the fields required:
    xgal/ygal/zgal, is_central, mhhalo, igrp.

    Args:
        iz_path: Path to snapshot directory
        ivol: Subvolume number
        centrals_only: If True, keep only central galaxies (is_central==1)
        mhalo_min: Minimum halo mass (mhhalo) threshold in Msun. None = no cut.
        mstar_min_log10: Minimum log10(mstar [Msun/h]). None = no cut.

    Returns:
        positions: (N,3) array in native units (Mpc/h)
        groupids: (N,) array with group membership for each galaxy
        z: best-effort redshift if available
    """
    f = open_galaxies_hdf5(iz_path, ivol=ivol)
    if f is None:
        raise FileNotFoundError(f"Missing or unreadable galaxies.hdf5 at {iz_path}/ivol{ivol}")

    try:
        g = get_output_group(f)
        if g is None:
            raise RuntimeError("No OutputNNN group found in HDF5 file")

        # Positions
        x = np.asarray(g['xgal'])
        y = np.asarray(g['ygal'])
        z_pos = np.asarray(g['zgal'])

        # Mask
        n = len(x)
        mask = np.ones(n, dtype=bool)

        if centrals_only:
            is_central = np.asarray(g['is_central'])
            mask &= (is_central == 1)

        if mhalo_min is not None:
            mhalo_key = 'mhhalo' if 'mhhalo' in g else 'mhalo'
            mhalo = np.asarray(g[mhalo_key])
            mask &= (mhalo >= mhalo_min)

        if mstar_min_log10 is not None:
            logh = np.log10(0.7)
            m_disk = np.asarray(g['mstars_disk']) if 'mstars_disk' in g else np.zeros(n)
            m_bulge = np.asarray(g['mstars_bulge']) if 'mstars_bulge' in g else np.zeros(n)
            mstar_log10 = np.log10(m_disk + m_bulge + 1e-30) - logh
            mask &= (mstar_log10 >= mstar_min_log10)

        # Group IDs — unique identifier per halo
        if 'igrp' in g:
            groupids = np.asarray(g['igrp'])[mask].astype(int)
        elif 'ihhalo' in g:
            groupids = np.asarray(g['ihhalo'])[mask].astype(int)
        else:
            # Each galaxy in its own group
            groupids = np.where(mask)[0].astype(int)

        positions = np.vstack([x[mask], y[mask], z_pos[mask]]).T.astype(np.float64, copy=False)

        # Redshift
        from ...utils.read_galaxies import _get_redshift_from_file, _get_redshift_from_zsnap
        z_val = _get_redshift_from_file(f) or _get_redshift_from_zsnap(iz_path, ivol)

        return positions, groupids, z_val

    finally:
        try:
            f.close()
        except Exception:
            pass


def compute_weighted_xi_corrfunc(
    positions: np.ndarray,
    groupids: np.ndarray,
    boxsize: float,
    rbins: Optional[np.ndarray] = None,
    sampling_fraction: float = 1.0,
    nthreads: int = 4,
) -> pd.DataFrame:
    """Compute weighted 2PCF accounting for incomplete group sampling.

    Corrects for the case where only a fraction of groups are sampled. Uses marked
    pair counting to identify same-group pairs, then applies an unbiased estimator.

    The correction is based on:
    - DD_all: all data-data pairs observed
    - DD_same: pairs within the same group (from marked TPCF)
    - f_group: fraction of groups sampled
    
    Returns: xi = (1/f^2 * DD_all - (1/f - 1/f^2) * DD_same) / RR_full - 1

    Args:
        positions: (N,3) array with coordinates
        groupids: (N,) array with group/halo ID for each galaxy
        boxsize: Side length of the periodic box
        rbins: Radial bin edges. Defaults to DEFAULT_RBINS
        sampling_fraction: Fraction of groups sampled (f_group in [0,1]).
                          1.0 = complete sample, <1.0 = incomplete
        nthreads: Number of OpenMP threads for halotools

    Returns:
        DataFrame with columns ['r', 'xi'] and metadata in df.attrs
    """
    if rbins is None:
        rbins = DEFAULT_RBINS
    rbins = np.asarray(rbins, dtype=float)
    
    # For periodic geometry, rmax must be < boxsize/2 to avoid double-counting
    rmax_periodic = boxsize / 2.0
    rbins = rbins[rbins <= rmax_periodic]
    
    if len(rbins) < 2:
        raise ValueError(f"No valid rbins within periodic limit (rmax={rmax_periodic:.2f}). Cannot compute correlation.")

    ngal = positions.shape[0]
    if ngal < 2:
        # Not enough galaxies
        r_centers = 0.5 * (rbins[:-1] + rbins[1:])
        df = pd.DataFrame({
            'r': r_centers,
            'xi': np.full_like(r_centers, np.nan),
        })
        df.attrs.update({'rbins': rbins, 'ngal': ngal})
        return df

    # Ensure sampling_fraction is in valid range
    if not (0 < sampling_fraction <= 1.0):
        raise ValueError(f"sampling_fraction must be in (0, 1], got {sampling_fraction}")

    # DD pairs: all observed pairs
    DD_all_obs = np.diff(npairs_3d(positions, positions, rbins) / 2)

    # Same-group fraction using marked TPCF
    # Create marks: each galaxy marked with its group ID (normalized) and a weight of 1
    _, groupid_compact = np.unique(groupids, return_inverse=True)
    marks = np.vstack((groupid_compact.astype(float), np.ones_like(groupid_compact, dtype=float))).T

    M_eq = marked_tpcf(
        positions, rbins,
        marks1=marks,
        period=boxsize,
        normalize_by="number_counts",
        weight_func_id=3,  # weight_func_id=3 uses the first mark as ID matching
        num_threads=nthreads,
    )

    DD_same_obs = M_eq * DD_all_obs

    # Correct for incomplete sampling:
    # DD_full_hat = (1/f²)*DD_cross + (1/f)*DD_same
    #             = (1/f²)*(DD_all - DD_same) + (1/f)*DD_same
    #             = (1/f²)*DD_all + (1/f - 1/f²)*DD_same
    inv_f = 1.0 / sampling_fraction
    inv_f2 = inv_f * inv_f
    DD_full_hat = inv_f2 * DD_all_obs + (inv_f - inv_f2) * DD_same_obs

    # RR pairs: analytic random expectation
    Vbox = boxsize ** 3
    N_obs = positions.shape[0]
    N_full = N_obs / sampling_fraction

    shell_vol = (4.0 * np.pi / 3.0) * (rbins[1:] ** 3 - rbins[:-1] ** 3)
    RR_full = (N_full * (N_full - 1) / 2.0) * (shell_vol / Vbox)

    # Correlation function
    xi_full = DD_full_hat / RR_full - 1.0
    r_centers = 0.5 * (rbins[:-1] + rbins[1:])

    df = pd.DataFrame({'r': r_centers, 'xi': xi_full})
    df.attrs.update({
        'rbins': rbins, 
        'ngal': ngal,
        'sampling_fraction': sampling_fraction,
        'n_groups': len(np.unique(groupids)),
    })
    return df


def weighted_correlation_given_redshift_and_subvolume(
    iz_path: str,
    ivol: int,
    rbins: Optional[np.ndarray] = None,
    nthreads: int = 4,
    centrals_only: bool = True,
    mhalo_min: Optional[float] = None,
    mstar_min_log10: Optional[float] = None,
    sampling_fraction: Optional[float] = None,
    boxsize: Optional[float] = None,
) -> Optional[pd.DataFrame]:
    """Compute weighted (bias-corrected) 2PCF for (snapshot, ivol).

    Corrects for incomplete group/halo sampling using marked pair counting.

    Args:
        iz_path: Path to snapshot directory (e.g., str(get_base_dir()/"iz207"))
        ivol: Subvolume number
        rbins: Radial bin edges (Mpc). Defaults to DEFAULT_RBINS
        nthreads: Number of OpenMP threads for halotools
        centrals_only: If True, keep only central galaxies (is_central==1)
        mhalo_min: Minimum halo mass (mhalo) in Msun. None = no cut.
        sampling_fraction: Fraction of groups sampled in [0,1].
                          If None, assumes complete sample (f=1.0)
        boxsize: Full simulation box side length in Mpc/h. If None, read from
                 the HDF5 Parameters group.

    Returns:
        DataFrame with columns ['r', 'xi'] and metadata in df.attrs.
        Returns None if unavailable.
    """
    try:
        pos, groupids, z_val = _load_positions_and_groupids_from_hdf5(
            iz_path, ivol, centrals_only=centrals_only, mhalo_min=mhalo_min,
            mstar_min_log10=mstar_min_log10,
        )

        L = boxsize
        if L is None:
            gal_file = os.path.join(iz_path, f'ivol{ivol}', 'galaxies.hdf5')
            try:
                import h5py
                with h5py.File(gal_file, 'r') as f:
                    if 'Parameters' in f and 'volume' in f['Parameters']:
                        v_ivol = float(np.array(f['Parameters']['volume']))
                        n_sub = int(np.array(f['Parameters'].get('n_subvolumes', 1024)))
                        L = float((v_ivol * n_sub) ** (1.0 / 3.0))
            except Exception:
                pass

        if L is None or not np.isfinite(L) or L <= 0:
            raise RuntimeError(f"Cannot determine box size for {iz_path}/ivol{ivol}")

        f_group = sampling_fraction if sampling_fraction is not None else 1.0

        res = compute_weighted_xi_corrfunc(
            pos, groupids, boxsize=L, rbins=rbins,
            sampling_fraction=f_group, nthreads=nthreads
        )
        res.attrs.update({
            'z': z_val,
            'ivol': ivol,
            'boxsize': L,
            'ngal': res.attrs.get('ngal'),
            'n_groups': res.attrs.get('n_groups'),
            'sampling_fraction': res.attrs.get('sampling_fraction'),
            'rbins': res.attrs.get('rbins'),
        })
        return res

    except (FileNotFoundError, RuntimeError, KeyError):
        return None


def avg_weighted_correlation_given_redshift_and_subvolumes(
    iz_num: int,
    ivols: List[int],
    rbins: Optional[np.ndarray] = None,
    nthreads: int = 16,
    base_dir: Optional[str] = None,
    centrals_only: bool = True,
    mhalo_min: Optional[float] = None,
    mstar_min_log10: Optional[float] = None,
    sampling_fraction: Optional[float] = None,
    boxsize: Optional[float] = None,
    n_total_subvolumes: int = 1024,
) -> Optional[pd.DataFrame]:
    """Compute weighted 2PCF by combining galaxies from multiple subvolumes.

    Combines all galaxy positions and group IDs from multiple subvolumes 
    into a single box, then computes the weighted correlation function.

    Args:
        iz_num: Numeric snapshot identifier (e.g. 207 for 'iz207')
        ivols: List of subvolume indices to combine
        rbins: Optional radial bin edges (defaults to DEFAULT_RBINS)
        nthreads: Number of OpenMP threads for halotools
        base_dir: Optional base directory; defaults to configured base dir
        centrals_only: If True, only include central galaxies (is_central=1)
        mhalo_min: Minimum halo mass (mhalo) in Msun. None = no cut
        sampling_fraction: Fraction of groups sampled in [0,1].
                          If None, inferred as len(ivols)/n_total_subvolumes
        boxsize: Full simulation box side length in Mpc/h. If None, read from
                 the HDF5 Parameters group (volume key).
        n_total_subvolumes: Total number of subvolumes in the simulation (default 1024).
                            Used to infer sampling_fraction when not provided.

    Returns:
        DataFrame with columns ['r', 'xi'] and metadata in df.attrs.
        Returns None if no subvolume produced valid data.
    """
    if rbins is None:
        rbins = DEFAULT_RBINS
    if base_dir is None:
        base_dir = str(get_base_dir())

    iz_path = os.path.join(base_dir, f'iz{iz_num}')
    if not os.path.isdir(iz_path):
        return None

    all_positions = []
    all_groupids = []
    z = None
    L_box = boxsize  # may be None until we read from file
    group_id_offset = 0

    for iv in ivols:
        try:
            pos, groupids, z_val = _load_positions_and_groupids_from_hdf5(
                iz_path, iv, centrals_only=centrals_only, mhalo_min=mhalo_min,
                mstar_min_log10=mstar_min_log10,
            )

            if z is None and z_val is not None:
                z = z_val

            # Read box size from file once if not provided
            if L_box is None:
                gal_file = os.path.join(iz_path, f'ivol{iv}', 'galaxies.hdf5')
                try:
                    import h5py
                    with h5py.File(gal_file, 'r') as f:
                        if 'Parameters' in f and 'volume' in f['Parameters']:
                            v_ivol = float(np.array(f['Parameters']['volume']))
                            n_sub = int(np.array(f['Parameters'].get('n_subvolumes', n_total_subvolumes)))
                            L_box = float((v_ivol * n_sub) ** (1.0 / 3.0))
                except Exception:
                    pass

            # Make group IDs unique across subvolumes
            groupids_offsetted = groupids + group_id_offset
            group_id_offset = int(np.max(groupids_offsetted)) + 1

            all_positions.append(pos)
            all_groupids.append(groupids_offsetted)

        except (FileNotFoundError, RuntimeError, KeyError):
            continue

    if not all_positions:
        return None

    combined_positions = np.vstack(all_positions)
    combined_groupids = np.hstack(all_groupids)
    total_galaxies = combined_positions.shape[0]

    if L_box is None or not np.isfinite(L_box) or L_box <= 0:
        return None

    if sampling_fraction is None:
        sampling_fraction = min(1.0, len(all_positions) / n_total_subvolumes)

    res = compute_weighted_xi_corrfunc(
        combined_positions, combined_groupids, boxsize=L_box, rbins=rbins,
        sampling_fraction=sampling_fraction, nthreads=nthreads
    )

    res.attrs.update({
        'z': z,
        'iz': f'iz{iz_num}',
        'boxsize': L_box,
        'n_ivols': len(all_positions),
        'total_galaxies': total_galaxies,
        'rbins': rbins,
        'sampling_fraction': sampling_fraction,
        'method': 'combined_weighted_overlapping_subvolumes',
    })
    
    return res


def weighted_correlations_given_redshifts_and_subvolume(
    iz_nums: List[int],
    ivol: int,
    rbins: Optional[np.ndarray] = None,
    nthreads: int = 4,
    base_dir: Optional[str] = None,
    centrals_only: bool = True,
    mhalo_min: Optional[float] = None,
    sampling_fraction: Optional[float] = None,
) -> List[pd.DataFrame]:
    """Compute weighted correlation function for one subvolume across multiple snapshots.

    Args:
        iz_nums: List of numeric snapshot identifiers (e.g. [100, 120, 142])
        ivol: Subvolume index
        rbins: Optional radial bin edges (defaults to DEFAULT_RBINS)
        nthreads: Number of OpenMP threads for halotools
        base_dir: Optional base directory; defaults to configured base dir
        centrals_only: If True, only include central galaxies (is_central=1)
        mhalo_min: Minimum halo mass (mhalo) in Msun. None = no cut
        sampling_fraction: Fraction of groups sampled in [0,1]

    Returns:
        List of DataFrames, one per snapshot. Returns None for unavailable snapshots.
    """
    if rbins is None:
        rbins = DEFAULT_RBINS
    if base_dir is None:
        base_dir = str(get_base_dir())

    results = []
    for iz_num in iz_nums:
        iz_path = os.path.join(base_dir, f'iz{iz_num}')
        if not os.path.isdir(iz_path):
            continue
        
        res = weighted_correlation_given_redshift_and_subvolume(
            iz_path, ivol, rbins=rbins, nthreads=nthreads,
            centrals_only=centrals_only, mhalo_min=mhalo_min,
            sampling_fraction=sampling_fraction
        )
        if res is not None:
            res.attrs['iz'] = f'iz{iz_num}'
            results.append(res)
    
    return results


def avg_weighted_correlation_given_subvolume_and_redshifts(
    iz_nums: List[int],
    ivol: int,
    rbins: Optional[np.ndarray] = None,
    nthreads: int = 4,
    base_dir: Optional[str] = None,
    centrals_only: bool = True,
    mhalo_min: Optional[float] = None,
    sampling_fraction: Optional[float] = None,
) -> Optional[pd.DataFrame]:
    """Average weighted xi(r) across multiple redshifts for a single subvolume.

    Args:
        iz_nums: List of numeric snapshot identifiers (e.g., [100, 120, 142])
        ivol: Subvolume index to evaluate
        rbins: Optional radial bin edges; defaults to DEFAULT_RBINS
        nthreads: Number of OpenMP threads for halotools
        base_dir: Optional base directory for snapshots; defaults to configured base dir
        centrals_only: If True, only include central galaxies (is_central==1)
        mhalo_min: Minimum halo mass threshold in Msun; None applies no cut
        sampling_fraction: Fraction of groups sampled in [0,1]

    Returns:
        DataFrame with columns ['r', 'xi', 'xi_std'] and metadata in df.attrs.
        Returns None if no snapshots produced valid data.
    """
    if rbins is None:
        rbins = DEFAULT_RBINS
    if base_dir is None:
        base_dir = str(get_base_dir())

    per_xi: List[np.ndarray] = []
    r_ref: Optional[np.ndarray] = None
    used_iz: List[str] = []
    used_z: List[Optional[float]] = []

    for iz_num in iz_nums:
        iz_path = os.path.join(base_dir, f'iz{iz_num}')
        if not os.path.isdir(iz_path):
            continue

        res = weighted_correlation_given_redshift_and_subvolume(
            iz_path, ivol, rbins=rbins, nthreads=nthreads,
            centrals_only=centrals_only, mhalo_min=mhalo_min,
            sampling_fraction=sampling_fraction
        )

        if res is None:
            continue
        if r_ref is None:
            r_ref = res['r'].to_numpy()
        per_xi.append(res['xi'].to_numpy())
        used_iz.append(f'iz{iz_num}')
        used_z.append(res.attrs.get('z'))

    if not per_xi:
        return None

    per_xi_arr = np.vstack(per_xi)
    r = r_ref if r_ref is not None else 0.5 * (rbins[1:] + rbins[:-1])
    xi_mean = per_xi_arr.mean(axis=0)
    xi_std = per_xi_arr.std(axis=0)

    metadata = {
        'ivol': ivol,
        'n_used': per_xi_arr.shape[0],
        'used_iz': used_iz,
        'used_z': used_z,
        'rbins': rbins,
        'sampling_fraction': sampling_fraction,
    }
    df = pd.DataFrame({'r': r, 'xi': xi_mean, 'xi_std': xi_std})
    df.attrs.update(metadata)
    return df
