import os
from typing import Optional, Tuple, List

import numpy as np
from Corrfunc.theory.DD import DD as corrfunc_DD

import pandas as pd

from ...config import DEFAULT_RBINS, get_base_dir
from ...io.loaders import read_snapshot_data
from ...utils.read_galaxies import read_galaxy_positions


def _load_positions_from_hdf5(
    iz_path: str, 
    ivol: int,
    centrals_only: bool = False,
    mhalo_min: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[float]]:
    """Load galaxy positions (x,y,z) and redshift from an HDF5 subvolume.

    Args:
        iz_path: Path to snapshot directory
        ivol: Subvolume number
        centrals_only: If True, only include central galaxies (is_central=1)
        mhalo_min: Minimum halo mass (mhalo) threshold in Msun. None = no cut.

    Returns:
        positions: (N,3) array in the native units of the file (assumed Mpc or Mpc/h)
        z: best-effort redshift if available
    """
    return read_galaxy_positions(
        iz_path=iz_path,
        ivol=ivol,
        centrals_only=centrals_only,
        mhalo_min=mhalo_min,
    )


def compute_xi_corrfunc(
    positions: np.ndarray,
    boxsize: float,
    rbins: Optional[np.ndarray] = None,
    nthreads: int = 4,
) -> pd.DataFrame:
    """Compute the real-space two-point correlation xi(r) using Corrfunc.

    For periodic subvolumes, uses Corrfunc.theory.DD to count pairs
    with periodic boundary conditions, then uses Landy-Szalay estimator
    with analytic random pair counts.

    Args:
        positions: (N,3) array with coordinates (physical positions in subvolume)
        boxsize: Side length of the subvolume (same units as r)
        rbins: Radial bin edges. Defaults to config.DEFAULT_RBINS
        nthreads: Number of OpenMP threads for parallel execution

    Returns:
        DataFrame with columns ['r', 'xi'] and metadata in df.attrs
    """
    if rbins is None:
        rbins = DEFAULT_RBINS
    rbins = np.asarray(rbins, dtype=float)
    
    # For periodic geometry, rmax must be < boxsize/2 to avoid double-counting
    rmax_periodic = boxsize / 2.0
    rbins = rbins[rbins < rmax_periodic]
    
    if len(rbins) < 2:
        raise ValueError(f"rbins exceed periodic limit (rmax={rmax_periodic:.2f}). Cannot compute correlation.")
    
    ngal = positions.shape[0]
    if ngal < 2:
        # Not enough galaxies for correlation
        r_centers = 0.5 * (rbins[:-1] + rbins[1:])
        df = pd.DataFrame({
            'r': r_centers,
            'xi': np.full_like(r_centers, np.nan) - 1.0,  # xi 0 = uncorrelated
        })
        df.attrs.update({'rbins': rbins, 'ngal': ngal})
        return df

    # Use DD with periodic boundary conditions since subvolumes have periodic geometry
    # autocorr=1 means this is an auto-correlation (same catalog for both sets)
    # periodic=True is critical - subvolumes have toroidal topology with wraparound!
    results = corrfunc_DD(
        autocorr=1,
        nthreads=nthreads,
        binfile=rbins,
        X1=positions[:, 0],
        Y1=positions[:, 1],
        Z1=positions[:, 2],
        periodic=True,
        boxsize=boxsize,
        verbose=False,
        output_ravg=True,
    )

    # Extract pair counts and average r
    npairs = np.array([x['npairs'] for x in results], dtype=np.float64)
    ravg = np.array([x['ravg'] for x in results], dtype=np.float64)
    
    # Use ravg if available, otherwise use bin centers
    if np.all(ravg > 0):
        r = ravg
    else:
        r = 0.5 * (rbins[:-1] + rbins[1:])
    
    # Compute RR analytically for a periodic cubic volume
    # Number of random pairs in shell [r1, r2] for uniform density with periodic BC
    # For periodic geometry: max separation = boxsize/2 (shortest image convention)
    # RR(r) = n * (n-1) / 2 * V_shell / V_box for r <= boxsize/2
    # where V_shell = 4/3 * pi * (r2^3 - r1^3)
    volume = boxsize ** 3
    # n_rand = ngal  # Use same number density as data - not needed for analytical RR
    
    r1 = rbins[:-1]
    r2 = rbins[1:]
    
    # Volume of spherical shells (periodic: shells within boxsize/2 are unaffected)
    V_shell = (4.0 / 3.0) * np.pi * (r2**3 - r1**3)
    
    # Expected number of random pairs (normalized by total volume)
    # For autocorrelation with periodic BC: RR = n * (n-1) / 2 * V_shell / V_box
    # RR = n_rand * (n_rand - 1.0) / 2.0 * V_shell / volume
    
    # Landy-Szalay estimator: xi = (DD - 2*DR + RR) / RR
    # For auto-correlation with analytic RR: DD/RR - 1
    # Normalize DD: DD_normalized = DD / (n_gal * (n_gal - 1) / 2)
    # RR_normalized = RR / (n_rand * (n_rand - 1) / 2)
    DD_norm = npairs / (ngal * (ngal - 1.0) / 2.0)
    RR_norm = V_shell / volume
    
    # Avoid division by zero
    xi_vals = np.where(RR_norm > 0, DD_norm / RR_norm - 1.0, np.nan)

    df = pd.DataFrame({'r': r, 'xi': xi_vals})
    df.attrs.update({'rbins': rbins, 'ngal': ngal})
    return df


def correlation_given_redshift_and_subvolume(
    iz_path: str,
    ivol: int,
    rbins: Optional[np.ndarray] = None,
    nthreads: int = 4,
    centrals_only: bool = False,
    mhalo_min: Optional[float] = None,
) -> Optional[pd.DataFrame]:
    """High-level helper mirroring the HMF API: xi(r) for (snapshot, ivol).

    Attempts to read positions from the HDF5 subvolume, estimate boxsize from
    the recorded subvolume volume, and compute xi(r) using Corrfunc.

    Args:
        iz_path: Path to snapshot directory (e.g., str(get_base_dir()/"iz207"))
        ivol: Subvolume number
        rbins: Radial bin edges (Mpc). Defaults to config.DEFAULT_RBINS
        nthreads: Number of OpenMP threads for Corrfunc
        centrals_only: If True, only include central galaxies (is_central=1)
        mhalo_min: Minimum halo mass (mhalo) in Msun. None = no cut.

    Returns:
        DataFrame with columns ['r', 'xi'] and metadata in df.attrs.
        Returns None if unavailable.
    """
    try:
        # Load positions and redshift
        pos, z_val = _load_positions_from_hdf5(iz_path, ivol, centrals_only=centrals_only, mhalo_min=mhalo_min)

        # Get subvolume metadata
        meta = read_snapshot_data(iz_path, ivol)
        V_ivol = meta.get('V_ivol', None)
        
        # The HDF5 file contains galaxies from one subvolume (periodic subdomain).
        # Positions are in absolute simulation coordinates; use V_ivol to get
        # the actual subvolume box size for the periodic calculation.
        
        if V_ivol is not None and np.isfinite(V_ivol) and V_ivol > 0:
            L = float(V_ivol) ** (1.0 / 3.0)
        else:
            # Fallback: infer from position extent
            extent = np.ptp(pos, axis=0)
            L = float(np.max(extent))
        
        # Shift positions into [0, L) to match Corrfunc convention
        # Use modulo to wrap positions into the subvolume frame
        pos = np.fmod(pos, L)
        pos = np.where(pos < 0, pos + L, pos)  # Handle negative values

        if not np.isfinite(L) or L <= 0:
            raise RuntimeError(f"Invalid subvolume box size for {iz_path}/ivol{ivol}: L={L}")

        res = compute_xi_corrfunc(pos, boxsize=L, rbins=rbins, nthreads=nthreads)
        
        # Metadata
        metadata = {
            'z': z_val if z_val is not None else meta.get('z'),
            'ivol': ivol,
            'V_ivol': V_ivol,
            'boxsize': L,
            'ngal': res.attrs.get('ngal'),
            'rbins': res.attrs.get('rbins'),
        }
        res.attrs.update(metadata)
        return res

    except (FileNotFoundError, RuntimeError, KeyError) as e:
        # Graceful failure to mirror other analysis helpers
        import traceback
        print(f"Warning: correlation could not be computed for {iz_path}/ivol{ivol}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


def halo_correlation_given_redshift_and_subvolume(
    iz_path: str,
    ivol: int,
    rbins: Optional[np.ndarray] = None,
    nthreads: int = 4,
    mhalo_min: Optional[float] = None,
) -> Optional[pd.DataFrame]:
    """Compute dark matter halo correlation function directly from GALFORM halo positions.

    Instead of using merger tree files, this function computes the 2-point correlation
    function of halos by reading their positions (and masses) directly from the
    galaxies.hdf5 file. Each galaxy represents a halo in GALFORM.

    Args:
        iz_path: Path to snapshot directory
        ivol: Subvolume number
        rbins: Radial bin edges (Mpc/h). Defaults to DEFAULT_RBINS
        nthreads: Number of OpenMP threads for Corrfunc
        mhalo_min: Optional minimum halo mass cut in Msun

    Returns:
        DataFrame with columns ['r', 'xi'] and metadata in df.attrs.
        Returns None if computation fails.
    """
    try:
        # Load halo positions and masses directly from GALFORM
        pos, z_val = _load_positions_from_hdf5(iz_path, ivol, centrals_only=False, mhalo_min=mhalo_min)

        # Get subvolume metadata
        meta = read_snapshot_data(iz_path, ivol)
        V_ivol = meta.get('V_ivol', None)
        
        if V_ivol is not None and np.isfinite(V_ivol) and V_ivol > 0:
            L = float(V_ivol) ** (1.0 / 3.0)
        else:
            extent = np.ptp(pos, axis=0)
            L = float(np.max(extent))
        
        # Shift positions into [0, L) for Corrfunc
        pos = np.fmod(pos, L)
        pos = np.where(pos < 0, pos + L, pos)

        if not np.isfinite(L) or L <= 0:
            raise RuntimeError(f"Invalid subvolume box size for {iz_path}/ivol{ivol}: L={L}")

        res = compute_xi_corrfunc(pos, boxsize=L, rbins=rbins, nthreads=nthreads)
        
        # Metadata
        metadata = {
            'z': z_val if z_val is not None else meta.get('z'),
            'ivol': ivol,
            'V_ivol': V_ivol,
            'boxsize': L,
            'nhalo': res.attrs.get('ngal'),  # Use ngal as count of halos
            'rbins': res.attrs.get('rbins'),
        }
        res.attrs.update(metadata)
        return res

    except (FileNotFoundError, RuntimeError, KeyError) as e:
        import traceback
        print(f"Warning: Halo correlation could not be computed from {iz_path}/ivol{ivol}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None

def avg_correlation_given_redshift_and_subvolumes(
    iz_num: int,
    ivols: List[int],
    rbins: Optional[np.ndarray] = None,
    nthreads: int = 4,
    base_dir: Optional[str] = None,
    centrals_only: bool = False,
    mhalo_min: Optional[float] = None,
) -> Optional[pd.DataFrame]:
    """Average correlation function over a provided list of subvolumes for a snapshot.

    Mirrors the HMF API: compute xi(r) for each requested ivol and average.

    Args:
        iz_num: Numeric snapshot identifier (e.g. 207 for 'iz207').
        ivols: List of subvolume indices to include in the average.
        rbins: Optional radial bin edges (defaults to DEFAULT_RBINS).
        nthreads: Number of OpenMP threads for Corrfunc.
        base_dir: Optional base directory; defaults to configured base dir.
        centrals_only: If True, only include central galaxies (is_central=1)
        mhalo_min: Minimum halo mass (mhalo) in Msun. None = no cut.
    Returns:
        DataFrame with columns ['r', 'xi', 'xi_std'] and metadata in df.attrs.
        Returns None if no subvolume produced valid data.
    """
    if rbins is None:
        rbins = DEFAULT_RBINS
    if base_dir is None:
        base_dir = str(get_base_dir())

    iz_path = os.path.join(base_dir, f'iz{iz_num}')
    if not os.path.isdir(iz_path):
        return None

    per_xi = []
    z = None
    r_ref = None

    for iv in ivols:
        res = correlation_given_redshift_and_subvolume(
            iz_path, iv, rbins=rbins, nthreads=nthreads, 
            centrals_only=centrals_only, mhalo_min=mhalo_min
        )
        if res is None:
            continue
        if r_ref is None:
            r_ref = res['r'].to_numpy()
        if z is None:
            z = res.attrs.get('z')
        per_xi.append(res['xi'].to_numpy())

    if not per_xi:
        return None

    per_xi = np.array(per_xi)
    r = r_ref if r_ref is not None else 0.5 * (rbins[1:] + rbins[:-1])
    xi_mean = per_xi.mean(axis=0)
    xi_std = per_xi.std(axis=0)
    
    metadata = {
        'iz': f'iz{iz_num}',
        'z': z,
        'n_used': per_xi.shape[0],
        'n_requested': len(ivols),
        'rbins': rbins,
    }
    df = pd.DataFrame({'r': r, 'xi': xi_mean, 'xi_std': xi_std})
    df.attrs.update(metadata)
    return df


def correlations_given_redshifts_and_subvolume(
    iz_nums: List[int],
    ivol: int,
    rbins: Optional[np.ndarray] = None,
    nthreads: int = 4,
    base_dir: Optional[str] = None,
    centrals_only: bool = False,
    mhalo_min: Optional[float] = None,
) -> List[pd.DataFrame]:
    """Compute correlation function for one subvolume across multiple snapshots.

    Args:
        iz_nums: List of numeric snapshot identifiers (e.g. [100, 120, 142]).
        ivol: Subvolume index.
        rbins: Optional radial bin edges (defaults to DEFAULT_RBINS).
        nthreads: Number of OpenMP threads for Corrfunc.
        base_dir: Optional base directory; defaults to configured base dir.
        centrals_only: If True, only include central galaxies (is_central=1)
        mhalo_min: Minimum halo mass (mhalo) in Msun. None = no cut.

    Returns:
        List of dictionaries, one per snapshot. Each contains:
            - 'iz': snapshot name (e.g. 'iz100')
            - 'z': redshift
            - 'r': radial bin centers
            - 'xi': correlation function
            - 'ngal': number of galaxies
            - 'boxsize': box size used
        Skips snapshots where data is unavailable.
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
        
        res = correlation_given_redshift_and_subvolume(
            iz_path, ivol, rbins=rbins, nthreads=nthreads,
            centrals_only=centrals_only, mhalo_min=mhalo_min
        )
        if res is not None:
            res.attrs['iz'] = f'iz{iz_num}'
            results.append(res)
    
    return results


def avg_correlation_given_subvolume_and_redshifts(
    iz_nums: List[int],
    ivol: int,
    rbins: Optional[np.ndarray] = None,
    nthreads: int = 4,
    base_dir: Optional[str] = None,
    centrals_only: bool = False,
    mhalo_min: Optional[float] = None,
) -> Optional[pd.DataFrame]:
    """Average xi(r) across multiple redshifts for a single subvolume.

    Args:
        iz_nums: List of numeric snapshot identifiers (e.g., [100, 120, 142]).
        ivol: Subvolume index to evaluate.
        rbins: Optional radial bin edges; defaults to ``DEFAULT_RBINS``.
        nthreads: Number of OpenMP threads for Corrfunc.
        base_dir: Optional base directory for snapshots; defaults to configured base dir.
        centrals_only: If True, only include central galaxies (is_central==1).
        mhalo_min: Minimum halo mass threshold in Msun; None applies no cut.
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

        res = correlation_given_redshift_and_subvolume(
            iz_path, ivol, rbins=rbins, nthreads=nthreads,
            centrals_only=centrals_only, mhalo_min=mhalo_min
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
    }
    df = pd.DataFrame({'r': r, 'xi': xi_mean, 'xi_std': xi_std})
    df.attrs.update(metadata)
    return df