import os
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
from Corrfunc.theory.xi import xi as corrfunc_xi

from ..config import DEFAULT_RBINS, get_base_dir
from ..io.loaders import open_galaxies_hdf5, get_output_group, read_snapshot_data


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
    f = open_galaxies_hdf5(iz_path, ivol=ivol)
    if f is None:
        raise FileNotFoundError(f"Missing or unreadable galaxies.hdf5 at {iz_path}/ivol{ivol}")
    try:
        g = get_output_group(f)
        if g is None:
            raise RuntimeError("No OutputNNN group found in HDF5 file")

        # Load position arrays (GALFORM uses xgal, ygal, zgal)
        if 'xgal' not in g or 'ygal' not in g or 'zgal' not in g:
            raise KeyError("Could not find xgal/ygal/zgal position arrays in Output group")
        
        x = np.asarray(g['xgal'])
        y = np.asarray(g['ygal'])
        z_arr = np.asarray(g['zgal'])

        # Ensure 1D arrays and consistent length
        x = np.ravel(x)
        y = np.ravel(y)
        z_arr = np.ravel(z_arr)
        n = min(x.size, y.size, z_arr.size)
        
        # Apply filtering if requested
        mask = np.ones(n, dtype=bool)
        
        if centrals_only:
            if 'is_central' not in g:
                raise KeyError("is_central field not found - cannot filter for centrals")
            is_central = np.ravel(g['is_central'][:n])
            mask &= (is_central == 1)
        
        if mhalo_min is not None:
            if 'mhalo' not in g:
                raise KeyError("mhalo field not found - cannot apply halo mass cut")
            mhalo = np.ravel(g['mhalo'][:n])
            mask &= (mhalo >= mhalo_min)
        
        # Apply mask to positions
        x = x[:n][mask]
        y = y[:n][mask]
        z_arr = z_arr[:n][mask]
        
        pos = np.vstack([x, y, z_arr]).T.astype(np.float64, copy=False)

        # Redshift (best-effort)
        z_val = None
        try:
            if 'Redshifts' in f and isinstance(f['Redshifts'], np.ndarray):
                z_val = float(np.ravel(f['Redshifts'])[0])
        except Exception:
            z_val = None

        return pos, z_val
    finally:
        try:
            f.close()
        except Exception:
            pass


def compute_xi_corrfunc(
    positions: np.ndarray,
    boxsize: float,
    rbins: Optional[np.ndarray] = None,
    nthreads: int = 4,
) -> Dict[str, np.ndarray]:
    """Compute the real-space two-point correlation xi(r) using Corrfunc.

    Uses Corrfunc's optimized pair counting with periodic boundary conditions.
    Much faster than FFT methods and doesn't require randoms.

    Args:
        positions: (N,3) array with coordinates in [0, boxsize)
        boxsize: Side length of the periodic domain (same units as r)
        rbins: Radial bin edges. Defaults to config.DEFAULT_RBINS
        nthreads: Number of OpenMP threads for parallel execution

    Returns:
        dict with keys: 'rbins', 'r', 'xi', 'ngal'
    """
    if rbins is None:
        rbins = DEFAULT_RBINS
    rbins = np.asarray(rbins, dtype=float)
    
    # Corrfunc requires rmax < boxsize/2 for periodic boxes
    # Truncate bins that exceed this limit
    max_allowed_r = boxsize / 2.0 * 0.99  # Use 99% to be safe
    if rbins[-1] > max_allowed_r:
        valid_bins = rbins <= max_allowed_r
        if np.sum(valid_bins) < 2:
            # Need at least 2 bin edges (1 bin)
            raise ValueError(f"Boxsize {boxsize:.2f} too small for requested rbins (need rmax < {max_allowed_r:.2f})")
        rbins = rbins[valid_bins]

    # Wrap positions into the box [0, L)
    pos = np.mod(positions, boxsize)
    
    ngal = pos.shape[0]
    if ngal < 2:
        # Not enough galaxies for correlation
        r_centers = 0.5 * (rbins[:-1] + rbins[1:])
        return {
            'rbins': rbins,
            'r': r_centers,
            'xi': np.full_like(r_centers, np.nan),
            'ngal': ngal,
        }

    # Compute xi(r) directly using Corrfunc
    results = corrfunc_xi(
        boxsize=boxsize,
        nthreads=nthreads,
        binfile=rbins,
        X=pos[:, 0],
        Y=pos[:, 1],
        Z=pos[:, 2],
        verbose=False,
        output_ravg=True,  # Get average r for each bin
    )

    # Extract correlation function values
    xi_vals = np.array([x['xi'] for x in results], dtype=np.float64)
    ravg = np.array([x['ravg'] for x in results], dtype=np.float64)
    
    # Use ravg if available, otherwise use bin centers
    if np.all(ravg > 0):
        r = ravg
    else:
        r = 0.5 * (rbins[:-1] + rbins[1:])
    
    return {
        'rbins': rbins,
        'r': r,
        'xi': xi_vals,
        'ngal': ngal,
    }


def correlation_given_redshift_and_subvolume(
    iz_path: str,
    ivol: int,
    rbins: Optional[np.ndarray] = None,
    nthreads: int = 4,
    centrals_only: bool = False,
    mhalo_min: Optional[float] = None,
) -> Optional[Dict[str, np.ndarray]]:
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
        dict with 'r', 'xi', 'z', 'ivol', 'V_ivol', 'boxsize', 'ngal'; or None if unavailable
    """
    try:
        # Load positions and redshift
        pos, z_val = _load_positions_from_hdf5(iz_path, ivol, centrals_only=centrals_only, mhalo_min=mhalo_min)

        # Get volumes using existing loader (also returns z if available)
        meta = read_snapshot_data(iz_path, ivol)
        V_ivol = meta.get('V_ivol', None)
        if V_ivol is None:
            # Fallback: infer from positions extent as a cube
            L_est = np.ptp(np.mod(pos, np.max(pos, axis=0) - np.min(pos, axis=0)), axis=0)
            L = float(np.max(L_est))
            V_ivol = L ** 3
        else:
            L = float(V_ivol) ** (1.0 / 3.0)

        res = compute_xi_corrfunc(pos, boxsize=L, rbins=rbins, nthreads=nthreads)
        out = {
            'r': res['r'],
            'xi': res['xi'],
            'rbins': res['rbins'],
            'ngal': res['ngal'],
            'z': z_val if z_val is not None else meta.get('z'),
            'ivol': ivol,
            'V_ivol': V_ivol,
            'boxsize': L,
        }
        return out
    except (FileNotFoundError, RuntimeError, KeyError) as e:
        # Graceful failure to mirror other analysis helpers
        print(f"Warning: correlation could not be computed for {iz_path}/ivol{ivol}: {e}")
        return None


def avg_correlation_given_redshift_and_subvolumes(
    iz_num: int,
    ivols: List[int],
    rbins: Optional[np.ndarray] = None,
    nthreads: int = 4,
    base_dir: Optional[str] = None,
    centrals_only: bool = False,
    mhalo_min: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
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
        Dictionary with keys:
            - 'iz': snapshot name (e.g. 'iz207')
            - 'z': redshift (from first successful subvolume)
            - 'r': radial bin centers
            - 'xi': mean correlation function across provided subvolumes
            - 'xi_std': standard deviation across provided subvolumes
            - 'n_used': number of successful subvolumes
            - 'n_requested': length of ivols list
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
            r_ref = res['r']
        if z is None:
            z = res['z']
        per_xi.append(res['xi'])

    if not per_xi:
        return None

    per_xi = np.array(per_xi)
    r = r_ref if r_ref is not None else 0.5 * (rbins[1:] + rbins[:-1])

    return {
        'iz': f'iz{iz_num}',
        'z': z,
        'r': r,
        'xi': per_xi.mean(axis=0),
        'xi_std': per_xi.std(axis=0),
        'n_used': per_xi.shape[0],
        'n_requested': len(ivols),
    }