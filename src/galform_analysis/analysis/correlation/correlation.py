import os
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
from Corrfunc.theory.DD import DD as corrfunc_DD

from ...config import DEFAULT_RBINS, get_base_dir
from ...io.loaders import open_galaxies_hdf5, get_output_group, read_snapshot_data


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

    For periodic subvolumes, uses Corrfunc.theory.DD to count pairs
    with periodic boundary conditions, then uses Landy-Szalay estimator
    with analytic random pair counts.

    Args:
        positions: (N,3) array with coordinates (physical positions in subvolume)
        boxsize: Side length of the subvolume (same units as r)
        rbins: Radial bin edges. Defaults to config.DEFAULT_RBINS
        nthreads: Number of OpenMP threads for parallel execution

    Returns:
        dict with keys: 'rbins', 'r', 'xi', 'ngal'
    """
    if rbins is None:
        rbins = DEFAULT_RBINS
    rbins = np.asarray(rbins, dtype=float)
    
    ngal = positions.shape[0]
    if ngal < 2:
        # Not enough galaxies for correlation
        r_centers = 0.5 * (rbins[:-1] + rbins[1:])
        return {
            'rbins': rbins,
            'r': r_centers,
            'xi': np.full_like(r_centers, np.nan),
            'ngal': ngal,
        }

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
    n_rand = ngal  # Use same number density as data
    
    r1 = rbins[:-1]
    r2 = rbins[1:]
    
    # Volume of spherical shells (periodic: shells within boxsize/2 are unaffected)
    V_shell = (4.0 / 3.0) * np.pi * (r2**3 - r1**3)
    
    # Expected number of random pairs (normalized by total volume)
    # For autocorrelation with periodic BC: RR = n * (n-1) / 2 * V_shell / V_box
    RR = n_rand * (n_rand - 1.0) / 2.0 * V_shell / volume
    
    # Landy-Szalay estimator: xi = (DD - 2*DR + RR) / RR
    # For auto-correlation with analytic RR: DD/RR - 1
    # Normalize DD: DD_normalized = DD / (n_gal * (n_gal - 1) / 2)
    # RR_normalized = RR / (n_rand * (n_rand - 1) / 2)
    DD_norm = npairs / (ngal * (ngal - 1.0) / 2.0)
    RR_norm = V_shell / volume
    
    # Avoid division by zero
    xi_vals = np.where(RR_norm > 0, DD_norm / RR_norm - 1.0, np.nan)
    
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
        
        # For periodic correlation functions, we need the full simulation box size
        # Positions are in absolute simulation coordinates
        # The simulation is P-Millennium L800 = 800 Mpc/h box
        # Infer from position extent with some margin (positions might not fill entire box)
        pos_max = float(np.max(pos))
        if pos_max > 600:  # Likely 800 Mpc/h box (L800)
            L = 800.0
        elif pos_max > 400:  # Likely 542 Mpc/h box (intermediate)
            L = 542.16  # Use exact max
        elif pos_max > 200:  # Likely 400 Mpc/h box (L400)
            L = 400.0
        else:
            # Small box or subvolume, use actual extent
            L = pos_max
        
        # V_ivol is subvolume volume, not the periodic box volume
        if V_ivol is None:
            V_ivol = L ** 3

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
        import traceback
        print(f"Warning: correlation could not be computed for {iz_path}/ivol{ivol}: {type(e).__name__}: {e}")
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


def correlations_given_redshifts_and_subvolume(
    iz_nums: List[int],
    ivol: int,
    rbins: Optional[np.ndarray] = None,
    nthreads: int = 4,
    base_dir: Optional[str] = None,
    centrals_only: bool = False,
    mhalo_min: Optional[float] = None,
) -> List[Dict[str, Any]]:
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
            res['iz'] = f'iz{iz_num}'
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
) -> Optional[Dict[str, Any]]:
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
        dict with keys: 'r', 'xi', 'xi_std', 'ivol', 'n_used', 'used_iz', 'used_z'
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
            iz_path,
            ivol,
            rbins=rbins,
            nthreads=nthreads,
            centrals_only=centrals_only,
            mhalo_min=mhalo_min,
        )

        if res is None:
            continue
        if r_ref is None:
            r_ref = res['r']
        per_xi.append(res['xi'])
        used_iz.append(f'iz{iz_num}')
        used_z.append(res.get('z'))

    if not per_xi:
        return None

    per_xi_arr = np.vstack(per_xi)
    r = r_ref if r_ref is not None else 0.5 * (rbins[1:] + rbins[:-1])

    return {
        'ivol': ivol,
        'r': r,
        'xi': per_xi_arr.mean(axis=0),
        'xi_std': per_xi_arr.std(axis=0),
        'n_used': per_xi_arr.shape[0],
        'used_iz': used_iz,
        'used_z': used_z,
    }