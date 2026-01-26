import os
from typing import Optional, Tuple, List

import numpy as np
from Corrfunc.theory.xi import xi as corrfunc_xi

import pandas as pd

from ...config import DEFAULT_RBINS, get_base_dir
from ...io.loaders import read_snapshot_data
from ...utils.read_galaxies import read_galaxy_positions, read_halo_positions


def _load_positions_from_hdf5(
    iz_path: str, 
    ivol: int,
    centrals_only: bool = True,
    mhalo_min: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[float]]:
    """Load galaxy positions (x,y,z) and redshift from an HDF5 subvolume.

    Always uses central galaxies only (is_central=1).
    The centrals_only parameter is kept for API compatibility but is always enforced as True.

    Args:
        iz_path: Path to snapshot directory
        ivol: Subvolume number
        centrals_only: Kept for API compatibility, always enforced as True
        mhalo_min: Minimum halo mass (mhalo) threshold in Msun. None = no cut.

    Returns:
        positions: (N,3) array in the native units of the file (assumed Mpc or Mpc/h)
        z: best-effort redshift if available
    """
    return read_galaxy_positions(
        iz_path=iz_path,
        ivol=ivol,
        centrals_only=True,  # Always True
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
    # Filter bin edges but ensure we keep at least 2 edges to form 1+ bins
    rmax_periodic = boxsize / 2.0
    rbins = rbins[rbins <= rmax_periodic]
    
    if len(rbins) < 2:
        raise ValueError(f"No valid rbins within periodic limit (rmax={rmax_periodic:.2f} Mpc/h). Cannot compute correlation.")

    
    ngal = positions.shape[0]
    if ngal < 2:
        # Not enough galaxies for correlation
        r_centers = 0.5 * (rbins[:-1] + rbins[1:])
        df = pd.DataFrame({
            'r': r_centers,
            'xi': np.full_like(r_centers, np.nan),
        })
        df.attrs.update({'rbins': rbins, 'ngal': ngal})
        return df

    # Use Corrfunc's xi calculator for periodic boxes to avoid manual normalization bugs
    # corrfunc_xi applies the Landy-Szalay estimator internally and returns xi directly.
    results = corrfunc_xi(
        boxsize=boxsize,
        nthreads=nthreads,
        binfile=rbins,
        X=positions[:, 0],
        Y=positions[:, 1],
        Z=positions[:, 2],
        output_ravg=True,
    )

    ravg = np.array([x['ravg'] for x in results], dtype=np.float64)
    xi_vals = np.array([x['xi'] for x in results], dtype=np.float64)

    # Use ravg if available, otherwise fall back to bin centers
    if np.all(np.isfinite(ravg) & (ravg > 0)):
        r = ravg
    else:
        r = 0.5 * (rbins[:-1] + rbins[1:])

    df = pd.DataFrame({'r': r, 'xi': xi_vals})
    df.attrs.update({'rbins': rbins, 'ngal': ngal})
    return df


def correlation_given_redshift_and_subvolume(
    iz_path: str,
    ivol: int,
    rbins: Optional[np.ndarray] = None,
    nthreads: int = 4,
    centrals_only: bool = True,
    mhalo_min: Optional[float] = None,
) -> Optional[pd.DataFrame]:
    """High-level helper mirroring the HMF API: xi(r) for (snapshot, ivol).

    Always uses central galaxies only (is_central=1).
    The centrals_only parameter is kept for API compatibility but is always enforced as True.

    Args:
        iz_path: Path to snapshot directory (e.g., str(get_base_dir()/"iz207"))
        ivol: Subvolume number
        rbins: Radial bin edges (Mpc). Defaults to config.DEFAULT_RBINS
        nthreads: Number of OpenMP threads for Corrfunc
        centrals_only: Kept for API compatibility, always enforced as True
        mhalo_min: Minimum halo mass (mhalo) in Msun. None = no cut.

    Returns:
        DataFrame with columns ['r', 'xi'] and metadata in df.attrs.
        Returns None if unavailable.
    """
    try:
        # Load positions and redshift (always central galaxies only)
        pos, z_val = _load_positions_from_hdf5(iz_path, ivol, centrals_only=True, mhalo_min=mhalo_min)

        # Get subvolume metadata
        meta = read_snapshot_data(iz_path, ivol)
        V_ivol = meta.get('V_ivol', None)
        
        # CRITICAL: Each subvolume is an INDEPENDENT REALIZATION of the full simulation box.
        # Positions are stored in full box coordinates (e.g., 0-542 Mpc/h for a 542³ box).
        # V_ivol represents the statistical volume (number of such realizations × full box volume),
        # NOT the size of a spatial tile.
        #
        # For correlation function calculation:
        # - Use the position range to infer the actual periodic box size
        # - Each subvolume spans the full box (they're overlapping realizations)
        
        # Infer box size from position extent
        extent = np.ptp(pos, axis=0)  # Range in each dimension
        L = float(np.max(extent))
        
        # Sanity check: positions should start near 0
        pos_min = np.min(pos, axis=0)
        if not np.all(pos_min >= -1.0):  # Allow small numerical errors
            # Shift to [0, L) if needed
            pos = pos - pos_min
        
        # Final wrap to handle any edge cases
        pos = np.fmod(pos, L)
        pos = np.where(pos < 0, pos + L, pos)

        if not np.isfinite(L) or L <= 0:
            raise RuntimeError(f"Invalid box size for {iz_path}/ivol{ivol}: L={L}")

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
    mhhalo_min: Optional[float] = None,
) -> Optional[pd.DataFrame]:
    """Compute dark matter halo correlation function directly from GALFORM halo positions.

    DM halos are represented by central galaxies of main halos (is_central=1, ihhalo=1)
    from the galaxies.hdf5 file. Uses host halo mass (mhhalo) for filtering.

    Args:
        iz_path: Path to snapshot directory
        ivol: Subvolume number
        rbins: Radial bin edges (Mpc/h). Defaults to DEFAULT_RBINS
        nthreads: Number of OpenMP threads for Corrfunc
        mhhalo_min: Optional minimum host halo mass cut in Msun

    Returns:
        DataFrame with columns ['r', 'xi'] and metadata in df.attrs.
        Returns None if computation fails.
    """
    try:
        # Load DM halo positions from GALFORM galaxies.hdf5
        # Uses centrals of main halos as halo representatives
        pos, z_val = read_halo_positions(iz_path, ivol, mhhalo_min=mhhalo_min)

        # Get subvolume metadata
        meta = read_snapshot_data(iz_path, ivol)
        V_ivol = meta.get('V_ivol', None)
        
        # Infer box size from position extent (same logic as galaxy correlation)
        extent = np.ptp(pos, axis=0)
        L = float(np.max(extent))
        
        # Ensure positions are in [0, L) range
        pos_min = np.min(pos, axis=0)
        if not np.all(pos_min >= -1.0):
            pos = pos - pos_min
        
        pos = np.fmod(pos, L)
        pos = np.where(pos < 0, pos + L, pos)

        if not np.isfinite(L) or L <= 0:
            raise RuntimeError(f"Invalid box size for {iz_path}/ivol{ivol}: L={L}")

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
    """Compute 2PCF by combining galaxies from multiple subvolumes into one box.

    CRITICAL: Subvolumes are overlapping realizations of the SAME spatial volume.
    Each subvolume samples 1/1024 of the galaxy population in the same spatial box.
    When combining N subvolumes, we get N/1024 of the full population in the SAME box.
    
    Correct approach: Combine ALL galaxy positions from multiple subvolumes into
    a single box, then compute xi(r) once on this denser population. This gives
    the correlation function for a population N times denser than a single subvolume.

    Args:
        iz_num: Numeric snapshot identifier (e.g. 207 for 'iz207').
        ivols: List of subvolume indices to combine.
        rbins: Optional radial bin edges (defaults to DEFAULT_RBINS).
        nthreads: Number of OpenMP threads for Corrfunc.
        base_dir: Optional base directory; defaults to configured base dir.
        centrals_only: If True, only include central galaxies (is_central=1)
        mhalo_min: Minimum halo mass (mhalo) in Msun. None = no cut.
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

    # Combine all galaxy positions from multiple subvolumes
    all_positions = []
    z = None
    V_ivol = None
    L_box = None
    
    for iv in ivols:
        try:
            # Load positions and metadata for this subvolume
            pos, z_val = _load_positions_from_hdf5(iz_path, iv, centrals_only=True, mhalo_min=mhalo_min)
            meta = read_snapshot_data(iz_path, iv)
            
            if z is None:
                z = z_val if z_val is not None else meta.get('z')
            if V_ivol is None:
                V_ivol = meta.get('V_ivol')
            
            # Infer box size from first subvolume
            if L_box is None:
                extent = np.ptp(pos, axis=0)
                L_box = float(np.max(extent))
                # Ensure positions are wrapped to [0, L)
                pos_min = np.min(pos, axis=0)
                if not np.all(pos_min >= -1.0):
                    pos = pos - pos_min
                pos = np.fmod(pos, L_box)
                pos = np.where(pos < 0, pos + L_box, pos)
            else:
                # Wrap positions for consistency
                pos_min = np.min(pos, axis=0)
                if not np.all(pos_min >= -1.0):
                    pos = pos - pos_min
                pos = np.fmod(pos, L_box)
                pos = np.where(pos < 0, pos + L_box, pos)
            
            all_positions.append(pos)
            
        except (FileNotFoundError, RuntimeError, KeyError) as e:
            continue
    
    if not all_positions:
        return None
    
    # Combine all positions into one dataset (same box, more galaxies)
    combined_positions = np.vstack(all_positions)
    total_galaxies = combined_positions.shape[0]
    
    if not np.isfinite(L_box) or L_box <= 0:
        return None
    
    # Compute xi(r) on the combined population
    res = compute_xi_corrfunc(combined_positions, boxsize=L_box, rbins=rbins, nthreads=nthreads)
    
    # Update metadata
    res.attrs['z'] = z
    res.attrs['iz'] = f'iz{iz_num}'
    res.attrs['V_ivol'] = V_ivol
    res.attrs['boxsize'] = L_box
    res.attrs['n_used'] = len(all_positions)
    res.attrs['n_ivols'] = len(all_positions)
    res.attrs['total_galaxies'] = total_galaxies
    res.attrs['rbins'] = rbins
    res.attrs['method'] = 'combined_overlapping_subvolumes'
    
    return res


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