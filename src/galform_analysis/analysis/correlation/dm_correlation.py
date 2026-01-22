"""
Dark matter halo two-point correlation function calculations.

This module computes the 2PCF for dark matter halos from merger tree files,
mirroring the API of correlation.py but for DM halos instead of galaxies.
"""

import os
from typing import Dict, Optional, Tuple, List, Any
import warnings

import numpy as np
import h5py
from Corrfunc.theory.DD import DD as corrfunc_DD

from ...config import DEFAULT_RBINS


def _normalize_positions_units(
    positions: np.ndarray,
    boxsize: Optional[float],
    unit_hint: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[float]]:
    """Normalize halo positions to Mpc/h when tree files are in kpc/h.

    Heuristic: if boxsize is very large (e.g., > 10,000), assume kpc/h and
    convert to Mpc/h by dividing by 1000. This matches P-Millennium trees
    that often store positions in kpc/h with Lbox ~ 800000.
    """
    if boxsize is None:
        return positions, boxsize

    # Explicit unit hint if provided by file attributes
    if unit_hint is not None:
        hint = unit_hint.lower()
        if 'kpc' in hint:
            return positions / 1000.0, boxsize / 1000.0
        if 'mpc' in hint:
            return positions, boxsize

    # Heuristic based on boxsize magnitude
    if boxsize > 1.0e4:
        warnings.warn(
            f"Large boxsize detected ({boxsize:.1f}); assuming kpc/h and converting to Mpc/h.")
        return positions / 1000.0, boxsize / 1000.0

    return positions, boxsize


def _extract_length_unit_hint(f: h5py.File) -> Optional[str]:
    """Attempt to extract a length unit hint from common HDF5 locations."""
    # Check units group attributes or datasets
    if 'units' in f:
        units_group = f['units']
        for key in ('LengthUnit', 'lengthUnit', 'length', 'Length', 'posUnit', 'PositionUnit'):
            if key in units_group.attrs:
                return str(units_group.attrs[key])
            if key in units_group:
                try:
                    return str(np.asarray(units_group[key]).ravel()[0])
                except Exception:
                    continue

    # Check file-level attributes
    for key in ('LengthUnit', 'lengthUnit', 'length', 'Length', 'posUnit', 'PositionUnit'):
        if key in f.attrs:
            return str(f.attrs[key])

    return None


def _load_aquarius_format(
    f: h5py.File,
    snapshot_idx: Optional[int] = None,
    mhalo_min: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[float], Optional[float], Optional[str]]:
    """Load halos from AQUARIUS-style P-Millennium tree format.
    
    Args:
        f: Open HDF5 file handle
        snapshot_idx: Snapshot to extract (if None, uses final snapshot from treeIndex)
        mhalo_min: Minimum particle count (proxy for mass)
    
    Returns:
        positions, boxsize, redshift
    """
    halo_trees = f['haloTrees']
    
    # Get snapshot information
    if snapshot_idx is None:
        # Use final snapshot from trees
        tree_index = f['treeIndex']
        final_snaps = np.asarray(tree_index['finalSnapshot'])
        snapshot_idx = int(np.max(final_snaps))
    
    # Filter for halos at the desired snapshot
    snapshot_numbers = np.asarray(halo_trees['snapshotNumber'])
    mask = (snapshot_numbers == snapshot_idx)
    
    # Further filter for FoF centres (primary halos, not subhalos)
    if 'isFoFCentre' in halo_trees:
        is_fof_centre = np.asarray(halo_trees['isFoFCentre'])
        mask &= (is_fof_centre == 1)
    
    # Apply particle count cut as proxy for mass if requested
    if mhalo_min is not None and 'np' in halo_trees:
        np_particles = np.asarray(halo_trees['np'])
        # Rough conversion: assume particle mass ~1e9 Msun for P-Millennium
        particle_mass = 1.06e9  # Msun/h for P-Millennium
        min_particles = int(mhalo_min / particle_mass)
        mask &= (np_particles >= min_particles)
    
    # Load positions for selected halos
    positions = np.asarray(halo_trees['position'])[mask]
    
    # Get box size
    boxsize = None
    if 'simulation' in f and 'boxSize' in f['simulation'].attrs:
        boxsize = float(f['simulation'].attrs['boxSize'])
    elif 'simulation' in f and 'Lbox' in f['simulation'].attrs:
        boxsize = float(f['simulation'].attrs['Lbox'])
    elif 'simulation' in f and 'boxSize' in f['simulation']:
        boxsize = float(np.asarray(f['simulation']['boxSize']).ravel()[0])
    elif 'simulation' in f and 'Lbox' in f['simulation']:
        boxsize = float(np.asarray(f['simulation']['Lbox']).ravel()[0])
    else:
        # Estimate from position extent
        extent = np.ptp(positions, axis=0)
        boxsize = float(np.max(extent))
        warnings.warn(f"BoxSize not in file attributes, estimated: {boxsize:.1f}")
    
    # Get redshift
    redshift = None
    if 'outputTimes' in f:
        output_times = f['outputTimes']
        snap_nums = np.asarray(output_times['snapshotNumber'])
        redshifts = np.asarray(output_times['redshift'])
        idx = np.where(snap_nums == snapshot_idx)[0]
        if len(idx) > 0:
            redshift = float(redshifts[idx[0]])
    
    unit_hint = _extract_length_unit_hint(f)
    return positions.astype(np.float64), boxsize, redshift, unit_hint


def _load_halo_positions_from_hdf5(
    tree_file: str,
    snapshot_idx: Optional[int] = None,
    mhalo_min: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[float], Optional[float], Optional[str]]:
    """Load dark matter halo positions from HDF5 merger tree file.
    
    Supports AQUARIUS-style P-Millennium trees and generic HDF5 formats.
    
    Args:
        tree_file: Path to HDF5 merger tree file
        snapshot_idx: Snapshot index to load (if None, uses final snapshot)
        mhalo_min: Minimum halo mass threshold in Msun. None = no cut.
    
    Returns:
        positions: (N,3) array of halo positions in Mpc/h
        boxsize: Simulation box size in Mpc/h
        redshift: Redshift of the snapshot
    """
    if not os.path.exists(tree_file):
        raise FileNotFoundError(f"Merger tree file not found: {tree_file}")
    
    try:
        with h5py.File(tree_file, 'r') as f:
            # Check for AQUARIUS-style P-Millennium format
            if 'haloTrees' in f and 'position' in f['haloTrees']:
                return _load_aquarius_format(f, snapshot_idx, mhalo_min)
            
            # Try common HDF5 structures for merger trees
            pos_fields = ['Pos', 'Position', 'position', 'x', 'coordinates']
            pos = None
            
            # Check in root or common group names
            search_groups = [f, f.get('Halos'), f.get('Subhalos'), f.get('Trees'), f.get('haloTrees')]
            search_groups = [g for g in search_groups if g is not None]
            
            for group in search_groups:
                for field in pos_fields:
                    if field in group:
                        pos_data = np.asarray(group[field])
                        if pos_data.ndim == 2 and pos_data.shape[1] == 3:
                            pos = pos_data
                            break
                        elif pos_data.ndim == 1:
                            # May need to combine x, y, z arrays
                            continue
                if pos is not None:
                    break
            
            # Try separate x, y, z arrays if combined position not found
            if pos is None:
                for group in search_groups:
                    if all(k in group for k in ['x', 'y', 'z']):
                        x = np.asarray(group['x']).ravel()
                        y = np.asarray(group['y']).ravel()
                        z = np.asarray(group['z']).ravel()
                        n = min(len(x), len(y), len(z))
                        pos = np.vstack([x[:n], y[:n], z[:n]]).T
                        break
            
            if pos is None:
                # List available datasets to help debugging
                available = []
                for group in search_groups:
                    if hasattr(group, 'keys'):
                        available.extend(list(group.keys())[:10])
                raise KeyError(
                    f"Could not find halo position arrays in {tree_file}. "
                    f"Searched for: {pos_fields}. Available fields: {available}"
                )
            
            # Apply mass cut if requested
            if mhalo_min is not None:
                mass_fields = ['Mvir', 'M200', 'Mhalo', 'mass', 'M_Crit200', 'np']
                mass = None
                for group in search_groups:
                    for field in mass_fields:
                        if field in group:
                            mass = np.asarray(group[field]).ravel()
                            break
                    if mass is not None:
                        break
                
                if mass is not None:
                    mask = mass[:len(pos)] >= mhalo_min
                    pos = pos[mask]
                else:
                warnings.warn("Could not find mass field for mass cut. Proceeding without cut.")
            boxsize = None
            boxsize_fields = ['BoxSize', 'boxsize', 'Lbox', 'L']
            for field in boxsize_fields:
                if field in f.attrs:
                    boxsize = float(f.attrs[field])
                    break
                elif field in f:
                    boxsize = float(np.asarray(f[field]))
                    break
            
            # If boxsize not found, estimate from position extent
            if boxsize is None:
                extent = np.ptp(pos, axis=0)
                boxsize = float(np.max(extent))
                warnings.warn(f"BoxSize not found in file, estimated from extent: {boxsize:.1f}")
            
            # Try to get redshift
            redshift = None
            redshift_fields = ['Redshift', 'redshift', 'z', 'Redshifts']
            for field in redshift_fields:
                if field in f.attrs:
                    redshift = float(f.attrs[field])
                    break
                elif field in f:
                    z_data = np.asarray(f[field])
                    if z_data.size > 0:
                        redshift = float(z_data.ravel()[0])
                        break
            
            unit_hint = _extract_length_unit_hint(f)
            return pos.astype(np.float64, copy=False), boxsize, redshift, unit_hint
            
    except Exception as e:
        raise RuntimeError(f"Error loading halo positions from {tree_file}: {e}") from e


def _load_halo_positions_from_binary(
    tree_file: str,
    snapshot_idx: Optional[int] = None,
    mhalo_min: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[float], Optional[float], Optional[str]]:
    """Load dark matter halo positions from binary merger tree files (Millennium format).
    
    This is a placeholder for binary tree formats. Implementation depends on
    the specific binary format used (e.g., old Millennium trees).
    
    Args:
        tree_file: Path to binary merger tree file
        snapshot_idx: Snapshot index to load
        mhalo_min: Minimum halo mass threshold in Msun
    
    Returns:
        positions: (N,3) array of halo positions
        boxsize: Simulation box size
        redshift: Redshift of snapshot
    """
    raise NotImplementedError(
        "Binary merger tree format not yet implemented. "
        "Please provide HDF5 format trees or implement binary reader for your format."
    )


def compute_xi_corrfunc(
    positions: np.ndarray,
    boxsize: float,
    rbins: Optional[np.ndarray] = None,
    nthreads: int = 4,
) -> Dict[str, np.ndarray]:
    """Compute the real-space two-point correlation xi(r) for dark matter halos.
    
    Uses Corrfunc.theory.DD with periodic boundary conditions and Landy-Szalay
    estimator with analytic random pair counts.
    
    Args:
        positions: (N,3) array with halo coordinates (Mpc/h)
        boxsize: Side length of the simulation box (Mpc/h)
        rbins: Radial bin edges (Mpc/h). Defaults to config.DEFAULT_RBINS
        nthreads: Number of OpenMP threads for parallel execution
    
    Returns:
        dict with keys: 'rbins', 'r', 'xi', 'nhalo'
    """
    if rbins is None:
        rbins = DEFAULT_RBINS
    rbins = np.asarray(rbins, dtype=float)
    
    # For periodic geometry, rmax must be < boxsize/2 to avoid double-counting
    rmax_periodic = boxsize / 2.0
    rbins = rbins[rbins < rmax_periodic]
    
    if len(rbins) < 2:
        raise ValueError(
            f"rbins exceed periodic limit (rmax={rmax_periodic:.2f}). "
            "Cannot compute correlation."
        )
    
    nhalo = positions.shape[0]
    if nhalo < 2:
        # Not enough halos for correlation
        r_centers = 0.5 * (rbins[:-1] + rbins[1:])
        return {
            'rbins': rbins,
            'r': r_centers,
            'xi': np.full_like(r_centers, np.nan),
            'nhalo': nhalo,
        }
    
    # Ensure positions are in [0, boxsize)
    pos_wrapped = np.fmod(positions, boxsize)
    pos_wrapped = np.where(pos_wrapped < 0, pos_wrapped + boxsize, pos_wrapped)
    
    # Use DD with periodic boundary conditions
    results = corrfunc_DD(
        autocorr=1,
        nthreads=nthreads,
        binfile=rbins,
        X1=pos_wrapped[:, 0],
        Y1=pos_wrapped[:, 1],
        Z1=pos_wrapped[:, 2],
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
    
    # Compute RR analytically for periodic cubic volume
    volume = boxsize ** 3
    # n_rand = nhalo  # Not used - RR computed analytically
    
    r1 = rbins[:-1]
    r2 = rbins[1:]
    V_shell = (4.0 / 3.0) * np.pi * (r2**3 - r1**3)
    
    # Landy-Szalay estimator: (DD - 2*DR + RR) / RR
    # For auto-correlation with analytic RR: DD/RR - 1
    DD_norm = npairs / (nhalo * (nhalo - 1.0) / 2.0)
    RR_norm = V_shell / volume
    
    # Avoid division by zero
    xi_vals = np.where(RR_norm > 0, DD_norm / RR_norm - 1.0, np.nan)
    
    return {
        'rbins': rbins,
        'r': r,
        'xi': xi_vals,
        'nhalo': nhalo,
    }


def dm_correlation_from_tree_file(
    tree_file: str,
    rbins: Optional[np.ndarray] = None,
    nthreads: int = 4,
    snapshot_idx: Optional[int] = None,
    mhalo_min: Optional[float] = None,
    file_format: str = 'auto',
    boxsize_override: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Compute DM halo 2PCF from a merger tree file.
    
    High-level function mirroring correlation_given_redshift_and_subvolume API.
    
    Args:
        tree_file: Path to merger tree file (HDF5 or binary)
        rbins: Radial bin edges (Mpc/h). Defaults to DEFAULT_RBINS
        nthreads: Number of OpenMP threads for Corrfunc
        snapshot_idx: Snapshot index (if needed for multi-snapshot files)
        mhalo_min: Minimum halo mass in Msun. None = no cut.
        file_format: 'hdf5', 'binary', or 'auto' (auto-detect from extension)
        boxsize_override: If set, use this boxsize (Mpc/h) instead of the tree file boxsize
    
    Returns:
        dict with 'r', 'xi', 'z', 'boxsize', 'nhalo'; or None if unavailable
    """
    try:
        # Auto-detect format
        if file_format == 'auto':
            if tree_file.endswith(('.hdf5', '.h5')):
                file_format = 'hdf5'
            else:
                file_format = 'binary'
        
        # Load halo positions
        if file_format == 'hdf5':
            pos, boxsize, redshift, unit_hint = _load_halo_positions_from_hdf5(
                tree_file, snapshot_idx, mhalo_min
            )
        elif file_format == 'binary':
            pos, boxsize, redshift, unit_hint = _load_halo_positions_from_binary(
                tree_file, snapshot_idx, mhalo_min
            )
        else:
            raise ValueError(f"Unknown file_format: {file_format}")
        
        # Normalize units if necessary (kpc/h -> Mpc/h)
        pos, boxsize = _normalize_positions_units(pos, boxsize, unit_hint)

        # Override boxsize if provided (e.g., use subvolume size)
        if boxsize_override is not None and boxsize_override > 0:
            extent = np.ptp(pos, axis=0)
            extent_max = float(np.max(extent))
            if extent_max > boxsize_override * 1.2:
                warnings.warn(
                    f"Positions span ~{extent_max:.1f} which exceeds override boxsize "
                    f"{boxsize_override:.1f}. Check tree file/subvolume mapping."
                )
            boxsize = float(boxsize_override)

        # If positions occupy a much smaller region than boxsize, assume subvolume tree
        if boxsize is not None and boxsize > 0:
            extent = np.ptp(pos, axis=0)
            extent_max = float(np.max(extent))
            if extent_max > 0 and boxsize / extent_max > 1.5:
                warnings.warn(
                    f"Positions span only ~{extent_max:.1f} of boxsize {boxsize:.1f}; "
                    "using extent as boxsize for subvolume tree."
                )
                boxsize = extent_max

        if boxsize is None or boxsize <= 0:
            raise RuntimeError(f"Invalid box size: {boxsize}")
        
        # Compute correlation function
        res = compute_xi_corrfunc(pos, boxsize=boxsize, rbins=rbins, nthreads=nthreads)
        
        out = {
            'r': res['r'],
            'xi': res['xi'],
            'rbins': res['rbins'],
            'nhalo': res['nhalo'],
            'z': redshift,
            'boxsize': boxsize,
            'tree_file': tree_file,
        }
        return out
        
    except (FileNotFoundError, RuntimeError, KeyError, NotImplementedError) as e:
        import traceback
        print(f"Warning: DM correlation could not be computed from {tree_file}: "
              f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return None


def avg_dm_correlation_from_tree_files(
    tree_files: List[str],
    rbins: Optional[np.ndarray] = None,
    nthreads: int = 4,
    snapshot_idx: Optional[int] = None,
    mhalo_min: Optional[float] = None,
    file_format: str = 'auto',
) -> Optional[Dict[str, Any]]:
    """Average DM halo 2PCF over multiple merger tree files.
    
    Mirrors avg_correlation_given_redshift_and_subvolumes API.
    
    Args:
        tree_files: List of merger tree file paths
        rbins: Radial bin edges. Defaults to DEFAULT_RBINS
        nthreads: Number of OpenMP threads
        snapshot_idx: Snapshot index (if multi-snapshot files)
        mhalo_min: Minimum halo mass in Msun
        file_format: 'hdf5', 'binary', or 'auto'
    
    Returns:
        Dictionary with keys:
            - 'r': radial bin centers
            - 'xi': mean correlation function
            - 'xi_std': standard deviation
            - 'z': redshift (from first file)
            - 'n_used': number of successful files
            - 'n_requested': total files requested
        Returns None if no file produced valid data.
    """
    if rbins is None:
        rbins = DEFAULT_RBINS
    
    results = []
    redshift = None
    
    for tree_file in tree_files:
        res = dm_correlation_from_tree_file(
            tree_file,
            rbins=rbins,
            nthreads=nthreads,
            snapshot_idx=snapshot_idx,
            mhalo_min=mhalo_min,
            file_format=file_format,
        )
        if res is not None:
            results.append(res)
            if redshift is None and res.get('z') is not None:
                redshift = res['z']
    
    if not results:
        print("Warning: No successful DM correlation calculations.")
        return None
    
    # Stack xi values and compute mean/std
    xi_stack = np.array([r['xi'] for r in results])
    r_mean = results[0]['r']  # Assume all use same bins
    xi_mean = np.nanmean(xi_stack, axis=0)
    xi_std = np.nanstd(xi_stack, axis=0)
    
    return {
        'r': r_mean,
        'xi': xi_mean,
        'xi_std': xi_std,
        'z': redshift,
        'n_used': len(results),
        'n_requested': len(tree_files),
        'rbins': results[0]['rbins'],
    }
