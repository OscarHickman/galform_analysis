"""Data loading utilities for GALFORM HDF5 outputs."""

import os
import re
import glob
import h5py
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any

from ..config import N_SUBVOLUMES


def get_completed_subvolumes(iz_path: str) -> List[int]:
    """Scan all subvolumes in a snapshot directory and return list of ivol numbers
    where CompletionFlag=1 in the galaxies.hdf5 file.
    
    This function pre-scans all subvolumes to check CompletionFlag, which can be used
    to get an accurate count before processing. For faster iteration, consider using
    glob.glob directly and handling failures with try/except during data reading.
    
    Args:
        iz_path: Path to the snapshot directory (e.g., 'iz99')
        
    Returns:
        List of ivol numbers that have CompletionFlag=1
    """
    ivol_dirs = sorted(glob.glob(os.path.join(iz_path, 'ivol*')))
    completed = []
    
    for ivol_dir in ivol_dirs:
        ivol_num = int(Path(ivol_dir).name.replace('ivol', ''))
        fpath = os.path.join(ivol_dir, 'galaxies.hdf5')
        
        if not os.path.exists(fpath) or not _is_hdf5_file(fpath):
            continue
            
        try:
            with h5py.File(fpath, 'r') as f:
                # Check for CompletionFlag
                if 'CompletionFlag' in f:
                    flag = f['CompletionFlag'][()]
                    if flag == 1:
                        completed.append(ivol_num)
        except Exception:
            continue
    
    return completed


def _is_hdf5_file(path: str) -> bool:
    """Check if a file is an HDF5 file by its signature."""
    try:
        with open(path, 'rb') as f:
            sig = f.read(8)
        return sig == b'\x89HDF\r\n\x1a\n'
    except Exception:
        return False


def open_galaxies_hdf5(iz_path: str, ivol: int = 0) -> Optional[h5py.File]:
    """Open a galaxies.hdf5 file, returning the h5py.File object or None.
    
    Args:
        iz_path: Path to the snapshot directory
        ivol: Subvolume number
        
    Returns:
        h5py.File object or None if file cannot be opened
    """
    fpath = os.path.join(iz_path, f"ivol{ivol}", "galaxies.hdf5")
    if not os.path.exists(fpath) or not _is_hdf5_file(fpath):
        return None
    try:
        return h5py.File(fpath, 'r')
    except OSError:
        return None


def get_output_group(f: Optional[h5py.File]) -> Optional[h5py.Group]:
    """Get the first 'Output' group from an HDF5 file.
    
    Args:
        f: HDF5 file object
        
    Returns:
        Output group or None if not found
    """
    if not f:
        return None
    outs = sorted([k for k in f.keys() if re.match(r'^Output\d+$', k)])
    return f[outs[0]] if outs else None


def _get_redshift_from_file(f: Optional[h5py.File]) -> Optional[float]:
    """Attempt to read redshift from 'Redshifts' or 'Output_Times' datasets."""
    if not f:
        return None
    try:
        if 'Redshifts' in f:
            obj = f['Redshifts']
            # Case A: dataset-like
            if isinstance(obj, h5py.Dataset):
                z0 = obj[0]
                if isinstance(z0, (bytes, np.bytes_)):
                    z0 = z0.decode('utf-8')
                return float(z0)
            # Case B: group with keys that are stringified redshifts
            if isinstance(obj, h5py.Group):
                vals = []
                for k in obj.keys():
                    try:
                        vals.append(float(k))
                    except Exception:
                        continue
                if vals:
                    # choose the smallest redshift value as a representative for this file
                    return float(sorted(vals)[0])
    except Exception:
        pass
    try:
        if 'Output_Times' in f:
            arr = np.array(f['Output_Times'])
            # Some files store strings like ['aout','nout',...], ignore non-numeric entries
            for x in arr.flat:
                try:
                    return float(x)
                except Exception:
                    continue
    except Exception:
        pass
    return None


def _get_redshift_from_zsnap(iz_path: str, ivol: int) -> Optional[float]:
    """Read redshift from a zsnap.dat file."""
    zfile = os.path.join(iz_path, f"ivol{ivol}", "zsnap.dat")
    if os.path.exists(zfile):
        try:
            with open(zfile, 'r') as f:
                line = f.readline().strip()
                return float(line)
        except Exception:
            return None
    return None


def _get_first_array(group: h5py.Group, candidates: List[str], 
                     default: Optional[np.ndarray] = None) -> np.ndarray:
    """Helper to robustly fetch arrays by trying multiple candidate keys."""
    for name in candidates:
        if name in group:
            try:
                return np.array(group[name])
            except Exception:
                continue
    return np.array([]) if default is None else default


def read_snapshot_data(iz_path: str, ivol: int = 0) -> Dict[str, Any]:
    """Read key galaxy properties from a single snapshot subvolume.
    
    Args:
        iz_path: Path to the snapshot directory
        ivol: Subvolume number
        
    Returns:
        Dictionary containing galaxy data with keys:
            - 'file': h5py.File object (needs to be closed!)
            - 'group': Output group
            - 'mstar': Stellar masses
            - 'mhalo': Halo masses
            - 'sfr': Star formation rates
            - 'Lg', 'Lr': g-band and r-band luminosities (if available)
            - 'z': Redshift
            - 'V_total': Total volume
            - 'V_ivol': Subvolume volume
            
    Raises:
        FileNotFoundError: If HDF5 file cannot be read
        RuntimeError: If no Output group found
    """
    f = open_galaxies_hdf5(iz_path, ivol=ivol)
    if f is None:
        raise FileNotFoundError(f"Unreadable or missing HDF5 for {iz_path}/ivol{ivol}")

    g = get_output_group(f)
    if g is None:
        f.close()
        raise RuntimeError("No OutputNNN group found")

    data = {'file': f, 'group': g, 'iz': Path(iz_path).name, 'ivol': ivol}

    # Stellar mass, halo mass, and SFR
    m_disk = _get_first_array(g, ['mstars_disk'])
    m_bulge = _get_first_array(g, ['mstars_bulge'])
    if m_disk.size and m_bulge.size:
        data['mstar'] = m_disk + m_bulge
    else:
        # Fallbacks if split masses are unavailable
        data['mstar'] = _get_first_array(g, ['mstars', 'StellarMass', 'Mstar', 'mstars_allburst'])

    data['mhalo'] = _get_first_array(g, ['mhalo', 'mchalo', 'Mhalo', 'M_Halo'])
    data['sfr'] = _get_first_array(g, ['mstardot', 'Sfr', 'sfr', 'sfr_disk'])

    # Band luminosities
    data['Lg'] = data['Lr'] = None
    if 'Bands' in f and 'bandname' in f['Bands']:
        names = [n.decode('utf-8') if isinstance(n, (bytes, np.bytes_)) else str(n)
                 for n in np.array(f['Bands']['bandname'])]
        
        def idx_for(label_candidates):
            for t in label_candidates:
                for i, nm in enumerate(names, start=1):
                    if t.lower() in nm.lower():
                        return i
            return None

        ig = idx_for(["sdss-g", "sdss g", " g ", "_g", "sdss_g"])
        ir = idx_for(["sdss-r", "sdss r", " r ", "_r", "sdss_r"])

        if ig is not None and 'Bands' in g:
            key_disk = f'Band{ig:03d}_Lum_Disk'
            key_bulge = f'Band{ig:03d}_Lum_Bulge'
            if key_disk in g['Bands'] and key_bulge in g['Bands']:
                data['Lg'] = np.array(g['Bands'][key_disk]) + np.array(g['Bands'][key_bulge])
        if ir is not None and 'Bands' in g:
            key_disk = f'Band{ir:03d}_Lum_Disk'
            key_bulge = f'Band{ir:03d}_Lum_Bulge'
            if key_disk in g['Bands'] and key_bulge in g['Bands']:
                data['Lr'] = np.array(g['Bands'][key_disk]) + np.array(g['Bands'][key_bulge])

    # Redshift
    data['z'] = _get_redshift_from_file(f) or _get_redshift_from_zsnap(iz_path, ivol)

    # Volume
    data['V_total'] = data['V_ivol'] = None
    if 'Parameters' in f and 'volume' in f['Parameters']:
        # The 'volume' parameter in GALFORM HDF5 files stores the PER-SUBVOLUME volume
        # NOT the total simulation box volume
        V_ivol = float(np.array(f['Parameters']['volume']))
        data['V_ivol'] = V_ivol
        
        # Calculate total volume by multiplying by number of subvolumes
        n_subvol = None
        if 'n_subvolumes' in f['Parameters']:
            n_subvol = int(np.array(f['Parameters']['n_subvolumes']))
        else:
            # Use the configured value from config.py
            n_subvol = N_SUBVOLUMES
        
        if n_subvol and n_subvol > 0:
            data['V_total'] = V_ivol * n_subvol
        else:
            # Fallback: if not specified, assume this is the total
            data['V_total'] = V_ivol

    return data


def close_snapshot(obj: Dict[str, Any]) -> None:
    """Safely close the HDF5 file associated with a snapshot data object.
    
    Args:
        obj: Dictionary returned by read_snapshot_data
    """
    try:
        if 'file' in obj and obj['file']:
            obj['file'].close()
    except Exception:
        pass
