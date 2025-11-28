"""Analysis functions for aggregating GALFORM data across subvolumes."""

import os
import glob
import numpy as np
import h5py
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd


try:
    from ..io.loaders import read_snapshot_data, close_snapshot
    from galform_analysis.config import get_base_dir
except ImportError:
    # If running as script, use absolute imports
    import sys
    parent_dir = str(Path(__file__).resolve().parents[3])
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from galform_analysis.io.loaders import read_snapshot_data, close_snapshot
    from galform_analysis.config import get_base_dir

def completed_galaxies(basedir: str = get_base_dir()) -> pd.DataFrame:
    """Scan base directory and return DataFrame of all completed galaxy files.
    
    Looks through all iz*/ivol* directories and checks CompletionFlag in galaxies.hdf5 files.
    
    Args:
        basedir: Base directory containing iz* snapshot folders
        
    Returns:
        DataFrame with columns:
            - iz: Snapshot name (e.g., 'iz100')
            - iz_num: Numeric iz value (e.g., 100)
            - ivol: Subvolume number
            - path: Full path to the galaxies.hdf5 file
            - completed: Whether CompletionFlag==1
    """
    records = []
    
    # Find all iz* directories
    iz_dirs = sorted(glob.glob(os.path.join(basedir, 'iz*')))
    
    for iz_dir in iz_dirs:
        iz_name = Path(iz_dir).name
        iz_records = []  # Track records for this redshift only
        
        # Extract numeric iz value
        try:
            iz_num = int(iz_name.replace('iz', ''))
        except ValueError:
            continue
        
        ivol_dirs = sorted(glob.glob(os.path.join(iz_dir, 'ivol*')))
        
        for ivol_dir in ivol_dirs:
            ivol_name = Path(ivol_dir).name

            try:
                ivol_num = int(ivol_name.replace('ivol', ''))
            except ValueError:
                continue
            
            # Check for galaxies.hdf5 file
            gal_file = os.path.join(ivol_dir, 'galaxies.hdf5')
            
            if not os.path.exists(gal_file):
                continue
            
            # Quick file size check - empty or very small files are incomplete
            try:
                file_size = os.path.getsize(gal_file)
                if file_size < 1000:  # Less than 1KB is definitely incomplete
                    record = {
                        'iz': iz_name,
                        'iz_num': iz_num,
                        'ivol': ivol_num,
                        'path': gal_file,
                        'completed': False
                    }
                    records.append(record)
                    iz_records.append(record)
                    continue
            except OSError:
                continue
            
            # Try to open the file - if it fails with serialization error, it's incomplete
            completed = False
            
            try:
                # Use swmr mode for faster read access
                with h5py.File(gal_file, 'r', swmr=True):
                    # If we can open it without error, it's completed
                    completed = True
            except (OSError, KeyError, RuntimeError) as e:
                # Check if it's the specific serialization error indicating incomplete file
                if "Can't deserialize" in str(e) or "bad object header" in str(e):
                    completed = False
                else:
                    # Other errors might be temporary, but mark as incomplete
                    completed = False
            
            record = {
                'iz': iz_name,
                'iz_num': iz_num,
                'ivol': ivol_num,
                'path': gal_file,
                'completed': completed
            }
            records.append(record)
            iz_records.append(record)
        completed_count = sum(r['completed'] for r in iz_records)
        checked_count = len(iz_records)
        total_ivol_dirs = len(ivol_dirs)
        skipped_missing = total_ivol_dirs - checked_count
        print(
            f"{iz_name}: found {total_ivol_dirs} ivol dirs; "
            f"checked {checked_count} galaxies.hdf5; "
            f"completed {completed_count}/{checked_count}; "
            f"skipped {skipped_missing} without galaxies.hdf5"
        )    
    
    df = pd.DataFrame(records)
    
    # Sort by iz_num and ivol
    if not df.empty:
        df = df.sort_values(['iz_num', 'ivol']).reset_index(drop=True)
    
    return df

def aggregate_snapshot(iz_path: str) -> Optional[Dict[str, Any]]:
    """Aggregate mstar, mhalo, and volume from all ivols in a snapshot.
    
    Args:
        iz_path: Path to the snapshot directory
        
    Returns:
        Dictionary with keys: 'iz', 'z', 'volume', 'mstar', 'mhalo'
        Returns None if no data found
    """
    ivol_paths = sorted(glob.glob(os.path.join(iz_path, 'ivol*')))
    if not ivol_paths:
        return None
    
    all_mstar, all_mhalo = [], []
    total_vol = 0
    z = None

    for ivp in ivol_paths:
        iv = int(Path(ivp).name.replace('ivol', ''))
        try:
            data = read_snapshot_data(iz_path, ivol=iv)
            if data.get('V_ivol') and data['V_ivol'] > 0:
                total_vol += data['V_ivol']
            if z is None:
                z = data.get('z')
            
            mstar = data.get('mstar')
            mhalo = data.get('mhalo')
            if mstar is not None:
                all_mstar.append(mstar)
            if mhalo is not None:
                all_mhalo.append(mhalo)

            close_snapshot(data)
        except Exception:
            continue
            
    if not all_mstar and not all_mhalo:
        return None

    return {
        'iz': Path(iz_path).name,
        'z': z,
        'volume': total_vol,
        'mstar': np.concatenate(all_mstar) if all_mstar else np.array([]),
        'mhalo': np.concatenate(all_mhalo) if all_mhalo else np.array([])
    }


if __name__ == "__main__":
    # import sys
    # from pathlib import Path
    # parent_dir = str(Path(__file__).resolve().parents[3])
    # if parent_dir not in sys.path:
    #     sys.path.insert(0, parent_dir)
    from galform_analysis.config import get_base_dir
    
    base_dir = get_base_dir()
    df = completed_galaxies(str(base_dir))