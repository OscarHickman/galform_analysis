"""Analysis functions for aggregating GALFORM data across subvolumes."""

import os
import glob
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

from ..io.loaders import read_snapshot_data, close_snapshot


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
