"""Configuration module for galform_analysis.

This module manages paths and constants for GALFORM output analysis.
Set BASE_DIR to point to your GALFORM output directory before running analyses.
"""

import os
from pathlib import Path
import numpy as np

# ==============================================================================
# BASE DIRECTORY CONFIGURATION
# ==============================================================================

# Default base directory - override this or set via environment variable
_DEFAULT_BASE_DIR = '/cosma5/data/durham/dc-hick2/Galform_Out/L800/gp14'

# Check for environment variable override
BASE_DIR = os.environ.get('GALFORM_BASE_DIR', _DEFAULT_BASE_DIR)


def set_base_dir(path: str) -> None:
    """Set the base directory for GALFORM outputs.
    
    Args:
        path: Path to the GALFORM output directory
    """
    global BASE_DIR
    BASE_DIR = str(Path(path).resolve())


def get_base_dir() -> Path:
    """Get the current base directory as a Path object.
    
    Returns:
        Path object pointing to the base directory
    """
    return Path(BASE_DIR)


# ==============================================================================
# REDSHIFT MAPPING
# ==============================================================================

def load_redshift_mapping():
    """Load redshift mapping from redshift_list.txt.
    
    Returns:
        dict: Mapping from iz number (int) to redshift (float)
    """
    redshift_file = Path(__file__).parent / 'redshift_list.txt'
    z_map = {}
    
    if not redshift_file.exists():
        print(f"Warning: {redshift_file} not found. Redshift mapping unavailable.")
        return z_map
    
    with open(redshift_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    iz_num = int(parts[0])
                    z_val = float(parts[1])
                    z_map[iz_num] = z_val
                except ValueError:
                    continue
    
    return z_map


def get_snapshot_redshift(snapshot_name):
    """Get redshift for a snapshot name like 'iz99' or 'iz132'.
    
    Args:
        snapshot_name: Snapshot name (e.g., 'iz99')
    
    Returns:
        float or None: Redshift value, or None if not found
    """
    z_map = load_redshift_mapping()
    
    # Extract iz number from snapshot name
    import re
    match = re.search(r'iz(\d+)', snapshot_name)
    if match:
        iz_num = int(match.group(1))
        return z_map.get(iz_num)
    
    return None


def find_snapshot_at_redshift(target_z, tolerance=0.1):
    """Find the snapshot closest to a target redshift.
    
    Args:
        target_z: Target redshift
        tolerance: Maximum allowed difference
    
    Returns:
        str or None: Snapshot name (e.g., 'iz99'), or None if not found
    """
    z_map = load_redshift_mapping()
    
    best_match = None
    min_diff = float('inf')
    
    for iz_num, z_val in z_map.items():
        diff = abs(z_val - target_z)
        if diff < min_diff:
            min_diff = diff
            best_match = iz_num
    
    if min_diff <= tolerance and best_match is not None:
        return f'iz{best_match}'
    
    return None


# ==============================================================================
# COSMOLOGY PARAMETERS
# ==============================================================================

class Cosmology:
    """Cosmological parameters for the simulation."""
    
    OMEGA_M = 0.307
    OMEGA_L = 0.693
    OMEGA_B = 0.04825
    H0 = 67.77
    h = H0 / 100.0
    SIGMA_8 = 0.8288
    DELTA_C = 1.686
    F_B = OMEGA_B / OMEGA_M


# ==============================================================================
# ANALYSIS CONSTANTS
# ==============================================================================

# Simulation volume parameters
N_SUBVOLUMES = 1024  # Total number of subvolumes in the simulation

# Default binning for correlation functions
DEFAULT_RBINS = np.logspace(-1, 1.5, 21)  # Mpc

# SFR conversion factor
SFR_CONVERSION = 1.0  # Msun/yr per code unit

# Default mass bins for mass functions (log10 M_sun)
DEFAULT_STELLAR_MASS_BINS = np.arange(8.0, 12.6, 0.2)
DEFAULT_HALO_MASS_BINS = np.arange(10.0, 15.5, 0.2)

# Default sSFR bins (log10 yr^-1)
DEFAULT_SSFR_BINS = np.arange(-10.0, 5.0, 0.1)
