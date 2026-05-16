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

_REDSHIFT_LISTS_DIR = Path(__file__).parent / 'redshift_lists'


def load_redshift_mapping(sim_name: str) -> dict[int, float]:
    """Load redshift mapping for the given N-body simulation.

    Args:
        sim_name: Simulation name (e.g. 'L800', 'Mill1', 'Mill2').
                  Must match a file in src/redshift_lists/<sim_name>.txt.

    Returns:
        dict: Mapping from iz number (int) to redshift (float)
    """
    redshift_file = _REDSHIFT_LISTS_DIR / f'{sim_name}.txt'

    if not redshift_file.exists():
        available = [p.stem for p in _REDSHIFT_LISTS_DIR.glob('*.txt')]
        raise FileNotFoundError(
            f"No redshift list for simulation '{sim_name}'. "
            f"Available: {available}. "
            f"Add {redshift_file} to register a new simulation."
        )

    z_map = {}
    with open(redshift_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    z_map[int(parts[0])] = float(parts[1])
                except ValueError:
                    continue

    return z_map


def get_snapshot_redshift(snapshot_name: str, sim_name: str) -> float | None:
    """Get redshift for a snapshot name like 'iz99' or 'iz132'.

    Args:
        snapshot_name: Snapshot name (e.g., 'iz99')
        sim_name: N-body simulation name (e.g. 'L800', 'Mill1', 'Mill2')

    Returns:
        float or None: Redshift value, or None if not found
    """
    import re
    if not snapshot_name.startswith('iz'):
        snapshot_name = f'iz{snapshot_name}'
    z_map = load_redshift_mapping(sim_name)
    match = re.search(r'iz(\d+)', snapshot_name)
    if match:
        return z_map.get(int(match.group(1)))
    return None


def find_snapshot_at_redshift(target_z: float, sim_name: str, tolerance: float = 0.1) -> str | None:
    """Find the snapshot closest to a target redshift.

    Args:
        target_z: Target redshift
        sim_name: N-body simulation name (e.g. 'L800', 'Mill1', 'Mill2')
        tolerance: Maximum allowed difference

    Returns:
        str or None: Snapshot name (e.g., 'iz99'), or None if not found
    """
    z_map = load_redshift_mapping(sim_name)

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

# Default mass bins for mass functions
# GALFORM stores halo/stellar masses in M_sun/h.
# log10(M) bins here are log10(M_sun/h).
DEFAULT_STELLAR_MASS_BINS = np.arange(8.0, 12.6, 0.2)  # log10(M_star [M_sun/h])
DEFAULT_HALO_MASS_BINS = np.arange(10.0, 15.5, 0.2)    # log10(M_halo [M_sun/h])

# Default sSFR bins (log10 yr^-1)
DEFAULT_SSFR_BINS = np.arange(-10.0, 5.0, 0.1)

