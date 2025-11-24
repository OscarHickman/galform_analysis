"""galform_analysis - A Python library for GALFORM simulation analysis.

This library provides tools for analyzing GALFORM galaxy formation simulation outputs,
including:
- Reading HDF5 snapshot data
- Computing mass functions (stellar and halo)
- Aggregating data across subvolumes

Quick Start:
    >>> from galform_analysis.config import set_base_dir
    >>> from galform_analysis.analysis.smf import compute_smf_avg_by_snapshot
    >>> 
    >>> # Set your GALFORM output directory
    >>> set_base_dir('/path/to/galform/output')
    >>> 
    >>> # Compute stellar mass function
    >>> smf = compute_smf_avg_by_snapshot('iz99')

Configuration:
    Set the BASE_DIR for your GALFORM outputs:
    - Via Python: galform_analysis.config.set_base_dir('/path')
    - Via environment: export GALFORM_BASE_DIR=/path
    - Edit config.py directly
"""

__version__ = "0.1.0"

# Import key modules for convenience
from . import config
from . import io
from . import analysis

# Expose commonly used functions at package level
from .config import (
    set_base_dir, 
    get_base_dir, 
    Cosmology,
    load_redshift_mapping,
    get_snapshot_redshift,
    find_snapshot_at_redshift,
)
from .io import read_snapshot_data, close_snapshot
from .analysis import (
    aggregate_snapshot,
    compute_smf_avg_by_snapshot,
    compute_hmf_avg_by_snapshot,
)

__all__ = [
    '__version__',
    # Submodules
    'config',
    'io',
    'analysis',
    # Common functions
    'set_base_dir',
    'get_base_dir',
    'Cosmology',
    'load_redshift_mapping',
    'get_snapshot_redshift',
    'find_snapshot_at_redshift',
    'read_snapshot_data',
    'close_snapshot',
    'aggregate_snapshot',
    'compute_smf_avg_by_snapshot',
    'compute_hmf_avg_by_snapshot',
]