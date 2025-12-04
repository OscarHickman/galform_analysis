"""galform_analysis - A Python library for GALFORM simulation analysis.

This library provides tools for analyzing GALFORM galaxy formation simulation outputs,
including:
- Reading HDF5 snapshot data
- Computing mass functions (stellar and halo)
- Aggregating data across subvolumes

Quick Start:
    >>> from galform_analysis.config import set_base_dir
    >>> from galform_analysis.analysis.hmf import avg_hmf_given_redshift_and_subvolumes
    >>> from galform_analysis.analysis.smf import avg_smf_given_redshift_and_subvolumes
    >>> 
    >>> # Set your GALFORM output directory
    >>> set_base_dir('/path/to/galform/output')
    >>> 
    >>> # Compute stellar mass function
    >>> smf = avg_smf_given_redshift_and_subvolumes(iz_num=99, ivols=[0, 1, 2])

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
    # HMF functions
    hmf_given_redshift_and_subvolume,
    hmfs_given_redshifts_and_subvolume,
    avg_hmf_given_redshift_and_subvolumes,
    avg_hmf_given_redshifts_and_subvolume,
    plot_hmf_convergence_by_subvolumes,
    plot_hmf_convergence_by_redshift,
    # SMF functions
    smf_given_redshift_and_subvolume,
    smfs_given_redshifts_and_subvolume,
    avg_smf_given_redshift_and_subvolumes,
    avg_smf_given_redshifts_and_subvolume,
    plot_smf_convergence_by_subvolumes,
    plot_smf_convergence_by_redshift,
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
    # HMF functions
    'hmf_given_redshift_and_subvolume',
    'hmfs_given_redshifts_and_subvolume',
    'avg_hmf_given_redshift_and_subvolumes',
    'avg_hmf_given_redshifts_and_subvolume',
    'plot_hmf_convergence_by_subvolumes',
    'plot_hmf_convergence_by_redshift',
    # SMF functions
    'smf_given_redshift_and_subvolume',
    'smfs_given_redshifts_and_subvolume',
    'avg_smf_given_redshift_and_subvolumes',
    'avg_smf_given_redshifts_and_subvolume',
    'plot_smf_convergence_by_subvolumes',
    'plot_smf_convergence_by_redshift',
]