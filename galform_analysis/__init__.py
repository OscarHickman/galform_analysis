"""galform_analysis — Python library for GALFORM simulation analysis."""

__version__ = "0.1.9"

from galform_analysis import analysis, config
from galform_analysis.analysis import (
    aggregate_snapshot,
    avg_hmf_given_redshift_and_subvolumes,
    avg_hmf_given_redshifts_and_subvolume,
    avg_smf_given_redshift_and_subvolumes,
    avg_smf_given_redshifts_and_subvolume,
    hmf_given_redshift_and_subvolume,
    hmfs_given_redshifts_and_subvolume,
    smf_given_redshift_and_subvolume,
    smfs_given_redshifts_and_subvolume,
)
from galform_analysis.analysis.correlation import (
    avg_correlation_given_redshift_and_subvolumes,
    avg_correlation_given_subvolume_and_redshifts,
    avg_galaxy_bias_over_subvolumes,
    compute_galaxy_bias,
    compute_matter_xi,
    compute_xi_corrfunc,
    correlation_given_redshift_and_subvolume,
    correlations_given_redshifts_and_subvolume,
    halo_correlation_given_redshift_and_subvolume,
    matter_xi_at_snapshot,
    matter_xi_at_snapshots,
    satellite_central_cross_correlation,
)
from galform_analysis.config import (
    SimulationConfig,
    find_snapshot_at_redshift,
    get_base_dir,
    get_snapshot_redshift,
    load_redshift_mapping,
    load_sim_config,
    set_base_dir,
)
from galform_analysis.readers import close_snapshot, read_snapshot_data

__all__ = [
    "__version__",
    # Submodules
    "config",
    "analysis",
    # Configuration
    "set_base_dir",
    "get_base_dir",
    "SimulationConfig",
    "load_sim_config",
    "load_redshift_mapping",
    "get_snapshot_redshift",
    "find_snapshot_at_redshift",
    # I/O
    "read_snapshot_data",
    "close_snapshot",
    # Aggregation
    "aggregate_snapshot",
    # HMF
    "hmf_given_redshift_and_subvolume",
    "hmfs_given_redshifts_and_subvolume",
    "avg_hmf_given_redshift_and_subvolumes",
    "avg_hmf_given_redshifts_and_subvolume",
    # SMF
    "smf_given_redshift_and_subvolume",
    "smfs_given_redshifts_and_subvolume",
    "avg_smf_given_redshift_and_subvolumes",
    "avg_smf_given_redshifts_and_subvolume",
    # Galaxy 2PCF
    "compute_xi_corrfunc",
    "correlation_given_redshift_and_subvolume",
    "correlations_given_redshifts_and_subvolume",
    "avg_correlation_given_redshift_and_subvolumes",
    "avg_correlation_given_subvolume_and_redshifts",
    # Halo 2PCF (halo tracer — not matter xi)
    "halo_correlation_given_redshift_and_subvolume",
    # Linear matter correlation function
    "compute_matter_xi",
    "matter_xi_at_snapshot",
    "matter_xi_at_snapshots",
    # Galaxy bias
    "compute_galaxy_bias",
    "avg_galaxy_bias_over_subvolumes",
    # Satellite–central cross-correlation
    "satellite_central_cross_correlation",
]
