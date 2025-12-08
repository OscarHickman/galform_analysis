"""Analysis subpackage for GALFORM data processing."""

from .aggregation import aggregate_snapshot, completed_galaxies, incomplete_subvolumes
from .mass_functions import (
    hmf_given_redshift_and_subvolume,
    hmfs_given_redshifts_and_subvolume,
    avg_hmf_given_redshift_and_subvolumes,
    avg_hmf_given_redshifts_and_subvolume,
    plot_hmf_convergence_by_subvolumes,
    plot_hmf_convergence_by_redshift,
    smf_given_redshift_and_subvolume,
    smfs_given_redshifts_and_subvolume,
    avg_smf_given_redshift_and_subvolumes,
    avg_smf_given_redshifts_and_subvolume,
    plot_smf_convergence_by_subvolumes,
    plot_smf_convergence_by_redshift,
)
from .galaxy_efficiency import (
    compute_efficiency_vs_mass,
    process_efficiency_redshifts,
    plot_efficiency_vs_mass,
    save_efficiency_data,
    find_peak_efficiency,
)
from .correlation import (
    compute_xi_corrfunc,
    correlation_given_redshift_and_subvolume,
    avg_correlation_given_redshift_and_subvolumes,
    plot_correlation_convergence_by_subvolumes,
    plot_correlation_convergence_by_redshift,
)

__all__ = [
    'aggregate_snapshot',
    'completed_galaxies',
    'incomplete_subvolumes',
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
    # Galaxy efficiency functions
    'compute_efficiency_vs_mass',
    'process_efficiency_redshifts',
    'plot_efficiency_vs_mass',
    'save_efficiency_data',
    'find_peak_efficiency',
    # Correlation functions
    'compute_xi_corrfunc',
    'correlation_given_redshift_and_subvolume',
    'avg_correlation_given_redshift_and_subvolumes',
    'plot_correlation_convergence_by_subvolumes',
    'plot_correlation_convergence_by_redshift',
]
