"""Analysis subpackage for GALFORM data processing."""

from .aggregation import aggregate_snapshot
from .hmf import (
    hmf_given_redshift_and_subvolume,
    hmfs_given_redshifts_and_subvolume,
    avg_hmf_given_redshift_and_subvolumes,
    avg_hmf_given_redshifts_and_subvolume,
    plot_hmf_convergence_by_subvolumes,
    plot_hmf_convergence_by_redshift,
)
from .smf import (
    smf_given_redshift_and_subvolume,
    smfs_given_redshifts_and_subvolume,
    avg_smf_given_redshift_and_subvolumes,
    avg_smf_given_redshifts_and_subvolume,
    plot_smf_convergence_by_subvolumes,
    plot_smf_convergence_by_redshift,
)

__all__ = [
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
