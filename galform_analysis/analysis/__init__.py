"""Analysis subpackage for GALFORM data processing."""

from .aggregation import aggregate_snapshot, completed_galaxies, incomplete_subvolumes
from .mass_functions import (
    hmf_given_redshift_and_subvolume,
    hmfs_given_redshifts_and_subvolume,
    avg_hmf_given_redshift_and_subvolumes,
    avg_hmf_given_redshifts_and_subvolume,
    smf_given_redshift_and_subvolume,
    smfs_given_redshifts_and_subvolume,
    avg_smf_given_redshift_and_subvolumes,
    avg_smf_given_redshifts_and_subvolume,
    hod_given_redshift_and_subvolume,
    hods_given_redshifts_and_subvolume,
    avg_hod_given_redshift_and_subvolumes,
    avg_hod_given_redshifts_and_subvolume,
)
from .correlation import (
    compute_xi_corrfunc,
    correlation_given_redshift_and_subvolume,
    avg_correlation_given_redshift_and_subvolumes,
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
    # SMF functions
    'smf_given_redshift_and_subvolume',
    'smfs_given_redshifts_and_subvolume',
    'avg_smf_given_redshift_and_subvolumes',
    'avg_smf_given_redshifts_and_subvolume',
    # HOD functions
    'hod_given_redshift_and_subvolume',
    'hods_given_redshifts_and_subvolume',
    'avg_hod_given_redshift_and_subvolumes',
    'avg_hod_given_redshifts_and_subvolume',
    # Correlation functions
    'compute_xi_corrfunc',
    'correlation_given_redshift_and_subvolume',
    'avg_correlation_given_redshift_and_subvolumes',
]
