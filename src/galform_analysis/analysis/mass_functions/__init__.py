"""Mass function analysis subpackage (HMF and SMF)."""

from .hmf import (
    hmf_given_redshift_and_subvolume,
    hmfs_given_redshifts_and_subvolume,
    avg_hmf_given_redshift_and_subvolumes,
    avg_hmf_given_redshifts_and_subvolume,
    compute_hmf_from_aggregated,
)
from .smf import (
    smf_given_redshift_and_subvolume,
    smfs_given_redshifts_and_subvolume,
    avg_smf_given_redshift_and_subvolumes,
    avg_smf_given_redshifts_and_subvolume,
    compute_smf_from_aggregated,
)
from .plot_hmf import (
    plot_hmf_convergence_by_subvolumes,
    plot_hmf_convergence_by_redshift,
)
from .plot_smf import (
    plot_smf_convergence_by_subvolumes,
    plot_smf_convergence_by_redshift,
)
from .theoretical_hmf import (
    compute_theoretical_hmfs,
    get_mvir_to_m200c_ratio,
    create_theoretical_hmf,
    interpolate_hmf_to_bins,
)

__all__ = [
    # HMF functions
    'hmf_given_redshift_and_subvolume',
    'hmfs_given_redshifts_and_subvolume',
    'avg_hmf_given_redshift_and_subvolumes',
    'avg_hmf_given_redshifts_and_subvolume',
    'compute_hmf_from_aggregated',
    # SMF functions
    'smf_given_redshift_and_subvolume',
    'smfs_given_redshifts_and_subvolume',
    'avg_smf_given_redshift_and_subvolumes',
    'avg_smf_given_redshifts_and_subvolume',
    'compute_smf_from_aggregated',
    # HMF plotting functions
    'plot_hmf_convergence_by_subvolumes',
    'plot_hmf_convergence_by_redshift',
    # SMF plotting functions
    'plot_smf_convergence_by_subvolumes',
    'plot_smf_convergence_by_redshift',
    # Theoretical HMF functions
    'compute_theoretical_hmfs',
    'get_mvir_to_m200c_ratio',
    'create_theoretical_hmf',
    'interpolate_hmf_to_bins',
]
