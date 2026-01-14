"""Correlation function analysis subpackage."""

from .correlation import (
    compute_xi_corrfunc,
    correlation_given_redshift_and_subvolume,
    avg_correlation_given_redshift_and_subvolumes,
    correlations_given_redshifts_and_subvolume,
    avg_correlation_given_subvolume_and_redshifts,
)
from .plot_correlation import (
    plot_correlation_convergence_by_subvolumes,
    plot_correlation_convergence_by_redshift,
    plot_single_correlation,
    plot_correlation_multi_redshift,
    plot_avg_correlation_over_redshifts,
)
from .dm_correlation import (
    dm_correlation_from_tree_file,
    avg_dm_correlation_from_tree_files,
)
from .galaxy_bias import (
    compute_galaxy_bias,
    avg_galaxy_bias_over_subvolumes,
)

__all__ = [
    'compute_xi_corrfunc',
    'correlation_given_redshift_and_subvolume',
    'avg_correlation_given_redshift_and_subvolumes',
    'correlations_given_redshifts_and_subvolume',
    'avg_correlation_given_subvolume_and_redshifts',
    'plot_correlation_convergence_by_subvolumes',
    'plot_correlation_convergence_by_redshift',
    'plot_single_correlation',
    'plot_correlation_multi_redshift',
    'plot_avg_correlation_over_redshifts',
    'dm_correlation_from_tree_file',
    'avg_dm_correlation_from_tree_files',
    'compute_galaxy_bias',
    'avg_galaxy_bias_over_subvolumes',
]
