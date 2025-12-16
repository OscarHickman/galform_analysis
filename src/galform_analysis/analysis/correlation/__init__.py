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
]
