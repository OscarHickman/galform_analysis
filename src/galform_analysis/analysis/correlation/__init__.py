"""Correlation function analysis subpackage."""

from .correlation import (
    compute_xi_corrfunc,
    correlation_given_redshift_and_subvolume,
    avg_correlation_given_redshift_and_subvolumes,
)
from .plot_correlation import (
    plot_correlation_convergence_by_subvolumes,
    plot_correlation_convergence_by_redshift,
)

__all__ = [
    'compute_xi_corrfunc',
    'correlation_given_redshift_and_subvolume',
    'avg_correlation_given_redshift_and_subvolumes',
    'plot_correlation_convergence_by_subvolumes',
    'plot_correlation_convergence_by_redshift',
]
