"""Correlation function analysis subpackage."""

from .correlation import (
    compute_xi_corrfunc,
    correlation_given_redshift_and_subvolume,
    avg_correlation_given_redshift_and_subvolumes,
    correlations_given_redshifts_and_subvolume,
    avg_correlation_given_subvolume_and_redshifts,
)
from .dm_correlation import (
    dm_correlation_from_tree_file,
    avg_dm_correlation_from_tree_files,
)
from .galaxy_bias import (
    compute_galaxy_bias,
    avg_galaxy_bias_over_subvolumes,
)
from .satellite_cross_correlation import (
    satellite_central_cross_correlation,
    compute_xi_cross_corrfunc,
)

__all__ = [
    'compute_xi_corrfunc',
    'correlation_given_redshift_and_subvolume',
    'avg_correlation_given_redshift_and_subvolumes',
    'correlations_given_redshifts_and_subvolume',
    'avg_correlation_given_subvolume_and_redshifts',
    'dm_correlation_from_tree_file',
    'avg_dm_correlation_from_tree_files',
    'compute_galaxy_bias',
    'avg_galaxy_bias_over_subvolumes',
    'satellite_central_cross_correlation',
    'compute_xi_cross_corrfunc',
]
