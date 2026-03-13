"""Correlation function analysis subpackage."""

from .correlation import (
    compute_xi_corrfunc,
    correlation_given_redshift_and_subvolume,
    avg_correlation_given_redshift_and_subvolumes,
    correlations_given_redshifts_and_subvolume,
    avg_correlation_given_subvolume_and_redshifts,
    notebook_style_correlation_for_nvolumes,
)

try:
    from .group_sampling_correlation import (
        compute_group_sampling_corrected_xi as compute_group_sampling_corrected_xi,
        compute_notebook_style_correlations_for_nvolumes as compute_notebook_style_correlations_for_nvolumes,
        compute_notebook_style_standard_xi as compute_notebook_style_standard_xi,
    )
    _HAS_GROUP_SAMPLING = True
except Exception:  # pragma: no cover - optional dependency path
    _HAS_GROUP_SAMPLING = False
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
    'notebook_style_correlation_for_nvolumes',
    'dm_correlation_from_tree_file',
    'avg_dm_correlation_from_tree_files',
    'compute_galaxy_bias',
    'avg_galaxy_bias_over_subvolumes',
    'satellite_central_cross_correlation',
    'compute_xi_cross_corrfunc',
]

if _HAS_GROUP_SAMPLING:
    __all__.extend([
        'compute_group_sampling_corrected_xi',
        'compute_notebook_style_correlations_for_nvolumes',
        'compute_notebook_style_standard_xi',
    ])
