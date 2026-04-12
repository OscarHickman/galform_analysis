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

try:
    from .halo_sampling_correction import (
        compute_halo_sampling_corrected_xi as compute_halo_sampling_corrected_xi,
        compute_halo_sampling_correlations_for_nvolumes as compute_halo_sampling_correlations_for_nvolumes,
        load_halo_sampled_galaxies as load_halo_sampled_galaxies,
    )
    _HAS_HALO_SAMPLING = True
except Exception:  # pragma: no cover - optional dependency path
    _HAS_HALO_SAMPLING = False

try:
    from .subvol_weighted_correction import (
        compute_weighted_xi_for_n_list as compute_weighted_xi_for_n_list,
        compute_weighted_xi_from_catalogue as compute_weighted_xi_from_catalogue,
        compute_weighted_wp_for_n_list as compute_weighted_wp_for_n_list,
        compute_weighted_wp_from_catalogue as compute_weighted_wp_from_catalogue,
        load_subvolume_galaxies as load_subvolume_galaxies,
    )
    _HAS_SUBVOL_WEIGHTED = True
except Exception:  # pragma: no cover - optional dependency path
    _HAS_SUBVOL_WEIGHTED = False

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

if _HAS_HALO_SAMPLING:
    __all__.extend([
        'compute_halo_sampling_corrected_xi',
        'compute_halo_sampling_correlations_for_nvolumes',
        'load_halo_sampled_galaxies',
    ])

if _HAS_SUBVOL_WEIGHTED:
    __all__.extend([
        'compute_weighted_xi_for_n_list',
        'compute_weighted_xi_from_catalogue',
        'compute_weighted_wp_for_n_list',
        'compute_weighted_wp_from_catalogue',
        'load_subvolume_galaxies',
    ])
