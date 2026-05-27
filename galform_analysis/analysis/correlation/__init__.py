"""Correlation function analysis subpackage."""

from .correlation import (
    avg_correlation_given_redshift_and_subvolumes,
    avg_correlation_given_subvolume_and_redshifts,
    compute_xi_corrfunc,
    correlation_given_redshift_and_subvolume,
    correlations_given_redshifts_and_subvolume,
)

try:
    from .subvol_weighted_correction import (
        compute_weighted_wp_for_n_list as compute_weighted_wp_for_n_list,
    )
    from .subvol_weighted_correction import (
        compute_weighted_wp_from_catalogue as compute_weighted_wp_from_catalogue,
    )
    from .subvol_weighted_correction import (
        compute_weighted_xi_for_n_list as compute_weighted_xi_for_n_list,
    )
    from .subvol_weighted_correction import (
        compute_weighted_xi_from_catalogue as compute_weighted_xi_from_catalogue,
    )
    from .subvol_weighted_correction import (
        load_subvolume_galaxies as load_subvolume_galaxies,
    )

    _HAS_SUBVOL_WEIGHTED = True
except Exception:  # pragma: no cover - optional dependency path
    _HAS_SUBVOL_WEIGHTED = False

from .dm_correlation import (
    avg_dm_correlation_from_tree_files,
    dm_correlation_from_tree_file,
)
from .galaxy_bias import (
    avg_galaxy_bias_over_subvolumes,
    compute_galaxy_bias,
)
from .satellite_cross_correlation import (
    compute_xi_cross_corrfunc,
    satellite_central_cross_correlation,
)

__all__ = [
    "compute_xi_corrfunc",
    "correlation_given_redshift_and_subvolume",
    "avg_correlation_given_redshift_and_subvolumes",
    "correlations_given_redshifts_and_subvolume",
    "avg_correlation_given_subvolume_and_redshifts",
    "dm_correlation_from_tree_file",
    "avg_dm_correlation_from_tree_files",
    "compute_galaxy_bias",
    "avg_galaxy_bias_over_subvolumes",
    "satellite_central_cross_correlation",
    "compute_xi_cross_corrfunc",
]

if _HAS_SUBVOL_WEIGHTED:
    __all__.extend(
        [
            "compute_weighted_xi_for_n_list",
            "compute_weighted_xi_from_catalogue",
            "compute_weighted_wp_for_n_list",
            "compute_weighted_wp_from_catalogue",
            "load_subvolume_galaxies",
        ]
    )
