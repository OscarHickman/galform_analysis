"""Correlation function analysis subpackage."""

from .correlation import (
    avg_correlation_given_redshift_and_subvolumes,
    avg_correlation_given_subvolume_and_redshifts,
    compute_xi_corrfunc,
    correlation_given_redshift_and_subvolume,
    correlations_given_redshifts_and_subvolume,
    halo_correlation_given_redshift_and_subvolume,
)
from .dm_correlation import (
    matter_xi_at_snapshot,
    matter_xi_at_snapshots,
)
from .galaxy_bias import (
    avg_galaxy_bias_over_subvolumes,
    compute_galaxy_bias,
)
from .matter_xi import compute_matter_xi
from .n_point_bruteforce import compute_npoint_counts, sugc_weights_npcf
from .satellite_cross_correlation import (
    compute_xi_cross_corrfunc,
    satellite_central_cross_correlation,
)
from .three_point_bruteforce import compute_triplet_counts, sugc_weights
from .three_point_sugc import compute_3pcf_counts_with_sugc

try:
    from .subvol_weighted_correction import (
        compute_weighted_wp_for_n_list,
        compute_weighted_wp_from_catalogue,
        compute_weighted_xi_for_n_list,
        compute_weighted_xi_from_catalogue,
        load_subvolume_galaxies,
    )
    from ..redshift_space_distortions.subvol_weighted_multipoles import (
        compute_direct_rsd_multipoles,
        compute_standard_rsd_multipoles,
        compute_weighted_direct_rsd_multipoles,
        compute_weighted_rsd_multipoles,
    )

    _HAS_SUBVOL_WEIGHTED = True
except Exception:
    _HAS_SUBVOL_WEIGHTED = False

__all__ = [
    # Galaxy 2PCF
    "compute_xi_corrfunc",
    "correlation_given_redshift_and_subvolume",
    "correlations_given_redshifts_and_subvolume",
    "avg_correlation_given_redshift_and_subvolumes",
    "avg_correlation_given_subvolume_and_redshifts",
    # Halo 2PCF (halo tracer xi, not matter xi)
    "halo_correlation_given_redshift_and_subvolume",
    # Linear matter correlation function (correct DM reference for bias)
    "compute_matter_xi",
    "matter_xi_at_snapshot",
    "matter_xi_at_snapshots",
    # Galaxy bias
    "compute_galaxy_bias",
    "avg_galaxy_bias_over_subvolumes",
    # Satellite–central cross-correlation
    "satellite_central_cross_correlation",
    "compute_xi_cross_corrfunc",
    # N-point / 3PCF (SUGC-based)
    "compute_3pcf_counts_with_sugc",
    # N-point / 3PCF (brute-force, for small samples / validation)
    "compute_triplet_counts",
    "sugc_weights",
    "compute_npoint_counts",
    "sugc_weights_npcf",
]

if _HAS_SUBVOL_WEIGHTED:
    __all__.extend(
        [
            "load_subvolume_galaxies",
            "compute_weighted_xi_from_catalogue",
            "compute_weighted_xi_for_n_list",
            "compute_weighted_wp_from_catalogue",
            "compute_weighted_wp_for_n_list",
            "compute_weighted_rsd_multipoles",
            "compute_standard_rsd_multipoles",
            "compute_direct_rsd_multipoles",
            "compute_weighted_direct_rsd_multipoles",
        ]
    )
