"""Analysis subpackage for GALFORM data processing."""

from .aggregation import aggregate_snapshot
from .smf import compute_smf_avg_by_snapshot
from .hmf import compute_hmf_avg_by_snapshot
from .convergence import plot_hmf_convergence, plot_smf_convergence

__all__ = [
    'aggregate_snapshot',
    'compute_smf_avg_by_snapshot',
    'compute_hmf_avg_by_snapshot',
    'plot_hmf_convergence',
    'plot_smf_convergence',
]
