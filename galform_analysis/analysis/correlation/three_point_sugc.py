"""3PCF triplet counting via SUGC with subvolume-weighted correction."""

import numpy as np

try:
    import sugc
except ImportError:
    sugc = None


def compute_3pcf_counts_with_sugc(
    positions: np.ndarray,
    labels: np.ndarray,
    rbins: np.ndarray,
    m_selected: int,
    k_total: int = 1024,
    boxsize: float = 542.16,
) -> dict:
    """Compute triplet counts decomposed by subvolume origin using SUGC.

    Args:
        positions: (N, 3) galaxy positions
        labels: (N,) subvolume IDs in [0, k_total)
        rbins: bin edges for the max side of the triangle
        m_selected: number of subvolumes selected
        k_total: total number of subvolumes
        boxsize: simulation box size (Mpc/h)

    Returns:
        dict with keys: r, t_sss, t_ssd, t_ddd, t_corr, t_total, weights
    """
    if sugc is None:
        raise ImportError("sugc package is not installed.")

    T_by_s, _ = sugc.count_npoint(
        positions.astype(np.float64),
        labels.astype(np.int32),
        rbins.astype(np.float64),
        float(boxsize),
        3,
    )

    m = float(m_selected)
    k = float(k_total)
    w_sss = (m / k) ** 2
    w_ssd = (m**2 * (k - 1)) / (k**2 * (m - 1)) if m > 1 else 0.0
    w_ddd = (m**2 * (k - 1) * (k - 2)) / (k**2 * (m - 1) * (m - 2)) if m > 2 else 0.0

    t_sss = T_by_s[0]
    t_ssd = T_by_s[1]
    t_ddd = T_by_s[2]
    t_corr = w_sss * t_sss + w_ssd * t_ssd + w_ddd * t_ddd

    return {
        "r": 0.5 * (rbins[:-1] + rbins[1:]),
        "t_sss": t_sss,
        "t_ssd": t_ssd,
        "t_ddd": t_ddd,
        "t_corr": t_corr,
        "t_total": t_sss + t_ssd + t_ddd,
        "weights": (w_sss, w_ssd, w_ddd),
    }
