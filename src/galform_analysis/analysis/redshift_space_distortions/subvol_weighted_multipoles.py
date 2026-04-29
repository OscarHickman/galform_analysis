"""Redshift-space correlation multipoles for subvolume-tagged sub-sampling.

Extends the auto/cross weighted pair-count correction to redshift space
distortions (RSD). Pairs are binned in (s, mu), where s is the 
redshift-space separation and mu is the cosine of the line-of-sight angle.

For m selected subvolumes out of k total, corrected pair counts are
DD_corr(s, mu) = alpha * DD_auto(s, mu) + beta * DD_cross(s, mu),

where
alpha = m / k
beta  = m (k - 1) / [k (m - 1)]   (m > 1).

The redshift-space multipoles (monopole xi_0, quadrupole xi_2) are then
obtained by integrating either the Landy-Szalay xi(s, mu) or a direct
periodic-box estimator multiplied by the appropriate Legendre polynomials
over mu.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from Corrfunc.theory.DDsmu import DDsmu

from ..correlation.subvol_weighted_correction import _choose2, _pick_partition_labels

def _counts_to_grid_smu(result: np.ndarray, n_s_bins: int, n_mu_bins: int) -> np.ndarray:
    """Convert Corrfunc DDsmu output into a [n_s_bins, n_mu_bins] array."""
    npairs = np.asarray(result["npairs"], dtype=np.float64)
    expected = n_s_bins * n_mu_bins
    if npairs.size != expected:
        raise RuntimeError(
            f"Unexpected DDsmu output size: got {npairs.size}, expected {expected}"
        )
    return npairs.reshape(n_s_bins, n_mu_bins)


def _analytic_rr_smu(
    s_bins: np.ndarray,
    mu_max: float,
    n_mu_bins: int,
    boxsize: float,
    n_points: int,
) -> np.ndarray:
    """Analytic RR normalization for a periodic cubic box in (s, mu) bins."""
    s_bins = np.asarray(s_bins, dtype=np.float64)
    if s_bins.ndim != 1 or len(s_bins) < 2:
        raise ValueError("s_bins must be a 1D array with at least two edges")
    if mu_max <= 0.0:
        raise ValueError("mu_max must be positive")
    if n_mu_bins < 1:
        raise ValueError("n_mu_bins must be >= 1")

    volume = float(boxsize) ** 3
    shell_volume = (4.0 / 3.0) * np.pi * (s_bins[1:] ** 3 - s_bins[:-1] ** 3)
    mu_bin_fraction = 1.0 / float(n_mu_bins)
    rr_shell = shell_volume / volume
    return rr_shell[:, None] * np.full((1, int(n_mu_bins)), mu_bin_fraction, dtype=np.float64)


def _project_rsd_multipoles(
    xi_grid: np.ndarray,
    mu_max: float,
    n_mu_bins: int,
    s_bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project xi(s, mu) grids to monopole and quadrupole."""
    s_bins = np.asarray(s_bins, dtype=np.float64)
    mu_bins = np.linspace(0.0, mu_max, n_mu_bins + 1)
    mu_mid = 0.5 * (mu_bins[:-1] + mu_bins[1:])
    dmu = mu_max / n_mu_bins

    l0 = np.ones_like(mu_mid)
    l2 = 0.5 * (3.0 * mu_mid**2 - 1.0)

    xi0 = np.nansum(xi_grid * l0 * dmu, axis=1)
    xi2 = 5.0 * np.nansum(xi_grid * l2 * dmu, axis=1)
    s_mid = 0.5 * (s_bins[:-1] + s_bins[1:])
    return s_mid, xi0, xi2

def _paircounts_smu_auto(
    positions: np.ndarray,
    s_bins: np.ndarray,
    mu_max: float,
    n_mu_bins: int,
    boxsize: float,
    nthreads: int,
) -> np.ndarray:
    """Auto pair counts DD(s, mu) for one sample."""
    n_s_bins = len(s_bins) - 1
    if positions.shape[0] < 2:
        return np.zeros((n_s_bins, n_mu_bins), dtype=np.float64)

    res = DDsmu(
        autocorr=1,
        nthreads=nthreads,
        binfile=s_bins,
        mu_max=mu_max,
        nmu_bins=n_mu_bins,
        X1=positions[:, 0],
        Y1=positions[:, 1],
        Z1=positions[:, 2],
        periodic=True,
        boxsize=boxsize,
    )
    return _counts_to_grid_smu(res, n_s_bins=n_s_bins, n_mu_bins=n_mu_bins)

def _paircounts_smu_cross(
    positions_a: np.ndarray,
    positions_b: np.ndarray,
    s_bins: np.ndarray,
    mu_max: float,
    n_mu_bins: int,
    boxsize: float,
    nthreads: int,
) -> np.ndarray:
    """Cross pair counts D1D2(s, mu) between two samples."""
    n_s_bins = len(s_bins) - 1
    if positions_a.shape[0] == 0 or positions_b.shape[0] == 0:
        return np.zeros((n_s_bins, n_mu_bins), dtype=np.float64)

    res = DDsmu(
        autocorr=0,
        nthreads=nthreads,
        binfile=s_bins,
        mu_max=mu_max,
        nmu_bins=n_mu_bins,
        X1=positions_a[:, 0],
        Y1=positions_a[:, 1],
        Z1=positions_a[:, 2],
        X2=positions_b[:, 0],
        Y2=positions_b[:, 1],
        Z2=positions_b[:, 2],
        periodic=True,
        boxsize=boxsize,
    )
    return _counts_to_grid_smu(res, n_s_bins=n_s_bins, n_mu_bins=n_mu_bins)

def compute_weighted_rsd_multipoles(
    galaxy_pos: np.ndarray,
    galaxy_labels: np.ndarray,
    random_pos: np.ndarray,
    s_bins: np.ndarray,
    mu_max: float = 1.0,
    n_mu_bins: int = 120,
    k_total: int = 1024,
    boxsize: float = 800.0,
    nthreads: int = 4,
) -> dict:
    """Compute weighted separation/mu grids and Legendre multipoles.
    
    Returns standard and corrected monopole (xi_0) and quadrupole (xi_2).
    """
    galaxy_pos = np.ascontiguousarray(galaxy_pos, dtype=np.float64)
    random_pos = np.ascontiguousarray(random_pos, dtype=np.float64)
    galaxy_labels = np.asarray(galaxy_labels, dtype=np.int64)
    s_bins = np.asarray(s_bins, dtype=np.float64)

    nd = float(galaxy_pos.shape[0])
    nr = float(random_pos.shape[0])
    
    unique_labels = np.unique(galaxy_labels)
    m_selected = len(unique_labels)
    
    n_s_bins = len(s_bins) - 1
    s_mid = 0.5 * (s_bins[:-1] + s_bins[1:])
    mu_bins = np.linspace(0, mu_max, n_mu_bins + 1)
    mu_mid = 0.5 * (mu_bins[:-1] + mu_bins[1:])
    dmu = mu_max / n_mu_bins

    # 1. Random computations
    rr_counts = _paircounts_smu_auto(random_pos, s_bins, mu_max, n_mu_bins, boxsize, nthreads)
    rr_norm = rr_counts / _choose2(nr)
    
    dr_counts = _paircounts_smu_cross(
        galaxy_pos, random_pos, s_bins, mu_max, n_mu_bins, boxsize, nthreads
    )
    dr_norm = dr_counts / (nd * nr)

    # 2. Total data computations
    dd_total = _paircounts_smu_auto(galaxy_pos, s_bins, mu_max, n_mu_bins, boxsize, nthreads)
    dd_total_norm = dd_total / _choose2(nd)
    
    # 3. Standard LS estimator
    with np.errstate(divide="ignore", invalid="ignore"):
        xi_standard = (dd_total_norm - 2.0 * dr_norm + rr_norm) / rr_norm

    # 4. Auto-pair accumulation
    dd_auto = np.zeros((n_s_bins, n_mu_bins), dtype=np.float64)
    for label in unique_labels:
        mask = (galaxy_labels == label)
        pos_sub = galaxy_pos[mask]
        dd_auto += _paircounts_smu_auto(pos_sub, s_bins, mu_max, n_mu_bins, boxsize, nthreads)

    # 5. Cross-pair derivation and Correction applying
    dd_cross = dd_total - dd_auto
    
    if m_selected < 2:
        alpha = float(m_selected) / float(k_total)
        beta = np.nan
        xi_corrected = np.full_like(xi_standard, np.nan)
    else:
        alpha = float(m_selected) / float(k_total)
        beta = float(m_selected * (k_total - 1)) / float(k_total * (m_selected - 1))
        dd_corr = alpha * dd_auto + beta * dd_cross
        dd_corr_norm = dd_corr / _choose2(nd)
        
        with np.errstate(divide="ignore", invalid="ignore"):
            xi_corrected = (dd_corr_norm - 2.0 * dr_norm + rr_norm) / rr_norm
            
    # Clean zero-division grid areas
    xi_standard[~np.isfinite(xi_standard)] = np.nan
    xi_corrected[~np.isfinite(xi_corrected)] = np.nan

    # 6. Legendre Polynomial Integrations for Multipoles
    # L0 = 1
    # L2 = 0.5 * (3 * mu^2 - 1)
    # xi_ell(s) = (2ell + 1) * \int_0^1 d\mu xi(s, \mu) L_ell(\mu)
    
    L0 = np.ones_like(mu_mid)
    L2 = 0.5 * (3.0 * mu_mid**2 - 1.0)
    
    # xi_0(s) = 1 * sum(xi * 1 * dmu)
    xi0_standard = np.nansum(xi_standard * L0 * dmu, axis=1)
    xi0_corrected = np.nansum(xi_corrected * L0 * dmu, axis=1)
    
    # xi_2(s) = 5 * sum(xi * L2 * dmu)
    xi2_standard = 5.0 * np.nansum(xi_standard * L2 * dmu, axis=1)
    xi2_corrected = 5.0 * np.nansum(xi_corrected * L2 * dmu, axis=1)

    return {
        "s": s_mid,
        "xi0_standard": xi0_standard,
        "xi2_standard": xi2_standard,
        "xi0_corrected": xi0_corrected,
        "xi2_corrected": xi2_corrected,
        "xi_standard_grid": xi_standard,
        "xi_corrected_grid": xi_corrected,
        "alpha": alpha,
        "beta": beta,
        "m_selected": int(m_selected),
        "k_total": int(k_total),
    }


def compute_standard_rsd_multipoles(
    galaxy_pos: np.ndarray,
    random_pos: np.ndarray,
    s_bins: np.ndarray,
    mu_max: float = 1.0,
    n_mu_bins: int = 120,
    boxsize: float = 800.0,
    nthreads: int = 4,
) -> dict:
    """Compute standard Landy-Szalay RSD multipoles without subvolume correction."""
    galaxy_pos = np.ascontiguousarray(galaxy_pos, dtype=np.float64)
    random_pos = np.ascontiguousarray(random_pos, dtype=np.float64)
    s_bins = np.asarray(s_bins, dtype=np.float64)

    nd = float(galaxy_pos.shape[0])
    nr = float(random_pos.shape[0])

    n_s_bins = len(s_bins) - 1
    s_mid = 0.5 * (s_bins[:-1] + s_bins[1:])
    mu_bins = np.linspace(0.0, mu_max, n_mu_bins + 1)
    mu_mid = 0.5 * (mu_bins[:-1] + mu_bins[1:])
    dmu = mu_max / n_mu_bins

    rr_counts = _paircounts_smu_auto(random_pos, s_bins, mu_max, n_mu_bins, boxsize, nthreads)
    rr_norm = rr_counts / _choose2(nr)

    dr_counts = _paircounts_smu_cross(
        galaxy_pos, random_pos, s_bins, mu_max, n_mu_bins, boxsize, nthreads
    )
    dr_norm = dr_counts / (nd * nr)

    dd_counts = _paircounts_smu_auto(galaxy_pos, s_bins, mu_max, n_mu_bins, boxsize, nthreads)
    dd_norm = dd_counts / _choose2(nd)

    with np.errstate(divide="ignore", invalid="ignore"):
        xi_grid = (dd_norm - 2.0 * dr_norm + rr_norm) / rr_norm

    xi_grid[~np.isfinite(xi_grid)] = np.nan

    l0 = np.ones_like(mu_mid)
    l2 = 0.5 * (3.0 * mu_mid**2 - 1.0)

    xi0 = np.nansum(xi_grid * l0 * dmu, axis=1)
    xi2 = 5.0 * np.nansum(xi_grid * l2 * dmu, axis=1)

    return {
        "s": s_mid,
        "xi0": xi0,
        "xi2": xi2,
        "xi_grid": xi_grid,
        "ngal": int(nd),
        "nrandom": int(nr),
    }


def compute_direct_rsd_multipoles(
    galaxy_pos: np.ndarray,
    s_bins: np.ndarray,
    mu_max: float = 1.0,
    n_mu_bins: int = 120,
    boxsize: float = 800.0,
    nthreads: int = 4,
) -> dict:
    """Compute periodic-box RSD multipoles without a random catalog."""
    galaxy_pos = np.ascontiguousarray(galaxy_pos, dtype=np.float64)
    s_bins = np.asarray(s_bins, dtype=np.float64)

    nd = float(galaxy_pos.shape[0])

    dd_counts = _paircounts_smu_auto(galaxy_pos, s_bins, mu_max, n_mu_bins, boxsize, nthreads)
    dd_norm = dd_counts / _choose2(nd)

    rr_norm = _analytic_rr_smu(s_bins, mu_max, n_mu_bins, boxsize, int(nd))
    with np.errstate(divide="ignore", invalid="ignore"):
        xi_grid = dd_norm / rr_norm - 1.0

    xi_grid[~np.isfinite(xi_grid)] = np.nan
    s_mid, xi0, xi2 = _project_rsd_multipoles(xi_grid, mu_max, n_mu_bins, s_bins)

    return {
        "s": s_mid,
        "xi0": xi0,
        "xi2": xi2,
        "xi_grid": xi_grid,
        "ngal": int(nd),
        "nrandom": 0,
    }


def compute_weighted_direct_rsd_multipoles(
    galaxy_pos: np.ndarray,
    galaxy_labels: np.ndarray,
    s_bins: np.ndarray,
    mu_max: float = 1.0,
    n_mu_bins: int = 120,
    k_total: int = 1024,
    boxsize: float = 800.0,
    nthreads: int = 4,
) -> dict:
    """Compute weighted periodic-box RSD multipoles without random catalogs."""
    galaxy_pos = np.ascontiguousarray(galaxy_pos, dtype=np.float64)
    galaxy_labels = np.asarray(galaxy_labels, dtype=np.int64)
    s_bins = np.asarray(s_bins, dtype=np.float64)

    nd = float(galaxy_pos.shape[0])
    unique_labels = np.unique(galaxy_labels)
    m_selected = len(unique_labels)

    n_s_bins = len(s_bins) - 1

    dd_total = _paircounts_smu_auto(galaxy_pos, s_bins, mu_max, n_mu_bins, boxsize, nthreads)
    dd_total_norm = dd_total / _choose2(nd)

    dd_auto = np.zeros((n_s_bins, n_mu_bins), dtype=np.float64)
    for label in unique_labels:
        mask = galaxy_labels == label
        pos_sub = galaxy_pos[mask]
        dd_auto += _paircounts_smu_auto(pos_sub, s_bins, mu_max, n_mu_bins, boxsize, nthreads)

    dd_cross = dd_total - dd_auto

    if m_selected < 2:
        alpha = float(m_selected) / float(k_total)
        beta = np.nan
        xi_grid = np.full_like(dd_total_norm, np.nan)
    else:
        alpha = float(m_selected) / float(k_total)
        beta = float(m_selected * (k_total - 1)) / float(k_total * (m_selected - 1))
        dd_corr = alpha * dd_auto + beta * dd_cross
        dd_corr_norm = dd_corr / _choose2(nd)

        rr_norm = _analytic_rr_smu(s_bins, mu_max, n_mu_bins, boxsize, int(nd))
        with np.errstate(divide="ignore", invalid="ignore"):
            xi_grid = dd_corr_norm / rr_norm - 1.0

    xi_grid[~np.isfinite(xi_grid)] = np.nan
    s_mid, xi0, xi2 = _project_rsd_multipoles(xi_grid, mu_max, n_mu_bins, s_bins)

    return {
        "s": s_mid,
        "xi0": xi0,
        "xi2": xi2,
        "xi_grid": xi_grid,
        "alpha": alpha,
        "beta": beta,
        "m_selected": int(m_selected),
        "k_total": int(k_total),
        "ngal": int(nd),
        "nrandom": 0,
    }
