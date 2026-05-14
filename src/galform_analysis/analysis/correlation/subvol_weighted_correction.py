"""Projected 2PCF correction for subvolume-tagged sub-sampling.

Implements the auto/cross weighted pair-count correction described in the
project note:

- DD_auto: sum of pair counts computed within each selected subvolume.
- DD_total: pair counts of the combined catalogue of selected subvolumes.
- DD_cross = DD_total - DD_auto.

For m selected subvolumes out of k total, corrected pair counts are

DD_corr = alpha * DD_auto + beta * DD_cross,

where

alpha = m / k,
beta  = m (k - 1) / [k (m - 1)]   (m > 1).

The projected correlation function is then obtained by integrating the
Landy-Szalay xi(r_p, pi) along pi.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import polars as pl
from Corrfunc.theory.DD import DD
from Corrfunc.theory.DDrppi import DDrppi

from ...utils.read_galaxies import read_galaxy_arrays


_HALO_ID_FIELDS = ("ihalof", "ihhalo", "DHaloID", "TreeID", "SubhaloID")


def _pick_partition_labels(catalogue: pl.DataFrame) -> np.ndarray:
    """Return the label vector used for auto/cross decomposition."""
    if "partition_label" in catalogue.columns:
        return catalogue["partition_label"].to_numpy(dtype=np.int64)
    if "subvol_rank" in catalogue.columns:
        return catalogue["subvol_rank"].to_numpy(dtype=np.int64)
    raise KeyError("Catalogue is missing both 'partition_label' and 'subvol_rank'")


def _select_halo_id_array(arrays: dict[str, np.ndarray]) -> tuple[Optional[np.ndarray], Optional[str]]:
    """Choose an informative halo identifier array from loaded GALFORM fields."""
    for key in _HALO_ID_FIELDS:
        if key not in arrays:
            continue
        arr = np.asarray(arrays[key])
        if arr.size == 0:
            continue

        if arr.dtype.kind in "iu":
            valid = arr[arr >= 0]
        else:
            valid = arr[np.isfinite(arr)]

        if valid.size == 0:
            continue
        if np.unique(valid).size > 1:
            return arr.astype(np.int64, copy=False), key

    return None, None


def _counts_to_grid(result: np.ndarray, n_rp_bins: int, n_pi_bins: int) -> np.ndarray:
    """Convert Corrfunc DDrppi output into a [n_rp_bins, n_pi_bins] array."""
    npairs = np.asarray(result["npairs"], dtype=np.float64)
    expected = n_rp_bins * n_pi_bins
    if npairs.size != expected:
        raise RuntimeError(
            f"Unexpected DDrppi output size: got {npairs.size}, expected {expected}"
        )
    return npairs.reshape(n_rp_bins, n_pi_bins)


def _paircounts_rppi_auto(
    positions: np.ndarray,
    rp_bins: np.ndarray,
    pimax: int,
    boxsize: float,
    nthreads: int,
) -> np.ndarray:
    """Auto pair counts DD(r_p, pi) for one sample."""
    n_rp_bins = len(rp_bins) - 1
    n_pi_bins = int(pimax)

    if positions.shape[0] < 2:
        return np.zeros((n_rp_bins, n_pi_bins), dtype=np.float64)

    res = DDrppi(
        autocorr=1,
        nthreads=nthreads,
        pimax=pimax,
        binfile=rp_bins,
        X1=positions[:, 0],
        Y1=positions[:, 1],
        Z1=positions[:, 2],
        periodic=True,
        boxsize=boxsize,
    )
    return _counts_to_grid(res, n_rp_bins=n_rp_bins, n_pi_bins=n_pi_bins)


def _paircounts_rppi_cross(
    positions_a: np.ndarray,
    positions_b: np.ndarray,
    rp_bins: np.ndarray,
    pimax: int,
    boxsize: float,
    nthreads: int,
) -> np.ndarray:
    """Cross pair counts D1D2(r_p, pi) between two samples."""
    n_rp_bins = len(rp_bins) - 1
    n_pi_bins = int(pimax)

    if positions_a.shape[0] == 0 or positions_b.shape[0] == 0:
        return np.zeros((n_rp_bins, n_pi_bins), dtype=np.float64)

    res = DDrppi(
        autocorr=0,
        nthreads=nthreads,
        pimax=pimax,
        binfile=rp_bins,
        X1=positions_a[:, 0],
        Y1=positions_a[:, 1],
        Z1=positions_a[:, 2],
        X2=positions_b[:, 0],
        Y2=positions_b[:, 1],
        Z2=positions_b[:, 2],
        periodic=True,
        boxsize=boxsize,
    )
    return _counts_to_grid(res, n_rp_bins=n_rp_bins, n_pi_bins=n_pi_bins)


def _paircounts_r_auto(
    positions: np.ndarray,
    rbins: np.ndarray,
    boxsize: float,
    nthreads: int,
) -> np.ndarray:
    """Auto pair counts DD(r) for one sample."""
    n_bins = len(rbins) - 1
    if positions.shape[0] < 2:
        return np.zeros(n_bins, dtype=np.float64)

    res = DD(
        autocorr=1,
        nthreads=nthreads,
        binfile=rbins,
        X1=positions[:, 0],
        Y1=positions[:, 1],
        Z1=positions[:, 2],
        periodic=True,
        boxsize=boxsize,
    )
    npairs = np.asarray(res["npairs"], dtype=np.float64)
    if npairs.size != n_bins:
        raise RuntimeError(f"Unexpected DD output size: got {npairs.size}, expected {n_bins}")
    return npairs


def _paircounts_r_cross(
    positions_a: np.ndarray,
    positions_b: np.ndarray,
    rbins: np.ndarray,
    boxsize: float,
    nthreads: int,
) -> np.ndarray:
    """Cross pair counts D1D2(r) between two samples."""
    n_bins = len(rbins) - 1
    if positions_a.shape[0] == 0 or positions_b.shape[0] == 0:
        return np.zeros(n_bins, dtype=np.float64)

    res = DD(
        autocorr=0,
        nthreads=nthreads,
        binfile=rbins,
        X1=positions_a[:, 0],
        Y1=positions_a[:, 1],
        Z1=positions_a[:, 2],
        X2=positions_b[:, 0],
        Y2=positions_b[:, 1],
        Z2=positions_b[:, 2],
        periodic=True,
        boxsize=boxsize,
    )
    npairs = np.asarray(res["npairs"], dtype=np.float64)
    if npairs.size != n_bins:
        raise RuntimeError(f"Unexpected DD cross output size: got {npairs.size}, expected {n_bins}")
    return npairs


def _choose2(n: int) -> float:
    """Number of unique unordered pairs from n points."""
    return 0.5 * n * (n - 1)


def load_subvolume_galaxies(
    base_dir: str,
    iz_num: int,
    ivols: Sequence[int],
    centrals_only: bool = False,
    mhalo_min: Optional[float] = None,
    mstar_min_log10: Optional[float] = None,
    partition_scheme: str = "ivol",
    k_total: int = 1024,
) -> pl.DataFrame:
    """Load galaxies for selected subvolumes and attach subvolume labels.

    Returns a DataFrame with columns: x, y, z, subvol_rank, partition_label, ivol.

        partition_scheme:
        - "ivol" (default): partition_label == subvol_rank.
            This follows the explicit implementation recipe where each galaxy is
            tagged by the selected subvolume catalogue it came from.
        - "halo_id_hash": optional diagnostic mode where
            partition_label = halo_id mod k_total.
    """
    if partition_scheme not in {"ivol", "halo_id_hash"}:
        raise ValueError("partition_scheme must be 'ivol' or 'halo_id_hash'")
    if int(k_total) < 2:
        raise ValueError("k_total must be >= 2")

    iz_path = str(Path(base_dir) / f"iz{iz_num}")
    chunks: list[pl.DataFrame] = []

    for subvol_rank, ivol in enumerate(ivols):
        extra_fields = _HALO_ID_FIELDS if partition_scheme == "halo_id_hash" else None
        arrays, _ = read_galaxy_arrays(
            iz_path=iz_path,
            ivol=int(ivol),
            fields=extra_fields,
            include_positions=True,
            include_derived=True,
            centrals_only=centrals_only,
            mhalo_min=mhalo_min,
        )

        if not arrays or "x" not in arrays or len(arrays["x"]) == 0:
            continue

        n = len(arrays["x"])
        keep = np.ones(n, dtype=bool)

        if mstar_min_log10 is not None and "mstar" in arrays:
            mstar = np.asarray(arrays["mstar"], dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                keep &= np.log10(np.clip(mstar, 1e-30, None)) >= float(mstar_min_log10)

        if np.count_nonzero(keep) == 0:
            continue

        n_keep = int(np.count_nonzero(keep))
        if partition_scheme == "ivol":
            labels = np.full(n_keep, int(subvol_rank), dtype=np.int64)
        else:
            halo_id, src = _select_halo_id_array(arrays)
            if halo_id is None:
                raise RuntimeError(
                    f"No informative halo ID field found for iz{iz_num}/ivol{ivol}; "
                    "cannot build halo_id_hash partitions"
                )
            labels = np.mod(np.abs(halo_id[keep]), int(k_total)).astype(np.int64, copy=False)

        chunks.append(
            pl.DataFrame(
                {
                    "x": np.asarray(arrays["x"], dtype=np.float64)[keep],
                    "y": np.asarray(arrays["y"], dtype=np.float64)[keep],
                    "z": np.asarray(arrays["z"], dtype=np.float64)[keep],
                    "subvol_rank": np.full(n_keep, int(subvol_rank), dtype=np.int64),
                    "partition_label": labels,
                    "ivol": np.full(n_keep, int(ivol), dtype=np.int64),
                    "partition_scheme": np.full(n_keep, partition_scheme),
                }
            )
        )

    if not chunks:
        return pl.DataFrame(schema={"x": pl.Float64, "y": pl.Float64, "z": pl.Float64, "subvol_rank": pl.Int64, "partition_label": pl.Int64, "ivol": pl.Int64, "partition_scheme": pl.Utf8})

    return pl.concat(chunks)


def compute_weighted_wp_from_catalogue(
    catalogue: pl.DataFrame,
    m_selected: int,
    k_total: int,
    rp_bins: np.ndarray,
    pimax: int = 40,
    boxsize: float = 542.16,
    random_multiplier: float = 3.0,
    random_seed: int = 12345,
    nthreads: int = 8,
) -> dict[str, np.ndarray | float | int]:
    """Compute standard and auto/cross-corrected projected correlation functions.

    Notes:
    - The correction is only defined for m_selected > 1.
    - For m_selected == 1, corrected outputs are returned as NaN.
    """
    rp_bins = np.asarray(rp_bins, dtype=np.float64)
    if rp_bins.ndim != 1 or len(rp_bins) < 2:
        raise ValueError("rp_bins must be a 1D array with at least two edges")
    if pimax < 1 or int(pimax) != pimax:
        raise ValueError("pimax must be a positive integer (pi bin width is 1)")

    n_pi_bins = int(pimax)
    n_rp_bins = len(rp_bins) - 1
    rp_mid = 0.5 * (rp_bins[:-1] + rp_bins[1:])

    if catalogue.is_empty():
        nan_wp = np.full(n_rp_bins, np.nan)
        nan_xi = np.full((n_rp_bins, n_pi_bins), np.nan)
        return {
            "rp": rp_mid,
            "wp_standard": nan_wp,
            "wp_corrected": nan_wp,
            "xi_standard_grid": nan_xi,
            "xi_corrected_grid": nan_xi,
            "alpha": np.nan,
            "beta": np.nan,
            "ngal": 0,
            "nrandom": 0,
            "m_selected": int(m_selected),
            "k_total": int(k_total),
        }

    pos = catalogue.select(["x", "y", "z"]).to_numpy()
    tags = _pick_partition_labels(catalogue)
    nd = pos.shape[0]

    dd_total = _paircounts_rppi_auto(pos, rp_bins=rp_bins, pimax=pimax, boxsize=boxsize, nthreads=nthreads)

    dd_auto = np.zeros_like(dd_total)
    for tag in range(int(m_selected)):
        mask = tags == tag
        if np.count_nonzero(mask) < 2:
            continue
        dd_auto += _paircounts_rppi_auto(
            pos[mask], rp_bins=rp_bins, pimax=pimax, boxsize=boxsize, nthreads=nthreads
        )
    dd_cross = dd_total - dd_auto

    nr = max(2, int(np.ceil(random_multiplier * nd)))
    rng = np.random.default_rng(random_seed)
    rnd = rng.uniform(0.0, boxsize, size=(nr, 3))

    dr = _paircounts_rppi_cross(
        pos,
        rnd,
        rp_bins=rp_bins,
        pimax=pimax,
        boxsize=boxsize,
        nthreads=nthreads,
    )
    rr = _paircounts_rppi_auto(rnd, rp_bins=rp_bins, pimax=pimax, boxsize=boxsize, nthreads=nthreads)

    dd_norm = dd_total / _choose2(nd)
    dr_norm = dr / (nd * nr)
    rr_norm = rr / _choose2(nr)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi_standard = (dd_norm - 2.0 * dr_norm + rr_norm) / rr_norm

    if int(m_selected) <= 1:
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

    wp_standard = 2.0 * np.nansum(xi_standard, axis=1)
    wp_corrected = 2.0 * np.nansum(xi_corrected, axis=1)

    # Preserve undefined rows as NaN (e.g. m_selected == 1 for corrected estimator).
    wp_standard[np.all(~np.isfinite(xi_standard), axis=1)] = np.nan
    wp_corrected[np.all(~np.isfinite(xi_corrected), axis=1)] = np.nan

    return {
        "rp": rp_mid,
        "wp_standard": wp_standard,
        "wp_corrected": wp_corrected,
        "xi_standard_grid": xi_standard,
        "xi_corrected_grid": xi_corrected,
        "alpha": alpha,
        "beta": beta,
        "ngal": int(nd),
        "nrandom": int(nr),
        "m_selected": int(m_selected),
        "k_total": int(k_total),
    }


def compute_weighted_xi_from_catalogue(
    catalogue: pl.DataFrame,
    m_selected: int,
    k_total: int,
    rbins: np.ndarray,
    boxsize: float = 542.16,
    random_multiplier: float = 3.0,
    random_seed: int = 12345,
    nthreads: int = 8,
) -> dict[str, np.ndarray | float | int]:
    """Compute standard and auto/cross-corrected real-space xi(r).

    This variant is useful for direct comparison with existing cached xi(r)
    CSV outputs (e.g. n_subvol=1024) without re-running a costly full-box job.
    """
    rbins = np.asarray(rbins, dtype=np.float64)
    if rbins.ndim != 1 or len(rbins) < 2:
        raise ValueError("rbins must be a 1D array with at least two edges")

    r_mid = 0.5 * (rbins[:-1] + rbins[1:])
    n_bins = len(rbins) - 1

    if catalogue.is_empty():
        nan = np.full(n_bins, np.nan)
        return {
            "r": r_mid,
            "xi_standard": nan,
            "xi_corrected": nan,
            "alpha": np.nan,
            "beta": np.nan,
            "ngal": 0,
            "nrandom": 0,
            "m_selected": int(m_selected),
            "k_total": int(k_total),
        }

    pos = catalogue.select(["x", "y", "z"]).to_numpy()
    tags = _pick_partition_labels(catalogue)
    nd = pos.shape[0]

    dd_total = _paircounts_r_auto(pos, rbins=rbins, boxsize=boxsize, nthreads=nthreads)

    dd_auto = np.zeros_like(dd_total)
    for tag in range(int(m_selected)):
        mask = tags == tag
        if np.count_nonzero(mask) < 2:
            continue
        dd_auto += _paircounts_r_auto(pos[mask], rbins=rbins, boxsize=boxsize, nthreads=nthreads)
    dd_cross = dd_total - dd_auto

    nr = max(2, int(np.ceil(random_multiplier * nd)))
    rng = np.random.default_rng(random_seed)
    rnd = rng.uniform(0.0, boxsize, size=(nr, 3))

    dr = _paircounts_r_cross(pos, rnd, rbins=rbins, boxsize=boxsize, nthreads=nthreads)
    rr = _paircounts_r_auto(rnd, rbins=rbins, boxsize=boxsize, nthreads=nthreads)

    dd_norm = dd_total / _choose2(nd)
    dr_norm = dr / (nd * nr)
    rr_norm = rr / _choose2(nr)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi_standard = (dd_norm - 2.0 * dr_norm + rr_norm) / rr_norm

    if int(m_selected) <= 1:
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

    xi_standard[~np.isfinite(xi_standard)] = np.nan
    xi_corrected[~np.isfinite(xi_corrected)] = np.nan

    return {
        "r": r_mid,
        "xi_standard": xi_standard,
        "xi_corrected": xi_corrected,
        "alpha": alpha,
        "beta": beta,
        "ngal": int(nd),
        "nrandom": int(nr),
        "m_selected": int(m_selected),
        "k_total": int(k_total),
    }


def compute_weighted_xi_for_n_list(
    base_dir: str,
    iz_num: int,
    n_subvol_list: Sequence[int],
    k_total: int = 1024,
    rbins: Optional[np.ndarray] = None,
    boxsize: float = 542.16,
    centrals_only: bool = False,
    mhalo_min: Optional[float] = None,
    mstar_min_log10: Optional[float] = None,
    random_multiplier: float = 3.0,
    random_seed: int = 12345,
    nthreads: int = 8,
    ivol_start: int = 0,
    load_n_subvolumes: Optional[int] = None,
    partition_scheme: str = "ivol",
) -> pl.DataFrame:
    """Compute standard and corrected xi(r) for multiple subvolume counts."""
    if rbins is None:
        rbins = np.logspace(-1.0, 1.5, 21)
    rbins = np.asarray(rbins, dtype=np.float64)

    n_vals = sorted({int(n) for n in n_subvol_list})
    if not n_vals or n_vals[0] < 1:
        raise ValueError("n_subvol_list must contain positive integers")

    max_n = max(n_vals)
    if load_n_subvolumes is None:
        load_n = int(max_n) if partition_scheme == "ivol" else int(k_total)
    else:
        load_n = int(load_n_subvolumes)
    if load_n < max_n:
        raise ValueError("load_n_subvolumes must be >= max(n_subvol_list)")

    ivols = list(range(int(ivol_start), int(ivol_start) + load_n))
    full_cat = load_subvolume_galaxies(
        base_dir=base_dir,
        iz_num=iz_num,
        ivols=ivols,
        centrals_only=centrals_only,
        mhalo_min=mhalo_min,
        mstar_min_log10=mstar_min_log10,
        partition_scheme=partition_scheme,
        k_total=k_total,
    )

    out_rows: list[dict[str, float | int]] = []
    label_col = "partition_label" if "partition_label" in full_cat.columns else "subvol_rank"
    for n in n_vals:
        sub_cat = full_cat.filter(pl.col(label_col) < n)
        result = compute_weighted_xi_from_catalogue(
            catalogue=sub_cat,
            m_selected=n,
            k_total=k_total,
            rbins=rbins,
            boxsize=boxsize,
            random_multiplier=random_multiplier,
            random_seed=random_seed + n,
            nthreads=nthreads,
        )

        r_mid = np.asarray(result["r"], dtype=np.float64)
        xi_std = np.asarray(result["xi_standard"], dtype=np.float64)
        xi_corr = np.asarray(result["xi_corrected"], dtype=np.float64)

        for bidx, (r, xstd, xcorr) in enumerate(zip(r_mid, xi_std, xi_corr)):
            out_rows.append(
                {
                    "iz": int(iz_num),
                    "n_subvol": int(n),
                    "bin_idx": int(bidx),
                    "r": float(r),
                    "xi_standard": float(xstd),
                    "xi_corrected": float(xcorr),
                    "alpha": float(result["alpha"]),
                    "beta": float(result["beta"]),
                    "ngal": int(result["ngal"]),
                    "nrandom": int(result["nrandom"]),
                }
            )

    return pl.DataFrame(out_rows)


def compute_weighted_wp_for_n_list(
    base_dir: str,
    iz_num: int,
    n_subvol_list: Sequence[int],
    k_total: int = 1024,
    rp_bins: Optional[np.ndarray] = None,
    pimax: int = 40,
    boxsize: float = 542.16,
    centrals_only: bool = False,
    mhalo_min: Optional[float] = None,
    mstar_min_log10: Optional[float] = None,
    random_multiplier: float = 3.0,
    random_seed: int = 12345,
    nthreads: int = 8,
    ivol_start: int = 0,
    load_n_subvolumes: Optional[int] = None,
    partition_scheme: str = "ivol",
) -> pl.DataFrame:
    """Compute standard and corrected wp for multiple selected subvolume counts.

    Subvolumes are chosen deterministically as ivol_start..ivol_start+n-1.
    """
    if rp_bins is None:
        rp_bins = np.logspace(-1.0, 1.5, 16)
    rp_bins = np.asarray(rp_bins, dtype=np.float64)

    n_vals = sorted({int(n) for n in n_subvol_list})
    if not n_vals or n_vals[0] < 1:
        raise ValueError("n_subvol_list must contain positive integers")

    max_n = max(n_vals)
    if load_n_subvolumes is None:
        load_n = int(max_n) if partition_scheme == "ivol" else int(k_total)
    else:
        load_n = int(load_n_subvolumes)
    if load_n < max_n:
        raise ValueError("load_n_subvolumes must be >= max(n_subvol_list)")

    ivols = list(range(int(ivol_start), int(ivol_start) + load_n))
    full_cat = load_subvolume_galaxies(
        base_dir=base_dir,
        iz_num=iz_num,
        ivols=ivols,
        centrals_only=centrals_only,
        mhalo_min=mhalo_min,
        mstar_min_log10=mstar_min_log10,
        partition_scheme=partition_scheme,
        k_total=k_total,
    )

    out_rows: list[dict[str, float | int]] = []
    label_col = "partition_label" if "partition_label" in full_cat.columns else "subvol_rank"
    for n in n_vals:
        sub_cat = full_cat.filter(pl.col(label_col) < n)
        result = compute_weighted_wp_from_catalogue(
            catalogue=sub_cat,
            m_selected=n,
            k_total=k_total,
            rp_bins=rp_bins,
            pimax=pimax,
            boxsize=boxsize,
            random_multiplier=random_multiplier,
            random_seed=random_seed + n,
            nthreads=nthreads,
        )

        rp_mid = np.asarray(result["rp"], dtype=np.float64)
        wp_std = np.asarray(result["wp_standard"], dtype=np.float64)
        wp_corr = np.asarray(result["wp_corrected"], dtype=np.float64)

        for bidx, (rp, wstd, wcorr) in enumerate(zip(rp_mid, wp_std, wp_corr)):
            out_rows.append(
                {
                    "iz": int(iz_num),
                    "n_subvol": int(n),
                    "bin_idx": int(bidx),
                    "rp": float(rp),
                    "wp_standard": float(wstd),
                    "wp_corrected": float(wcorr),
                    "alpha": float(result["alpha"]),
                    "beta": float(result["beta"]),
                    "ngal": int(result["ngal"]),
                    "nrandom": int(result["nrandom"]),
                }
            )

    return pl.DataFrame(out_rows)
