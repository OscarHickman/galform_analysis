import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from galform_analysis._optional import import_optional
from galform_analysis.config import DEFAULT_RBINS, get_base_dir
from galform_analysis.readers.loaders import close_snapshot, read_snapshot_data
from galform_analysis.utils.read_galaxies import (
    read_galaxy_positions,
    read_halo_positions,
)


def _load_positions_from_hdf5(
    iz_path: str,
    ivol: int,
    centrals_only: bool = True,
    mhalo_min: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[float]]:
    """Load galaxy positions (x,y,z) and redshift from an HDF5 subvolume.

    When centrals_only=True, uses only central galaxies (is_central=1).
    When centrals_only=False, uses all galaxies (centrals + satellites).

    Args:
        iz_path: Path to snapshot directory
        ivol: Subvolume number
        centrals_only: If True, keep only central galaxies (is_central==1)
        mhalo_min: Minimum halo mass (mhalo) threshold in Msun. None = no cut.

    Returns:
        positions: (N,3) array in the native units of the file (assumed Mpc or Mpc/h)
        z: best-effort redshift if available
    """
    return read_galaxy_positions(
        iz_path=iz_path,
        ivol=ivol,
        centrals_only=centrals_only,
        mhalo_min=mhalo_min,
    )


def _subvolume_metadata(iz_path: str, ivol: int) -> Dict[str, Any]:
    """Read z, V_ivol and the periodic box size of one subvolume.

    Each subvolume is an independent realisation of the full simulation box,
    so the box side is L = V_total^(1/3) with V_total = V_ivol * n_subvolumes.
    ``boxsize`` is None when the file has no ``Parameters/volume``.
    """
    meta = read_snapshot_data(iz_path, ivol)
    try:
        v_total = meta.get("V_total")
        return {
            "z": meta.get("z"),
            "V_ivol": meta.get("V_ivol"),
            "boxsize": float(v_total) ** (1.0 / 3.0) if v_total else None,
        }
    finally:
        close_snapshot(meta)


def _resolve_boxsize(
    pos: np.ndarray,
    boxsize: Optional[float],
    file_boxsize: Optional[float],
    where: str,
) -> float:
    """Pick the periodic box size: explicit argument, then file, then extent.

    The extent fallback underestimates L for sparse samples, so it warns.

    Raises:
        RuntimeError: If no valid (finite, positive) box size can be found.
        ValueError: If the positions do not fit in the box read from the file.
    """
    if boxsize is not None:
        L = float(boxsize)
    elif file_boxsize is not None:
        L = float(file_boxsize)
        # V_total relies on n_subvolumes, which falls back to N_SUBVOLUMES when
        # the file does not store it; catch a box that cannot hold the data.
        if len(pos) > 0 and np.max(pos) > L * (1.0 + 1e-3):
            raise ValueError(
                f"{where}: positions extend to {np.max(pos):.3f}, beyond the box "
                f"size L={L:.3f} derived from Parameters/volume. Pass boxsize= "
                "explicitly."
            )
    elif len(pos) > 0:
        L = float(np.max(np.ptp(pos, axis=0)))
        warnings.warn(
            f"{where}: no Parameters/volume in the file; inferring the box size "
            f"from the position extent (L={L:.3f}). Pass boxsize= to avoid "
            "biasing xi(r) low.",
            RuntimeWarning,
            stacklevel=3,
        )
    else:
        raise RuntimeError(f"Cannot determine the box size for {where}")

    if not np.isfinite(L) or L <= 0:
        raise RuntimeError(f"Invalid box size for {where}: L={L}")
    return L


def _wrap_into_box(pos: np.ndarray, boxsize: float) -> np.ndarray:
    """Map positions periodically into [0, boxsize)."""
    wrapped = np.mod(np.asarray(pos, dtype=np.float64), boxsize)
    # np.mod can round a tiny negative coordinate up to exactly boxsize.
    wrapped[wrapped >= boxsize] = 0.0
    return wrapped


def compute_xi_corrfunc(
    positions: np.ndarray,
    boxsize: float,
    rbins: Optional[np.ndarray] = None,
    nthreads: int = 4,
) -> pl.DataFrame:
    """Compute the real-space two-point correlation xi(r) using Corrfunc.

    For periodic subvolumes, uses Corrfunc.theory.DD to count pairs
    with periodic boundary conditions, then uses Landy-Szalay estimator
    with analytic random pair counts.

    Args:
        positions: (N,3) array with coordinates (physical positions in subvolume)
        boxsize: Side length of the subvolume (same units as r)
        rbins: Radial bin edges. Defaults to config.DEFAULT_RBINS
        nthreads: Number of OpenMP threads for parallel execution

    Returns:
        DataFrame with columns ['r', 'xi'] and metadata in df.attrs
    """
    if rbins is None:
        rbins = DEFAULT_RBINS
    rbins = np.asarray(rbins, dtype=float)

    # Corrfunc requires rmax < boxsize/2 for periodic boxes; drop larger edges.
    rmax_periodic = boxsize / 2.0
    rbins = rbins[rbins < rmax_periodic]

    if len(rbins) < 2:
        raise ValueError(
            f"No valid rbins within periodic limit (rmax={rmax_periodic:.2f} Mpc/h). "
            "Cannot compute correlation."
        )

    ngal = positions.shape[0]
    if ngal < 2:
        # Not enough galaxies for correlation
        r_centers = 0.5 * (rbins[:-1] + rbins[1:])
        df = pl.DataFrame(
            {
                "r": r_centers,
                "xi": np.full_like(r_centers, np.nan),
            }
        )
        df.attrs = {"rbins": rbins, "ngal": ngal}
        return df

    # Use Corrfunc's xi calculator for periodic boxes to avoid manual normalization bugs
    # corrfunc_xi applies the Landy-Szalay estimator internally and returns xi directly.
    corrfunc_xi = import_optional("Corrfunc.theory.xi").xi
    results = corrfunc_xi(
        boxsize=boxsize,
        nthreads=nthreads,
        binfile=rbins,
        X=positions[:, 0],
        Y=positions[:, 1],
        Z=positions[:, 2],
        output_ravg=True,
    )

    ravg = np.array([x["ravg"] for x in results], dtype=np.float64)
    xi_vals = np.array([x["xi"] for x in results], dtype=np.float64)

    # Use ravg if available, otherwise fall back to bin centers
    if np.all(np.isfinite(ravg) & (ravg > 0)):
        r = ravg
    else:
        r = 0.5 * (rbins[:-1] + rbins[1:])

    df = pl.DataFrame({"r": r, "xi": xi_vals})
    df.attrs = {"rbins": rbins, "ngal": ngal}
    return df


def _xi_for_subvolume(
    pos: np.ndarray,
    z_val: Optional[float],
    iz_path: str,
    ivol: int,
    rbins: Optional[np.ndarray],
    nthreads: int,
    boxsize: Optional[float],
) -> pl.DataFrame:
    """xi(r) of one subvolume's positions in its true periodic box."""
    meta = _subvolume_metadata(iz_path, ivol)
    L = _resolve_boxsize(pos, boxsize, meta["boxsize"], f"{iz_path}/ivol{ivol}")
    res = compute_xi_corrfunc(
        _wrap_into_box(pos, L), boxsize=L, rbins=rbins, nthreads=nthreads
    )
    res.attrs.update(
        {
            "z": z_val if z_val is not None else meta["z"],
            "ivol": ivol,
            "V_ivol": meta["V_ivol"],
            "boxsize": L,
        }
    )
    return res


def correlation_given_redshift_and_subvolume(
    iz_path: str,
    ivol: int,
    rbins: Optional[np.ndarray] = None,
    nthreads: int = 4,
    centrals_only: bool = True,
    mhalo_min: Optional[float] = None,
    boxsize: Optional[float] = None,
) -> Optional[pl.DataFrame]:
    """High-level helper mirroring the HMF API: xi(r) for (snapshot, ivol).

    When centrals_only=True, uses only central galaxies (is_central=1).
    When centrals_only=False, uses all galaxies (centrals + satellites).

    Each subvolume is an independent realisation of the full simulation box,
    so positions span the whole box. The periodic box size is taken from
    ``Parameters/volume`` (L = (V_ivol * n_subvolumes)^(1/3)) unless
    ``boxsize`` is given.

    Args:
        iz_path: Path to snapshot directory (e.g., str(get_base_dir()/"iz207"))
        ivol: Subvolume number
        rbins: Radial bin edges (Mpc/h). Defaults to config.DEFAULT_RBINS.
            Edges at or beyond boxsize/2 are dropped.
        nthreads: Number of OpenMP threads for Corrfunc
        centrals_only: If True, keep only central galaxies (is_central==1)
        mhalo_min: Minimum halo mass (mhalo) in Msun/h. None = no cut.
        boxsize: Periodic box side in Mpc/h. None = read it from the file.

    Returns:
        DataFrame with columns ['r', 'xi'] and metadata in df.attrs (xi is NaN
        when fewer than two galaxies are selected). Returns None if the file
        cannot be read.
    """
    try:
        pos, z_val = _load_positions_from_hdf5(
            iz_path, ivol, centrals_only=centrals_only, mhalo_min=mhalo_min
        )
        return _xi_for_subvolume(pos, z_val, iz_path, ivol, rbins, nthreads, boxsize)
    except (FileNotFoundError, RuntimeError, KeyError):
        # Graceful failure to mirror other analysis helpers
        return None


def halo_correlation_given_redshift_and_subvolume(
    iz_path: str,
    ivol: int,
    rbins: Optional[np.ndarray] = None,
    nthreads: int = 4,
    mhhalo_min: Optional[float] = None,
    boxsize: Optional[float] = None,
) -> Optional[pl.DataFrame]:
    """Compute dark matter halo correlation function from GALFORM halo positions.

    DM halos are represented by central galaxies of main halos (is_central=1, ihhalo=1)
    from the galaxies.hdf5 file. Uses host halo mass (mhhalo) for filtering.

    Args:
        iz_path: Path to snapshot directory
        ivol: Subvolume number
        rbins: Radial bin edges (Mpc/h). Defaults to DEFAULT_RBINS
        nthreads: Number of OpenMP threads for Corrfunc
        mhhalo_min: Optional minimum host halo mass cut in Msun/h
        boxsize: Periodic box side in Mpc/h. None = read it from the file.

    Returns:
        DataFrame with columns ['r', 'xi'] and metadata in df.attrs (xi is NaN
        when fewer than two halos are selected). Returns None if the file
        cannot be read.
    """
    try:
        pos, z_val = read_halo_positions(iz_path, ivol, mhhalo_min=mhhalo_min)
        res = _xi_for_subvolume(pos, z_val, iz_path, ivol, rbins, nthreads, boxsize)
        res.attrs["nhalo"] = res.attrs.get("ngal")
        return res
    except (FileNotFoundError, RuntimeError, KeyError):
        return None


def avg_correlation_given_redshift_and_subvolumes(
    iz_num: int,
    ivols: List[int],
    rbins: Optional[np.ndarray] = None,
    nthreads: int = 16,
    base_dir: Optional[str] = None,
    centrals_only: bool = True,
    mhalo_min: Optional[float] = None,
    boxsize: Optional[float] = None,
) -> Optional[pl.DataFrame]:
    """Compute 2PCF by combining galaxies from multiple subvolumes into one box.

    CRITICAL: Subvolumes are overlapping realizations of the SAME spatial volume.
    Each subvolume samples 1/1024 of the galaxy population in the same spatial box.
    When combining N subvolumes, we get N/1024 of the full population in the SAME box.

    Correct approach: Combine ALL galaxy positions from multiple subvolumes into
    a single box, then compute xi(r) once on this denser population. This gives
    the correlation function for a population N times denser than a single subvolume.

    Args:
        iz_num: Numeric snapshot identifier (e.g. 207 for 'iz207').
        ivols: List of subvolume indices to combine.
        rbins: Optional radial bin edges (defaults to DEFAULT_RBINS).
        nthreads: Number of OpenMP threads for Corrfunc.
        base_dir: Optional base directory; defaults to configured base dir.
        centrals_only: If True, only include central galaxies (is_central=1)
        mhalo_min: Minimum halo mass (mhalo) in Msun/h. None = no cut.
        boxsize: Periodic box side in Mpc/h. None = read it from the first
            usable subvolume's file.
    Returns:
        DataFrame with columns ['r', 'xi'] and metadata in df.attrs.
        Returns None if no subvolume contributed any galaxies.
    """
    if base_dir is None:
        base_dir = str(get_base_dir())

    iz_path = os.path.join(base_dir, f"iz{iz_num}")
    if not os.path.isdir(iz_path):
        return None

    all_positions = []
    z = None
    V_ivol = None
    L_box = None

    for iv in ivols:
        try:
            pos, z_val = _load_positions_from_hdf5(
                iz_path, iv, centrals_only=centrals_only, mhalo_min=mhalo_min
            )
            if len(pos) == 0:
                continue
            meta = _subvolume_metadata(iz_path, iv)
            if L_box is None:
                L_box = _resolve_boxsize(
                    pos, boxsize, meta["boxsize"], f"{iz_path}/ivol{iv}"
                )
        except (FileNotFoundError, RuntimeError, KeyError):
            continue

        if z is None:
            z = z_val if z_val is not None else meta["z"]
        if V_ivol is None:
            V_ivol = meta["V_ivol"]
        all_positions.append(_wrap_into_box(pos, L_box))

    if not all_positions:
        return None

    # Combine all positions into one dataset (same box, more galaxies)
    combined_positions = np.vstack(all_positions)

    res = compute_xi_corrfunc(
        combined_positions, boxsize=L_box, rbins=rbins, nthreads=nthreads
    )

    res.attrs.update(
        {
            "z": z,
            "iz": f"iz{iz_num}",
            "V_ivol": V_ivol,
            "boxsize": L_box,
            "n_used": len(all_positions),
            "n_ivols": len(all_positions),
            "total_galaxies": combined_positions.shape[0],
            "method": "combined_overlapping_subvolumes",
        }
    )
    return res


def correlations_given_redshifts_and_subvolume(
    iz_nums: List[int],
    ivol: int,
    rbins: Optional[np.ndarray] = None,
    nthreads: int = 4,
    base_dir: Optional[str] = None,
    centrals_only: bool = True,
    mhalo_min: Optional[float] = None,
    boxsize: Optional[float] = None,
) -> List[pl.DataFrame]:
    """Compute correlation function for one subvolume across multiple snapshots.

    Args:
        iz_nums: List of numeric snapshot identifiers (e.g. [100, 120, 142]).
        ivol: Subvolume index.
        rbins: Optional radial bin edges (defaults to DEFAULT_RBINS).
        nthreads: Number of OpenMP threads for Corrfunc.
        base_dir: Optional base directory; defaults to configured base dir.
        centrals_only: If True, only include central galaxies (is_central=1)
        mhalo_min: Minimum halo mass (mhalo) in Msun/h. None = no cut.
        boxsize: Periodic box side in Mpc/h. None = read it from each file.

    Returns:
        List of DataFrames from correlation_given_redshift_and_subvolume, one
        per available snapshot, each with ``attrs['iz']`` set (e.g. 'iz100').
        Snapshots whose data is unavailable are skipped.
    """
    if base_dir is None:
        base_dir = str(get_base_dir())

    results = []
    for iz_num in iz_nums:
        iz_path = os.path.join(base_dir, f"iz{iz_num}")
        if not os.path.isdir(iz_path):
            continue

        res = correlation_given_redshift_and_subvolume(
            iz_path,
            ivol,
            rbins=rbins,
            nthreads=nthreads,
            centrals_only=centrals_only,
            mhalo_min=mhalo_min,
            boxsize=boxsize,
        )
        if res is not None:
            res.attrs["iz"] = f"iz{iz_num}"
            results.append(res)

    return results


def avg_correlation_given_subvolume_and_redshifts(
    iz_nums: List[int],
    ivol: int,
    rbins: Optional[np.ndarray] = None,
    nthreads: int = 4,
    base_dir: Optional[str] = None,
    centrals_only: bool = True,
    mhalo_min: Optional[float] = None,
    boxsize: Optional[float] = None,
) -> Optional[pl.DataFrame]:
    """Average xi(r) across multiple redshifts for a single subvolume.

    Args:
        iz_nums: List of numeric snapshot identifiers (e.g., [100, 120, 142]).
        ivol: Subvolume index to evaluate.
        rbins: Optional radial bin edges; defaults to ``DEFAULT_RBINS``.
        nthreads: Number of OpenMP threads for Corrfunc.
        base_dir: Optional base directory for snapshots; defaults to configured
            base dir.
        centrals_only: If True, only include central galaxies (is_central==1).
        mhalo_min: Minimum halo mass threshold in Msun/h; None applies no cut.
        boxsize: Periodic box side in Mpc/h. None = read it from each file.
    Returns:
        DataFrame with columns ['r', 'xi', 'xi_std'] and metadata in df.attrs.
        ``r`` is taken from the first snapshot used. Snapshots with fewer than
        two selected galaxies are skipped. Returns None if no snapshot produced
        valid data.
    """
    results = correlations_given_redshifts_and_subvolume(
        iz_nums,
        ivol,
        rbins=rbins,
        nthreads=nthreads,
        base_dir=base_dir,
        centrals_only=centrals_only,
        mhalo_min=mhalo_min,
        boxsize=boxsize,
    )
    results = [res for res in results if res.attrs.get("ngal", 0) >= 2]
    if not results:
        return None

    per_xi_arr = np.vstack([res["xi"].to_numpy() for res in results])
    df = pl.DataFrame(
        {
            "r": results[0]["r"].to_numpy(),
            "xi": per_xi_arr.mean(axis=0),
            "xi_std": per_xi_arr.std(axis=0),
        }
    )
    df.attrs = {
        "ivol": ivol,
        "n_used": len(results),
        "used_iz": [res.attrs["iz"] for res in results],
        "used_z": [res.attrs.get("z") for res in results],
        "rbins": results[0].attrs["rbins"],
    }
    return df
