"""Centralized readers for galaxies.hdf5 files.

This module provides reusable helpers for loading galaxy data from a single
GALFORM subvolume into NumPy arrays or polars DataFrames, with consistent
filtering and metadata handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import polars as pl

from galform_analysis.readers.loaders import (
    get_output_group,
    open_galaxies_hdf5,
    read_volumes,
    resolve_redshift,
)

_POSITION_KEYS = (("xgal", "x"), ("ygal", "y"), ("zgal", "z"))


def _normalize_arrays(
    arrays: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], int]:
    """Flatten arrays to 1D and check they describe the same galaxies.

    ``None`` entries (fields absent from the file) are dropped.

    Raises:
        ValueError: If the remaining arrays have different lengths.
    """
    arrays = {k: np.ravel(v) for k, v in arrays.items() if v is not None}
    if not arrays:
        return {}, 0
    lengths = {k: len(v) for k, v in arrays.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"Galaxy arrays have inconsistent lengths: {lengths}")
    return arrays, next(iter(lengths.values()))


def _apply_mask(
    arrays: Dict[str, np.ndarray], mask: np.ndarray
) -> Dict[str, np.ndarray]:
    """Apply a boolean mask to all arrays."""
    return {k: v[mask] for k, v in arrays.items()}


def _first_present(g, candidates) -> Optional[np.ndarray]:
    """First of ``candidates`` present in group ``g``, or None if none is."""
    for name in candidates:
        if name in g:
            return np.asarray(g[name])
    return None


def _read_positions(g, arrays: Dict[str, np.ndarray]) -> None:
    for key, alias in _POSITION_KEYS:
        if key not in g:
            raise KeyError(
                "Could not find xgal/ygal/zgal position arrays in Output group"
            )
        arrays[alias] = np.asarray(g[key])


def _read_fields(g, arrays: Dict[str, np.ndarray], fields) -> None:
    for name in fields or ():
        if name not in arrays and name in g:
            arrays[name] = np.asarray(g[name])


def _cut_mask(
    arrays: Dict[str, np.ndarray], n: int, cuts: Dict[str, Optional[float]]
) -> np.ndarray:
    """Mask selecting rows with ``arrays[field] >= minimum`` for every cut."""
    mask = np.ones(n, dtype=bool)
    for field, minimum in cuts.items():
        if minimum is None:
            continue
        if field not in arrays:
            raise KeyError(f"{field} field not found - cannot apply {field} cut")
        mask &= arrays[field] >= minimum
    return mask


def _read_subvolume(
    iz_path: str, ivol: int, collect
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Open one galaxies.hdf5, run ``collect(g) -> (arrays, mask)``, add metadata."""
    f = open_galaxies_hdf5(iz_path, ivol=ivol)
    if f is None:
        raise FileNotFoundError(
            f"Missing or unreadable galaxies.hdf5 at {iz_path}/ivol{ivol}"
        )
    try:
        g = get_output_group(f)
        if g is None:
            raise RuntimeError("No OutputNNN group found in HDF5 file")
        arrays, mask = collect(g)
        meta: Dict[str, Any] = {
            "iz": Path(iz_path).name,
            "ivol": ivol,
            "z": resolve_redshift(f, iz_path, ivol),
            **read_volumes(f),
        }
        return _apply_mask(arrays, mask), meta
    finally:
        f.close()


def read_galaxy_arrays(
    iz_path: str,
    ivol: int = 0,
    fields: Optional[Iterable[str]] = None,
    include_positions: bool = True,
    include_derived: bool = True,
    centrals_only: bool = True,
    mhalo_min: Optional[float] = None,
    mstar_min: Optional[float] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Read galaxy data arrays from galaxies.hdf5 for one subvolume.

    When centrals_only=True, filters to central galaxies (is_central == 1).
    When centrals_only=False, returns all galaxies (centrals + satellites).
    For dark matter halos, use read_halo_arrays() instead.

    Derived fields that are absent from the file are left out of the result
    instead of being returned empty.

    Args:
        iz_path: Path to snapshot directory (e.g., /.../iz207)
        ivol: Subvolume index
        fields: Optional iterable of dataset names to pull directly from the
                output group.
        include_positions: Include x,y,z positions (xgal/ygal/zgal)
        include_derived: Include derived fields (mstar, mhalo, sfr, is_central)
        centrals_only: If True, keep only central galaxies (is_central==1)
        mhalo_min: Minimum subhalo mass (mhalo) threshold; None = no cut
        mstar_min: Minimum stellar mass (mstar) threshold in M_sun/h; None = no cut

    Returns:
        Tuple of (arrays, metadata). Metadata holds iz, ivol, z, V_ivol,
        V_total and n_subvolumes.

    Raises:
        FileNotFoundError: If the file is missing or unreadable.
        RuntimeError: If the file has no OutputNNN group.
        KeyError: If a field needed for a requested cut is missing.
        ValueError: If the per-galaxy arrays have inconsistent lengths.
    """

    def collect(g):
        arrays: Dict[str, np.ndarray] = {}
        if include_positions:
            _read_positions(g, arrays)
        if include_derived:
            if "mstars_disk" in g and "mstars_bulge" in g:
                arrays["mstar"] = np.asarray(g["mstars_disk"]) + np.asarray(
                    g["mstars_bulge"]
                )
            else:
                arrays["mstar"] = _first_present(
                    g, ["mstars", "StellarMass", "Mstar", "mstars_allburst"]
                )
            arrays["mhalo"] = _first_present(g, ["mhalo", "mchalo", "Mhalo", "M_Halo"])
            arrays["sfr"] = _first_present(g, ["mstardot", "Sfr", "sfr", "sfr_disk"])
            arrays["is_central"] = _first_present(g, ["is_central"])
        _read_fields(g, arrays, fields)

        arrays, n = _normalize_arrays(arrays)
        if centrals_only and "is_central" not in arrays:
            raise KeyError(
                "is_central field not found - cannot filter for central galaxies"
            )
        mask = _cut_mask(arrays, n, {"mhalo": mhalo_min, "mstar": mstar_min})
        if centrals_only:
            mask &= arrays["is_central"] == 1
        return arrays, mask

    return _read_subvolume(iz_path, ivol, collect)


def read_galaxies_dataframe(
    iz_path: str,
    ivol: int = 0,
    fields: Optional[Iterable[str]] = None,
    include_positions: bool = True,
    include_derived: bool = True,
    centrals_only: bool = True,
    mhalo_min: Optional[float] = None,
    return_metadata: bool = False,
    mstar_min: Optional[float] = None,
):
    """Read galaxies.hdf5 and return a Polars DataFrame.

    Args are the same as read_galaxy_arrays. If return_metadata is True,
    returns (df, metadata).
    """

    arrays, meta = read_galaxy_arrays(
        iz_path=iz_path,
        ivol=ivol,
        fields=fields,
        include_positions=include_positions,
        include_derived=include_derived,
        centrals_only=centrals_only,
        mhalo_min=mhalo_min,
        mstar_min=mstar_min,
    )

    df = pl.DataFrame(arrays)
    df.attrs = meta
    return (df, meta) if return_metadata else df


def read_halo_arrays(
    iz_path: str,
    ivol: int = 0,
    fields: Optional[Iterable[str]] = None,
    include_positions: bool = True,
    include_derived: bool = True,
    mhhalo_min: Optional[float] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Read DM halo data from galaxies.hdf5 for one subvolume.

    DM halos are represented by central galaxies (is_central=1), which includes both
    main FOF halos and subhalos. Each central galaxy represents the center of its
    (sub)halo. This gives ~96k halos matching the number of central galaxies.
    Uses host halo mass (mhhalo) for filtering.

    Args:
        iz_path: Path to snapshot directory (e.g., /.../iz207)
        ivol: Subvolume index
        fields: Optional iterable of dataset names to pull directly from the
                output group.
        include_positions: Include x,y,z positions (xgal/ygal/zgal)
        include_derived: Include derived fields (mhhalo, is_central)
        mhhalo_min: Minimum host halo mass (mhhalo) threshold; None = no cut

    Returns:
        Tuple of (arrays, metadata)
    """

    def collect(g):
        arrays: Dict[str, np.ndarray] = {}
        if include_positions:
            _read_positions(g, arrays)
        if include_derived:
            arrays["mhhalo"] = _first_present(g, ["mhhalo", "mhalo_host"])
            arrays["is_central"] = _first_present(g, ["is_central"])
        _read_fields(g, arrays, fields)

        arrays, n = _normalize_arrays(arrays)
        # Each central galaxy (is_central==1) represents its (sub)halo centre.
        if "is_central" not in arrays:
            raise KeyError("is_central field required for halo sample")
        mask = _cut_mask(arrays, n, {"mhhalo": mhhalo_min})
        mask &= arrays["is_central"] == 1
        return arrays, mask

    return _read_subvolume(iz_path, ivol, collect)


def read_halo_positions(
    iz_path: str,
    ivol: int,
    mhhalo_min: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[float]]:
    """Load DM halo (halo center) positions and redshift for a subvolume.

    Returns:
        positions: (N,3) array
        z: redshift (if available)
    """
    arrays, meta = read_halo_arrays(
        iz_path=iz_path,
        ivol=ivol,
        fields=None,
        include_positions=True,
        include_derived=True,
        mhhalo_min=mhhalo_min,
    )

    if not all(k in arrays for k in ("x", "y", "z")):
        raise KeyError("Missing position columns in galaxies.hdf5")

    pos = np.vstack([arrays["x"], arrays["y"], arrays["z"]]).T.astype(
        np.float64, copy=False
    )
    return pos, meta.get("z")


def read_galaxy_positions(
    iz_path: str,
    ivol: int,
    centrals_only: bool = True,
    mhalo_min: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[float]]:
    """Load galaxy positions and redshift for a subvolume.

    When centrals_only=True, returns only central galaxies (is_central == 1).
    When centrals_only=False, returns all galaxies (centrals + satellites).

    Returns:
        positions: (N,3) array
        z: redshift (if available)
    """
    arrays, meta = read_galaxy_arrays(
        iz_path=iz_path,
        ivol=ivol,
        fields=None,
        include_positions=True,
        include_derived=True,
        centrals_only=centrals_only,
        mhalo_min=mhalo_min,
    )

    if not all(k in arrays for k in ("x", "y", "z")):
        raise KeyError("Missing position columns in galaxies.hdf5")

    pos = np.vstack([arrays["x"], arrays["y"], arrays["z"]]).T.astype(
        np.float64, copy=False
    )
    return pos, meta.get("z")
