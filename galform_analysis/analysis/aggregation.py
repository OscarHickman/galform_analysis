"""Analysis functions for aggregating GALFORM data across subvolumes."""

import glob
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import h5py
import numpy as np
import polars as pl

from galform_analysis.config import get_base_dir
from galform_analysis.readers.loaders import (
    close_snapshot,
    get_completed_subvolumes,
    list_subvolume_dirs,
    read_completion_flag,
    read_snapshot_data,
)

# Files smaller than this cannot hold a finished GALFORM output.
_MIN_COMPLETE_FILE_BYTES = 1000

_COMPLETED_SCHEMA = {
    "iz": pl.Utf8,
    "iz_num": pl.Int64,
    "ivol": pl.Int64,
    "path": pl.Utf8,
    "completed": pl.Boolean,
}
_INCOMPLETE_SCHEMA = {
    "iz": pl.Utf8,
    "iz_num": pl.Int64,
    "ivol": pl.Int64,
    "path": pl.Utf8,
    "reason": pl.Utf8,
}


def _snapshot_dirs(
    basedir: Union[str, Path], iz_snapshots: Optional[List[int]]
) -> List[Tuple[str, int, str]]:
    """``(iz_name, iz_num, path)`` for each snapshot directory to scan."""
    if iz_snapshots is not None:
        paths = [os.path.join(basedir, f"iz{iz}") for iz in iz_snapshots]
    else:
        paths = glob.glob(os.path.join(basedir, "iz*"))

    found = []
    for path in paths:
        name = Path(path).name
        suffix = name[2:]
        if os.path.isdir(path) and suffix.isdigit():
            found.append((name, int(suffix), path))
    return sorted(found, key=lambda item: item[1])


def _subvolume_status(gal_file: str) -> str:
    """Classify a galaxies.hdf5 file.

    Returns one of ``"complete"``, ``"missing"``, ``"incomplete"`` (too small or
    CompletionFlag != 1), ``"corrupted"`` (cannot be opened as HDF5) or
    ``"inaccessible"``.
    """
    if not os.path.exists(gal_file):
        return "missing"
    try:
        if os.path.getsize(gal_file) < _MIN_COMPLETE_FILE_BYTES:
            return "incomplete"
    except OSError:
        return "inaccessible"
    try:
        with h5py.File(gal_file, "r", swmr=True) as f:
            flag = read_completion_flag(f)
    except (OSError, KeyError, RuntimeError, ValueError):
        return "corrupted"
    return "complete" if flag == 1 else "incomplete"


def _scan(
    basedir: Optional[Union[str, Path]], iz_snapshots: Optional[List[int]]
) -> Iterator[Dict[str, Any]]:
    """Yield one record (with a ``status``) per ``iz*/ivol<N>`` directory."""
    if basedir is None:
        basedir = get_base_dir()
    for iz_name, iz_num, iz_dir in _snapshot_dirs(basedir, iz_snapshots):
        for ivol_num, ivol_dir in list_subvolume_dirs(iz_dir):
            gal_file = os.path.join(ivol_dir, "galaxies.hdf5")
            yield {
                "iz": iz_name,
                "iz_num": iz_num,
                "ivol": ivol_num,
                "path": gal_file,
                "status": _subvolume_status(gal_file),
            }


def completed_galaxies(
    basedir: Optional[Union[str, Path]] = None,
    iz_snapshots: Optional[List[int]] = None,
) -> pl.DataFrame:
    """Scan base directory and return DataFrame of all galaxies.hdf5 files.

    Looks through all iz*/ivol* directories and reads the CompletionFlag of
    each galaxies.hdf5 file. Subvolumes without a galaxies.hdf5 are omitted.

    Args:
        basedir: Base directory containing iz* snapshot folders. Defaults to
            the configured base directory (``get_base_dir()``) at call time.
        iz_snapshots: Optional list of snapshot numbers (e.g., [82, 100, 105]).
                     If provided, only these snapshots will be scanned.
                     If None, all iz* directories are scanned.

    Returns:
        DataFrame sorted by (iz_num, ivol) with columns:
            - iz: Snapshot name (e.g., 'iz100')
            - iz_num: Numeric iz value (e.g., 100)
            - ivol: Subvolume number
            - path: Full path to the galaxies.hdf5 file
            - completed: Whether the file opens and has CompletionFlag == 1
    """
    records = []
    for rec in _scan(basedir, iz_snapshots):
        status = rec.pop("status")
        if status != "missing":
            records.append({**rec, "completed": status == "complete"})
    return pl.DataFrame(records, schema=_COMPLETED_SCHEMA, orient="row").sort(
        ["iz_num", "ivol"]
    )


def incomplete_subvolumes(
    basedir: Optional[Union[str, Path]] = None,
    iz_snapshots: Optional[List[int]] = None,
) -> pl.DataFrame:
    """Scan base directory and return DataFrame of incomplete/missing galaxy files.

    This is the complement of completed_galaxies(), plus subvolume directories
    that have no galaxies.hdf5 at all.

    Args:
        basedir: Base directory containing iz* snapshot folders. Defaults to
            the configured base directory (``get_base_dir()``) at call time.
        iz_snapshots: Optional list of snapshot numbers (e.g., [82, 100, 105]).
                     If provided, only these snapshots will be scanned.
                     If None, all iz* directories are scanned.

    Returns:
        DataFrame sorted by (iz_num, ivol) with columns:
            - iz: Snapshot name (e.g., 'iz100')
            - iz_num: Numeric iz value (e.g., 100)
            - ivol: Subvolume number
            - path: Path to the expected galaxies.hdf5 file (may not exist)
            - reason: 'missing', 'incomplete' (truncated or CompletionFlag != 1),
              'corrupted' (not readable as HDF5) or 'inaccessible'
    """
    records = []
    for rec in _scan(basedir, iz_snapshots):
        status = rec.pop("status")
        if status != "complete":
            records.append({**rec, "reason": status})
    return pl.DataFrame(records, schema=_INCOMPLETE_SCHEMA, orient="row").sort(
        ["iz_num", "ivol"]
    )


def _read_ivol_masses(
    iz_path: str, ivol: int
) -> Optional[Tuple[np.ndarray, np.ndarray, float, Optional[float]]]:
    """``(mstar, mhalo, V_ivol, z)`` for one subvolume, or None if unusable."""
    try:
        data = read_snapshot_data(iz_path, ivol=ivol)
    except (OSError, RuntimeError, KeyError, ValueError) as exc:
        warnings.warn(f"Skipping {iz_path}/ivol{ivol}: {exc}", stacklevel=3)
        return None
    try:
        mstar, mhalo, volume = data["mstar"], data["mhalo"], data["V_ivol"]
        z = data["z"]
    finally:
        close_snapshot(data)

    if not volume or volume <= 0:
        problem = "no positive Parameters/volume"
    elif mstar.size == 0 or mhalo.size == 0:
        problem = "missing stellar or halo masses"
    elif mstar.shape != mhalo.shape:
        problem = f"mstar has {mstar.size} entries but mhalo has {mhalo.size}"
    else:
        return mstar, mhalo, volume, z
    warnings.warn(f"Skipping {iz_path}/ivol{ivol}: {problem}", stacklevel=3)
    return None


def aggregate_snapshot(iz_path: str) -> Optional[Dict[str, Any]]:
    """Aggregate mstar, mhalo, and volume over the completed ivols of a snapshot.

    Only subvolumes whose galaxies.hdf5 has CompletionFlag == 1 are used.
    A completed subvolume is skipped, with a warning, if it has no positive
    ``Parameters/volume`` or its stellar and halo mass arrays are missing or
    of different lengths, so that ``mstar[i]`` and ``mhalo[i]`` always refer
    to the same galaxy and ``volume`` covers exactly the galaxies returned.

    Args:
        iz_path: Path to the snapshot directory

    Returns:
        Dictionary with keys: 'iz', 'z', 'volume', 'mstar', 'mhalo' and
        'ivols' (the subvolumes used, in numeric order).
        Returns None if no subvolume could be used.
    """
    all_mstar, all_mhalo, used = [], [], []
    total_vol = 0.0
    z = None

    for ivol in get_completed_subvolumes(iz_path):
        result = _read_ivol_masses(iz_path, ivol)
        if result is None:
            continue
        mstar, mhalo, volume, z_ivol = result
        all_mstar.append(mstar)
        all_mhalo.append(mhalo)
        used.append(ivol)
        total_vol += volume
        if z is None:
            z = z_ivol

    if not used:
        return None

    return {
        "iz": Path(iz_path).name,
        "z": z,
        "volume": total_vol,
        "mstar": np.concatenate(all_mstar),
        "mhalo": np.concatenate(all_mhalo),
        "ivols": used,
    }
