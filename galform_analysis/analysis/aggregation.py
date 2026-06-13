"""Analysis functions for aggregating GALFORM data across subvolumes."""

import glob
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import polars as pl

from galform_analysis.config import get_base_dir
from galform_analysis.readers.loaders import close_snapshot, read_snapshot_data


def completed_galaxies(
    basedir: str = get_base_dir(), iz_snapshots: Optional[List[int]] = None
) -> pl.DataFrame:
    """Scan base directory and return DataFrame of all completed galaxy files.

    Looks through all iz*/ivol* directories and checks CompletionFlag in
    galaxies.hdf5 files.

    Args:
        basedir: Base directory containing iz* snapshot folders
        iz_snapshots: Optional list of snapshot numbers (e.g., [82, 100, 105]).
                     If provided, only these snapshots will be scanned.
                     If None, all iz* directories are scanned.

    Returns:
        DataFrame with columns:
            - iz: Snapshot name (e.g., 'iz100')
            - iz_num: Numeric iz value (e.g., 100)
            - ivol: Subvolume number
            - path: Full path to the galaxies.hdf5 file
            - completed: Whether CompletionFlag==1
    """
    records = []

    if iz_snapshots is not None:
        iz_dirs = sorted(
            [
                os.path.join(basedir, f"iz{iz}")
                for iz in iz_snapshots
                if os.path.isdir(os.path.join(basedir, f"iz{iz}"))
            ]
        )
    else:
        iz_dirs = sorted(glob.glob(os.path.join(basedir, "iz*")))

    for iz_dir in iz_dirs:
        iz_name = Path(iz_dir).name
        try:
            iz_num = int(iz_name.replace("iz", ""))
        except ValueError:
            continue

        ivol_dirs = sorted(glob.glob(os.path.join(iz_dir, "ivol*")))

        for ivol_dir in ivol_dirs:
            ivol_name = Path(ivol_dir).name
            try:
                ivol_num = int(ivol_name.replace("ivol", ""))
            except ValueError:
                continue

            gal_file = os.path.join(ivol_dir, "galaxies.hdf5")
            if not os.path.exists(gal_file):
                continue

            try:
                if os.path.getsize(gal_file) < 1000:
                    records.append({"iz": iz_name, "iz_num": iz_num, "ivol": ivol_num, "path": gal_file, "completed": False})
                    continue
            except OSError:
                continue

            completed = False
            try:
                with h5py.File(gal_file, "r", swmr=True):
                    completed = True
            except (OSError, KeyError, RuntimeError):
                pass

            records.append({"iz": iz_name, "iz_num": iz_num, "ivol": ivol_num, "path": gal_file, "completed": completed})

    df = pl.DataFrame(records)

    if not df.is_empty():
        df = df.sort(["iz_num", "ivol"])

    return df


def incomplete_subvolumes(
    basedir: str = get_base_dir(), iz_snapshots: Optional[List[int]] = None
) -> pl.DataFrame:
    """Scan base directory and return DataFrame of incomplete/missing galaxy files.

    This is the complement of completed_galaxies(). Returns records for subvolumes
    where galaxies.hdf5 either doesn't exist or is incomplete/corrupted.

    Args:
        basedir: Base directory containing iz* snapshot folders
        iz_snapshots: Optional list of snapshot numbers (e.g., [82, 100, 105]).
                     If provided, only these snapshots will be scanned.
                     If None, all iz* directories are scanned.

    Returns:
        DataFrame with columns:
            - iz: Snapshot name (e.g., 'iz100')
            - iz_num: Numeric iz value (e.g., 100)
            - ivol: Subvolume number
            - path: Path to the expected galaxies.hdf5 file (may not exist)
            - reason: Why the file is incomplete ('missing', 'incomplete',
              or 'corrupted')
    """
    records = []

    if iz_snapshots is not None:
        iz_dirs = sorted(
            [
                os.path.join(basedir, f"iz{iz}")
                for iz in iz_snapshots
                if os.path.isdir(os.path.join(basedir, f"iz{iz}"))
            ]
        )
    else:
        iz_dirs = sorted(glob.glob(os.path.join(basedir, "iz*")))

    for iz_dir in iz_dirs:
        iz_name = Path(iz_dir).name
        try:
            iz_num = int(iz_name.replace("iz", ""))
        except ValueError:
            continue

        ivol_dirs = sorted(glob.glob(os.path.join(iz_dir, "ivol*")))

        for ivol_dir in ivol_dirs:
            ivol_name = Path(ivol_dir).name
            try:
                ivol_num = int(ivol_name.replace("ivol", ""))
            except ValueError:
                continue

            gal_file = os.path.join(ivol_dir, "galaxies.hdf5")

            if not os.path.exists(gal_file):
                records.append({"iz": iz_name, "iz_num": iz_num, "ivol": ivol_num, "path": gal_file, "reason": "missing"})
                continue

            try:
                if os.path.getsize(gal_file) < 1000:
                    records.append({"iz": iz_name, "iz_num": iz_num, "ivol": ivol_num, "path": gal_file, "reason": "incomplete"})
                    continue
            except OSError:
                records.append({"iz": iz_name, "iz_num": iz_num, "ivol": ivol_num, "path": gal_file, "reason": "inaccessible"})
                continue

            try:
                with h5py.File(gal_file, "r", swmr=True):
                    pass
            except (OSError, KeyError, RuntimeError):
                records.append({"iz": iz_name, "iz_num": iz_num, "ivol": ivol_num, "path": gal_file, "reason": "corrupted"})

    df = pl.DataFrame(records)

    if not df.is_empty():
        df = df.sort(["iz_num", "ivol"])

    return df


def aggregate_snapshot(iz_path: str) -> Optional[Dict[str, Any]]:
    """Aggregate mstar, mhalo, and volume from all ivols in a snapshot.

    Args:
        iz_path: Path to the snapshot directory

    Returns:
        Dictionary with keys: 'iz', 'z', 'volume', 'mstar', 'mhalo'
        Returns None if no data found
    """
    ivol_paths = sorted(glob.glob(os.path.join(iz_path, "ivol*")))
    if not ivol_paths:
        return None

    all_mstar, all_mhalo = [], []
    total_vol = 0
    z = None

    for ivp in ivol_paths:
        iv = int(Path(ivp).name.replace("ivol", ""))
        try:
            data = read_snapshot_data(iz_path, ivol=iv)
            if data.get("V_ivol") and data["V_ivol"] > 0:
                total_vol += data["V_ivol"]
            if z is None:
                z = data.get("z")

            mstar = data.get("mstar")
            mhalo = data.get("mhalo")
            if mstar is not None:
                all_mstar.append(mstar)
            if mhalo is not None:
                all_mhalo.append(mhalo)

            close_snapshot(data)
        except Exception:
            continue

    if not all_mstar and not all_mhalo:
        return None

    return {
        "iz": Path(iz_path).name,
        "z": z,
        "volume": total_vol,
        "mstar": np.concatenate(all_mstar) if all_mstar else np.array([]),
        "mhalo": np.concatenate(all_mhalo) if all_mhalo else np.array([]),
    }


