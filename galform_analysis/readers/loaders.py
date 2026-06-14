"""Data loading utilities for GALFORM HDF5 outputs."""

import glob
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np

from galform_analysis.config import N_SUBVOLUMES


def get_completed_subvolumes(iz_path: str) -> List[int]:
    """Return ivol numbers where CompletionFlag=1 in iz_path/ivol*/galaxies.hdf5."""
    ivol_dirs = sorted(glob.glob(os.path.join(iz_path, "ivol*")))
    completed = []

    for ivol_dir in ivol_dirs:
        ivol_num = int(Path(ivol_dir).name.replace("ivol", ""))
        fpath = os.path.join(ivol_dir, "galaxies.hdf5")

        if not os.path.exists(fpath) or not _is_hdf5_file(fpath):
            continue

        try:
            with h5py.File(fpath, "r") as f:
                # Check for CompletionFlag
                if "CompletionFlag" in f:
                    flag = f["CompletionFlag"][()]
                    if flag == 1:
                        completed.append(ivol_num)
        except Exception:
            continue

    return completed


def _is_hdf5_file(path: str) -> bool:
    """Check if a file is an HDF5 file by its signature."""
    try:
        with open(path, "rb") as f:
            sig = f.read(8)
        return sig == b"\x89HDF\r\n\x1a\n"
    except Exception:
        return False


def open_galaxies_hdf5(iz_path: str, ivol: int = 0) -> Optional[h5py.File]:
    """Open a galaxies.hdf5 file, returning the h5py.File object or None.

    Args:
        iz_path: Path to the snapshot directory
        ivol: Subvolume number

    Returns:
        h5py.File object or None if file cannot be opened
    """
    fpath = os.path.join(iz_path, f"ivol{ivol}", "galaxies.hdf5")
    if not os.path.exists(fpath):
        return None
    try:
        return h5py.File(fpath, "r")
    except (OSError, Exception):
        return None


def get_output_group(f: Optional[h5py.File]) -> Optional[h5py.Group]:
    """Return the highest-numbered OutputNNN group from an HDF5 file."""
    if not f:
        return None
    outs = [k for k in f.keys() if re.match(r"^Output\d+$", k)]
    if not outs:
        return None
    outs_sorted = sorted(
        outs, key=lambda x: int(re.search(r"Output(\d+)", x).group(1)), reverse=True
    )
    return f[outs_sorted[0]]


def _get_redshift_from_file(f: Optional[h5py.File]) -> Optional[float]:
    """Attempt to read redshift from the highest output group, 'Redshifts' or
    'Output_Times'.
    """
    if not f:
        return None
    try:
        # First, try to read from the highest-numbered Output group's redshift dataset
        g = get_output_group(f)
        if g is not None and "redshift" in g:
            val = g["redshift"]
            if isinstance(val, h5py.Dataset):
                return float(val[()])
    except Exception:
        pass

    try:
        if "Redshifts" in f:
            obj = f["Redshifts"]
            # Case A: dataset-like
            if isinstance(obj, h5py.Dataset):
                z0 = obj[0]
                if isinstance(z0, (bytes, np.bytes_)):
                    z0 = z0.decode("utf-8")
                return float(z0)
            # Case B: group with keys that are stringified redshifts
            if isinstance(obj, h5py.Group):
                vals = []
                for k in obj.keys():
                    try:
                        vals.append(float(k))
                    except Exception:
                        continue
                if vals:
                    # choose the smallest redshift value as a representative
                    # for this file
                    return float(sorted(vals)[0])
    except Exception:
        pass
    try:
        if "Output_Times" in f:
            arr = np.array(f["Output_Times"])
            # Some files store strings like ['aout','nout',...],
            # ignore non-numeric entries
            for x in arr.flat:
                try:
                    return float(x)
                except Exception:
                    continue
    except Exception:
        pass
    return None


def _get_redshift_from_zsnap(iz_path: str, ivol: int) -> Optional[float]:
    """Read redshift from a zsnap.dat file at either snapshot or subvolume level."""
    # Check parent snapshot directory first, then subvolume subdirectory
    paths = [
        os.path.join(iz_path, "zsnap.dat"),
        os.path.join(iz_path, f"ivol{ivol}", "zsnap.dat"),
    ]
    for zfile in paths:
        if os.path.exists(zfile):
            try:
                with open(zfile, "r") as f:
                    line = f.readline().strip()
                    # Try direct float conversion
                    try:
                        return float(line)
                    except ValueError:
                        # Try parsing "iz= 155 z= 1.496"
                        import re

                        match = re.search(r"z\s*=\s*([0-9.-]+)", line)
                        if match:
                            return float(match.group(1))
            except Exception:
                continue
    return None


def resolve_redshift(
    f: Optional[h5py.File], iz_path: str, ivol: int
) -> Optional[float]:
    """Resolve redshift robustly, avoiding falsy z=0.0 issues."""
    z = _get_redshift_from_file(f)
    if z is not None:
        return z
    return _get_redshift_from_zsnap(iz_path, ivol)


def _get_first_array(
    group: h5py.Group, candidates: List[str], default: Optional[np.ndarray] = None
) -> np.ndarray:
    """Helper to robustly fetch arrays by trying multiple candidate keys."""
    for name in candidates:
        if name in group:
            try:
                return np.array(group[name])
            except Exception:
                continue
    return np.array([]) if default is None else default


def read_snapshot_data(iz_path: str, ivol: int = 0) -> Dict[str, Any]:
    """Read key galaxy properties from a single snapshot subvolume.

    Returns dict with keys: file (must be closed!), group, mstar, mhalo, sfr,
    Lg, Lr, z, V_total, V_ivol. Raises FileNotFoundError / RuntimeError on failure.
    """
    f = open_galaxies_hdf5(iz_path, ivol=ivol)
    if f is None:
        raise FileNotFoundError(f"Unreadable or missing HDF5 for {iz_path}/ivol{ivol}")

    g = get_output_group(f)
    if g is None:
        f.close()
        raise RuntimeError("No OutputNNN group found")

    data = {"file": f, "group": g, "iz": Path(iz_path).name, "ivol": ivol}

    # Stellar mass, halo mass, and SFR
    m_disk = _get_first_array(g, ["mstars_disk"])
    m_bulge = _get_first_array(g, ["mstars_bulge"])
    if m_disk.size and m_bulge.size:
        data["mstar"] = m_disk + m_bulge
    else:
        # Fallbacks if split masses are unavailable
        data["mstar"] = _get_first_array(
            g, ["mstars", "StellarMass", "Mstar", "mstars_allburst"]
        )

    data["mhalo"] = _get_first_array(g, ["mhalo", "mchalo", "Mhalo", "M_Halo"])
    data["sfr"] = _get_first_array(g, ["mstardot", "Sfr", "sfr", "sfr_disk"])

    # Band luminosities
    data["Lg"] = data["Lr"] = None
    if "Bands" in f and "bandname" in f["Bands"]:
        names = [
            n.decode("utf-8") if isinstance(n, (bytes, np.bytes_)) else str(n)
            for n in np.array(f["Bands"]["bandname"])
        ]

        def idx_for(label_candidates):
            for t in label_candidates:
                for i, nm in enumerate(names, start=1):
                    if t.lower() in nm.lower():
                        return i
            return None

        ig = idx_for(["sdss-g", "sdss g", " g ", "_g", "sdss_g"])
        ir = idx_for(["sdss-r", "sdss r", " r ", "_r", "sdss_r"])

        if ig is not None and "Bands" in g:
            key_disk = f"Band{ig:03d}_Lum_Disk"
            key_bulge = f"Band{ig:03d}_Lum_Bulge"
            if key_disk in g["Bands"] and key_bulge in g["Bands"]:
                data["Lg"] = np.array(g["Bands"][key_disk]) + np.array(
                    g["Bands"][key_bulge]
                )
        if ir is not None and "Bands" in g:
            key_disk = f"Band{ir:03d}_Lum_Disk"
            key_bulge = f"Band{ir:03d}_Lum_Bulge"
            if key_disk in g["Bands"] and key_bulge in g["Bands"]:
                data["Lr"] = np.array(g["Bands"][key_disk]) + np.array(
                    g["Bands"][key_bulge]
                )

    # Redshift
    data["z"] = resolve_redshift(f, iz_path, ivol)

    data["V_total"] = data["V_ivol"] = None
    if "Parameters" in f and "volume" in f["Parameters"]:
        V_ivol = float(np.array(f["Parameters"]["volume"]))
        data["V_ivol"] = V_ivol
        n_subvol = (
            int(np.array(f["Parameters"]["n_subvolumes"]))
            if "n_subvolumes" in f["Parameters"]
            else N_SUBVOLUMES
        )
        data["V_total"] = V_ivol * n_subvol if n_subvol and n_subvol > 0 else V_ivol

    return data


def close_snapshot(obj: Dict[str, Any]) -> None:
    """Safely close the HDF5 file associated with a snapshot data object.

    Args:
        obj: Dictionary returned by read_snapshot_data
    """
    try:
        if "file" in obj and obj["file"]:
            obj["file"].close()
    except Exception:
        pass
