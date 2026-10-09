"""Data loading utilities for GALFORM HDF5 outputs."""

import glob
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

from galform_analysis.config import N_SUBVOLUMES

_IVOL_DIR_RE = re.compile(r"^ivol(\d+)$")
# "z = 1.496", but not the "z=" inside "iz= 155".
_ZSNAP_RE = re.compile(
    r"(?<![A-Za-z0-9_])z\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
)


def list_subvolume_dirs(iz_path: str) -> List[Tuple[int, str]]:
    """Return ``(ivol, path)`` for every ``ivol<N>`` directory, in numeric order.

    Entries that are not directories or whose suffix is not an integer (for
    example ``ivol_old`` or ``ivol7.tar``) are ignored.
    """
    found = []
    for path in glob.glob(os.path.join(iz_path, "ivol*")):
        match = _IVOL_DIR_RE.match(Path(path).name)
        if match and os.path.isdir(path):
            found.append((int(match.group(1)), path))
    return sorted(found)


def read_completion_flag(f: h5py.File) -> Optional[int]:
    """Return the file's CompletionFlag, or None if it has none."""
    if "CompletionFlag" not in f:
        return None
    flag = np.ravel(f["CompletionFlag"][()])
    return int(flag[0]) if flag.size else None


def get_completed_subvolumes(iz_path: str) -> List[int]:
    """Return ivol numbers, in numeric order, whose galaxies.hdf5 has
    CompletionFlag == 1."""
    completed = []
    for ivol_num, ivol_dir in list_subvolume_dirs(iz_path):
        fpath = os.path.join(ivol_dir, "galaxies.hdf5")
        if not os.path.exists(fpath) or not _is_hdf5_file(fpath):
            continue
        try:
            with h5py.File(fpath, "r") as f:
                if read_completion_flag(f) == 1:
                    completed.append(ivol_num)
        except (OSError, KeyError, ValueError):
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
        if not os.path.exists(zfile):
            continue
        try:
            with open(zfile, "r") as f:
                line = f.readline().strip()
        except OSError:
            continue
        try:
            return float(line)
        except ValueError:
            match = _ZSNAP_RE.search(line)
            if match:
                return float(match.group(1))
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


def read_volumes(f: h5py.File) -> Dict[str, Optional[float]]:
    """Read the subvolume volume and derived total volume from ``Parameters``.

    Returns:
        Dict with ``V_ivol`` (volume of this subvolume), ``n_subvolumes`` (from
        ``Parameters/n_subvolumes`` or the ``N_SUBVOLUMES`` fallback) and
        ``V_total = V_ivol * n_subvolumes``. All are None when the file has no
        ``Parameters/volume``.
    """
    out: Dict[str, Optional[float]] = {
        "V_ivol": None,
        "V_total": None,
        "n_subvolumes": None,
    }
    if "Parameters" not in f or "volume" not in f["Parameters"]:
        return out
    params = f["Parameters"]
    V_ivol = float(np.ravel(params["volume"][()])[0])
    n_subvol = (
        int(np.ravel(params["n_subvolumes"][()])[0])
        if "n_subvolumes" in params
        else N_SUBVOLUMES
    )
    out["V_ivol"] = V_ivol
    out["n_subvolumes"] = n_subvol
    out["V_total"] = V_ivol * n_subvol if n_subvol > 0 else V_ivol
    return out


def _band_index(names: List[str], band: str) -> Optional[int]:
    """1-based index of the SDSS ``band`` filter in ``names``, or None.

    Matches are tried from most to least specific so that, e.g., ``SDSS_r`` is
    preferred over a name that merely ends in ``_r``.
    """
    lowered = [n.strip().lower() for n in names]
    specific = re.compile(rf"sdss[-_ ]?{band}(?![a-z0-9])")
    generic = re.compile(rf"(?:^|[\s_\-]){band}(?:$|[\s_\-])")
    exact = {f"sdss-{band}", f"sdss_{band}", f"sdss {band}", f"sdss{band}"}
    for matches in (
        lambda nm: nm in exact,
        lambda nm: specific.search(nm) is not None,
        lambda nm: generic.search(nm) is not None,
    ):
        for i, nm in enumerate(lowered, start=1):
            if matches(nm):
                return i
    return None


def _band_luminosity(g: h5py.Group, index: Optional[int]) -> Optional[np.ndarray]:
    """Disk + bulge luminosity for band ``index``, or None if unavailable."""
    if index is None or "Bands" not in g:
        return None
    key_disk = f"Band{index:03d}_Lum_Disk"
    key_bulge = f"Band{index:03d}_Lum_Bulge"
    if key_disk not in g["Bands"] or key_bulge not in g["Bands"]:
        return None
    return np.array(g["Bands"][key_disk]) + np.array(g["Bands"][key_bulge])


def read_snapshot_data(iz_path: str, ivol: int = 0) -> Dict[str, Any]:
    """Read key galaxy properties from a single snapshot subvolume.

    Returns dict with keys: file (must be closed with ``close_snapshot``),
    group, iz, ivol, mstar, mhalo, sfr, Lg, Lr, z, V_ivol, V_total and
    n_subvolumes. Raises FileNotFoundError / RuntimeError on failure.
    """
    f = open_galaxies_hdf5(iz_path, ivol=ivol)
    if f is None:
        raise FileNotFoundError(f"Unreadable or missing HDF5 for {iz_path}/ivol{ivol}")

    try:
        g = get_output_group(f)
        if g is None:
            raise RuntimeError(f"No OutputNNN group found in {iz_path}/ivol{ivol}")

        data: Dict[str, Any] = {
            "file": f,
            "group": g,
            "iz": Path(iz_path).name,
            "ivol": ivol,
        }

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

        # SDSS g and r band luminosities
        data["Lg"] = data["Lr"] = None
        if "Bands" in f and "bandname" in f["Bands"]:
            names = [
                n.decode("utf-8") if isinstance(n, (bytes, np.bytes_)) else str(n)
                for n in np.array(f["Bands"]["bandname"])
            ]
            data["Lg"] = _band_luminosity(g, _band_index(names, "g"))
            data["Lr"] = _band_luminosity(g, _band_index(names, "r"))

        data["z"] = resolve_redshift(f, iz_path, ivol)
        data.update(read_volumes(f))
        return data
    except Exception:
        f.close()
        raise


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
