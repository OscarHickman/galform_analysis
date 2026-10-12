"""Readers for GALFORM per-band luminosities.

GALFORM lists its filters once per file in the ``Bands`` group (``bandname``,
``iselect``, ``idband``, ``lambda_eff``) and stores, in each output group,
``Bands/BandNNN_Lum_Disk`` and ``Bands/BandNNN_Lum_Bulge`` for every galaxy
(``NNN`` is the 1-based position in that table). These luminosities are
intrinsic (no dust attenuation) and in units where GALFORM's own conversion
gives the AB absolute magnitude ``M - 5 log10 h = -2.5 log10(L)``.

``iselect`` is 0 for a rest-frame band and 1 for an observer-frame band; other
values mark special outputs (bolometric, Lyman continuum, ...) that can only be
selected by index. A filter is named by the first word of its ``bandname``,
e.g. ``SDSS-r``, ``UKIRT-K``, ``Bj`` or ``GALEX-FUV``.

``Bands/a0_ab`` holds each band's Vega - AB offset: GALFORM writes Vega
magnitudes as ``M_Vega = M_AB + a0_ab``.
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import polars as pl

FRAMES: Dict[str, int] = {"rest": 0, "observer": 1}
COMPONENTS: Dict[str, Tuple[str, ...]] = {
    "total": ("Disk", "Bulge"),
    "disk": ("Disk",),
    "bulge": ("Bulge",),
}
SYSTEMS = ("AB", "Vega")


def _decode(values) -> List[str]:
    return [
        v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else str(v)
        for v in np.atleast_1d(values)
    ]


def _band_table(f: h5py.File) -> Dict[str, list]:
    """Columns of the file's band table, or raise KeyError if it has none."""
    if "Bands" not in f or "bandname" not in f["Bands"]:
        raise KeyError(f"{f.filename} has no Bands table")
    table = f["Bands"]
    names = [n.strip() for n in _decode(table["bandname"][()])]
    n_band = len(names)

    def column(key, fill):
        if key not in table:
            return [fill] * n_band
        values = np.atleast_1d(table[key][()]).tolist()
        if len(values) != n_band:
            raise ValueError(
                f"{f.filename}: Bands/{key} has {len(values)} entries but "
                f"bandname has {n_band}"
            )
        return values

    return {
        "name": names,
        "label": [n.split()[0] if n.split() else "" for n in names],
        "iselect": [int(v) for v in column("iselect", 0)],
        "idband": column("idband", None),
        "lambda_eff": [float(v) for v in column("lambda_eff", np.nan)],
        "a0_ab": [float(v) for v in column("a0_ab", np.nan)],
    }


def _frame_name(iselect: int) -> str:
    return {v: k for k, v in FRAMES.items()}.get(iselect, "other")


def list_bands(iz_path: str, ivol: int = 0) -> pl.DataFrame:
    """Table of the bands in one subvolume's galaxies.hdf5.

    Args:
        iz_path: Path to the snapshot directory (e.g. '/path/to/iz271').
        ivol: Subvolume index.

    Returns:
        polars DataFrame with columns ``index`` (1-based, as used in
        ``BandNNN``), ``label`` (the name to pass as ``band``), ``frame``
        ('rest', 'observer' or 'other'), ``lambda_eff`` [Angstrom],
        ``a0_ab`` (Vega - AB offset) and ``name`` (the full GALFORM
        description).

    Raises:
        FileNotFoundError: if the file does not exist.
        KeyError: if the file has no band table.
    """
    fpath = os.path.join(iz_path, f"ivol{ivol}", "galaxies.hdf5")
    if not os.path.exists(fpath):
        raise FileNotFoundError(fpath)
    with h5py.File(fpath, "r") as f:
        table = _band_table(f)
    return pl.DataFrame(
        {
            "index": list(range(1, len(table["name"]) + 1)),
            "label": table["label"],
            "frame": [_frame_name(s) for s in table["iselect"]],
            "lambda_eff": table["lambda_eff"],
            "a0_ab": table["a0_ab"],
            "name": table["name"],
        }
    )


def find_band(f: h5py.File, band: Union[str, int], frame: str = "rest") -> int:
    """1-based index of ``band`` in the file's band table.

    Args:
        f: Open galaxies.hdf5 file.
        band: Filter label (first word of ``bandname``, case-sensitive, e.g.
            'SDSS-r') or a 1-based band index, which is used as given.
        frame: 'rest' or 'observer'; ignored when ``band`` is an index.

    Raises:
        KeyError: if the file has no band table.
        ValueError: if the band or frame is unknown, the index is out of range,
            or the label names two different filters in the same frame.
    """
    if frame not in FRAMES:
        raise ValueError(f"frame must be one of {sorted(FRAMES)}, got {frame!r}")
    table = _band_table(f)
    n_band = len(table["name"])

    if isinstance(band, bool) or not isinstance(band, (str, int, np.integer)):
        raise TypeError(f"band must be a label or an int index, got {band!r}")
    if isinstance(band, (int, np.integer)):
        if not 1 <= band <= n_band:
            raise ValueError(f"band index {band} out of range 1..{n_band}")
        return int(band)

    wanted = FRAMES[frame]
    matches = [
        i
        for i, (label, sel) in enumerate(zip(table["label"], table["iselect"]))
        if label == band and sel == wanted
    ]
    if not matches:
        available = list(
            dict.fromkeys(
                label
                for label, sel in zip(table["label"], table["iselect"])
                if sel == wanted
            )
        )
        raise ValueError(
            f"band {band!r} not found in the {frame}-frame bands of {f.filename}; "
            f"available: {', '.join(available)}. Use list_bands() to see all "
            "bands, or pass an integer index."
        )
    ids = {table["idband"][i] for i in matches}
    if len(matches) > 1 and (len(ids) > 1 or None in ids):
        raise ValueError(
            f"band {band!r} is ambiguous in {f.filename}: indices "
            f"{[i + 1 for i in matches]} are different (or unidentified) filters; "
            "pass an index."
        )
    return matches[0] + 1


def band_label(f: h5py.File, index: int) -> Tuple[str, str]:
    """``(label, frame)`` of the 1-based band ``index``."""
    table = _band_table(f)
    return table["label"][index - 1], _frame_name(table["iselect"][index - 1])


def ab_to_system_offset(f: h5py.File, index: int, system: str = "AB") -> float:
    """Offset to add to an AB magnitude of band ``index`` to get ``system``.

    Args:
        f: Open galaxies.hdf5 file.
        index: 1-based band index.
        system: 'AB' (offset 0) or 'Vega' (GALFORM's ``Bands/a0_ab``).

    Raises:
        ValueError: if ``system`` is unknown.
        KeyError: if 'Vega' is requested and the file has no ``a0_ab``.
    """
    if system not in SYSTEMS:
        raise ValueError(f"system must be one of {list(SYSTEMS)}, got {system!r}")
    if system == "AB":
        return 0.0
    if "Bands" not in f or "a0_ab" not in f["Bands"]:
        raise KeyError(f"{f.filename} has no Bands/a0_ab for Vega magnitudes")
    return float(np.atleast_1d(f["Bands"]["a0_ab"][()])[index - 1])


def band_luminosity(
    g: h5py.Group, index: int, component: str = "total"
) -> Optional[np.ndarray]:
    """Luminosity of band ``index`` for every galaxy in output group ``g``.

    Args:
        g: An ``OutputNNN`` group.
        index: 1-based band index.
        component: 'total' (disk + bulge), 'disk' or 'bulge'.

    Returns:
        float64 array in GALFORM units, or None if the datasets are missing.
    """
    if component not in COMPONENTS:
        raise ValueError(
            f"component must be one of {sorted(COMPONENTS)}, got {component!r}"
        )
    if "Bands" not in g:
        return None
    keys = [f"Band{index:03d}_Lum_{part}" for part in COMPONENTS[component]]
    if any(key not in g["Bands"] for key in keys):
        return None
    return sum(np.asarray(g["Bands"][key], dtype=np.float64) for key in keys)


def luminosity_to_magnitude(lum: np.ndarray) -> np.ndarray:
    """AB absolute magnitude ``M - 5 log10 h`` from a GALFORM luminosity.

    Non-positive or non-finite luminosities give NaN.
    """
    lum = np.asarray(lum, dtype=np.float64)
    out = np.full(lum.shape, np.nan)
    ok = np.isfinite(lum) & (lum > 0)
    out[ok] = -2.5 * np.log10(lum[ok])
    return out
