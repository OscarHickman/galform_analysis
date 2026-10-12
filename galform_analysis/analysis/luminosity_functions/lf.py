"""Luminosity function (LF) computation utilities.

Magnitudes are GALFORM's intrinsic (dust-free) AB absolute magnitudes,
``M - 5 log10 h = -2.5 log10(L_disk + L_bulge)``, optionally converted to
Vega with GALFORM's ``a0_ab`` offsets, so phi is in (Mpc/h)^-3 mag^-1.
See :mod:`galform_analysis.readers.bands` for how bands are named and
selected.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import polars as pl

from galform_analysis.analysis.mass_functions._common import (
    _avg_phi_over_snapshots,
    _avg_phi_over_subvolumes,
)
from galform_analysis.config import DEFAULT_MAGNITUDE_BINS, get_base_dir
from galform_analysis.readers.bands import (
    COMPONENTS,
    SYSTEMS,
    ab_to_system_offset,
    band_label,
    band_luminosity,
    find_band,
    luminosity_to_magnitude,
)
from galform_analysis.readers.loaders import (
    get_output_group,
    open_galaxies_hdf5,
    read_volumes,
    resolve_redshift,
)

Band = Union[str, int]
_BAND_KEYS = ("band", "band_index", "frame", "component", "system")


def _check_choices(component: str, system: str) -> None:
    if component not in COMPONENTS:
        raise ValueError(
            f"component must be one of {sorted(COMPONENTS)}, got {component!r}"
        )
    if system not in SYSTEMS:
        raise ValueError(f"system must be one of {list(SYSTEMS)}, got {system!r}")


def _read_magnitudes(
    iz_path: str, ivol: int, band: Band, frame: str, component: str, system: str
) -> Optional[Dict[str, Any]]:
    """Magnitudes of one subvolume plus band, volume and redshift metadata.

    Returns None if the file or its band data is missing; raises ValueError
    if ``band`` is not in the file's band table.
    """
    f = open_galaxies_hdf5(iz_path, ivol=ivol)
    if f is None:
        return None
    with f:
        g = get_output_group(f)
        if g is None or "Bands" not in f or "bandname" not in f["Bands"]:
            return None
        index = find_band(f, band, frame=frame)
        lum = band_luminosity(g, index, component)
        if lum is None:
            return None
        label, band_frame = band_label(f, index)
        return {
            "mag": luminosity_to_magnitude(lum) + ab_to_system_offset(f, index, system),
            "band": label,
            "band_index": index,
            "frame": band_frame,
            "component": component,
            "system": system,
            "z": resolve_redshift(f, iz_path, ivol),
            "V_ivol": read_volumes(f)["V_ivol"],
        }


def lf_given_redshift_and_subvolume(
    iz_path: str,
    ivol: int,
    band: Band,
    bins: Optional[np.ndarray] = None,
    frame: str = "rest",
    component: str = "total",
    system: str = "AB",
) -> Optional[Dict[str, Any]]:
    """Compute the luminosity function of one band for a single subvolume.

    Args:
        iz_path: Path to snapshot directory (e.g. '/path/to/iz271').
        ivol: Subvolume index.
        band: Filter label such as 'SDSS-r', 'UKIRT-K' or 'Bj' (see
            ``list_bands``), or a 1-based band index.
        bins: Absolute magnitude (M - 5 log10 h) bin edges, increasing.
            Defaults to DEFAULT_MAGNITUDE_BINS.
        frame: 'rest' or 'observer'. Identical at z = 0.
        component: 'total' (disk + bulge), 'disk' or 'bulge'.
        system: 'AB' or 'Vega' magnitudes; ``bins`` are in this system.

    Returns:
        dict with keys: iz, ivol, z, band, band_index, frame, component,
        system, centers, phi [(Mpc/h)^-3 mag^-1], counts, V_ivol.
        None if data is missing or no galaxy has positive luminosity.

    Raises:
        ValueError: if ``band``, ``frame``, ``component`` or ``system`` is
            unknown, or ``bins`` do not increase. A band missing from one
            snapshot's filter list raises rather than skipping it, so that a
            typo is not mistaken for missing data; the multi-snapshot
            functions inherit this.
        KeyError: if Vega magnitudes are requested from a file without
            ``Bands/a0_ab``.
    """
    _check_choices(component, system)
    if bins is None:
        bins = DEFAULT_MAGNITUDE_BINS

    d = _read_magnitudes(iz_path, ivol, band, frame, component, system)
    if d is None or d["V_ivol"] is None or d["V_ivol"] <= 0:
        return None
    mag = d["mag"][np.isfinite(d["mag"])]
    if mag.size == 0:
        return None

    counts, edges = np.histogram(mag, bins=bins)
    phi = counts / (np.diff(edges) * d["V_ivol"])

    return {
        "iz": Path(iz_path).name,
        "ivol": ivol,
        "z": d["z"],
        **{key: d[key] for key in _BAND_KEYS},
        "centers": 0.5 * (edges[1:] + edges[:-1]),
        "phi": phi,
        "counts": counts,
        "V_ivol": d["V_ivol"],
    }


def lfs_given_redshifts_and_subvolume(
    ivol: int,
    iz_nums: List[int],
    band: Band,
    bins: Optional[np.ndarray] = None,
    frame: str = "rest",
    component: str = "total",
    system: str = "AB",
    base_dir: Optional[str] = None,
) -> Optional[pl.DataFrame]:
    """LF for one subvolume across multiple snapshots, as a long-form DataFrame.

    Args:
        ivol: Subvolume index.
        iz_nums: Snapshot numbers (e.g. [121, 155, 207, 271]).
        band, bins, frame, component, system: As for
            ``lf_given_redshift_and_subvolume``.
        base_dir: Base directory; defaults to configured base dir.

    Returns:
        polars DataFrame with columns: iz, iz_num, z, mag, phi, counts.
        None if no snapshot produced valid data.
    """
    if base_dir is None:
        base_dir = str(get_base_dir())

    frames = []
    for iz_num in iz_nums:
        iz_path = os.path.join(base_dir, f"iz{iz_num}")
        res = lf_given_redshift_and_subvolume(
            iz_path,
            ivol,
            band,
            bins=bins,
            frame=frame,
            component=component,
            system=system,
        )
        if res is None:
            continue
        n = len(res["centers"])
        frames.append(
            pl.DataFrame(
                {
                    "iz": [f"iz{iz_num}"] * n,
                    "iz_num": [iz_num] * n,
                    "z": [res["z"]] * n,
                    "mag": res["centers"],
                    "phi": res["phi"],
                    "counts": res["counts"],
                }
            )
        )

    if not frames:
        return None
    return pl.concat(frames)


def avg_lf_given_redshift_and_subvolumes(
    iz_num: int,
    ivols: List[int],
    band: Band,
    bins: Optional[np.ndarray] = None,
    frame: str = "rest",
    component: str = "total",
    system: str = "AB",
    base_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Average LF over a list of subvolumes for one snapshot.

    Args:
        iz_num: Snapshot number (e.g. 271 for 'iz271').
        ivols: Subvolume indices to average over.
        band, bins, frame, component, system: As for
            ``lf_given_redshift_and_subvolume``.
        base_dir: Base directory; defaults to configured base dir.

    Returns:
        dict with keys: iz, z, band, band_index, frame, component, system,
        centers,
        phi, phi_std, n_used, n_requested. ``phi_std`` is the scatter between
        subvolumes. None if no subvolume produced valid data.
    """
    if bins is None:
        bins = DEFAULT_MAGNITUDE_BINS
    if base_dir is None:
        base_dir = str(get_base_dir())

    res = _avg_phi_over_subvolumes(
        lf_given_redshift_and_subvolume,
        iz_num,
        ivols,
        bins,
        base_dir,
        band=band,
        frame=frame,
        component=component,
        system=system,
    )
    if res is None:
        return None
    first = res.pop("first")
    return {**res, **{key: first[key] for key in _BAND_KEYS}}


def avg_lf_given_redshifts_and_subvolume(
    ivol: int,
    iz_nums: List[int],
    band: Band,
    bins: Optional[np.ndarray] = None,
    frame: str = "rest",
    component: str = "total",
    system: str = "AB",
    base_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Average LF for one subvolume across multiple snapshots.

    Args:
        ivol: Subvolume index.
        iz_nums: Snapshot numbers to average over.
        band, bins, frame, component, system: As for
            ``lf_given_redshift_and_subvolume``.
        base_dir: Base directory; defaults to configured base dir.

    Returns:
        dict with keys: ivol, iz_list, z_list, band, band_index, frame,
        component, system (from the first usable snapshot), centers, phi,
        phi_std, n_used, n_requested.
        None if no snapshot produced valid data.
    """
    if bins is None:
        bins = DEFAULT_MAGNITUDE_BINS
    if base_dir is None:
        base_dir = str(get_base_dir())

    first: List[Dict[str, Any]] = []

    def single(iz_path: str, iv: int, **kwargs) -> Optional[Dict[str, Any]]:
        res = lf_given_redshift_and_subvolume(iz_path, iv, **kwargs)
        if res is not None and not first:
            first.append(res)
        return res

    res = _avg_phi_over_snapshots(
        single,
        ivol,
        iz_nums,
        bins,
        base_dir,
        band=band,
        frame=frame,
        component=component,
        system=system,
    )
    if res is None:
        return None
    return {**res, **{key: first[0][key] for key in _BAND_KEYS}}


def schechter_magnitude(
    mag: Union[float, np.ndarray], phi_star: float, M_star: float, alpha: float
) -> np.ndarray:
    """Schechter function per unit magnitude.

    phi(M) = 0.4 ln(10) phi* x^(alpha + 1) exp(-x),  x = 10^(0.4 (M* - M)).

    Args:
        mag: Absolute magnitude(s).
        phi_star: Normalisation, in the volume units wanted for phi.
        M_star: Characteristic magnitude, in the same convention as ``mag``.
        alpha: Faint-end slope.
    """
    x = 10.0 ** (0.4 * (M_star - np.asarray(mag, dtype=np.float64)))
    return 0.4 * np.log(10.0) * phi_star * x ** (alpha + 1.0) * np.exp(-x)
