"""Per-galaxy absolute magnitudes from galaxies.hdf5, with optional dust."""

from typing import Any, Dict, Optional, Union

from galform_analysis.dust.galaxy import (
    Inclination,
    band_attenuation,
    check_inclination,
    resolve_dust_model,
)
from galform_analysis.dust.models import DustModel

from .bands import (
    COMPONENTS,
    SYSTEMS,
    ab_to_system_offset,
    band_label,
    band_luminosity,
    find_band,
    luminosity_to_magnitude,
)
from .loaders import (
    get_output_group,
    open_galaxies_hdf5,
    read_volumes,
    resolve_redshift,
)

Band = Union[str, int]


def check_magnitude_options(
    component: str,
    system: str,
    dust: Optional[Union[str, DustModel]] = None,
    inclination: Inclination = "random",
) -> None:
    """Raise ValueError for an unknown component, system, dust model or
    inclination, before any file is read."""
    if component not in COMPONENTS:
        raise ValueError(
            f"component must be one of {sorted(COMPONENTS)}, got {component!r}"
        )
    if system not in SYSTEMS:
        raise ValueError(f"system must be one of {list(SYSTEMS)}, got {system!r}")
    if dust is not None:
        resolve_dust_model(dust)
        check_inclination(inclination)


def read_band_magnitudes(
    iz_path: str,
    ivol: int,
    band: Band,
    frame: str = "rest",
    component: str = "total",
    system: str = "AB",
    dust: Optional[Union[str, DustModel]] = None,
    inclination: Inclination = "random",
    seed: int = 0,
) -> Optional[Dict[str, Any]]:
    """Absolute magnitudes ``M - 5 log10 h`` of every galaxy in one subvolume.

    Args:
        iz_path: Path to the snapshot directory (e.g. '/path/to/iz271').
        ivol: Subvolume index.
        band: Filter label (e.g. 'SDSS-r') or 1-based band index.
        frame: 'rest' or 'observer'.
        component: 'total', 'disk' or 'bulge'.
        system: 'AB' or 'Vega'.
        dust: None for intrinsic magnitudes, or a dust model ('lacey16',
            'baugh05' or a ``DustModel``) for diffuse-dust attenuation.
        inclination: With dust: 'random' (isotropic, seeded), 'average'
            (isotropic mean attenuation), 'face-on' or degrees.
        seed: Seed for random inclinations; galaxy orientations are drawn
            from ``numpy.random.default_rng([seed, ivol])``, so they are the
            same in every band.

    Returns:
        dict with keys: mag (NaN where the luminosity is not positive), band,
        band_index, frame, component, system, dust (model name or None),
        inclination, z, V_ivol. None if the file or its band data is missing.

    Raises:
        ValueError: for an unknown band, frame, component, system, dust model
            or inclination.
        KeyError: if dust is requested from a file without the needed gas
            properties, or Vega magnitudes from one without ``Bands/a0_ab``.
        FileNotFoundError: if the dust attenuation table cannot be found.
    """
    check_magnitude_options(component, system, dust, inclination)
    f = open_galaxies_hdf5(iz_path, ivol=ivol)
    if f is None:
        return None
    with f:
        g = get_output_group(f)
        if g is None or "Bands" not in f or "bandname" not in f["Bands"]:
            return None
        index = find_band(f, band, frame=frame)
        z = resolve_redshift(f, iz_path, ivol)
        if dust is None:
            lum = band_luminosity(g, index, component)
        else:
            lum = _attenuated_luminosity(
                f, g, index, component, dust, z, inclination, (seed, ivol)
            )
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
            "dust": None if dust is None else resolve_dust_model(dust).name,
            "inclination": None if dust is None else inclination,
            "z": z,
            "V_ivol": read_volumes(f)["V_ivol"],
        }


def _attenuated_luminosity(f, g, index, component, dust, z, inclination, seed):
    disk = band_luminosity(g, index, "disk")
    bulge = band_luminosity(g, index, "bulge")
    if disk is None or bulge is None:
        return None
    atten_disk, atten_bulge = band_attenuation(
        f, g, index, dust, z, inclination=inclination, seed=seed
    )
    parts = {"disk": disk * atten_disk, "bulge": bulge * atten_bulge}
    if component == "total":
        return parts["disk"] + parts["bulge"]
    return parts[component]
