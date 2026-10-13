"""Dust attenuation of GALFORM galaxies read from galaxies.hdf5.

The attenuation itself is computed by GALFORM's own code, run from your
checkout (see :mod:`galform_analysis.dust.galform`).
"""

from typing import Dict, Optional, Tuple, Union

import h5py
import numpy as np

from .galform import dust_driver, dust_table, find_galform, run_dust_driver
from .models import DUST_MODELS, DustModel

Inclination = Union[str, float]
INCLINATIONS = ("random", "average", "face-on")
_DUST_FIELDS = (
    "mcold",
    "cold_metal",
    "rdisk",
    "rbulge",
    "mcold_burst",
    "metals_burst",
    "tburst",
    "tsfburst",
)


def resolve_dust_model(dust: Union[str, DustModel]) -> DustModel:
    """Return ``dust`` itself, or the preset of that name (e.g. 'lacey16')."""
    if isinstance(dust, DustModel):
        return dust
    if dust in DUST_MODELS:
        return DUST_MODELS[dust]
    raise ValueError(
        f"unknown dust model {dust!r}; use one of {sorted(DUST_MODELS)} or a DustModel"
    )


def check_inclination(inclination: Inclination) -> None:
    """Raise ValueError unless ``inclination`` is a known mode or 0-90 deg."""
    if isinstance(inclination, str):
        if inclination not in INCLINATIONS:
            raise ValueError(
                f"inclination must be one of {list(INCLINATIONS)} or degrees, "
                f"got {inclination!r}"
            )
    elif not 0.0 <= float(inclination) <= 90.0:
        raise ValueError(f"inclination must be in [0, 90] deg, got {inclination}")


def inclinations(
    inclination: Inclination, n: int, seed: Union[int, tuple] = 0
) -> Optional[np.ndarray]:
    """Per-galaxy inclinations in degrees, or None for the isotropic average.

    'random' draws isotropic orientations (cos i uniform), as GALFORM does,
    from ``numpy.random.default_rng(seed)``. The draw depends only on the seed
    and the number of galaxies, so with the per-subvolume seeds used by
    ``read_band_magnitudes`` the i-th galaxy of a subvolume gets the same
    orientation in every band (and in every snapshot).

    Raises:
        ValueError: for an unknown inclination or a negative seed.
    """
    check_inclination(inclination)
    if any(int(s) < 0 for s in np.atleast_1d(seed)):
        raise ValueError(f"seed must be non-negative, got {seed!r}")
    if inclination == "average":
        return None
    if inclination == "random":
        cos_i = np.random.default_rng(seed).uniform(size=n)
        return np.degrees(np.arccos(cos_i))
    if inclination == "face-on":
        return np.zeros(n)
    return np.full(n, float(inclination))


def _scalar(f: h5py.File, key: str) -> float:
    return float(np.ravel(f[key][()])[0])


def rest_wavelength(f: h5py.File, index: int, z: Optional[float]) -> float:
    """Rest-frame effective wavelength [A] of band ``index``; <= 0 for bands
    GALFORM does not attenuate (bolometric, Lyman continuum, ...).

    Raises:
        ValueError: for lightcone interpolation bands (``iselect = -1``),
            whose redshift offset is not in galaxies.hdf5.
    """
    table = f["Bands"]
    lam = float(np.ravel(table["lambda_eff"][()])[index - 1])
    iselect = int(np.ravel(table["iselect"][()])[index - 1])
    if iselect < 0:
        raise ValueError(
            f"band {index} is a lightcone interpolation band (iselect={iselect}); "
            "dust attenuation is not supported for it"
        )
    if iselect >= 2 or lam <= 0:
        return -1.0
    if iselect == 1:
        if (
            "stellar_pop_z_ref" in table
            and _scalar(f, "Bands/stellar_pop_z_ref") > -100
        ):
            raise ValueError(
                "GALFORM cannot attenuate observer-frame bands when "
                "stellar_pop_z_ref is set"
            )
        if z is None:
            raise ValueError("observer-frame dust attenuation needs the redshift")
        return lam / (1.0 + z)
    return lam


# Gas below this mass (Msun/h) is treated as metal-free, as in GALFORM.
_MCOLD_TINY = 1.0


def _metallicity(metals: np.ndarray, gas: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(gas > _MCOLD_TINY, np.maximum(metals, 0.0) / gas, 0.0)


def _driver_inputs(
    data: Dict[str, np.ndarray], model: DustModel, efold: float
) -> Dict[str, np.ndarray]:
    """Per-galaxy inputs to GALFORM's diffuse-dust routine.

    A fraction ``fcloud`` of the cold gas is in clouds, so the diffuse dust
    follows the rest. An ongoing burst (younger than ``efold`` e-folding
    times) has its own dust, in a disk ``rfacburst`` times the bulge size.
    """
    burst = (data["tburst"] < efold * data["tsfburst"]) & (model.rfacburst > 0)
    mburst = np.where(burst, data["mcold_burst"], 0.0)
    return {
        "mdisk": (1.0 - model.fcloud) * data["mcold"],
        "zdisk": _metallicity(data["cold_metal"], data["mcold"]),
        "rdisk": data["rdisk"],
        "rbulge": data["rbulge"],
        "mburst": (1.0 - model.fcloud) * mburst,
        "zburst": _metallicity(np.where(burst, data["metals_burst"], 0.0), mburst),
        "rburst": np.where(burst, model.rfacburst * data["rbulge"], 0.0),
        "burst": burst,
    }


def band_attenuation(
    f: h5py.File,
    g: h5py.Group,
    index: int,
    dust: Union[str, DustModel],
    z: Optional[float],
    inclination: Inclination = "random",
    seed: Union[int, tuple] = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Attenuation of disk and bulge light in band ``index`` by diffuse dust.

    Args:
        f: Open galaxies.hdf5 file.
        g: Its ``OutputNNN`` group.
        index: 1-based band index.
        dust: A ``DustModel`` or preset name ('lacey16', 'baugh05').
        z: Redshift of ``g`` (for observer-frame bands).
        inclination: 'random', 'average', 'face-on' or degrees.
        seed: Seed for 'random' inclinations; the same seed gives the same
            orientation to each galaxy in every band.

    Returns:
        ``(atten_disk, atten_bulge)`` per galaxy.

    Raises:
        KeyError: if a field the dust model needs is missing.
        FileNotFoundError: if ``GALFORM_DIR`` is not set, GALFORM has not
            been built, or the dust table is missing.
        RuntimeError: if building or running GALFORM's dust code fails.
    """
    model = resolve_dust_model(dust)
    missing = [k for k in _DUST_FIELDS if k not in g]
    if missing:
        raise KeyError(f"{f.filename}: dust model needs {', '.join(missing)}")
    data = {k: np.asarray(g[k][()], dtype=np.float64) for k in _DUST_FIELDS}
    n = data["mcold"].size
    lam = rest_wavelength(f, index, z)
    if lam <= 0:
        return np.ones(n), np.ones(n)

    galform = find_galform()
    atten, _ = run_dust_driver(
        dust_driver(galform),
        dust_table(galform, model.table),
        lam=np.array([lam]),
        h0=_scalar(f, "Parameters/h0"),
        inc_deg=inclinations(inclination, n, seed),
        **_driver_inputs(data, model, _scalar(f, "Parameters/efold")),
    )
    return atten[0, 0], atten[1, 0]
