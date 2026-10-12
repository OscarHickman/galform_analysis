"""Reader for GALFORM's diffuse-dust attenuation tables.

The tables (``Data/dust/dust_*.dat`` in a GALFORM checkout) hold the
radiative-transfer results of Ferrara et al. (1999): the attenuation of disk
and bulge light by dust in an exponential disk, as a function of central
V-band optical depth, wavelength, inclination and bulge-to-disk scale ratio.

This is a port of GALFORM's ``Dust_Read`` (``dust.F90``). The arrays are padded
exactly as GALFORM pads them, so that interpolation in
:mod:`galform_analysis.dust.diffuse` reproduces GALFORM's attenuations.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

# Padding values GALFORM adds around the tabulated ranges.
_LAMBDA_PAD_LO = 100.0  # Angstrom
_LAMBDA_PAD_HI = 1.0e5  # Angstrom
_TAUEFF_ZERO_ATTEN = 100.0  # taueff used where the attenuation is <= 0
_N_TRAPEZIA = 50  # points in GALFORM's angle average


@dataclass(frozen=True)
class AttenuationTable:
    """Padded attenuation table, indexed as GALFORM indexes it (0-based here).

    Attributes:
        inc: Inclinations [deg], ascending, with 0 prepended.
        invlam: 1 / wavelength [1/Angstrom], descending, padded at both ends.
        tauv0: Central face-on V-band optical depths, with 0 prepended.
        re_h: Bulge effective radius over disk scale length.
        taueff: ``-ln(attenuation)``, shape (ntauv, nlam, ninc, nreh, 2); the
            last axis is (disk light, bulge light).
        taueff_ave: As ``taueff`` but averaged over inclination, shape
            (ntauv, nlam, nreh, 2).
        label: First line of the file.
    """

    inc: np.ndarray
    invlam: np.ndarray
    tauv0: np.ndarray
    re_h: np.ndarray
    taueff: np.ndarray
    taueff_ave: np.ndarray
    label: str


def _header_int(line: str) -> int:
    return int(line.split("=")[1])


def _read_block(lines: List[str], pos: int, nlam: int, ntauv: int):
    """Read one ``inc=`` block; return (inc, lambda, tauv0, atten, next pos)."""
    while not lines[pos].lstrip().startswith("inc="):
        pos += 1
    inc = float(lines[pos].split("=")[1])
    pos += 1
    while not lines[pos].lstrip().startswith("tauV"):
        pos += 1
    lam = np.array(lines[pos].split()[1 : nlam + 1], dtype=float)
    rows = np.array(
        [lines[pos + 1 + k].split()[: nlam + 1] for k in range(ntauv)], dtype=float
    )
    return inc, lam, rows[:, 0], rows[:, 1:], pos + 1 + ntauv


def read_attenuation_table(path: Union[str, Path]) -> AttenuationTable:
    """Read and pad a GALFORM ``dust_*.dat`` attenuation table."""
    with open(path) as fh:
        lines = fh.read().splitlines()
    label = lines[0].strip()
    header = {
        ln.split("=")[0].strip(): ln for ln in lines[:12] if "=" in ln and ln.strip()
    }
    ninc, nlam = _header_int(header["ninc"]), _header_int(header["nlam"])
    ntauv, nreh = _header_int(header["ntauv"]), _header_int(header["nreh"])

    inc = np.zeros(ninc + 1)
    lam = np.zeros(nlam + 2)
    tauv0 = np.zeros(ntauv + 1)
    re_h = np.zeros(nreh)
    atten = np.zeros((ntauv + 1, nlam + 2, ninc + 1, nreh, 2))

    pos = next(i for i, ln in enumerate(lines) if ln.strip() == "disk")
    for jinc in range(1, ninc + 1):
        inc[jinc], lam[1:-1], tauv0[1:], block, pos = _read_block(
            lines, pos, nlam, ntauv
        )
        atten[1:, 1:-1, jinc, :, 0] = block[:, :, None]

    pos = next(i for i, ln in enumerate(lines) if ln.strip() == "bulge")
    for jreh in range(nreh):
        while not lines[pos].lstrip().startswith("re/h="):
            pos += 1
        re_h[jreh] = float(lines[pos].split("=")[1])
        for jinc in range(1, ninc + 1):
            inc[jinc], lam[1:-1], tauv0[1:], block, pos = _read_block(
                lines, pos, nlam, ntauv
            )
            atten[1:, 1:-1, jinc, jreh, 1] = block

    # Pad the boundaries as GALFORM does.
    lam[0], lam[-1] = _LAMBDA_PAD_LO, _LAMBDA_PAD_HI
    atten[0, 1:-1, 1:, :, :] = 1.0  # no attenuation at tauv0 = 0
    # ln(atten) proportional to 1/lambda beyond the tabulated wavelengths.
    atten[:, 0, 1:] = atten[:, 1, 1:] ** (lam[1] / lam[0])
    atten[:, -1, 1:] = atten[:, -2, 1:] ** (lam[-2] / lam[-1])
    atten[:, :, 0] = atten[:, :, 1]  # inc below the first tabulated value

    with np.errstate(divide="ignore"):
        taueff = np.where(atten > 0.0, -np.log(atten), _TAUEFF_ZERO_ATTEN)
    ave = _angle_average(inc, atten)
    with np.errstate(divide="ignore"):
        taueff_ave = np.where(ave > 0.0, -np.log(ave), _TAUEFF_ZERO_ATTEN)

    return AttenuationTable(
        inc=inc,
        invlam=1.0 / lam,
        tauv0=tauv0,
        re_h=re_h,
        taueff=taueff,
        taueff_ave=taueff_ave,
        label=label,
    )


def _angle_average(inc: np.ndarray, atten: np.ndarray) -> np.ndarray:
    """Isotropic average of ``atten`` over inclination (GALFORM's
    ``Atten_Average``): midpoint values on a uniform grid weighted by
    d(cos i), capped at 1."""
    dinc = 90.0 / (_N_TRAPEZIA - 1)
    lo = np.arange(_N_TRAPEZIA - 1) * dinc
    mid = lo + 0.5 * dinc
    weights = np.cos(np.radians(lo)) - np.cos(np.radians(lo + dinc))
    j = np.clip(np.searchsorted(inc, mid, side="left") - 1, 0, len(inc) - 2)
    f = ((mid - inc[j]) / (inc[j + 1] - inc[j]))[:, None, None]  # inc is axis 2
    at_mid = (1.0 - f) * np.take(atten, j, axis=2) + f * np.take(atten, j + 1, axis=2)
    total = np.tensordot(at_mid, weights, axes=([2], [0]))
    return np.minimum(total, 1.0)


def find_dust_table(name: str, data_dir: Optional[Union[str, Path]] = None) -> Path:
    """Locate a GALFORM dust table such as ``dust_MW_hz1.0.dat``.

    Looks for ``name`` as a path, then in ``<data_dir>/dust/``, then in
    ``$GALFORM_DATA_DIR/dust/`` (the GALFORM ``Data`` directory).

    Raises:
        FileNotFoundError: if the table is in none of these places.
    """
    candidates = [Path(name)]
    for root in (data_dir, os.environ.get("GALFORM_DATA_DIR")):
        if root:
            candidates.append(Path(root) / "dust" / name)
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"GALFORM dust table {name!r} not found (tried "
        f"{', '.join(str(p) for p in candidates)}). Pass data_dir= or set "
        "GALFORM_DATA_DIR to your GALFORM 'Data' directory."
    )
