"""Hand-built GALFORM catalogues with per-band luminosities.

Mirrors the real schema: the file-level ``Bands`` group holds the filter table
(``bandname``, ``iselect``, ``idband``, ``lambda_eff``) and each output group
holds ``Bands/BandNNN_Lum_Disk`` / ``_Lum_Bulge`` per galaxy. GALFORM converts
these with ``M_AB - 5 log10 h = -2.5 log10(L_disk + L_bulge)``.
"""

from pathlib import Path

import h5py
import numpy as np

# (bandname, iselect, idband): rest-frame (0), observer-frame (1) and special (2).
DEFAULT_BANDS = [
    ("UKIRT-K  UKIRT K filter", 0, 49),
    ("SDSS-r  SDSS r standard (filter+CCD +1.3 air mass)", 0, 204),
    ("V  RGO CCD V", 2, 52),
    ("UKIRT-K  UKIRT K filter", 1, 49),
    ("SDSS-r  SDSS r standard (filter+CCD +1.3 air mass)", 1, 204),
]
# Vega - AB offset of each band in DEFAULT_BANDS (GALFORM's Bands/a0_ab).
DEFAULT_A0_AB = [-1.87, 0.12, 0.0, -1.87, 0.12]


def magnitudes_to_luminosity(mags):
    """Luminosity in GALFORM units that gives absolute magnitude ``mags``."""
    return 10.0 ** (-0.4 * np.asarray(mags, dtype=np.float64))


def mags_at(mag_counts):
    """Magnitudes with ``n`` galaxies at ``mag`` for each (mag, n) pair."""
    return np.concatenate([np.full(n, float(m)) for m, n in mag_counts])


def write_band_catalogue(
    base,
    iz_num,
    ivol,
    *,
    lum_disk,
    lum_bulge=None,
    bands=DEFAULT_BANDS,
    a0_ab=DEFAULT_A0_AB,
    volume=1000.0,
    redshift=None,
    write_band_table=True,
    write_band_lums=True,
    gas=None,
    parameters=None,
    lambda_eff=None,
):
    """Write ``base/iz<iz_num>/ivol<ivol>/galaxies.hdf5`` and return its iz dir.

    ``lum_disk`` and ``lum_bulge`` map a 1-based band index to an array of
    luminosities; bands not listed get zeros. ``lum_bulge`` defaults to zeros.
    ``a0_ab`` is written as ``Bands/a0_ab`` unless it is None. ``gas`` maps
    output-group dataset names (e.g. ``mcold``) to per-galaxy arrays and
    ``parameters`` maps ``Parameters`` names (e.g. ``h0``) to scalars.
    ``lambda_eff`` defaults to 5000 A for every band.
    """
    lum_bulge = lum_bulge or {}
    n_gal = len(next(iter(lum_disk.values())))
    iz_dir = Path(base) / f"iz{iz_num}"
    ivol_dir = iz_dir / f"ivol{ivol}"
    ivol_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(ivol_dir / "galaxies.hdf5", "w") as f:
        params = f.create_group("Parameters")
        if volume is not None:
            params.create_dataset("volume", data=np.float64(volume))
        for name, value in (parameters or {}).items():
            params.create_dataset(name, data=np.float64(value))
        if write_band_table:
            table = f.create_group("Bands")
            table.create_dataset("bandname", data=[b[0].encode() for b in bands])
            table.create_dataset("iselect", data=np.array([b[1] for b in bands]))
            table.create_dataset("idband", data=np.array([b[2] for b in bands]))
            lam = np.full(len(bands), 5000.0) if lambda_eff is None else lambda_eff
            table.create_dataset("lambda_eff", data=np.asarray(lam, np.float32))
            table.create_dataset("nband", data=np.int32(len(bands)))
            if a0_ab is not None:
                table.create_dataset("a0_ab", data=np.asarray(a0_ab, np.float32))
        g = f.create_group("Output001")
        g.create_dataset("mstars_disk", data=np.ones(n_gal))
        g.create_dataset("mstars_bulge", data=np.ones(n_gal))
        if redshift is not None:
            g.create_dataset("redshift", data=np.float64(redshift))
        for name, arr in (gas or {}).items():
            g.create_dataset(name, data=np.asarray(arr, dtype=np.float32))
        if write_band_lums:
            lums = g.create_group("Bands")
            zeros = np.zeros(n_gal, dtype=np.float32)
            for i in range(1, len(bands) + 1):
                disk = np.asarray(lum_disk.get(i, zeros), dtype=np.float32)
                bulge = np.asarray(lum_bulge.get(i, zeros), dtype=np.float32)
                lums.create_dataset(f"Band{i:03d}_Lum_Disk", data=disk)
                lums.create_dataset(f"Band{i:03d}_Lum_Bulge", data=bulge)
    return iz_dir
