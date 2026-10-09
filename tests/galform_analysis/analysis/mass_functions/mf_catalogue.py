"""Hand-built GALFORM catalogues with known mass functions and occupations."""

from pathlib import Path

import h5py
import numpy as np


def write_catalogue(
    base,
    iz_num,
    ivol,
    *,
    mstar=None,
    mhalo=None,
    mhhalo=None,
    is_central=None,
    tree_mphalo=None,
    volume=1000.0,
    redshift=None,
    split_mstar=True,
):
    """Write ``base/iz<iz_num>/ivol<ivol>/galaxies.hdf5`` and return its iz dir.

    Only the datasets that are given are written, so tests can remove any field.
    ``mstar`` is split 1:3 into disk and bulge unless ``split_mstar`` is False,
    in which case it is written as ``mstars``.
    """
    iz_dir = Path(base) / f"iz{iz_num}"
    ivol_dir = iz_dir / f"ivol{ivol}"
    ivol_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(ivol_dir / "galaxies.hdf5", "w") as f:
        if volume is not None:
            f.create_group("Parameters").create_dataset(
                "volume", data=np.float64(volume)
            )
        if tree_mphalo is not None:
            f.create_group("Trees").create_dataset(
                "mphalo", data=np.asarray(tree_mphalo, dtype=np.float64)
            )
        g = f.create_group("Output001")
        if redshift is not None:
            g.create_dataset("redshift", data=np.float64(redshift))
        if mstar is not None:
            mstar = np.asarray(mstar, dtype=np.float64)
            if split_mstar:
                g.create_dataset("mstars_disk", data=0.25 * mstar)
                g.create_dataset("mstars_bulge", data=0.75 * mstar)
            else:
                g.create_dataset("mstars", data=mstar)
        for name, arr in (
            ("mhalo", mhalo),
            ("mhhalo", mhhalo),
            ("is_central", is_central),
        ):
            if arr is not None:
                g.create_dataset(name, data=np.asarray(arr))
    return iz_dir


def masses_at(log10m_counts):
    """Masses with ``n`` objects at 10**log10m for each (log10m, n) pair."""
    return np.concatenate([np.full(n, 10.0**lm) for lm, n in log10m_counts])
