"""Satellite–central cross-correlation utilities.

Compute the real-space cross-correlation between satellite galaxies and
central galaxies (or host halos) using GALFORM galaxies.hdf5 outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import polars as pl
from Corrfunc.theory.DD import DD as corrfunc_DD

from galform_analysis.config import DEFAULT_RBINS
from galform_analysis.readers.loaders import (
    open_galaxies_hdf5,
    get_output_group,
    _get_first_array,
    _get_redshift_from_file,
    _get_redshift_from_zsnap,
)


def _load_galaxy_positions(
    iz_path: str,
    ivol: int,
    select_centrals: bool,
    stellar_mass_min: Optional[float] = None,
    host_halo_mass_min: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[float]]:
    """Load galaxy positions for centrals or satellites from galaxies.hdf5.

    Args:
        iz_path: Snapshot directory
        ivol: Subvolume index
        select_centrals: True for centrals, False for satellites
        stellar_mass_min: Optional stellar mass cut (Msun/h)
        host_halo_mass_min: Optional host halo mass cut (mhhalo) in Msun/h

    Returns:
        positions: (N,3) array
        z: redshift (if available)
    """
    f = open_galaxies_hdf5(iz_path, ivol=ivol)
    if f is None:
        raise FileNotFoundError(f"Missing or unreadable galaxies.hdf5 at {iz_path}/ivol{ivol}")

    try:
        g = get_output_group(f)
        if g is None:
            raise RuntimeError("No OutputNNN group found in HDF5 file")

        for key in ("xgal", "ygal", "zgal"):
            if key not in g:
                raise KeyError("Could not find xgal/ygal/zgal position arrays in Output group")

        x = np.asarray(g["xgal"])
        y = np.asarray(g["ygal"])
        z = np.asarray(g["zgal"])

        if "is_central" not in g:
            raise KeyError("is_central field not found - cannot select centrals/satellites")
        is_central = np.asarray(g["is_central"]).astype(int, copy=False)

        m_disk = _get_first_array(g, ["mstars_disk"])
        m_bulge = _get_first_array(g, ["mstars_bulge"])
        if m_disk.size and m_bulge.size:
            mstar = m_disk + m_bulge
        else:
            mstar = _get_first_array(g, ["mstars", "StellarMass", "Mstar", "mstars_allburst"])

        mhhalo = _get_first_array(g, ["mhhalo", "mhalo_host"])

        n = min(len(x), len(y), len(z), len(is_central))
        x, y, z, is_central = x[:n], y[:n], z[:n], is_central[:n]
        if mstar.size:
            mstar = mstar[:n]
        if mhhalo.size:
            mhhalo = mhhalo[:n]

        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        if select_centrals:
            mask &= is_central == 1
        else:
            mask &= is_central == 0

        if stellar_mass_min is not None:
            if mstar.size == 0:
                raise KeyError("mstar field not found - cannot apply stellar mass cut")
            mask &= mstar >= stellar_mass_min

        if host_halo_mass_min is not None:
            if mhhalo.size == 0:
                raise KeyError("mhhalo field not found - cannot apply host halo mass cut")
            mask &= mhhalo >= host_halo_mass_min

        pos = np.vstack([x[mask], y[mask], z[mask]]).T.astype(np.float64, copy=False)
        z_val = _get_redshift_from_file(f) or _get_redshift_from_zsnap(iz_path, ivol)
        return pos, z_val
    finally:
        try:
            f.close()
        except Exception:
            pass


def compute_xi_cross_corrfunc(
    positions_a: np.ndarray,
    positions_b: np.ndarray,
    boxsize: float,
    rbins: Optional[np.ndarray] = None,
    nthreads: int = 4,
) -> pl.DataFrame:
    """Compute cross-correlation xi(r) between two samples using Corrfunc.DD."""
    if rbins is None:
        rbins = DEFAULT_RBINS
    rbins = np.asarray(rbins, dtype=float)

    rmax_periodic = boxsize / 2.0
    rbins = rbins[rbins <= rmax_periodic]
    if len(rbins) < 2:
        raise ValueError(
            f"No valid rbins within periodic limit (rmax={rmax_periodic:.2f})."
        )

    n1 = positions_a.shape[0]
    n2 = positions_b.shape[0]
    if n1 < 1 or n2 < 1:
        r_centers = 0.5 * (rbins[:-1] + rbins[1:])
        df = pl.DataFrame({
            "r": r_centers,
            "xi": np.full_like(r_centers, np.nan),
            "npairs": np.zeros_like(r_centers, dtype=float),
        })
        df.attrs = {"rbins": rbins, "n1": n1, "n2": n2, "boxsize": boxsize}
        return df

    pos_a = np.fmod(positions_a, boxsize)
    pos_a = np.where(pos_a < 0, pos_a + boxsize, pos_a)
    pos_b = np.fmod(positions_b, boxsize)
    pos_b = np.where(pos_b < 0, pos_b + boxsize, pos_b)

    results = corrfunc_DD(
        autocorr=0,
        nthreads=nthreads,
        binfile=rbins,
        X1=pos_a[:, 0],
        Y1=pos_a[:, 1],
        Z1=pos_a[:, 2],
        X2=pos_b[:, 0],
        Y2=pos_b[:, 1],
        Z2=pos_b[:, 2],
        periodic=True,
        boxsize=boxsize,
        verbose=False,
        output_ravg=True,
    )

    npairs = np.array([x["npairs"] for x in results], dtype=np.float64)
    ravg = np.array([x["ravg"] for x in results], dtype=np.float64)

    if np.all(np.isfinite(ravg) & (ravg > 0)):
        r = ravg
    else:
        r = 0.5 * (rbins[:-1] + rbins[1:])

    volume = boxsize ** 3
    r1 = rbins[:-1]
    r2 = rbins[1:]
    V_shell = (4.0 / 3.0) * np.pi * (r2**3 - r1**3)

    DD_norm = npairs / (n1 * n2)
    RR_norm = V_shell / volume

    xi_vals = np.where(RR_norm > 0, DD_norm / RR_norm - 1.0, np.nan)

    df = pl.DataFrame({"r": r, "xi": xi_vals, "npairs": npairs})
    df.attrs = {"rbins": rbins, "n1": n1, "n2": n2, "boxsize": boxsize}
    return df


def satellite_central_cross_correlation(
    iz_path: str,
    ivol: int,
    rbins: Optional[np.ndarray] = None,
    nthreads: int = 4,
    satellite_stellar_mass_min: Optional[float] = None,
    central_stellar_mass_min: Optional[float] = None,
    host_halo_mass_min: Optional[float] = None,
    boxsize_override: Optional[float] = None,
) -> Optional[pl.DataFrame]:
    """Compute cross-correlation between satellites and centrals for one subvolume."""
    try:
        pos_sat, z_sat = _load_galaxy_positions(
            iz_path,
            ivol,
            select_centrals=False,
            stellar_mass_min=satellite_stellar_mass_min,
            host_halo_mass_min=host_halo_mass_min,
        )
        pos_cen, z_cen = _load_galaxy_positions(
            iz_path,
            ivol,
            select_centrals=True,
            stellar_mass_min=central_stellar_mass_min,
            host_halo_mass_min=host_halo_mass_min,
        )

        if pos_sat.size == 0 or pos_cen.size == 0:
            return None

        if boxsize_override is not None:
            boxsize = float(boxsize_override)
        else:
            extent = np.ptp(np.vstack([pos_sat, pos_cen]), axis=0)
            boxsize = float(np.max(extent))

        if not np.isfinite(boxsize) or boxsize <= 0:
            raise RuntimeError(f"Invalid box size for {iz_path}/ivol{ivol}: {boxsize}")

        df = compute_xi_cross_corrfunc(
            pos_sat,
            pos_cen,
            boxsize=boxsize,
            rbins=rbins,
            nthreads=nthreads,
        )

        df.attrs.update({
            "iz": Path(iz_path).name,
            "ivol": ivol,
            "z": z_sat if z_sat is not None else z_cen,
            "n_sat": pos_sat.shape[0],
            "n_cen": pos_cen.shape[0],
        })
        return df

    except (FileNotFoundError, RuntimeError, KeyError):
        return None
