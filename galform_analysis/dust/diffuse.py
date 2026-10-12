"""Attenuation of starlight by diffuse dust, vectorised over galaxies.

A port of GALFORM's ``Dust_Atten`` (``dust.F90``) and ``Calc_Atten_Diffuse``
(``dust.attenuate.F90``), including its extrapolation choices (``OPT_TAU=1``:
power-law in the central optical depth; ``OPT_RE=3``: endpoint value for small
bulges, no bulge attenuation for large ones).

Dust lives in the gas disk with the disk's scale length; its central face-on
V-band optical depth is ``tau_V0 = C_V * Z_cold * Sigma_gas0``. An ongoing
burst is a second dust disk of radius ``rfacburst * rbulge`` that attenuates
only bulge (burst) light.
"""

from typing import Optional, Tuple

import numpy as np

from .table import AttenuationTable

# GALFORM constants (data.numerical_constants.F90, galaxy_profiles.sersic.F90).
RDISK_HALF_SCALE = 1.678346990  # exponential disk half-mass radius / scale length
SERSIC_N4_RHALF_IN_RE = 1.34807965  # de Vaucouleurs 3D half-mass radius / Re
_MCOLD_MIN_FOR_SIGMA = 1.0e-4  # Msun/h; below this the gas disk is ignored
_AV_EBV, _NH_EBV, _Z_LOCAL_ISM = 3.10, 5.8e21, 0.02  # Savage & Mathis (1979)
_X_HYDROGEN, _M_ATOMIC_G = 0.707, 1.66053873e-24
_MSOLAR_G, _PC2CM = 1.9891e33, 3.0856775807e18

#: tau_V per unit metal surface density, (Msun/pc^2)^-1 (GALFORM's ``cv_dust``).
CV_DUST = (
    (1.0 / (2.5 * np.log10(np.e)))
    * (_AV_EBV / _NH_EBV)
    * (_X_HYDROGEN / _M_ATOMIC_G)
    * (_MSOLAR_G / _PC2CM**2)
    / _Z_LOCAL_ISM
)


def _locate(xx: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Numerical Recipes ``locate`` for an ascending array: the 1-based index
    j with xx(j) < x <= xx(j+1), i.e. the number of entries below x."""
    return np.searchsorted(xx, x, side="left")


def _inclination_factors(inc: np.ndarray, inc_deg: np.ndarray):
    """0-based lower index and weight in ``inc``, fixed at the endpoints."""
    n = len(inc)
    j = _locate(inc, inc_deg)
    j = np.where((j == 0) & (inc_deg == inc[0]), 1, j)
    j = np.where((j == n) & (inc_deg == inc[-1]), n - 1, j)
    below, above = j == 0, j >= n
    j = np.clip(j, 1, n - 1)
    f = (inc_deg - inc[j - 1]) / (inc[j] - inc[j - 1])
    f = np.where(below, 0.0, np.where(above, 1.0, f))
    return j - 1, f


def _wavelength_factors(invlam: np.ndarray, lam: float):
    """0-based lower index and weight in the descending ``invlam`` array."""
    x = 1.0 / lam
    n = len(invlam)
    j = n - int(np.searchsorted(invlam[::-1], x, side="left"))  # count(>= x)
    if j == n and x == invlam[-1]:
        j = n - 1
    if j == 0 or j >= n:
        raise ValueError(
            f"wavelength {lam:g} A is outside the dust table range "
            f"[{1.0 / invlam[0]:g}, {1.0 / invlam[-1]:g}] A"
        )
    return j - 1, (x - invlam[j - 1]) / (invlam[j] - invlam[j - 1])


def _tauv_factors(tauv0: np.ndarray, tau: np.ndarray):
    """Lower 0-based index, abscissa and extrapolation flag in the tau_V0 grid.

    Above the grid GALFORM extrapolates as a power law, i.e. linearly in
    ln(tau_V0).
    """
    n = len(tauv0)
    j = _locate(tauv0, tau)
    tau = np.where(j == 0, 0.0, tau)
    j = np.where(j == 0, 1, j)
    extrap = j == n
    lo = np.where(extrap, n - 1, j) - 1
    with np.errstate(divide="ignore"):
        x = np.where(extrap, np.log(np.where(extrap, tau, 1.0)), tau)
    return lo, x, extrap


class _Interpolator:
    """taueff at given tau_V0 rows, interpolated in wavelength and inclination."""

    def __init__(self, table, lam, inc_deg):
        self.table = table
        self.l0, self.fl = _wavelength_factors(table.invlam, lam)
        self.ave = inc_deg is None
        if not self.ave:
            self.i0, self.fi = _inclination_factors(table.inc, inc_deg)

    def taueff(self, row, reh, comp):
        l0, fl = self.l0, self.fl
        if self.ave:
            t = self.table.taueff_ave
            return (1 - fl) * t[row, l0, reh, comp] + fl * t[row, l0 + 1, reh, comp]
        t, i0, fi = self.table.taueff, self.i0, self.fi
        return (
            (1 - fi) * (1 - fl) * t[row, l0, i0, reh, comp]
            + fi * (1 - fl) * t[row, l0, i0 + 1, reh, comp]
            + (1 - fi) * fl * t[row, l0 + 1, i0, reh, comp]
            + fi * fl * t[row, l0 + 1, i0 + 1, reh, comp]
        )

    def along_tauv(self, lo, x, log_xa, extrap_max, reh, comp):
        """Interpolate taueff in tau_V0 between rows ``lo`` and ``lo + 1``."""
        ya0 = self.taueff(lo, reh, comp)
        ya1 = self.taueff(lo + 1, reh, comp)
        xa0, xa1 = self.table.tauv0[lo], self.table.tauv0[lo + 1]
        with np.errstate(divide="ignore"):
            xa0 = np.where(log_xa, np.log(np.where(log_xa, xa0, 1.0)), xa0)
            xa1 = np.where(log_xa, np.log(np.where(log_xa, xa1, 1.0)), xa1)
        fx = (x - xa0) / (xa1 - xa0)
        y = (1 - fx) * ya0 + fx * ya1
        return np.where(extrap_max, np.maximum(y, ya1), y)


def dust_atten(
    table: AttenuationTable,
    mcold: np.ndarray,
    zcold: np.ndarray,
    rdisk: np.ndarray,
    rbulge: np.ndarray,
    lam: float,
    h0: float,
    inc_deg: Optional[np.ndarray] = None,
    bulge: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Attenuation of disk and bulge light by a dusty exponential disk.

    Args:
        table: Attenuation table from ``read_attenuation_table``.
        mcold: Dusty gas mass in the disk [Msun/h].
        zcold: Gas metallicity (metal mass fraction).
        rdisk, rbulge: Disk and bulge half-mass radii [Mpc/h].
        lam: Rest-frame wavelength [Angstrom]; <= 0 means no attenuation.
        h0: Hubble parameter h.
        inc_deg: Inclination per galaxy in degrees (0 = face-on), or None for
            the isotropic average.
        bulge: False to skip the bulge attenuation (returned as ones).

    Returns:
        ``(atten_disk, atten_bulge, tauv0)``: factors multiplying the disk and
        bulge luminosities (can slightly exceed 1 from scattering), and the
        central face-on V-band optical depth.
    """
    mcold, zcold, rdisk, rbulge = (
        np.asarray(a, dtype=np.float64) for a in (mcold, zcold, rdisk, rbulge)
    )
    h = rdisk / RDISK_HALF_SCALE
    re = rbulge / SERSIC_N4_RHALF_IN_RE
    with np.errstate(divide="ignore", invalid="ignore"):
        re_h1 = np.where((h > 0) & (re > 0), re / h, np.where(re > 0, 1.0e10, 0.0))
        has_gas = (h > 0) & (mcold > _MCOLD_MIN_FOR_SIGMA)
        sig_gas0 = np.where(
            has_gas, (mcold / h0) / (2.0 * np.pi * (1.0e6 * h / h0) ** 2), 0.0
        )
    tauv0 = CV_DUST * zcold * sig_gas0
    if np.any(tauv0 < 0):
        raise ValueError("negative dust optical depth: check mcold and zcold >= 0")
    ones = np.ones_like(tauv0)
    if lam <= 0:
        return ones, ones.copy(), tauv0
    if inc_deg is not None:
        inc_deg = np.broadcast_to(np.asarray(inc_deg, dtype=np.float64), tauv0.shape)
        if np.any((inc_deg < -1e-3) | (inc_deg > 90.0 + 1e-3)):
            raise ValueError("inclinations must be between 0 and 90 degrees")

    interp = _Interpolator(table, lam, inc_deg)
    lo, x, extrap = _tauv_factors(table.tauv0, tauv0)
    taueff_disk = interp.along_tauv(lo, x, extrap, extrap, 0, 0)
    if not bulge:
        return np.exp(-taueff_disk), ones, tauv0
    taueff_bulge = _bulge_taueff(
        table, interp, tauv0, lo, x, extrap, re_h1, (re == 0) | (h == 0)
    )
    return np.exp(-taueff_disk), np.exp(-taueff_bulge), tauv0


def _bulge_taueff(table, interp, tauv0, lo, x, extrap, re_h1, no_geometry):
    """Bulge-light taueff, interpolated logarithmically in re/h."""
    re_h, nreh = table.re_h, len(table.re_h)
    jreh = _locate(re_h, re_h1)
    inside = (jreh > 0) & (jreh < nreh)
    small = (jreh == 0) & ~no_geometry  # bulge smaller than tabulated
    out = np.zeros_like(tauv0)

    if np.any(inside):
        k0 = np.clip(jreh - 1, 0, nreh - 2)
        safe_re_h = np.where(inside, re_h1, re_h[k0])
        freh = np.log(safe_re_h / re_h[k0]) / np.log(re_h[k0 + 1] / re_h[k0])
        t0 = interp.along_tauv(lo, x, extrap, extrap, k0, 1)
        t1 = interp.along_tauv(lo, x, extrap, extrap, k0 + 1, 1)
        out = np.where(inside, (1 - freh) * t0 + freh * t1, out)

    if np.any(small):
        # Endpoint re/h, with the bulge seeing the disk's own tau_V0, so the
        # tau_V0 interpolation factors are the disk's.
        t = interp.along_tauv(lo, x, extrap, extrap, 0, 1)
        out = np.where(small, t, out)
    return out


def diffuse_attenuation(
    table: AttenuationTable,
    *,
    mcold: np.ndarray,
    cold_metal: np.ndarray,
    rdisk: np.ndarray,
    rbulge: np.ndarray,
    mcold_burst: np.ndarray,
    metals_burst: np.ndarray,
    burst: np.ndarray,
    fcloud: float,
    rfacburst: float,
    lam: float,
    h0: float,
    inc_deg: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Net diffuse-dust attenuation of disk and bulge light (Calc_Atten_Diffuse).

    A fraction ``fcloud`` of the cold gas is in molecular clouds, so the
    diffuse dust mass is ``(1 - fcloud)`` times the gas mass, in the disk and
    in any ongoing burst (``burst`` true).

    Returns:
        ``(atten_disk, atten_bulge)`` factors per galaxy.
    """
    mcold = np.asarray(mcold, dtype=np.float64)
    # Clip rounding-level negative metal masses, as GALFORM does for bursts.
    zcold = _metallicity(np.maximum(np.asarray(cold_metal), 0.0), mcold)
    disk, bulge, _ = dust_atten(
        table, (1.0 - fcloud) * mcold, zcold, rdisk, rbulge, lam, h0, inc_deg
    )
    burst = np.asarray(burst, dtype=bool) & (rfacburst > 0)
    if not np.any(burst):
        return disk, bulge

    rbulge = np.asarray(rbulge, dtype=np.float64)
    mburst = np.where(burst, np.asarray(mcold_burst, dtype=np.float64), 0.0)
    zburst = _metallicity(np.maximum(np.asarray(metals_burst), 0.0), mburst)
    burst_atten, _, _ = dust_atten(
        table,
        (1.0 - fcloud) * mburst,
        zburst,
        np.where(burst, rfacburst * rbulge, 0.0),
        rbulge,
        lam,
        h0,
        inc_deg,
        bulge=False,
    )
    # Burst dust attenuates only the burst (bulge) light.
    return disk, bulge * np.where(burst, burst_atten, 1.0)


# Gas below this mass (Msun/h) is treated as metal-free (GALFORM's MCOLD_TINY).
_MCOLD_TINY = 1.0


def _metallicity(metals, gas) -> np.ndarray:
    gas = np.asarray(gas, dtype=np.float64)
    metals = np.asarray(metals, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(gas > _MCOLD_TINY, metals / gas, 0.0)
