"""Theoretical halo mass function calculations with configurable mass definitions.

This module provides utilities for computing theoretical HMF predictions using
the hmf library while maintaining consistency with GALFORM's mass definition (Mvir).

Includes the GPS+ (Generalized Press-Schechter + triaxial collapse) model from:
Fernández-García et al. (2025), "A redshift-independent theoretical halo mass
function validated with the Uchuu simulations", arXiv:2512.05847

Units:
- Masses are in M_sun/h, the unit of GALFORM's ``mhalo`` and of the hmf library,
  so theory grids can be compared directly with GALFORM mass functions.
- dn/dlog10M is in (Mpc/h)^-3, matching ``phi`` from the HMF helpers.
- Conversions between mass definitions assume an NFW profile with the
  Duffy et al. (2008) concentration-mass relations.
- GPS+ uses the M200m definition; conversion to Mvir may reduce accuracy.
"""

from typing import Any, Dict, Optional

import numpy as np

from galform_analysis._optional import import_optional

MASS_DEFINITION_MVIR = "virial"
MASS_DEFINITION_M200C = "200c"
MASS_DEFINITION_M200M = "200m"

# Default cosmology (L800 / Planck 2013 as used by EAGLE and P-Millennium).
# Every theory curve is computed in this cosmology so it can be compared with
# L800 GALFORM output; n_s is not stored in the simulation config.
_OMEGA_M = 0.307
_OMEGA_L = 0.693
_OMEGA_B = 0.0482519
_HUBBLE_H = 0.6777
_SIGMA_8 = 0.8288
_N_S = 0.9611
_T_CMB = 2.7255

# Duffy et al. (2008), Table 1, full sample (z = 0-2): c = A (M/M_pivot)^B (1+z)^C
_DUFFY_M_PIVOT = 2e12  # M_sun/h
_DUFFY_200C = (5.71, -0.084, -0.47)  # c200c(M200c)
_DUFFY_200M = (10.14, -0.081, -1.01)  # c200m(M200m)
_DUFFY_VIR = (7.85, -0.081, -0.71)  # cvir(Mvir)


def _omega_m_z(z: float) -> float:
    """Matter density parameter at redshift z (flat LCDM)."""
    a3 = _OMEGA_M * (1.0 + z) ** 3
    return a3 / (a3 + _OMEGA_L)


def _delta_vir_crit(z: float) -> float:
    """Bryan & Norman (1998) virial overdensity relative to the critical density."""
    x = _omega_m_z(z) - 1.0
    return 18.0 * np.pi**2 + 82.0 * x - 39.0 * x**2


def _duffy_concentration(mass, z: float, params) -> np.ndarray:
    a, b, c = params
    conc = a * (np.asarray(mass, dtype=float) / _DUFFY_M_PIVOT) ** b * (1.0 + z) ** c
    return np.clip(conc, 2.0, 20.0)


def _nfw_mu(x: np.ndarray) -> np.ndarray:
    return np.log1p(x) - x / (1.0 + x)


def _nfw_mass_ratio(conc: np.ndarray, delta_from: float, delta_to: float) -> np.ndarray:
    """Return M_to / M_from for NFW haloes.

    ``conc`` is the concentration r_from / r_s at overdensity ``delta_from``;
    both overdensities are relative to the same reference density. Solves
    mu(c y) / (mu(c) y^3) = delta_to / delta_from for y = r_to / r_from by
    bisection (the left side decreases monotonically in y).
    """
    conc = np.asarray(conc, dtype=float)
    target = delta_to / delta_from
    mu_c = _nfw_mu(conc)
    lo = np.full_like(conc, np.log(1e-3))
    hi = np.full_like(conc, np.log(1e3))
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        y = np.exp(mid)
        too_dense = _nfw_mu(conc * y) / (mu_c * y**3) > target
        lo = np.where(too_dense, mid, lo)
        hi = np.where(too_dense, hi, mid)
    y = np.exp(0.5 * (lo + hi))
    return _nfw_mu(conc * y) / mu_c


def _convert_mass_definition(
    log10m: np.ndarray, dndlog10m: np.ndarray, ratio: np.ndarray
):
    """Shift a mass function to a new mass definition, with the Jacobian.

    dn/dlog10M_new = dn/dlog10M_old / (dlog10M_new / dlog10M_old).
    """
    log10m_new = log10m + np.log10(ratio)
    jacobian = np.gradient(log10m_new, log10m) if log10m.size > 1 else 1.0
    return log10m_new, dndlog10m / jacobian


def _so_definition(mdef, z: float):
    """Overdensity (relative to critical) and Duffy+08 parameters of an hmf mdef.

    hmf evaluates each fitting function in the mass definition it was
    calibrated in (``MassFunction.mdef``): SOMean(200) for Tinker08 and PS,
    SOVirial for SMT.
    """
    kind = type(mdef).__name__
    overdensity = getattr(mdef, "params", {}).get("overdensity", 200.0)
    if kind == "SOVirial":
        return _delta_vir_crit(z), _DUFFY_VIR, "Mvir"
    if kind == "SOCritical" and overdensity == 200:
        return 200.0, _DUFFY_200C, "M200c"
    if kind == "SOMean" and overdensity == 200:
        return 200.0 * _omega_m_z(z), _DUFFY_200M, "M200m"
    raise ValueError(f"Unsupported hmf mass definition {mdef!r}")


def _trapezoid(y, x, axis=-1):
    """np.trapezoid (NumPy >= 2.0), falling back to np.trapz for NumPy 1.x."""
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return integrate(y, x, axis=axis)


def get_concentration(mass: np.ndarray, z: float) -> np.ndarray:
    """
    Get the NFW concentration c200c = r_200c / r_s from M200c and redshift.

    Uses the Duffy et al. (2008) relation for the M200c definition (Table 1,
    full sample): c = 5.71 (M / 2e12 M_sun/h)^-0.084 (1+z)^-0.47, clipped to
    [2, 20].

    Args:
        mass: Halo mass M200c in M_sun/h (array or scalar)
        z: Redshift

    Returns:
        NFW concentration c200c (float for scalar input, else array)
    """
    c200c = _duffy_concentration(np.atleast_1d(mass), z, _DUFFY_200C)
    if np.isscalar(mass):
        return float(c200c[0])
    return c200c


def get_mvir_to_m200c_ratio(z: float, mass: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Get the Mvir/M200c conversion factor at a given redshift and halo mass.

    Uses the concentration-mass relation to properly account for how the ratio
    varies with both redshift and mass. This avoids artificial spreads at high masses.

    Assumes an NFW profile with concentration c200c(M200c, z) from
    :func:`get_concentration`. The virial radius encloses a mean density of
    Delta_vir(z) rho_c (Bryan & Norman 1998), so r_vir / r_200c = y solves

        mu(c y) / (mu(c) y^3) = Delta_vir(z) / 200,  mu(x) = ln(1+x) - x/(1+x)

    and M_vir / M_200c = mu(c y) / mu(c).

    Args:
        z: Redshift
        mass: Halo mass M200c in M_sun/h (array-like, required)

    Returns:
        M_vir / M_200c ratio (float for scalar input, else array)

    Notes:
        Delta_vir < 200 at all redshifts, so the ratio is always > 1. It is
        about 1.2-1.4 at z = 0 and approaches 1 at high redshift, where
        Delta_vir -> 18 pi^2 ~ 178.
    """
    if mass is None:
        raise ValueError(
            "mass parameter is required for accurate Mvir/M200c conversion"
        )

    c200c = get_concentration(np.atleast_1d(mass), z)
    ratio = _nfw_mass_ratio(c200c, 200.0, _delta_vir_crit(z))

    if np.isscalar(mass):
        return float(ratio[0])
    return ratio


def create_theoretical_hmf(
    z: float,
    mmin: float = 9.0,
    mmax: float = 15.0,
    dlog10m: float = 0.01,
    use_mvir: bool = True,
    model: str = "Tinker08",
) -> Dict[str, Any]:
    """
    Generate theoretical HMF at a given redshift using hmf library.

    Args:
        z: Redshift
        mmin: Minimum log10(M/M_sun) for theory grid
        mmax: Maximum log10(M/M_sun) for theory grid
        dlog10m: Spacing in log10(M) for theory grid
        use_mvir: If True, return the HMF in the Mvir definition
                  (GALFORM-compatible). If False, return it in M200c.
        model: HMF model to use ('Tinker08', 'PS', 'SMT', etc.)

    Returns:
        Dictionary with keys:
            - 'z': redshift
            - 'log10M': log10(M / [M_sun/h]) mass grid
            - 'dndlog10m': dn/dlog10m in (Mpc/h)^-3
            - 'model': model name
            - 'mass_definition': 'Mvir' or 'M200c'
            - 'native_mass_definition': definition hmf computed the fit in
            - 'mass_ratio': M_out / M_native at each grid point
            - 'h_hubble': Hubble parameter h of the cosmology

    Raises:
        ImportError: If the optional ``hmf`` package is not installed.
        ValueError: If hmf cannot build the mass function.

    Notes:
        hmf returns each fit in its native definition (M200m for Tinker08 and
        PS, Mvir for SMT). Each mass is converted to the requested definition
        assuming an NFW profile with the Duffy et al. (2008) concentration of
        the native definition, and dn/dlog10M is corrected by the Jacobian of
        the mass mapping. The cosmology is the L800 one (module constants).
    """
    MassFunction = import_optional("hmf").MassFunction
    FlatLambdaCDM = import_optional("astropy.cosmology").FlatLambdaCDM
    try:
        hmf_calc = MassFunction(
            z=z,
            Mmin=mmin,
            Mmax=mmax,
            dlog10m=dlog10m,
            hmf_model=model,
            cosmo_model=FlatLambdaCDM(
                H0=100.0 * _HUBBLE_H, Om0=_OMEGA_M, Ob0=_OMEGA_B, Tcmb0=_T_CMB
            ),
            sigma_8=_SIGMA_8,
            n=_N_S,
            transfer_params={"extrapolate_with_eh": True},
        )
        log10m_native = np.log10(hmf_calc.m)
        dndlog10m_native = np.array(hmf_calc.dndlog10m, dtype=float)
    except Exception as e:
        raise ValueError(f"Failed to create MassFunction at z={z}: {e}") from e

    delta_native, duffy, native_name = _so_definition(hmf_calc.mdef, z)
    if use_mvir:
        delta_out, out_name = _delta_vir_crit(z), "Mvir"
    else:
        delta_out, out_name = 200.0, "M200c"

    conc = _duffy_concentration(10**log10m_native, z, duffy)
    ratio = _nfw_mass_ratio(conc, delta_native, delta_out)
    log10M, dndlog10m = _convert_mass_definition(log10m_native, dndlog10m_native, ratio)
    return {
        "z": z,
        "log10M": log10M,
        "dndlog10m": dndlog10m,
        "model": model,
        "mass_definition": out_name,
        "native_mass_definition": native_name,
        "mass_ratio": ratio,
        "h_hubble": _HUBBLE_H,
    }


def interpolate_hmf_to_bins(theory_hmf: Dict[str, Any], bins: np.ndarray) -> np.ndarray:
    """
    Interpolate theoretical HMF to specified mass bins in log-space.

    Args:
        theory_hmf: Dictionary from create_theoretical_hmf()
        bins: Mass bin edges in log10(M / [M_sun/h])

    Returns:
        Array of dN/dlog10m values at bin centers
    """
    centers = 0.5 * (bins[:-1] + bins[1:])

    # Interpolate in log-space (more accurate for power-law-like functions)
    log10M_theory = theory_hmf["log10M"]
    dndlog10m_theory = theory_hmf["dndlog10m"]

    # Mask for valid (finite, positive) values
    mask = (
        np.isfinite(log10M_theory)
        & np.isfinite(dndlog10m_theory)
        & (dndlog10m_theory > 0)
    )

    if np.count_nonzero(mask) < 2:
        # Not enough valid points
        return np.full_like(centers, np.nan, dtype=float)

    # Log-space interpolation
    log_interp = np.interp(
        centers, log10M_theory[mask], np.log10(dndlog10m_theory[mask])
    )
    result = 10**log_interp

    return result


def compute_theoretical_hmfs(
    z: float, bins: np.ndarray, use_mvir: bool = True, include_ps_plus: bool = True
) -> Dict[str, np.ndarray]:
    """
    Compute multiple theoretical HMF models at a given redshift.

    Args:
        z: Redshift
        bins: Mass bin edges in log10(M / [M_sun/h])
        use_mvir: If True, convert all to Mvir definition (GALFORM-compatible)
        include_ps_plus: If True, include GPS+ (Fernández-García et al. 2025)

    Raises:
        ImportError: If an optional dependency (``science`` extra) is missing.
            Other failures of an individual model give NaN for that model.

    Returns:
        Dictionary with keys for each model:
            - 'PS' (Press-Schechter)
            - 'SMT' (Sheth-Mo-Tormen)
            - 'Tinker08'
            - 'GPS+' (Generalized PS + triaxial collapse, if include_ps_plus=True)

        Each value is an array of dN/dlog10m at bin centers
    """
    models = {}

    # Standard models
    model_names = ["PS", "SMT", "Tinker08"]

    for model_name in model_names:
        try:
            theory_hmf = create_theoretical_hmf(
                z=z, use_mvir=use_mvir, model=model_name
            )
            dndlog10m = interpolate_hmf_to_bins(theory_hmf, bins)
            models[model_name] = dndlog10m
        except ImportError:
            raise
        except Exception:
            models[model_name] = np.full(len(bins) - 1, np.nan)

    # GPS+ from Fernández-García et al. (2025)
    if include_ps_plus:
        try:
            ps_plus_hmf = create_press_schechter_plus(z=z, use_mvir=use_mvir)
            dndlog10m = interpolate_hmf_to_bins(ps_plus_hmf, bins)
            models["GPS+"] = dndlog10m
        except ImportError:
            raise
        except Exception:
            models["GPS+"] = np.full(len(bins) - 1, np.nan)

    return models


def _camb_sigma_function(omega_m: float):
    """sigma(R, z) of the linear CAMB P(k) in the L800 cosmology.

    Returns a function ``sigma(R, growth)`` for R in Mpc/h, where ``growth``
    is the linear growth factor D(z)/D(0). The spectrum is normalised to
    sigma_8 = ``_SIGMA_8``. This replaces colossus's ``model="camb"``, which
    asks CAMB >= 2 for a 2-point spectrum (rejected by CAMB) whenever colossus
    has no cached sigma(R).
    """
    camb = import_optional("camb")
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=100.0 * _HUBBLE_H,
        ombh2=_OMEGA_B * _HUBBLE_H**2,
        omch2=(omega_m - _OMEGA_B) * _HUBBLE_H**2,
        omk=0.0,
        mnu=0.0,
        TCMB=_T_CMB,
    )
    pars.InitPower.set_params(ns=_N_S)
    pars.set_matter_power(redshifts=[0.0], kmax=1e3)
    pars.NonLinear = camb.model.NonLinear_none
    k, _, pk = camb.get_results(pars).get_matter_power_spectrum(
        minkh=1e-4, maxkh=1e3, npoints=4000
    )
    pk = pk[0]
    ln_k = np.log(k)

    def sigma_unnormalised(R: np.ndarray) -> np.ndarray:
        kr = np.outer(np.atleast_1d(R), k)
        window = 3.0 * (np.sin(kr) - kr * np.cos(kr)) / kr**3
        integrand = k**3 * pk * window**2 / (2.0 * np.pi**2)
        return np.sqrt(_trapezoid(integrand, ln_k, axis=-1))

    norm = _SIGMA_8 / sigma_unnormalised(np.array([8.0]))[0]

    def sigma(R, growth: float) -> np.ndarray:
        return norm * growth * sigma_unnormalised(R)

    return sigma


def create_press_schechter_plus(
    z: float,
    mmin: float = 9.0,
    mmax: float = 15.0,
    dlog10m: float = 0.01,
    use_mvir: bool = True,
    mdef: str = "m200b",
) -> Dict[str, Any]:
    """
    Create GPS+ (Generalized Press-Schechter + triaxial collapse) HMF.

    Implements the theoretical framework from Fernández-García et al. (2025):
    "A redshift-independent theoretical halo mass function validated with
    the Uchuu simulations"
    arXiv:2512.05847

    This implementation matches the exact GitHub code from https://github.com/uchuuproject/HMF_GPSplus

    This model uses triaxial collapse physics and achieves ~5-10% accuracy across
    log(M) = 6.5-16 and z = 0-20. It has no explicit redshift dependence - evolution
    enters solely through sigma(M,z).

    Key features:
    - Uses m200b mass definition (200 times background density) by default
    - Fitted parameters A=1.089, B=0.652, D=1.0, E=0.17, F=0.087 from Uchuu
      simulations
    - Mass-dependent functions b(M) and c(M) encode power spectrum shape
    - Modified variance sigma_mod includes correction term U(sigma) for
      improved accuracy
    - Outperforms Sheth-Tormen at z > 2 (ST deviates 70-80%, GPS+ ~5-10%)

    Args:
        z: Redshift
        mmin: Minimum log10(M / [M_sun/h]) for theory grid
        mmax: Maximum log10(M / [M_sun/h]) for theory grid
        dlog10m: Spacing in log10(M) for theory grid
        use_mvir: If True, convert from m200b to Mvir definition (NOT RECOMMENDED)
        mdef: Mass definition; only 'm200b' is supported

    Returns:
        Dictionary with same format as create_theoretical_hmf

    Notes:
        The paper uses m200b (background density). Using Mvir may reduce accuracy.
        Implementation uses the exact HaloMassFunction class from GitHub.
    """
    cosmology = import_optional("colossus.cosmology.cosmology")
    quad = import_optional("scipy.integrate").quad
    erfc = import_optional("scipy.special").erfc

    # HaloMassFunction class - exact implementation from GitHub
    class HaloMassFunction:
        def __init__(self, omega_m=0.3089, z=0, mdef="m200b"):
            self.omega_m = omega_m
            self.z = z
            self.rho_crit = 277536627245.708  # M_sun / (h Mpc)^3
            self.rho_m = omega_m * self.rho_crit
            # persistence="" keeps colossus from caching sigma(R) on disk,
            # where it would outlive changes to the cosmology or P(k).
            self.cosmo = cosmology.setCosmology(
                "galform_analysis_L800",
                persistence="",
                params={
                    "flat": True,
                    "H0": 100.0 * _HUBBLE_H,
                    "Om0": omega_m,
                    "Ob0": _OMEGA_B,
                    "sigma8": _SIGMA_8,
                    "ns": _N_S,
                },
            )
            self.D0 = self.D_unnormalized(0.0)
            self._sigma = _camb_sigma_function(omega_m)
            self._growth = self.cosmo.growthFactor(z)
            self.mdef = mdef

            if self.mdef == "m200b":
                self.aa, self.bb, self.DD, self.EE, self.FF = (
                    1.089,
                    0.652,
                    1.0,
                    0.17,
                    0.087,
                )
            else:
                raise ValueError(f"mdef '{self.mdef}' no válido. Usa 'm200b' o 'mvir'.")

        def RtoM(self, M):
            return (3 * M / (4 * np.pi * self.omega_m * self.rho_crit)) ** (1 / 3)

        def E(self, z):
            return np.sqrt(self.omega_m * (1 + z) ** 3 + (1 - self.omega_m))

        def D_unnormalized(self, z):
            integral, _ = quad(lambda zp: (1 + zp) / (self.E(zp) ** 3), z, np.inf)
            return (5 * self.omega_m * self.E(z) / 2) * integral

        def sigma(self, M):
            M = np.atleast_1d(M)
            R = self.RtoM(M)
            sigma_std = self._sigma(R, self._growth)
            x = sigma_std / 1.676
            U2 = (-0.00221 * x**3 + 0.03835 * x**2 + 0.17810 * x - 0.01507) ** 2
            sigma_mod = np.sqrt(sigma_std**2 + U2)
            return sigma_mod[0] if np.isscalar(M) else sigma_mod

        def b(self, m_val):
            m = np.array([1e16, 1e15, 1e14, 6.5e10, 1e10, 1e9, 1e8, 1e7, 1e6])
            b = np.array(
                [0.5259, 0.415, 0.328, 0.1764, 0.1552, 0.1308, 0.1179, 0.1045, 0.094]
            )
            coeffs = np.polyfit(np.log10(m), np.log10(b), 4)
            return 10 ** np.polyval(coeffs, np.log10(m_val))

        def c(self, m_val):
            m = np.array(
                [
                    3e15,
                    3e14,
                    3e13,
                    3e12,
                    3e11,
                    3e10,
                    3e9,
                    3e8,
                    3e7,
                    3e6,
                    1e10,
                    1e9,
                    1e8,
                    1e7,
                    1e6,
                ]
            )
            b = np.array(
                [
                    0.613,
                    0.474,
                    0.373,
                    0.301,
                    0.249,
                    0.209,
                    0.1794,
                    0.1560,
                    0.1355,
                    0.1223,
                    0.1942,
                    0.168,
                    0.1466,
                    0.1298,
                    0.1161,
                ]
            )
            coeffs = np.polyfit(np.log10(m), np.log10(b), 4)
            return 10 ** np.polyval(coeffs, np.log10(m_val))

        def F(self, m_array):
            m_array = np.atleast_1d(m_array)
            R = self.RtoM(m_array)
            b_val = self.b(m_array)
            sig = self.sigma(m_array)
            x = self._sigma(R, self._growth) / 1.676

            term1 = (1 + 0.845 * x - 0.04 * x**2 + 0.0025 * x**3) ** self.bb
            term2 = (
                self.aa * 1.365 * (1 + self.EE * b_val - self.FF * b_val**2) ** self.DD
            )
            delta_c = term1 * term2

            c_m = self.c(m_array)
            cte = delta_c / (np.sqrt(2) * sig)

            xi = np.linspace(0, 1, 1000)
            xi2 = xi**2
            xi_mat = xi[np.newaxis, :]
            c_m_mat = c_m[:, np.newaxis]
            cte_mat = cte[:, np.newaxis]
            integrand = (
                erfc(
                    cte_mat
                    * np.sqrt(
                        (1 - np.exp(-c_m_mat * xi_mat**2))
                        / (1 + np.exp(-c_m_mat * xi_mat**2))
                    )
                )
                * xi2
            )
            integral_result = _trapezoid(integrand, xi, axis=1)
            V = 3 * integral_result

            F_val = erfc(0.98 * cte) / V
            return F_val if F_val.size > 1 else F_val[0]

        def n0(self, m):  # returns dn/dlnM
            s = 0.01
            Fm = self.F(m)
            Fm_s = self.F((1 + s) * m)
            der = (Fm - Fm_s) / s
            return der * self.rho_m / (m * (1 + s / 2))

    # Create GPS+ model
    log10M = np.arange(mmin, mmax + dlog10m / 2.0, dlog10m)
    M = 10**log10M

    hmf_model = HaloMassFunction(omega_m=_OMEGA_M, z=z, mdef=mdef)
    dn_dlnM = hmf_model.n0(M)  # dn/dlnM
    dndlog10m = dn_dlnM * np.log(10.0)  # Convert to dn/dlog10M

    result = {
        "z": z,
        "log10M": log10M,
        "dndlog10m": dndlog10m,
        "model": "GPS+",
        "mass_definition": mdef,
        "h_hubble": hmf_model.cosmo.h,
    }

    # Convert from m200b to Mvir if requested (NOT RECOMMENDED). NFW profile
    # with the Duffy et al. (2008) c200m(M200m) relation; 200 x the mean
    # density is 200 Omega_m(z) x the critical density.
    if use_mvir and mdef == "m200b":
        c200m = _duffy_concentration(M, z, _DUFFY_200M)
        ratio_mvir_to_m200b = _nfw_mass_ratio(
            c200m, 200.0 * _omega_m_z(z), _delta_vir_crit(z)
        )
        log10M_mvir, dndlog10m_mvir = _convert_mass_definition(
            log10M, dndlog10m, ratio_mvir_to_m200b
        )

        result.update(
            {
                "log10M": log10M_mvir,
                "dndlog10m": dndlog10m_mvir,
                "mass_definition": "Mvir",
                "ratio_mvir_to_m200b": ratio_mvir_to_m200b,
            }
        )

    return result


def get_mass_definition_info() -> Dict[str, str]:
    """
    Return information about mass definitions used in this module.

    Returns:
        Dictionary documenting which mass definitions are used
    """
    return {
        "GALFORM": "Mvir (virial mass, Δ ≈ 178.65)",
        "Theory_Default": "Native fit definition (M200m: Tinker08, PS; Mvir: SMT)",
        "This_Module": "Mvir (converted from the native definition)",
        "Conversion_Method": "NFW profile with Duffy et al. (2008) concentrations",
        "Ratio_z0": "Mvir/M200c ≈ 1.2-1.4",
        "Ratio_z05": "Mvir/M200c ≈ 1.1-1.2",
    }
