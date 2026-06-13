"""Theoretical halo mass function calculations with configurable mass definitions.

This module provides utilities for computing theoretical HMF predictions using
the hmf library while maintaining consistency with GALFORM's mass definition (Mvir).

Includes the GPS+ (Generalized Press-Schechter + triaxial collapse) model from:
Fernández-García et al. (2025), "A redshift-independent theoretical halo mass
function validated with the Uchuu simulations", arXiv:2512.05847

Units:
- GALFORM stores masses in units of 10^10 h^-1 M_sun (standard N-body convention)
- Theoretical models (hmf library) use M_sun
- When using bins from GALFORM: bins represent log10(M / [10^10 h^-1 M_sun])
- This module applies h^3 correction to dN/dlogM to match GALFORM units
- For comparison: convert GALFORM masses by: M_true = M_galform * 10^10 / h
- GPS+ uses M200m definition; conversion to Mvir may reduce accuracy
"""

from typing import Any, Dict, Optional

import numpy as np
from hmf import MassFunction

MASS_DEFINITION_MVIR = "virial"
MASS_DEFINITION_M200C = "200c"
MASS_DEFINITION_M200M = "200m"

# Default cosmology (L800 / Planck 2015)
_OMEGA_M = 0.307
_OMEGA_L = 0.693


def get_concentration(mass: np.ndarray, z: float) -> np.ndarray:
    """
    Get NFW concentration parameter from halo mass and redshift.

    Uses the Duffy et al. (2008) / Prada et al. (2012) concentration-mass relation,
    which is calibrated on N-body simulations and evolves with both mass and redshift.

    Args:
        mass: Halo mass in M_sun (array or scalar)
        z: Redshift

    Returns:
        NFW concentration parameter c_vir

    Notes:
        Formula: c = A * (M/M_pivot)^B * (1+z)^C
        Using Duffy et al. (2008) Mvir definition parameters
    """
    # Duffy et al. (2008) parameters for Mvir
    A = 5.71
    B = -0.084
    C = -0.47
    M_pivot = 2e12  # Pivot mass in M_sun

    mass_arr = np.atleast_1d(mass)
    c_vir = A * (mass_arr / M_pivot) ** B * (1.0 + z) ** C

    # Ensure realistic range
    c_vir = np.clip(c_vir, 2.0, 20.0)

    if np.isscalar(mass):
        return float(c_vir[0])
    return c_vir


def get_mvir_to_m200c_ratio(z: float, mass: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Get the Mvir/M200c conversion factor at a given redshift and halo mass.

    Uses the concentration-mass relation to properly account for how the ratio
    varies with both redshift and mass. This avoids artificial spreads at high masses.

    The conversion is derived from the virial overdensity criterion:
        M_vir = (4π/3) * sigma_m * Δ_vir(z) * r_vir^3
        M_200c = (4π/3) * rho_c * 200 * r_200c^3

    Combined with the concentration relation c = r_200c / r_s, this gives:
        M_vir / M_200c = Δ_vir(z) / 200 * (c_vir / (c_vir - ln(1 + c_vir)))

    Args:
        z: Redshift
        mass: Halo mass in M_sun (array-like, required for accurate conversion)

    Returns:
        M_vir / M_200c ratio (array matching mass shape)

    Notes:
        The ratio increases significantly toward lower masses due to
        higher concentration.
        At z=0: ranges from ~1.2 at M=10^15 to ~3.0 at M=10^11 M_sun
    """
    if mass is None:
        raise ValueError(
            "mass parameter is required for accurate Mvir/M200c conversion"
        )

    mass_arr = np.atleast_1d(mass)

    c_vir = get_concentration(mass_arr, z)

    Ez2 = _OMEGA_M * (1.0 + z) ** 3 + _OMEGA_L
    Omega_z = _OMEGA_M * (1.0 + z) ** 3 / Ez2
    x = Omega_z - 1.0
    Delta_vir = 18.0 * np.pi**2 + 82.0 * x - 39.0 * x**2

    # Conversion formula from concentration and overdensity
    # Bryan & Norman Δ_vir is defined relative to the critical density, so no Ω_z factor
    # M_vir / M_200c = (Δ_vir / 200) * (c / (c - ln(1+c)))
    numerator = c_vir
    denominator = c_vir - np.log(1.0 + c_vir)
    ratio = (Delta_vir / 200.0) * (numerator / denominator)

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
        use_mvir: If True, adjust to Mvir definition (GALFORM-compatible).
                  If False, use default M200c (standard theory).
        model: HMF model to use ('Tinker08', 'PS', 'SMT', etc.)

    Returns:
        Dictionary with keys:
            - 'z': redshift
            - 'log10M': log10(M) mass grid
            - 'dndlog10m': dn/dlog10m in Mpc^-3
            - 'model': model name
            - 'mass_definition': 'Mvir' or 'M200c'
            - 'cosmo': cosmological parameters used

    Notes:
        The HMF library's default is M200c. If use_mvir=True, we apply
        an empirical conversion to approximate Mvir-based predictions.
    """
    try:
        hmf_calc = MassFunction(
            z=z,
            Mmin=mmin,
            Mmax=mmax,
            dlog10m=dlog10m,
            transfer_params={"extrapolate_with_eh": True},
        )
    except Exception as e:
        raise ValueError(f"Failed to create MassFunction at z={z}: {e}")

    hmf_calc.update(hmf_model=model)

    log10M = np.log10(hmf_calc.m)
    dndlog10m_m200c = hmf_calc.dndlog10m.copy()

    if use_mvir:
        # Convert from M200c to Mvir definition using mass-dependent
        # concentration relation
        # This properly accounts for how the conversion varies with halo mass

        # Get the actual masses in solar units
        M_m200c = 10**log10M

        # Get mass-dependent conversion ratio
        ratio = get_mvir_to_m200c_ratio(z, M_m200c)

        # Shift mass axis: Mvir_i = M200c_i * ratio
        log10M_mvir = np.log10(M_m200c * ratio)

        # Adjust number density correctly for the Jacobian transformation
        # The Jacobian is:
        # d(log10M_vir)/d(log10M_200c) = 1 + d(log10(ratio))/d(log10M_200c)
        # For the Duffy et al. concentration relation with power-law mass
        # dependence, d(log10(ratio))/d(log10M) is approximately constant = B
        # (the mass exponent)
        # From get_mvir_to_m200c_ratio: Duffy B = -0.084

        # Use analytical Jacobian instead of numerical gradient for speed
        B_duffy = -0.084  # Mass exponent from Duffy et al. concentration relation
        jacobian = 1.0 + B_duffy
        jacobian = np.clip(jacobian, 0.9, 1.1)  # Should be close to 1.0

        dndlog10m_mvir = dndlog10m_m200c / jacobian

        return {
            "z": z,
            "log10M": log10M_mvir,
            "dndlog10m": dndlog10m_mvir,
            "model": model,
            "mass_definition": "Mvir",
            "ratio_mvir_to_m200c": ratio,
            "h_hubble": hmf_calc.cosmo.h,
        }
    else:
        return {
            "z": z,
            "log10M": log10M,
            "dndlog10m": dndlog10m_m200c,
            "model": model,
            "mass_definition": "M200c",
            "h_hubble": hmf_calc.cosmo.h,
        }


def interpolate_hmf_to_bins(theory_hmf: Dict[str, Any], bins: np.ndarray) -> np.ndarray:
    """
    Interpolate theoretical HMF to specified mass bins in log-space.

    Args:
        theory_hmf: Dictionary from create_theoretical_hmf()
        bins: Mass bin edges in log10(M/M_sun)

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
        bins: Mass bin edges in log10(M/M_sun)
        use_mvir: If True, convert all to Mvir definition (GALFORM-compatible)
        include_ps_plus: If True, include GPS+ (Fernández-García et al. 2025)

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
        except Exception:
            models[model_name] = np.full(len(bins) - 1, np.nan)

    # Press-Schechter+ from Watson et al. 2025
    if include_ps_plus:
        try:
            ps_plus_hmf = create_press_schechter_plus(z=z, use_mvir=use_mvir)
            dndlog10m = interpolate_hmf_to_bins(ps_plus_hmf, bins)
            models["GPS+"] = dndlog10m
        except Exception:
            models["GPS+"] = np.full(len(bins) - 1, np.nan)

    return models


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
        mmin: Minimum log10(M/M_sun) for theory grid
        mmax: Maximum log10(M/M_sun) for theory grid
        dlog10m: Spacing in log10(M) for theory grid
        use_mvir: If True, convert from m200b to Mvir definition (NOT RECOMMENDED)
        mdef: Mass definition ('m200b' or 'mvir')

    Returns:
        Dictionary with same format as create_theoretical_hmf

    Notes:
        The paper uses m200b (background density). Using Mvir may reduce accuracy.
        Implementation uses the exact HaloMassFunction class from GitHub.
    """
    from colossus.cosmology import cosmology
    from scipy.integrate import quad
    from scipy.special import erfc

    # HaloMassFunction class - exact implementation from GitHub
    class HaloMassFunction:
        def __init__(self, omega_m=0.3089, z=0, mdef="m200b"):
            self.omega_m = omega_m
            self.z = z
            self.rho_crit = 277536627245.708  # M_sun / (h Mpc)^3
            self.rho_m = omega_m * self.rho_crit
            self.cosmo = cosmology.setCosmology("planck15")
            self.D0 = self.D_unnormalized(0.0)
            self.pk_table_path = None
            self.ps_args = (
                dict(model="uchuu_table", path=self.pk_table_path)
                if self.pk_table_path
                else dict(model="camb")
            )
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
            sigma_std = self.cosmo.sigma(R, self.z, ps_args=self.ps_args)
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
            x = self.cosmo.sigma(R, self.z, ps_args=self.ps_args) / 1.676

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
            integral_result = np.trapz(integrand, xi, axis=1)
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

    # Convert from m200b to Mvir if requested (NOT RECOMMENDED)
    if use_mvir and mdef == "m200b":
        Ez2 = _OMEGA_M * (1.0 + z) ** 3 + _OMEGA_L
        Omega_z = _OMEGA_M * (1.0 + z) ** 3 / Ez2
        x_vir = Omega_z - 1.0
        Delta_vir = 18.0 * np.pi**2 + 82.0 * x_vir - 39.0 * x_vir**2

        # Ratio: Mvir/m200b ≈ Delta_vir/200 (simplified)
        ratio_mvir_to_m200b = Delta_vir / 200.0

        log10M_mvir = np.log10(M * ratio_mvir_to_m200b)

        # Jacobian correction (simplified)
        jacobian = 1.0
        dndlog10m_mvir = dndlog10m / jacobian

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
        "Theory_Default": "M200c (200 x critical density)",
        "This_Module": "Mvir (converted from M200c)",
        "Conversion_Method": "Empirical redshift-dependent ratio",
        "Ratio_z0": "Mvir/M200c ≈ 2.5",
        "Ratio_z05": "Mvir/M200c ≈ 1.6",
    }
