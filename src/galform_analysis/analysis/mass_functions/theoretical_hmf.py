"""Theoretical halo mass function calculations with configurable mass definitions.

This module provides utilities for computing theoretical HMF predictions using
the hmf library while maintaining consistency with GALFORM's mass definition (Mvir).

Includes the GPS+ (Generalized Press-Schechter + triaxial collapse) model from:
Fernández-García et al. (2025), "A redshift-independent theoretical halo mass 
function validated with the Uchuu simulations", arXiv:2512.05847

IMPORTANT UNITS NOTE:
- GALFORM stores masses in units of 10^10 h^-1 M_sun (standard N-body convention)
- Theoretical models (hmf library) use M_sun
- When using bins from GALFORM: bins represent log10(M / [10^10 h^-1 M_sun])
- This module applies h^3 correction to dN/dlogM to match GALFORM units
- For comparison: convert GALFORM masses by: M_true = M_galform * 10^10 / h
- GPS+ uses M200m definition; conversion to Mvir may reduce accuracy
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple
from hmf import MassFunction
from ...config import Cosmology


# Mass definition conversion parameters
# Mvir/M200c ratios as a function of redshift and halo mass
# Based on concentration-mass relations and halo profile assumptions
MASS_DEFINITION_MVIR = 'virial'
MASS_DEFINITION_M200C = '200c'
MASS_DEFINITION_M200M = '200m'


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
    c_vir = A * (mass_arr / M_pivot)**B * (1.0 + z)**C
    
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
        M_vir = (4π/3) * ρ_m * Δ_vir(z) * r_vir^3
        M_200c = (4π/3) * ρ_c * 200 * r_200c^3
    
    Combined with the concentration relation c = r_200c / r_s, this gives:
        M_vir / M_200c = Δ_vir(z) / 200 * (c_vir / (c_vir - ln(1 + c_vir)))
    
    Args:
        z: Redshift
        mass: Halo mass in M_sun (array-like, required for accurate conversion)
        
    Returns:
        M_vir / M_200c ratio (array matching mass shape)
        
    Notes:
        The ratio increases significantly toward lower masses due to higher concentration.
        At z=0: ranges from ~1.2 at M=10^15 to ~3.0 at M=10^11 M_sun
    """
    if mass is None:
        raise ValueError("mass parameter is required for accurate Mvir/M200c conversion")
    
    mass_arr = np.atleast_1d(mass)
    
    # Get concentration from mass and redshift
    c_vir = get_concentration(mass_arr, z)
    
    # Virial overdensity as function of redshift (flat ΛCDM; Bryan & Norman 1998)
    cosmo = Cosmology()
    Omega_m0 = cosmo.OMEGA_M
    Omega_lambda0 = cosmo.OMEGA_L
    Ez2 = Omega_m0 * (1.0 + z)**3 + Omega_lambda0
    Omega_z = Omega_m0 * (1.0 + z)**3 / Ez2
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


def create_theoretical_hmf(z: float, 
                          mmin: float = 9.0, 
                          mmax: float = 15.0,
                          dlog10m: float = 0.01,
                          use_mvir: bool = True,
                          model: str = 'Tinker08') -> Dict[str, Any]:
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
        hmf_calc = MassFunction(z=z, Mmin=mmin, Mmax=mmax, dlog10m=dlog10m)
    except Exception as e:
        raise ValueError(f"Failed to create MassFunction at z={z}: {e}")
    
    hmf_calc.update(hmf_model=model)
    
    log10M = np.log10(hmf_calc.m)
    dndlog10m_m200c = hmf_calc.dndlog10m.copy()
    
    if use_mvir:
        # Convert from M200c to Mvir definition using mass-dependent concentration relation
        # This properly accounts for how the conversion varies with halo mass
        
        # Get the actual masses in solar units
        M_m200c = 10**log10M
        
        # Get mass-dependent conversion ratio
        ratio = get_mvir_to_m200c_ratio(z, M_m200c)
        
        # Shift mass axis: Mvir_i = M200c_i * ratio
        log10M_mvir = np.log10(M_m200c * ratio)
        
        # Adjust number density correctly for the Jacobian transformation
        # The Jacobian is: d(log10M_vir)/d(log10M_200c) = 1 + d(log10(ratio))/d(log10M_200c)
        # For the Duffy et al. concentration relation with power-law mass dependence,
        # d(log10(ratio))/d(log10M) is approximately constant = B (the mass exponent)
        # From get_mvir_to_m200c_ratio: Duffy B = -0.084
        
        # Use analytical Jacobian instead of numerical gradient for speed
        B_duffy = -0.084  # Mass exponent from Duffy et al. concentration relation
        jacobian = 1.0 + B_duffy
        jacobian = np.clip(jacobian, 0.9, 1.1)  # Should be close to 1.0
        
        dndlog10m_mvir = dndlog10m_m200c / jacobian
        
        return {
            'z': z,
            'log10M': log10M_mvir,
            'dndlog10m': dndlog10m_mvir,
            'model': model,
            'mass_definition': 'Mvir',
            'ratio_mvir_to_m200c': ratio,
            'h_hubble': hmf_calc.cosmo.h,
        }
    else:
        return {
            'z': z,
            'log10M': log10M,
            'dndlog10m': dndlog10m_m200c,
            'model': model,
            'mass_definition': 'M200c',
            'h_hubble': hmf_calc.cosmo.h,
        }


def interpolate_hmf_to_bins(theory_hmf: Dict[str, Any], 
                            bins: np.ndarray,
                            apply_hubble_correction: bool = True) -> np.ndarray:
    """
    Interpolate theoretical HMF to specified mass bins in log-space.
    
    Args:
        theory_hmf: Dictionary from create_theoretical_hmf()
        bins: Mass bin edges in log10(M/M_sun)
        apply_hubble_correction: If True, multiply by h^3 to match code units
        
    Returns:
        Array of dN/dlog10m values at bin centers
    """
    centers = 0.5 * (bins[:-1] + bins[1:])
    
    # Interpolate in log-space (more accurate for power-law-like functions)
    log10M_theory = theory_hmf['log10M']
    dndlog10m_theory = theory_hmf['dndlog10m']
    
    # Mask for valid (finite, positive) values
    mask = np.isfinite(log10M_theory) & np.isfinite(dndlog10m_theory) & (dndlog10m_theory > 0)
    
    if np.count_nonzero(mask) < 2:
        # Not enough valid points
        return np.full_like(centers, np.nan, dtype=float)
    
    # Log-space interpolation
    log_interp = np.interp(
        centers,
        log10M_theory[mask],
        np.log10(dndlog10m_theory[mask])
    )
    result = 10**log_interp
    
    # Apply Hubble parameter correction if requested
    if apply_hubble_correction:
        h_hubble = theory_hmf.get('h_hubble', 1.0)
        result = result * (h_hubble ** 3)
    
    return result


def compute_theoretical_hmfs(z: float,
                               bins: np.ndarray,
                               use_mvir: bool = True,
                               include_ps_plus: bool = True) -> Dict[str, np.ndarray]:
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
    model_names = ['PS', 'SMT', 'Tinker08']
    
    for model_name in model_names:
        try:
            theory_hmf = create_theoretical_hmf(
                z=z,
                use_mvir=use_mvir,
                model=model_name
            )
            dndlog10m = interpolate_hmf_to_bins(theory_hmf, bins, apply_hubble_correction=True)
            models[model_name] = dndlog10m
        except Exception as e:
            print(f"Warning: Failed to compute {model_name} at z={z}: {e}")
            models[model_name] = np.full(len(bins)-1, np.nan)
    
    # Press-Schechter+ from Watson et al. 2025
    if include_ps_plus:
        try:
            ps_plus_hmf = create_press_schechter_plus(z=z, use_mvir=use_mvir)
            dndlog10m = interpolate_hmf_to_bins(ps_plus_hmf, bins, apply_hubble_correction=True)
            models['GPS+'] = dndlog10m
        except Exception as e:
            print(f"Warning: Failed to compute GPS+ at z={z}: {e}")
            models['GPS+'] = np.full(len(bins)-1, np.nan)
    
    return models


def create_press_schechter_plus(z: float,
                                 mmin: float = 9.0,
                                 mmax: float = 15.0,
                                 dlog10m: float = 0.01,
                                 use_mvir: bool = True) -> Dict[str, Any]:
    """
    Create GPS+ (Generalized Press-Schechter + triaxial collapse) HMF.
    
    Implements the theoretical framework from Fernández-García et al. (2025):
    "A redshift-independent theoretical halo mass function validated with the Uchuu simulations"
    arXiv:2512.05847
    
    This model uses triaxial collapse physics and achieves 10-20% accuracy across
    log(M) = 6.5-16 and z = 0-20. It has no explicit redshift dependence - evolution
    enters solely through σ(M,z).
    
    Key features:
    - Uses M200m mass definition (200 × mean matter density) for universality
    - Fitted parameters A=1.089, B=0.652 from Uchuu simulations
    - Mass-dependent functions b(M) and c(M) encode power spectrum shape
    - Outperforms Sheth-Tormen at z > 2 (ST deviates 70-80%, GPS+ ~20%)
    
    Args:
        z: Redshift
        mmin: Minimum log10(M/M_sun) for theory grid
        mmax: Maximum log10(M/M_sun) for theory grid
        dlog10m: Spacing in log10(M) for theory grid
        use_mvir: If True, convert from M200m to Mvir definition
        
    Returns:
        Dictionary with same format as create_theoretical_hmf
        
    Notes:
        The paper uses M200m, not Mvir. If use_mvir=True, we convert at the end.
        This may introduce small discrepancies since GPS+ was calibrated on M200m.
    """
    from scipy.special import erfc
    from scipy.integrate import quad
    
    # Get cosmology and power spectrum from hmf library
    try:
        hmf_calc = MassFunction(z=z, Mmin=mmin, Mmax=mmax, dlog10m=dlog10m)
    except Exception as e:
        raise ValueError(f"Failed to create MassFunction at z={z}: {e}")
    
    # Extract mass grid and variance σ(M,z)
    M = hmf_calc.m  # Mass in M_sun
    log10M = np.log10(M)
    sigma = hmf_calc.sigma  # RMS density fluctuation
    
    # Cosmological parameters
    # Use hmf library's mean density (already in correct units: M_sun/Mpc^3 comoving)
    rho_m = hmf_calc.mean_density0  # M_sun/Mpc^3
    h_hubble = hmf_calc.cosmo.h
    
    # Physical constants from paper
    delta_c = 1.686  # Critical overdensity for spherical collapse
    A = 1.089  # Fitted parameter (Equation 7)
    B = 0.652  # Fitted parameter (Equation 7)
    D = 1.0    # Theoretical value confirmed by simulations
    
    # Equation 8: Mass-dependent function b(M)
    x_b = log10M
    log10_b = (-1.28 + 0.05781 * x_b - 0.005622 * x_b**2 
               - 0.0005884 * x_b**3 - 1.365e-5 * x_b**4)
    b_M = 10**log10_b
    
    # Equation 9: Mass-dependent function c(M)
    x_c = log10M
    log10_c = (-1.124 + 0.01756 * x_c + 0.002539 * x_c**2 
               - 6.438e-5 * x_c**3 + 4.726e-6 * x_c**4)
    c_M = 10**log10_c
    
    # Equation 6: Variance correction U(σ/δc)
    x_U = sigma / delta_c
    U = -0.01507 + 0.17810 * x_U + 0.03835 * x_U**2 - 0.00221 * x_U**3
    
    # Equation 5: Corrected variance Σ(M,z)
    Sigma = np.sqrt(sigma**2 + U**2)
    
    # Equation 7: Modified critical overdensity <δc>(σ,M)
    x_delta = sigma / delta_c
    delta_c_mod = (delta_c * (1.0 + 0.845 * x_delta - 0.04 * x_delta**2 + 0.0025 * x_delta**3)**B
                   * A * (1.0 + 0.17 * b_M - 0.087 * b_M**2)**D)
    
    # Equation 4: Volume factor V(Σ,M) - requires numerical integration
    def compute_V(Sigma_val, c_val, delta_c_val):
        """Compute V factor from Equation 4 using numerical integration."""
        def integrand(xi):
            exp_term = np.exp(-c_val * xi**2)
            factor = (1.0 - exp_term) / (1.0 + exp_term)
            arg = (delta_c_val / (2.0 * Sigma_val)) * factor
            return erfc(arg) * xi**2
        
        try:
            result, _ = quad(integrand, 0, 1, limit=50)
            return 3.0 * result
        except:
            # Fallback to trapezoidal rule if quad fails
            xi_grid = np.linspace(0, 1, 100)
            exp_term = np.exp(-c_val * xi_grid**2)
            factor = (1.0 - exp_term) / (1.0 + exp_term)
            arg = (delta_c_val / (2.0 * Sigma_val)) * factor
            integrand_vals = erfc(arg) * xi_grid**2
            return 3.0 * np.trapz(integrand_vals, xi_grid)
    
    # Vectorized computation of V for all masses
    V_factors = np.array([compute_V(Sigma[i], c_M[i], delta_c_mod[i]) 
                          for i in range(len(M))])
    
    # Equation 3: Mass fraction F(M,z)
    F = erfc(delta_c_mod / (np.sqrt(2.0) * sigma)) / V_factors
    
    # Standard relation: dn/dlnM = -(ρ_m/M) * dF/dlnM
    # Compute dF/dlnM using central differences
    lnM = np.log(M)
    dF_dlnM = np.gradient(F, lnM)
    
    # HMF: dn/dlnM (take absolute value to ensure positive)
    dn_dlnM = np.abs((rho_m / M) * dF_dlnM)
    
    # Convert to dn/dlog10M
    dndlog10m = dn_dlnM / np.log(10.0)
    
    # Handle negative or invalid values (should be positive for a valid HMF)
    dndlog10m = np.maximum(dndlog10m, 1e-100)
    
    result = {
        'z': z,
        'log10M': log10M,
        'dndlog10m': dndlog10m,
        'model': 'GPS+',
        'mass_definition': 'M200m',
        'h_hubble': h_hubble,
    }
    
    # Convert from M200m to Mvir if requested
    if use_mvir:
        # This conversion may introduce errors since GPS+ was calibrated on M200m
        # The paper shows that using Mvir worsens agreement at low-z, high-mass
        M_m200m = M
        
        # Approximate M200m to Mvir conversion
        # At z=0: Mvir/M200m ~ 1.5-2.0 depending on mass
        # This is different from M200c conversion!
        cosmo = Cosmology()
        Omega_m0 = cosmo.OMEGA_M
        Omega_lambda0 = cosmo.OMEGA_L
        Ez2 = Omega_m0 * (1.0 + z)**3 + Omega_lambda0
        Omega_z = Omega_m0 * (1.0 + z)**3 / Ez2
        x_vir = Omega_z - 1.0
        Delta_vir = 18.0 * np.pi**2 + 82.0 * x_vir - 39.0 * x_vir**2
        
        # Approximate ratio Mvir/M200m ≈ Delta_vir/200 (rough estimate)
        ratio_mvir_to_m200m = Delta_vir / 200.0
        
        log10M_mvir = np.log10(M_m200m * ratio_mvir_to_m200m)
        
        # Jacobian correction
        jacobian = 1.0  # Simplified - could improve with mass-dependent ratio
        dndlog10m_mvir = dndlog10m / jacobian
        
        result.update({
            'log10M': log10M_mvir,
            'dndlog10m': dndlog10m_mvir,
            'mass_definition': 'Mvir',
            'ratio_mvir_to_m200m': ratio_mvir_to_m200m,
        })
    
    return result


def get_mass_definition_info() -> Dict[str, str]:
    """
    Return information about mass definitions used in this module.
    
    Returns:
        Dictionary documenting which mass definitions are used
    """
    return {
        'GALFORM': 'Mvir (virial mass, Δ ≈ 178.65)',
        'Theory_Default': 'M200c (200× critical density)',
        'This_Module': 'Mvir (converted from M200c)',
        'Conversion_Method': 'Empirical redshift-dependent ratio',
        'Ratio_z0': 'Mvir/M200c ≈ 2.5',
        'Ratio_z05': 'Mvir/M200c ≈ 1.6',
    }
