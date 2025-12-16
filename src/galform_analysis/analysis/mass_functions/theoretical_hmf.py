"""Theoretical halo mass function calculations with configurable mass definitions.

This module provides utilities for computing theoretical HMF predictions using
the hmf library while maintaining consistency with GALFORM's mass definition (Mvir).
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
                               use_mvir: bool = True) -> Dict[str, np.ndarray]:
    """
    Compute multiple theoretical HMF models at a given redshift.
    
    Args:
        z: Redshift
        bins: Mass bin edges in log10(M/M_sun)
        use_mvir: If True, convert all to Mvir definition (GALFORM-compatible)
        
    Returns:
        Dictionary with keys for each model:
            - 'PS' (Press-Schechter)
            - 'SMT' (Sheth-Tormen)
            - 'Tinker08'
            
        Each value is an array of dN/dlog10m at bin centers
    """
    models = {}
    
    for model_name in ['PS', 'SMT', 'Tinker08']:
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
    
    return models


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
