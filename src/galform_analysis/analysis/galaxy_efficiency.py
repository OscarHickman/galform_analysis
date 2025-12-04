"""
Galaxy formation efficiency analysis.

This module provides functions to compute and visualize galaxy formation
efficiency (stellar mass / halo mass) as a function of halo mass across
different redshifts.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import os

from ..config import get_base_dir, get_snapshot_redshift
from .aggregation import aggregate_snapshot


def compute_efficiency_vs_mass(agg_data: Dict, mass_bins: np.ndarray) -> Optional[Dict]:
    """
    Compute median efficiency in halo mass bins.
    
    Parameters
    ----------
    agg_data : dict
        Aggregated galaxy data with 'mstar' and 'mhalo' arrays
    mass_bins : np.ndarray
        Bin edges for log10(halo mass)
        
    Returns
    -------
    dict or None
        Dictionary with efficiency statistics:
        - 'centers': bin centers
        - 'eta_med': median efficiency
        - 'eta_p16': 16th percentile
        - 'eta_p84': 84th percentile
        - 'z': redshift
    """
    mstar = agg_data['mstar']
    mhalo = agg_data['mhalo']
    
    # Filter for valid galaxies
    sel = (mstar > 0) & (mhalo > 0) & np.isfinite(mstar) & np.isfinite(mhalo)
    mstar, mhalo = mstar[sel], mhalo[sel]
    
    if len(mstar) == 0:
        return None
    
    # Compute efficiency
    eta = mstar / mhalo
    logMh = np.log10(mhalo)
    
    # Bin and compute statistics
    centers = 0.5 * (mass_bins[1:] + mass_bins[:-1])
    eta_med = np.full_like(centers, np.nan)
    eta_p16 = np.full_like(centers, np.nan)
    eta_p84 = np.full_like(centers, np.nan)
    
    for i in range(len(centers)):
        mask = (logMh >= mass_bins[i]) & (logMh < mass_bins[i+1])
        if np.any(mask):
            vals = eta[mask]
            eta_med[i] = np.median(vals)
            eta_p16[i] = np.percentile(vals, 16)
            eta_p84[i] = np.percentile(vals, 84)
    
    return {
        'centers': centers,
        'eta_med': eta_med,
        'eta_p16': eta_p16,
        'eta_p84': eta_p84,
        'z': agg_data.get('z')
    }


def process_efficiency_redshifts(
    redshifts: List[str],
    mass_bins: np.ndarray,
    base_dir: Optional[Path] = None,
    verbose: bool = True
) -> List[Dict]:
    """
    Process multiple redshifts and compute efficiency for each.
    
    Parameters
    ----------
    redshifts : list of str
        List of redshift names (e.g., ['iz82', 'iz100'])
    mass_bins : np.ndarray
        Bin edges for log10(halo mass)
    base_dir : Path, optional
        Base directory containing redshift data. If None, uses get_base_dir()
    verbose : bool, default True
        Whether to print progress messages
        
    Returns
    -------
    list of dict
        List of efficiency results for each redshift
    """
    if base_dir is None:
        base_dir = get_base_dir()
    
    results = []
    
    for redshift in redshifts:
        iz_path = base_dir / redshift
        
        if not iz_path.exists():
            if verbose:
                print(f"Skipping {redshift} - path not found")
            continue
        
        z = get_snapshot_redshift(redshift)
        label = f"z={z:.2f}" if z is not None else redshift
        
        if verbose:
            print(f"\n Processing {redshift} ({label})...")
        
        # Aggregate data from all subvolumes
        agg = aggregate_snapshot(str(iz_path))
        if agg is None:
            if verbose:
                print("No data found")
            continue
        
        if verbose:
            print(f"Loaded {len(agg['mstar'])} galaxies at z≈{agg.get('z', '?'):.2f}")
        
        # Compute efficiency
        eff = compute_efficiency_vs_mass(agg, mass_bins)
        if eff is None:
            if verbose:
                print("Could not compute efficiency")
            continue
        
        results.append(eff)
        
        if verbose:
            print(f"Efficiency computed for {np.sum(np.isfinite(eff['eta_med']))} mass bins")
    
    if verbose:
        print(f"\n Successfully processed {len(results)} redshifts")
    
    return results


def plot_efficiency_vs_mass(
    results: List[Dict],
    redshifts: List[str],
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot galaxy formation efficiency vs halo mass for multiple redshifts.
    
    Parameters
    ----------
    results : list of dict
        List of efficiency results from process_efficiency_redshifts
    redshifts : list of str
        Corresponding redshifts names
    figsize : tuple, default (12, 8)
        Figure size
    save_path : str, optional
        Path to save the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color map for different redshifts
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    for i, (result, redshifts) in enumerate(zip(results, redshifts)):
        if result is None:
            continue
        
        z = result.get('z')
        label = f"z={z:.2f}" if z is not None else redshifts
        
        # Plot median with shaded region for 16-84 percentile range
        valid = np.isfinite(result['eta_med'])
        ax.plot(result['centers'][valid], result['eta_med'][valid],
                color=colors[i], label=label, linewidth=2)
        ax.fill_between(result['centers'][valid],
                        result['eta_p16'][valid],
                        result['eta_p84'][valid],
                        color=colors[i], alpha=0.2)
    
    ax.set_xlabel(r'$\log_{10}(M_{\rm halo} / M_\odot)$', fontsize=14)
    ax.set_ylabel(r'$\eta = M_* / M_{\rm halo}$', fontsize=14)
    ax.set_title('Galaxy Formation Efficiency vs Halo Mass', fontsize=16, fontweight='bold')
    ax.legend(loc='best', frameon=True, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved plot to {save_path}")
    
    return fig


def save_efficiency_data(
    results: List[Dict],
    redshifts: List[str],
    output_dir: str = '_plots/_plots_data/efficiency'
) -> None:
    """
    Save efficiency data to CSV files.
    
    Parameters
    ----------
    results : list of dict
        List of efficiency results from process_efficiency_redshifts
    redshifts : list of str
        Corresponding redshift names
    output_dir : str, default '_plots/_plots_data/efficiency'
        Directory to save CSV files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    saved_count = 0
    for redshift, result in zip(redshifts, results):
        if result is not None:
            df_out = pd.DataFrame({
                'log_Mhalo': result['centers'],
                'eta_median': result['eta_med'],
                'eta_p16': result['eta_p16'],
                'eta_p84': result['eta_p84']
            })
            output_path = os.path.join(output_dir, f'galaxy_efficiency_{redshift}.csv')
            df_out.to_csv(output_path, index=False)
            saved_count += 1
    
    print(f"✓ Saved efficiency data for {saved_count} redshifts to {output_dir}")


def find_peak_efficiency(results: List[Dict], redshifts: List[str]) -> pd.DataFrame:
    """
    Find the halo mass where efficiency peaks for each redshift.
    
    Parameters
    ----------
    results : list of dict
        List of efficiency results from process_efficiency_redshifts
    redshifts : list of str
        Corresponding redshift names
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: redshift, redshift, peak_mass, peak_efficiency
    """
    peak_data = []
    
    for redshift, result in zip(redshifts, results):
        if result is None:
            continue
        
        valid = np.isfinite(result['eta_med'])
        if not np.any(valid):
            continue
        
        peak_idx = np.nanargmax(result['eta_med'])
        peak_mass = result['centers'][peak_idx]
        peak_eff = result['eta_med'][peak_idx]
        z = result.get('z')
        
        peak_data.append({
            'redshift_name': redshift,
            'redshift': z,
            'peak_mass': peak_mass,
            'peak_efficiency': peak_eff
        })
    
    return pd.DataFrame(peak_data)
