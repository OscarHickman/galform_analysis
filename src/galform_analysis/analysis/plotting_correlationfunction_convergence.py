"""Correlation function convergence testing utilities."""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
import random
from matplotlib.patches import Patch

from galform_analysis.config import DEFAULT_RBINS
from .correlation import avg_correlation_given_redshift_and_subvolumes


def plot_correlation_convergence_by_subvolumes(
    base_dir,
    df_completed: Optional[pd.DataFrame],
    iz_snapshots: List[str], 
    n_subvolumes: Optional[List[int]] = None,
    n_iterations: int = 1,
    rbins: Optional[np.ndarray] = None,
    nmesh: int = 128,
    outdir: str = 'plots/correlation',
    do_save: bool = True,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    panel_size: tuple = (7, 5)
) -> Dict[str, List[Dict[str, Any]]]:
    """Plot correlation function convergence with varying subvolume sample sizes.

    Args:
        base_dir: Base directory containing snapshot subdirectories
        df_completed: DataFrame with completed galaxy files (from completed_galaxies())
                     If provided, only completed subvolumes will be sampled
        iz_snapshots: List of snapshot numbers (e.g., ['82', '100', '207'])
        n_subvolumes: List of subvolume counts to test
        n_iterations: Number of random iterations per subvolume sample size
        rbins: Radial bin edges (Mpc). Defaults to DEFAULT_RBINS
        nmesh: Mesh size for FFT grid
        outdir: Output directory for figure and data
        do_save: Save figure and CSVs if True
        xlim: Tuple (xmin,xmax) for x-axis limits
        ylim: Tuple (ymin,ymax) for y-axis limits (log scale)
        panel_size: (width, height) for each subplot panel

    Returns:
        Dict mapping panel label to list of correlation function result dicts.
    """
    if rbins is None:
        rbins = DEFAULT_RBINS
    
    if n_subvolumes is None:
        n_subvolumes = [1, 2, 5, 10, 20]
    
    os.makedirs(outdir, exist_ok=True)
    data_dir = 'plots/_plots_data/correlation'
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Testing correlation function convergence with {len(n_subvolumes)} sample sizes: {n_subvolumes}")
    print(f"Averaging over {n_iterations} iteration(s) per sample size")
    
    results_by_panel = {}
    
    for n in n_subvolumes:
        print(f"\n=== Computing with n={n} subvolume(s) ===")
        correlations = []
        
        for iz_num in iz_snapshots:
            # Get completed subvolumes for this redshift
            if df_completed is not None:
                iz_name = f'iz{iz_num}'
                iz_completed = df_completed[(df_completed['iz'] == iz_name) & (df_completed['completed'])]
                available_ivols = sorted(iz_completed['ivol'].unique())
            else:
                # Fallback: scan for available subvolumes
                iz_path = os.path.join(str(base_dir), f'iz{iz_num}')
                if not os.path.isdir(iz_path):
                    continue
                import glob
                ivol_dirs = glob.glob(os.path.join(iz_path, 'ivol*'))
                available_ivols = [int(Path(d).name.replace('ivol', '')) for d in ivol_dirs]
                available_ivols = sorted(available_ivols)
            
            if len(available_ivols) < n:
                print(f"  iz{iz_num}: skipped (only {len(available_ivols)} available, need {n})")
                continue
            
            print(f"  iz{iz_num}: ", end='')
            
            # Perform n_iterations with random sampling
            iteration_results = []
            for iteration in range(n_iterations):
                # Randomly sample n subvolumes
                sampled_ivols = random.sample(available_ivols, n)
                corr = avg_correlation_given_redshift_and_subvolumes(
                    iz_num=int(iz_num), 
                    ivols=sampled_ivols, 
                    rbins=rbins,
                    nthreads=nmesh,
                    base_dir=str(base_dir)
                )
                
                if corr:
                    iteration_results.append(corr)
            
            if not iteration_results:
                print("no data")
                continue
            
            # Average over iterations
            if n_iterations > 1:
                # Average xi values across iterations
                xi_avg = np.mean([c['xi'] for c in iteration_results], axis=0)
                xi_std = np.std([c['xi'] for c in iteration_results], axis=0)
                
                averaged_corr = {
                    'iz': f'iz{iz_num}',
                    'z': iteration_results[0].get('z'),
                    'r': iteration_results[0]['r'],
                    'xi': xi_avg,
                    'xi_std': xi_std,
                    'n_used': n,
                    'n_iterations': n_iterations
                }
            else:
                # Single iteration, just use that result
                averaged_corr = iteration_results[0]
                averaged_corr['n_iterations'] = 1
            
            correlations.append(averaged_corr)
            print(f"done ({n} ivols × {n_iterations} iterations)")
        
        results_by_panel[str(n)] = correlations
    
    # Create grid of subplots; scale by panel_size
    n_plots = len(results_by_panel)
    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size[0]*ncols, panel_size[1]*nrows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    cmap = plt.get_cmap('viridis')
    
    for idx, (panel_label, correlations) in enumerate(results_by_panel.items()):
        ax: plt.Axes = axes[idx]
        
        if not correlations:
            ax.text(0.5, 0.5, f'No data for {panel_label}', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(panel_label, fontsize=14)
            continue
        
        any_shaded = False
        for i, c in enumerate(correlations):
            color = cmap(i / (len(correlations) - 1 if len(correlations) > 1 else 1))
            if c['z'] is not None and not np.isnan(c['z']):
                label = f"z={c['z']:.2f}"
            else:
                label = c['iz']
            
            # Plot smooth line with markers
            ax.plot(
                c['r'],
                c['xi'],
                color=color,
                lw=2,
                marker='o',
                markersize=3,
                label=label,
                alpha=0.9,
            )
            
            # Show uncertainty if available and n > 1 (standard error of the mean)
            if 'xi_std' in c and c['xi_std'] is not None and c['n_used'] > 1:
                xi_sem = c['xi_std'] / np.sqrt(c['n_used'])
                ax.fill_between(
                    c['r'],
                    c['xi'] - xi_sem,
                    c['xi'] + xi_sem,
                    color=color,
                    alpha=0.15,
                    linewidth=0,
                )
                any_shaded = True
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        if ylim:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(bottom=0.01)
        if xlim:
            ax.set_xlim(*xlim)
        else:
            ax.set_xlim(left=0.1, right=50)
        ax.set_ylabel(r'$\xi(r)$', fontsize=11)
        ax.set_xlabel(r'$r$ [Mpc]', fontsize=11)
        ax.set_title(f'n={panel_label}', fontsize=14)
        ax.grid(False)
        handles, labels = ax.get_legend_handles_labels()
        if any_shaded:
            sigma_patch = Patch(facecolor='gray', edgecolor='none', alpha=0.15, label='±1 SEM')
            handles.append(sigma_patch)
            labels.append('±1 SEM')
        ax.legend(handles, labels, fontsize=8, ncol=1, loc='best')
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle('Correlation Function Convergence with Increasing Subvolume Sample Size', 
                fontsize=16, y=0.995)
    plt.tight_layout()
    
    if do_save:
        fp = os.path.join(outdir, 'correlation_convergence.png')
        plt.savefig(fp, dpi=150, bbox_inches='tight')
        print(f"\nSaved convergence plot to {fp}")
        # Save plotted data as CSV
        for panel_label, correlations in results_by_panel.items():
            for c in correlations:
                snap_label = f"z{c['z']:.2f}" if c['z'] is not None and not np.isnan(c['z']) else c['iz']
                df = np.stack([c['r'], c['xi'], c.get('xi_std', np.full_like(c['xi'], np.nan))], axis=1)
                header = 'r,xi,xi_std'
                safe_panel = panel_label.replace(',', '_').replace(' ', '_')
                fname = f"correlation_convergence_{safe_panel}_{snap_label}.csv"
                np.savetxt(os.path.join(data_dir, fname), df, delimiter=',', header=header, comments='')
        print(f"Saved correlation data to {data_dir}")
    
    plt.show()
    return results_by_panel


def plot_correlation_convergence_by_redshift(
    base_dir,
    df_completed: Optional[pd.DataFrame],
    iz_snapshots: List[int],
    n_subvolumes: Optional[List[int]] = None,
    n_iterations: int = 1,
    rbins: Optional[np.ndarray] = None,
    nmesh: int = 128,
    outdir: str = 'plots/correlation',
    do_save: bool = True,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    panel_size: tuple = (7, 5)
) -> Dict[str, Dict[str, Any]]:
    """Plot correlation function convergence organized by redshift.

    Each panel shows one redshift with multiple lines for different
    subvolume counts, illustrating convergence as more subvolumes are
    averaged.

    Args:
        base_dir: Base directory containing snapshot subdirectories
        df_completed: DataFrame with completed galaxy files
        iz_snapshots: List of snapshot numbers (e.g., [82, 100, 207])
        n_subvolumes: Subvolume counts to test per snapshot panel
        n_iterations: Number of random iterations per subvolume sample size
        rbins: Radial bin edges (Mpc). Defaults to DEFAULT_RBINS
        nmesh: Mesh size for FFT grid
        outdir: Directory to save figure
        do_save: Whether to save figure and CSVs
        xlim: x-axis limits
        ylim: y-axis limits (log scale)
        panel_size: (width, height) for each subplot panel

    Returns:
        Dict keyed by redshift label with per-n sample correlation function results
    """
    if rbins is None:
        rbins = DEFAULT_RBINS
    
    if n_subvolumes is None:
        n_subvolumes = [1, 2, 5, 10]

    os.makedirs(outdir, exist_ok=True)
    data_dir = 'plots/_plots_data/correlation_by_redshift'
    os.makedirs(data_dir, exist_ok=True)

    sorted_snapshots = sorted(iz_snapshots)
    results_by_z: Dict[str, Dict[str, Any]] = {}

    print(f"Computing correlation convergence by redshift with n_subvolumes={n_subvolumes}")
    
    for iz_num in sorted_snapshots:
        # Get completed subvolumes for this redshift
        if df_completed is not None:
            iz_name = f'iz{iz_num}'
            iz_completed = df_completed[(df_completed['iz'] == iz_name) & (df_completed['completed'])]
            available_ivols = sorted(iz_completed['ivol'].unique())
        else:
            # Fallback: scan for available subvolumes
            iz_path = os.path.join(str(base_dir), f'iz{iz_num}')
            if not os.path.isdir(iz_path):
                continue
            import glob
            ivol_dirs = glob.glob(os.path.join(iz_path, 'ivol*'))
            available_ivols = [int(Path(d).name.replace('ivol', '')) for d in ivol_dirs]
            available_ivols = sorted(available_ivols)
        
        if not available_ivols:
            print(f"\niz{iz_num}: no available subvolumes")
            continue
            
        print(f"\nProcessing iz{iz_num} ({len(available_ivols)} available subvolumes)...")

        per_n_results: Dict[int, Dict[str, Any]] = {}
        
        for n in n_subvolumes:
            if len(available_ivols) < n:
                print(f"  n={n}: skipped (only {len(available_ivols)} available)")
                continue
                
            # Average over iterations
            iteration_results = []
            for iteration in range(n_iterations):
                sampled_ivols = random.sample(available_ivols, n)
                corr = avg_correlation_given_redshift_and_subvolumes(
                    iz_num=iz_num,
                    ivols=sampled_ivols,
                    rbins=rbins,
                    nthreads=nmesh,
                    base_dir=str(base_dir)
                )
                if corr:
                    iteration_results.append(corr)
            
            if not iteration_results:
                print(f"  n={n}: no data")
                continue
            
            # Average over iterations
            if n_iterations > 1:
                xi_avg = np.mean([c['xi'] for c in iteration_results], axis=0)
                xi_std = np.std([c['xi'] for c in iteration_results], axis=0)
                averaged_corr = {
                    'iz': f'iz{iz_num}',
                    'z': iteration_results[0].get('z'),
                    'r': iteration_results[0]['r'],
                    'xi': xi_avg,
                    'xi_std': xi_std,
                    'n_used': n,
                    'n_iterations': n_iterations
                }
            else:
                averaged_corr = iteration_results[0]
                averaged_corr['n_iterations'] = 1
            
            per_n_results[n] = averaged_corr
            print(f"  n={n}: done ({averaged_corr['n_used']} ivols × {n_iterations} iterations)")

        z_val = per_n_results[list(per_n_results.keys())[0]]['z'] if per_n_results else None
        z_label = f"z={z_val:.2f}" if z_val is not None and not np.isnan(z_val) else f'iz{iz_num}'
        
        results_by_z[z_label] = {
            'z': z_val,
            'iz': f'iz{iz_num}',
            'correlations': per_n_results
        }

    # Layout: one panel per redshift
    n_panels = len(results_by_z)
    if n_panels == 0:
        print('No redshift panels produced any data.')
        return results_by_z

    ncols = min(3, n_panels)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows))
    if n_panels == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    cmap = plt.get_cmap('viridis')

    for idx, (z_label, data) in enumerate(results_by_z.items()):
        ax: plt.Axes = axes[idx]
        corr_dict = data['correlations']

        if not corr_dict:
            ax.text(0.5, 0.5, f'No data for {z_label}', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(z_label, fontsize=14)
            continue

        any_shaded = False
        for i, key in enumerate(sorted(corr_dict.keys(), key=lambda k: str(k))):
            c = corr_dict[key]
            color = cmap(i / (len(corr_dict) - 1 if len(corr_dict) > 1 else 1))
            label = f'n={key}' if isinstance(key, (int, float)) else str(key)
            ax.plot(
                c['r'],
                c['xi'],
                color=color,
                lw=2,
                marker='o',
                markersize=3,
                label=label,
                alpha=0.9,
            )
            # Use n_used to decide if uncertainty shading is meaningful (standard error of the mean)
            if 'xi_std' in c and c['xi_std'] is not None and c.get('n_used', 1) > 1:
                xi_sem = c['xi_std'] / np.sqrt(c.get('n_used', 1))
                ax.fill_between(
                    c['r'],
                    c['xi'] - xi_sem,
                    c['xi'] + xi_sem,
                    color=color,
                    alpha=0.18,
                    linewidth=0,
                )
                any_shaded = True

        ax.set_xscale('log')
        ax.set_yscale('log')
        if ylim:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(bottom=0.01)
        if xlim:
            ax.set_xlim(*xlim)
        else:
            ax.set_xlim(left=0.1, right=50)
        ax.set_xlabel(r'$r$ [Mpc]', fontsize=11)
        ax.set_ylabel(r'$\xi(r)$', fontsize=11)
        ax.set_title(z_label, fontsize=14)
        handles, labels = ax.get_legend_handles_labels()
        if any_shaded:
            sigma_patch = Patch(facecolor='gray', edgecolor='none', alpha=0.18, label='±1 SEM')
            handles.append(sigma_patch)
            labels.append('±1 SEM')
        ax.legend(handles, labels, fontsize=8, loc='best')
        ax.legend(handles, labels, fontsize=8, loc='best')

    # Hide any unused axes
    for j in range(n_panels, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Correlation Function Convergence by Redshift', fontsize=16, y=0.995)
    plt.tight_layout()

    if do_save:
        fp = os.path.join(outdir, 'correlation_convergence_by_redshift.png')
        plt.savefig(fp, dpi=150, bbox_inches='tight')
        print(f"\nSaved convergence-by-redshift plot to {fp}")
        # Save per-redshift data
        for z_label, data in results_by_z.items():
            for key, c in data['correlations'].items():
                df = np.stack([c['r'], c['xi'], c.get('xi_std', np.full_like(c['xi'], np.nan))], axis=1)
                header = 'r,xi,xi_std'
                safe_label = z_label.replace('=', '').replace('.', 'p')
                key_str = str(key).replace(',', '_').replace(' ', '_').replace('=', '')
                fname = f"correlation_by_z_{safe_label}_{key_str}.csv"
                np.savetxt(os.path.join(data_dir, fname), df, delimiter=',', header=header, comments='')
        print(f"Saved per-redshift correlation data to {data_dir}")

    plt.show()
    return results_by_z
