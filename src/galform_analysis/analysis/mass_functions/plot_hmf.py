"""HMF convergence testing utilities."""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
import random
from matplotlib.patches import Patch

from ...config import DEFAULT_HALO_MASS_BINS
from .hmf import avg_hmf_given_redshift_and_subvolumes, hmf_given_redshift_and_subvolume

def plot_hmf_convergence_by_subvolumes(
    base_dir,
    df_completed: Optional[pd.DataFrame],
    iz_snapshots: List[str], 
    n_subvolumes: Optional[List[int]] = None,
    n_iterations: int = 1,
    bins: Optional[np.ndarray] = None,
    outdir: str = '_plots/convergence',
    do_save: bool = False,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    panel_size: tuple = (7, 5)
) -> Dict[str, List[Dict[str, Any]]]:
    """Plot HMF convergence with varying subvolume sample sizes.

    Args:
        base_dir: Base directory containing snapshot subdirectories
        df_completed: DataFrame with completed galaxy files (from completed_galaxies())
                     If provided, only completed subvolumes will be sampled
        iz_snapshots: List of snapshot numbers (e.g., [82, 100, 120, 155])
        n_subvolumes: List of subvolume counts to test
        n_iterations: Number of random iterations per subvolume sample size
        bins: log10(M_halo) bin edges (default from config)
        outdir: Output directory for figure and data
        do_save: Save figure and CSVs if True
        xlim: Tuple (xmin,xmax) for x-axis limits
        ylim: Tuple (ymin,ymax) for y-axis limits (log scale)
        panel_size: (width, height) for each subplot panel

    Returns:
        Dict mapping panel label to list of HMF result dicts.
    """
    if bins is None:
        bins = DEFAULT_HALO_MASS_BINS
    
    os.makedirs(outdir, exist_ok=True)
    data_dir = '_plots/_plots_data/convergence'
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Testing convergence with {len(n_subvolumes)} sample sizes: {n_subvolumes}")
    print(f"Averaging over {n_iterations} iteration(s) per sample size")
    
    results_by_panel = {}
    
    for n in n_subvolumes:
        print(f"\n=== Computing with n={n} subvolume(s) ===")
        hmfs = []
        
        for iz_num in iz_snapshots:
            # Get completed subvolumes for this redshift
            if df_completed is not None:
                iz_name = f'iz{iz_num}'
                iz_completed = df_completed[(df_completed['iz'] == iz_name) & (df_completed['completed'])]
                available_ivols = sorted(iz_completed['ivol'].unique())
            else:
                # Fallback: scan for available subvolumes
                print("failed to parse completed df")
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
            iteration_hmfs = []
            for iteration in range(n_iterations):
                # Randomly sample n subvolumes
                sampled_ivols = random.sample(available_ivols, n)
                hmf = avg_hmf_given_redshift_and_subvolumes(
                    iz_num=iz_num, 
                    ivols=sampled_ivols, 
                    bins=bins, 
                    base_dir=str(base_dir)
                )
                
                if hmf:
                    iteration_hmfs.append(hmf)
            
            if not iteration_hmfs:
                print("no data")
                continue
            
            # Average over iterations
            if n_iterations > 1:
                # Average phi values across iterations
                phi_avg = np.mean([h['phi'] for h in iteration_hmfs], axis=0)
                phi_std = np.std([h['phi'] for h in iteration_hmfs], axis=0)
                
                averaged_hmf = {
                    'iz': f'iz{iz_num}',
                    'z': iteration_hmfs[0].get('z'),
                    'centers': iteration_hmfs[0]['centers'],
                    'phi': phi_avg,
                    'phi_std': phi_std,
                    'n_used': n,
                    'n_iterations': n_iterations
                }
            else:
                # Single iteration, just use that result
                averaged_hmf = iteration_hmfs[0]
                averaged_hmf['n_iterations'] = 1
            
            hmfs.append(averaged_hmf)
            print(f"done ({n} ivols × {n_iterations} iterations)")
        
        results_by_panel[str(n)] = hmfs
    
    # Create grid of subplots; scale by panel_size
    n_plots = len(results_by_panel)
    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size[0]*ncols, panel_size[1]*nrows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    cmap = plt.colormaps['viridis']
    
    for idx, (panel_label, hmfs) in enumerate(results_by_panel.items()):
        ax: plt.Axes = axes[idx]
        
        if not hmfs:
            ax.text(0.5, 0.5, f'No data for {panel_label}', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(panel_label, fontsize=14)
            continue
        
        any_shaded = False
        for i, h in enumerate(hmfs):
            color = cmap(i / (len(hmfs) - 1 if len(hmfs) > 1 else 1))
            if h['z'] is not None and not np.isnan(h['z']):
                label = f"z={h['z']:.2f}"
            else:
                label = h['iz']
            
            # Plot smooth line with markers (no step style)
            ax.plot(
                h['centers'],
                h['phi'],
                color=color,
                lw=2,
                marker='o',
                markersize=3,
                label=label,
                alpha=0.9,
            )
            
            # Show uncertainty if available and n > 1
            if 'phi_std' in h and h['phi_std'] is not None and h['n_used'] > 1:
                ax.fill_between(
                    h['centers'],
                    np.maximum(h['phi'] - h['phi_std'], 1e-10),
                    h['phi'] + h['phi_std'],
                    color=color,
                    alpha=0.15,
                    linewidth=0,
                )
                any_shaded = True
        
        ax.set_yscale('log')
        if ylim:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(bottom=1e-5)
        if xlim:
            ax.set_xlim(*xlim)
        else:
            ax.set_xlim(left=7)
        ax.set_ylabel(r'$\Phi$ [Mpc$^{-3}$ dex$^{-1}$]', fontsize=11)
        ax.set_xlabel(r'$\log_{10}(M_{\rm halo}/M_\odot)$', fontsize=11)
        ax.set_title(panel_label, fontsize=14)
        ax.grid(True, which='both', alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if any_shaded:
            sigma_patch = Patch(facecolor='gray', edgecolor='none', alpha=0.15, label='±1σ')
            handles.append(sigma_patch)
            labels.append('±1σ')
        ax.legend(handles, labels, fontsize=8, ncol=1, loc='best')
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle('HMF Convergence with Increasing Subvolume Sample Size', 
                fontsize=16, y=0.995)
    plt.tight_layout()
    
    if do_save:
        fp = os.path.join(outdir, 'hmf_convergence.png')
        plt.savefig(fp, dpi=150, bbox_inches='tight')
        print(f"\nSaved convergence plot to {fp}")
        # Save plotted data as CSV
        for panel_label, hmfs in results_by_panel.items():
            for h in hmfs:
                snap_label = f"z{h['z']:.2f}" if h['z'] is not None and not np.isnan(h['z']) else h['iz']
                df = np.stack([h['centers'], h['phi'], h.get('phi_std', np.full_like(h['phi'], np.nan))], axis=1)
                header = 'logM,phi,phi_std'
                safe_panel = panel_label.replace(',', '_').replace(' ', '_')
                fname = f"hmf_convergence_{safe_panel}_{snap_label}.csv"
                np.savetxt(os.path.join(data_dir, fname), df, delimiter=',', header=header, comments='')
        print(f"Saved HMF data to {data_dir}")
    
    plt.show()
    return results_by_panel


def plot_hmf_convergence_by_redshift(
    base_dir,
    df_completed: Optional[pd.DataFrame],
    iz_snapshots: List[int],
    n_subvolumes: Optional[List[int]] = None,
    n_iterations: int = 1,
    bins: Optional[np.ndarray] = None,
    outdir: str = '_plots/convergence',
    do_save: bool = False,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    panel_size: tuple = (7, 5)
) -> Dict[str, Dict[str, Any]]:
    """Plot HMF convergence organized by redshift.

    Each panel shows one redshift with multiple lines for different
    subvolume counts, illustrating convergence as more subvolumes are
    averaged.

    Args:
        base_dir: Base directory containing snapshot subdirectories
        df_completed: DataFrame with completed galaxy files
        iz_snapshots: List of snapshot numbers (e.g., [82, 100, 120, 155])
        n_subvolumes: Subvolume counts to test per snapshot panel
        n_iterations: Number of random iterations per subvolume sample size
        bins: log10(M_halo) bin edges (default from config)
        outdir: Directory to save figure
        do_save: Whether to save figure and CSVs
        xlim: x-axis limits
        ylim: y-axis limits (log scale)
        panel_size: (width, height) for each subplot panel

    Returns:
        Dict keyed by redshift label with per-n sample HMF results
    """
    if bins is None:
        bins = DEFAULT_HALO_MASS_BINS
    
    if n_subvolumes is None:
        n_subvolumes = [1, 2, 5, 10]

    os.makedirs(outdir, exist_ok=True)
    data_dir = '_plots/_plots_data/convergence_by_redshift'
    os.makedirs(data_dir, exist_ok=True)

    sorted_snapshots = sorted(iz_snapshots)
    results_by_z: Dict[str, Dict[str, Any]] = {}

    print(f"Computing convergence by redshift with n_subvolumes={n_subvolumes}")
    
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
            iteration_hmfs = []
            for iteration in range(n_iterations):
                sampled_ivols = random.sample(available_ivols, n)
                hmf = avg_hmf_given_redshift_and_subvolumes(
                    iz_num=iz_num,
                    ivols=sampled_ivols,
                    bins=bins,
                    base_dir=str(base_dir)
                )
                if hmf:
                    iteration_hmfs.append(hmf)
            
            if not iteration_hmfs:
                print(f"  n={n}: no data")
                continue
            
            # Average over iterations
            if n_iterations > 1:
                phi_avg = np.mean([h['phi'] for h in iteration_hmfs], axis=0)
                phi_std = np.std([h['phi'] for h in iteration_hmfs], axis=0)
                averaged_hmf = {
                    'iz': f'iz{iz_num}',
                    'z': iteration_hmfs[0].get('z'),
                    'centers': iteration_hmfs[0]['centers'],
                    'phi': phi_avg,
                    'phi_std': phi_std,
                    'n_used': n,
                    'n_iterations': n_iterations
                }
            else:
                averaged_hmf = iteration_hmfs[0]
                averaged_hmf['n_iterations'] = 1
            
            per_n_results[n] = averaged_hmf
            print(f"  n={n}: done ({averaged_hmf['n_used']} ivols × {n_iterations} iterations)")

        z_val = per_n_results[list(per_n_results.keys())[0]]['z'] if per_n_results else None
        z_label = f"z={z_val:.2f}" if z_val is not None and not np.isnan(z_val) else f'iz{iz_num}'
        
        results_by_z[z_label] = {
            'z': z_val,
            'iz': f'iz{iz_num}',
            'hmfs': per_n_results
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

    cmap = plt.colormaps['viridis']

    for idx, (z_label, data) in enumerate(results_by_z.items()):
        ax: plt.Axes = axes[idx]
        hmfs_dict = data['hmfs']

        if not hmfs_dict:
            ax.text(0.5, 0.5, f'No data for {z_label}', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(z_label, fontsize=14)
            continue

        any_shaded = False
        for i, key in enumerate(sorted(hmfs_dict.keys(), key=lambda k: str(k))):
            h = hmfs_dict[key]
            color = cmap(i / (len(hmfs_dict) - 1 if len(hmfs_dict) > 1 else 1))
            label = f'n={key}' if isinstance(key, (int, float)) else str(key)
            ax.plot(
                h['centers'],
                h['phi'],
                color=color,
                lw=2,
                marker='o',
                markersize=3,
                label=label,
                alpha=0.9,
            )
            # Use n_used to decide if uncertainty shading is meaningful; avoids undefined variable 'n'
            if 'phi_std' in h and h['phi_std'] is not None and h.get('n_used', 1) > 1:
                ax.fill_between(
                    h['centers'],
                    np.maximum(h['phi'] - h['phi_std'], 1e-10),
                    h['phi'] + h['phi_std'],
                    color=color,
                    alpha=0.18,
                    linewidth=0,
                )
                any_shaded = True

        ax.set_yscale('log')
        if ylim:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(bottom=1e-5)
        if xlim:
            ax.set_xlim(*xlim)
        else:
            ax.set_xlim(left=bins.min(), right=bins.max())
        ax.set_xlabel(r'$\log_{10}(M_{\rm halo}/M_\odot)$', fontsize=11)
        ax.set_ylabel(r'$\Phi$ [Mpc$^{-3}$ dex$^{-1}$]', fontsize=11)
        ax.set_title(z_label, fontsize=14)
        ax.grid(True, which='both', alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if any_shaded:
            sigma_patch = Patch(facecolor='gray', edgecolor='none', alpha=0.18, label='±1σ')
            handles.append(sigma_patch)
            labels.append('±1σ')
        ax.legend(handles, labels, fontsize=8, loc='best')

    # Hide any unused axes
    for j in range(n_panels, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('HMF Convergence by Redshift', fontsize=16, y=0.995)
    plt.tight_layout()

    if do_save:
        fp = os.path.join(outdir, 'hmf_convergence_by_redshift.png')
        plt.savefig(fp, dpi=150, bbox_inches='tight')
        print(f"\nSaved convergence-by-redshift plot to {fp}")
        # Save per-redshift data
        for z_label, data in results_by_z.items():
            for key, h in data['hmfs'].items():
                df = np.stack([h['centers'], h['phi'], h.get('phi_std', np.full_like(h['phi'], np.nan))], axis=1)
                header = 'logM,phi,phi_std'
                safe_label = z_label.replace('=', '').replace('.', 'p')
                key_str = str(key).replace(',', '_').replace(' ', '_').replace('=', '')
                fname = f"hmf_by_z_{safe_label}_{key_str}.csv"
                np.savetxt(os.path.join(data_dir, fname), df, delimiter=',', header=header, comments='')
        print(f"Saved per-redshift HMF data to {data_dir}")

    plt.show()
    return results_by_z


def plot_single_hmf_given_redshift_and_subvolume(
    base_dir,
    snapshot: str,
    ivol: int,
    *,
    halo_mass_lower_limit: Optional[float] = None,
    show_plot: bool = True,
    do_save: bool = False,
    save_path: Optional[str] = None,
):
    """
    Plot the HMF for a given snapshot (e.g., 'iz100') and subvolume.

    Parameters:
    - base_dir: Path-like base directory for data
    - snapshot: Snapshot folder name, e.g., 'iz100'
    - ivol: Subvolume index
    - halo_mass_lower_limit: Optional lower halo mass cut
    - show_plot: If True, display the figure
    - do_save: If True, save figure to save_path
    - save_path: Path to save figure when do_save=True

    Returns (fig, ax) or (None, None) if no data.
    """
    iz_path = os.path.join(str(base_dir), snapshot)
    result = hmf_given_redshift_and_subvolume(
        iz_path,
        ivol,
        halo_mass_lower_limit=halo_mass_lower_limit,
    )

    if result is None:
        return None, None

    z_val = result.get('z')
    centers = result['centers']
    phi = result['phi']
    counts = result['counts']
    V_ivol = result.get('V_ivol')

    valid = np.isfinite(phi) & (phi > 0)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_yscale('log')
    ax.plot(centers[valid], phi[valid], 'o-', label=f'HMF ivol {ivol}')
    ax.set_xlabel(r'$\log_{10}(M_{\rm halo} \, [10^{10} h^{-1} M_\odot])$')
    ax.set_ylabel(r'$\Phi$ [Mpc$^{-3}$ dex$^{-1}$]')
    title = f"HMF: {snapshot} ivol {ivol}"
    ax.set_title(title)
    ax.legend()

    if do_save and save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    # Don't call plt.show() - let the caller handle display
    # In notebooks, the returned figure will be automatically displayed
    # In scripts, the caller can call plt.show() if needed

    return fig, ax


def plot_hmf_with_theory(
    base_dir,
    snapshot: str,
    ivol: int,
    *,
    halo_mass_lower_limit: Optional[float] = None,
    theory_bins: Optional[np.ndarray] = None,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    do_save: bool = False,
    save_path: Optional[str] = None,
):
    """
    Plot the HMF for a given snapshot and subvolume with theoretical model comparisons.

    Parameters:
    - base_dir: Path-like base directory for data
    - snapshot: Snapshot folder name, e.g., 'iz100'
    - ivol: Subvolume index
    - halo_mass_lower_limit: Optional lower halo mass cut
    - theory_bins: Bin edges for theoretical HMFs (default: np.arange(9.0, 15.0, 0.1))
    - xlim: Tuple (xmin, xmax) for x-axis limits
    - ylim: Tuple (ymin, ymax) for y-axis limits
    - show_plot: If True, display the figure
    - do_save: If True, save figure to save_path
    - save_path: Path to save figure when do_save=True

    Returns (fig, ax) or (None, None) if no data.
    """
    from .theoretical_hmf import compute_theoretical_hmfs
    
    iz_path = os.path.join(str(base_dir), snapshot)
    result = hmf_given_redshift_and_subvolume(
        iz_path,
        ivol,
        halo_mass_lower_limit=halo_mass_lower_limit,
    )

    if result is None:
        return None, None

    z_val = result.get('z')
    centers = result['centers']
    phi = result['phi']
    counts = result['counts']
    V_ivol = result.get('V_ivol')
    
    # Compute theoretical HMFs
    if theory_bins is None:
        theory_bins = np.arange(9.0, 15.0, 0.1)
    
    theoretical = compute_theoretical_hmfs(
        z=z_val,
        bins=theory_bins,
        use_mvir=True
    )
    
    # Compute bin centers for theory curves
    log10M_theory = (theory_bins[:-1] + theory_bins[1:]) / 2
    phi_ps = theoretical['PS']
    phi_smt = theoretical['SMT']
    phi_tinker = theoretical['Tinker08']
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # GALFORM data (mask out non-positive values)
    valid = np.isfinite(phi) & (phi > 0)
    ax.plot(centers[valid], phi[valid], 'o-', label=f'GALFORM {snapshot} ivol{ivol}', 
            linewidth=2, markersize=5, color='black', zorder=10)
    
    # Theory curves (mask non-positive values independently)
    valid_tinker = np.isfinite(phi_tinker) & (phi_tinker > 0)
    ax.plot(log10M_theory[valid_tinker], phi_tinker[valid_tinker], '--', label='Tinker+08', linewidth=2, alpha=0.7)
    valid_ps = np.isfinite(phi_ps) & (phi_ps > 0)
    ax.plot(log10M_theory[valid_ps], phi_ps[valid_ps], '--', label='Press-Schechter', linewidth=2, alpha=0.7)
    valid_smt = np.isfinite(phi_smt) & (phi_smt > 0)
    ax.plot(log10M_theory[valid_smt], phi_smt[valid_smt], '--', label='Sheth-Mo-Tormen', linewidth=2, alpha=0.7, color='green')
    
    ax.set_yscale('log')
    if xlim:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(10, 14)
    if ylim:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(1e-5, 1e-1)
    
    ax.set_xlabel(r'$\log_{10}(M_{\rm halo} \, [10^{10} h^{-1} M_\odot])$', fontsize=13)
    ax.set_ylabel(r'$\Phi$ [Mpc$^{-3}$ dex$^{-1}$]', fontsize=13)
    
    title = f"Halo Mass Function: {snapshot} ivol {ivol}"
    if z_val is not None:
        title += f" (z={z_val:.2f})"
    ax.set_title(title, fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    if do_save and save_path:
        plt.savefig(save_path, bbox_inches='tight')
        
    return fig, ax


def plot_hmf_multiple_redshifts(
    base_dir,
    ivol: int,
    iz_nums: list,
    *,
    halo_mass_lower_limit: Optional[float] = None,
    figsize: tuple = (8, 6),
    show_plot: bool = True,
    do_save: bool = False,
    save_path: Optional[str] = None,
):
    """
    Plot HMFs for multiple redshifts on the same axes.

    Parameters:
    - base_dir: Path-like base directory for data
    - ivol: Subvolume index
    - iz_nums: List of snapshot indices (e.g., [82, 100, 120])
    - halo_mass_lower_limit: Optional lower halo mass cut
    - figsize: Figure size tuple (width, height)
    - show_plot: If True, display the figure
    - do_save: If True, save figure to save_path
    - save_path: Path to save figure when do_save=True

    Returns (fig, ax, df, results_by_z) or (None, None, None, None) if no data.
    """
    from .hmf import hmfs_given_redshifts_and_subvolume
    
    # Collect results for each redshift
    df, results_by_z = hmfs_given_redshifts_and_subvolume(
        ivol,
        iz_nums,
        base_dir=base_dir,
        halo_mass_lower_limit=halo_mass_lower_limit,
    )
    
    if not results_by_z:
        return None, None, None, None
    
    # Plot each redshift separately on same axes
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_by_z)))
    
    for res, color in zip(results_by_z, colors):
        label = f"z={res['z']:.2f} ({res['iz']})"
        centers = res['centers']
        phi = res['phi']
        valid = np.isfinite(phi) & (phi > 0)
        ax.plot(centers[valid], phi[valid], 'o-', label=label, color=color, markersize=4)
    
    ax.set_yscale('log')
    ax.set_xlabel(r'$\log_{10}(M_{\rm halo} \, [10^{10} h^{-1} M_\odot])$', fontsize=13)
    ax.set_ylabel(r'$\Phi$ [Mpc$^{-3}$ dex$^{-1}$]', fontsize=13)
    ax.set_title(f'Halo Mass Function for ivol {ivol} at Different Redshifts', fontsize=15)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if do_save and save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig, ax, df, results_by_z
