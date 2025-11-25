"""Subvolume convergence testing utilities."""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Dict, Any

from .smf import compute_smf_avg_by_snapshot
from .hmf import compute_hmf_avg_by_snapshot


def plot_hmf_convergence(
    iz_paths: List[str],
    redshifts: Optional[List[float]],
    ivol_sets: Optional[List[List[int]]] = None,
    ivol_labels: Optional[List[str]] = None,
    n_samples: Optional[List[int]] = None,
    bins: Optional[np.ndarray] = None,
    outdir: str = 'plots/convergence',
    do_save: bool = True,
    random_seed: int = 42,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    panel_size: tuple = (7, 5)
) -> Dict[str, List[Dict[str, Any]]]:
    """Plot HMF convergence with explicit subvolume selections or sample sizes.

    Modes:
      1. Explicit: Provide `ivol_sets` (list of lists of ivol indices). Each panel
         corresponds to one ivol set.
      2. Legacy: Provide `n_samples` (list of counts). Each panel corresponds to
         a random sample of that many subvolumes.

    Args:
        iz_paths: Snapshot directory paths (iz*)
        redshifts: Matching redshift values (can be None entries)
        ivol_sets: List of lists of ivol indices to use explicitly per panel
        ivol_labels: Labels for panels (same length as ivol_sets)
        n_samples: Legacy mode list of sample sizes (ignored if ivol_sets provided)
        bins: log10(M_halo) bin edges (default np.arange(7.0,15.5,0.2))
        outdir: Output directory for figure and data
        do_save: Save figure and CSVs if True
        random_seed: Seed for sampling (legacy mode)
        xlim: Tuple (xmin,xmax) for x-axis limits
        ylim: Tuple (ymin,ymax) for y-axis limits (log scale)

    Returns:
        Dict mapping panel label to list of HMF result dicts.
    """
    if bins is None:
        bins = np.arange(7.0, 15.5, 0.2)
    
    os.makedirs(outdir, exist_ok=True)
    data_dir = 'plots/_plots_data/convergence'
    os.makedirs(data_dir, exist_ok=True)
    
    # Sort paths by iz number
    try:
        sorted_indices = sorted(range(len(iz_paths)), 
                              key=lambda i: int(Path(iz_paths[i]).name.replace('iz', '')))
        sorted_paths = [iz_paths[i] for i in sorted_indices]
    except Exception:
        sorted_paths = list(iz_paths)
    
    if ivol_sets:
        labels = ivol_labels if ivol_labels and len(ivol_labels) == len(ivol_sets) else [f"ivols={','.join(map(str,s))}" for s in ivol_sets]
        print(f"Testing convergence with explicit ivol sets: {labels}")
        results_by_panel: Dict[str, List[Dict[str, Any]]] = {}
        for label, ivols in zip(labels, ivol_sets):
            print(f"\nComputing with explicit subvolumes {ivols} ({label})...")
            hmfs = []
            for idx, p in enumerate(sorted_paths):
                iz_name = Path(p).name
                print(f"  [{idx+1}/{len(sorted_paths)}] {iz_name}...", end=' ')
                hmf = compute_hmf_avg_by_snapshot(str(p), bins=bins, ivol_list=ivols)
                if hmf:
                    hmfs.append(hmf)
                    print(f"done ({hmf['n_used']} ivols)")
                else:
                    print("no data")
            results_by_panel[label] = hmfs
    else:
        if n_samples is None:
            n_samples = [1, 5, 10, 50, 100]
        print(f"Testing convergence with {len(n_samples)} sample sizes: {n_samples}")
        results_by_panel = {}
        for n in n_samples:
            print(f"\nComputing with {n} subvolume(s)...")
            hmfs = []
            for idx, p in enumerate(sorted_paths):
                iz_name = Path(p).name
                print(f"  [{idx+1}/{len(sorted_paths)}] {iz_name}...", end=' ')
                hmf = compute_hmf_avg_by_snapshot(str(p), bins=bins, ivol_sample=n, random_seed=random_seed)
                if hmf:
                    hmfs.append(hmf)
                    print(f"done ({hmf['n_used']} ivols)")
                else:
                    print("no data")
            results_by_panel[str(n)] = hmfs
    
    # Create grid of subplots; scale by panel_size
    n_plots = len(results_by_panel)
    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size[0]*ncols, panel_size[1]*nrows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    cmap = plt.get_cmap('viridis')
    
    for idx, (panel_label, hmfs) in enumerate(results_by_panel.items()):
        ax = axes[idx]
        
        if not hmfs:
            ax.text(0.5, 0.5, f'No data for {panel_label}', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(panel_label, fontsize=14)
            continue
        
        for i, h in enumerate(hmfs):
            color = cmap(i / (len(hmfs) - 1 if len(hmfs) > 1 else 1))
            if h['z'] is not None and not np.isnan(h['z']):
                label = f"z={h['z']:.2f}"
            else:
                label = h['iz']
            
            # Plot line with steps
            ax.plot(h['centers'], h['phi'], drawstyle='steps-mid', 
                    color=color, lw=2, label=label, alpha=0.8)
            
            # Show uncertainty if available and n > 1
            if 'phi_std' in h and h['phi_std'] is not None and h['n_used'] > 1:
                ax.fill_between(h['centers'], 
                               np.maximum(h['phi'] - h['phi_std'], 1e-10),
                               h['phi'] + h['phi_std'],
                               color=color, alpha=0.15, linewidth=0, step='mid')
        
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
        ax.legend(fontsize=8, ncol=1, loc='best')
    
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


def plot_smf_convergence(
    iz_paths: List[str],
    redshifts: Optional[List[float]] = None,
    n_samples: List[int] = [1, 5, 10, 50, 100],
    bins: np.ndarray = None,
        outdir: str = 'plots/convergence',
    do_save: bool = True,
    random_seed: int = 42,
    panel_size: tuple = (7, 5)
) -> Dict[int, List[Dict[str, Any]]]:
    """Plot SMF convergence as more subvolumes are included.

    Creates separate panels for each n_sample showing how SMF averages
    across redshifts change as more subvolumes are included.
    
    Args:
        iz_paths: List of snapshot directory paths
        redshifts: List of redshifts (optional, for labeling)
        n_samples: List of subvolume counts to test
        bins: log10(M_star) bin edges, defaults to np.arange(8.0, 12.5, 0.2)
        outdir: Directory to save figures
        do_save: Whether to save the figure
        random_seed: Seed for reproducible sampling
        
    Returns:
        Dictionary mapping n_sample to list of SMF results
    """
    if bins is None:
        bins = np.arange(8.0, 12.5, 0.2)
    
    os.makedirs(outdir, exist_ok=True)
    data_dir = 'plots/_plots_data/convergence'
    os.makedirs(data_dir, exist_ok=True)
    
    # Sort paths by iz number
    try:
        sorted_indices = sorted(range(len(iz_paths)), 
                              key=lambda i: int(Path(iz_paths[i]).name.replace('iz', '')))
        sorted_paths = [iz_paths[i] for i in sorted_indices]
    except Exception:
        sorted_paths = list(iz_paths)
    
    print(f"Testing convergence with {len(n_samples)} sample sizes: {n_samples}")
    
    # Compute for each n_sample
    results_by_n = {}
    for n in n_samples:
        print(f"\nComputing with {n} subvolume(s)...")
        smfs = []
        for idx, p in enumerate(sorted_paths):
            iz_name = Path(p).name
            print(f"  [{idx+1}/{len(sorted_paths)}] {iz_name}...", end=' ')
            
            smf = compute_smf_avg_by_snapshot(
                str(p),
                bins=bins,
                ivol_sample=n,
                random_seed=random_seed
            )
            
            if smf:
                smfs.append(smf)
                print(f"done ({smf['n_used']} ivols)")
            else:
                print("no data")
        
        results_by_n[n] = smfs
    
    # Create grid of subplots; scale by panel_size
    n_plots = len(n_samples)
    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size[0]*ncols, panel_size[1]*nrows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    cmap = plt.get_cmap('viridis')
    
    for idx, n in enumerate(n_samples):
        ax = axes[idx]
        smfs = results_by_n[n]
        
        if not smfs:
            ax.text(0.5, 0.5, f'No data for n={n}', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{n} subvolume{"s" if n > 1 else ""}', fontsize=14)
            continue
        
        for i, s in enumerate(smfs):
            color = cmap(i / (len(smfs) - 1 if len(smfs) > 1 else 1))
            if s['z'] is not None and not np.isnan(s['z']):
                label = f"z={s['z']:.2f}"
            else:
                label = s['iz']
            
            # Plot line with steps
            ax.plot(s['centers'], s['phi'], drawstyle='steps-mid', 
                   color=color, lw=2, label=label, alpha=0.8)
            
            # Show uncertainty if available and n > 1
            if 'phi_std' in s and s['phi_std'] is not None and n > 1:
                ax.fill_between(s['centers'], 
                               np.maximum(s['phi'] - s['phi_std'], 1e-10),
                               s['phi'] + s['phi_std'],
                               color=color, alpha=0.15, linewidth=0, step='mid')
        
        ax.set_yscale('log')
        ax.set_ylim(bottom=1e-5)
        ax.set_xlim(8, 12)
        ax.set_ylabel(r'$\Phi$ [Mpc$^{-3}$ dex$^{-1}$]', fontsize=11)
        ax.set_xlabel(r'$\log_{10}(M_*/M_\odot)$', fontsize=11)
        ax.set_title(f'{n} subvolume{"s" if n > 1 else ""}', fontsize=14)
        ax.grid(True, which='both', alpha=0.25)
        ax.legend(fontsize=8, ncol=1, loc='best')
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle('SMF Convergence with Increasing Subvolume Sample Size', 
                fontsize=16, y=0.995)
    plt.tight_layout()
    
    if do_save:
        fp = os.path.join(outdir, 'smf_convergence.png')
        plt.savefig(fp, dpi=150, bbox_inches='tight')
        print(f"\nSaved convergence plot to {fp}")
        # Save plotted data as CSV
        for n, smfs in results_by_n.items():
            for i, s in enumerate(smfs):
                label = f"z{s['z']:.2f}" if s['z'] is not None and not np.isnan(s['z']) else s['iz']
                df = np.stack([s['centers'], s['phi'], s.get('phi_std', np.full_like(s['phi'], np.nan))], axis=1)
                header = 'logM,phi,phi_std'
                fname = f"smf_convergence_n{n}_{label}.csv"
                np.savetxt(os.path.join(data_dir, fname), df, delimiter=',', header=header, comments='')
        print(f"Saved SMF data to {data_dir}")
    
    plt.show()
    return results_by_n


def plot_hmf_convergence_by_redshift(
    iz_paths: List[str],
    redshifts: List[Optional[float]],
    ivol_sets: Optional[List[List[int]]] = None,
    ivol_labels: Optional[List[str]] = None,
    n_samples: Optional[List[int]] = None,
    bins: Optional[np.ndarray] = None,
    outdir: str = 'plots/convergence',
    do_save: bool = True,
    random_seed: int = 42,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    panel_size: tuple = (7, 5)
) -> Dict[str, Dict[str, Any]]:
    """Plot HMF convergence organized by redshift.

    Each panel shows one redshift with multiple lines for different
    subvolume counts, illustrating convergence as more subvolumes are
    averaged.

    Args:
        iz_paths: List of snapshot directory paths (iz*)
        redshifts: Optional list of redshift values matching iz_paths
        n_samples: Subvolume counts to test per snapshot panel
        bins: log10(M_halo) bin edges (default: np.arange(7.0, 15.5, 0.2))
        outdir: Directory to save figure
        do_save: Whether to save figure and CSVs
        random_seed: Random seed for subvolume sampling

    Returns:
        Dict keyed by redshift label with per-n sample HMF results
    """
    if bins is None:
        bins = np.arange(7.0, 15.5, 0.2)

    os.makedirs(outdir, exist_ok=True)
    data_dir = 'plots/_plots_data/convergence_by_redshift'
    os.makedirs(data_dir, exist_ok=True)

    # Sort paths by iz index for consistent ordering
    try:
        sorted_indices = sorted(range(len(iz_paths)),
                                key=lambda i: int(Path(iz_paths[i]).name.replace('iz', '')))
        sorted_paths = [iz_paths[i] for i in sorted_indices]
        sorted_z = [redshifts[i] if redshifts is not None else None for i in sorted_indices]
    except Exception:
        sorted_paths = list(iz_paths)
        sorted_z = redshifts if redshifts is not None else [None] * len(sorted_paths)

    results_by_z: Dict[str, Dict[str, Any]] = {}

    mode_desc = f"ivol_sets={ivol_sets}" if ivol_sets else f"n_samples={n_samples}"
    print(f"Computing convergence by redshift with {mode_desc}")
    for idx, p in enumerate(sorted_paths):
        iz_name = Path(p).name
        z_val = sorted_z[idx]
        z_label = f"z={z_val:.2f}" if z_val is not None and not np.isnan(z_val) else iz_name
        print(f"\nProcessing {iz_name} ({z_label})...")

        per_n_results: Dict[int, Dict[str, Any]] = {}
        if ivol_sets:
            labels = ivol_labels if ivol_labels and len(ivol_labels) == len(ivol_sets) else [f"ivols={','.join(map(str,s))}" for s in ivol_sets]
            for label, ivols in zip(labels, ivol_sets):
                hmf = compute_hmf_avg_by_snapshot(str(p), bins=bins, ivol_list=ivols)
                if hmf:
                    per_n_results[label] = hmf
                    print(f"  {label}: done ({hmf['n_used']} ivols used)")
                else:
                    print(f"  {label}: no data")
        else:
            if n_samples is None:
                n_samples = [1, 2, 5, 10]
            for n in n_samples:
                hmf = compute_hmf_avg_by_snapshot(str(p), bins=bins, ivol_sample=n, random_seed=random_seed)
                if hmf:
                    per_n_results[n] = hmf
                    print(f"  n={n}: done ({hmf['n_used']} ivols used)")
                else:
                    print(f"  n={n}: no data")

        results_by_z[z_label] = {
            'z': z_val,
            'iz': iz_name,
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

    cmap = plt.get_cmap('viridis')

    for idx, (z_label, data) in enumerate(results_by_z.items()):
        ax = axes[idx]
        hmfs_dict = data['hmfs']

        if not hmfs_dict:
            ax.text(0.5, 0.5, f'No data for {z_label}', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(z_label, fontsize=14)
            continue

        for i, key in enumerate(sorted(hmfs_dict.keys(), key=lambda k: str(k))):
            h = hmfs_dict[key]
            color = cmap(i / (len(hmfs_dict) - 1 if len(hmfs_dict) > 1 else 1))
            label = f'n={key}' if isinstance(key, (int, float)) else str(key)
            ax.plot(h['centers'], h['phi'], drawstyle='steps-mid', color=color, lw=2, label=label, alpha=0.85)
            # Use n_used to decide if uncertainty shading is meaningful; avoids undefined variable 'n'
            if 'phi_std' in h and h['phi_std'] is not None and h.get('n_used', 1) > 1:
                ax.fill_between(h['centers'],
                                np.maximum(h['phi'] - h['phi_std'], 1e-10),
                                h['phi'] + h['phi_std'],
                                color=color, alpha=0.18, linewidth=0, step='mid')

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
        ax.legend(fontsize=8, loc='best')

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
