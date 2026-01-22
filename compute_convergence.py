#!/usr/bin/env python
"""
Compute HMF and 2PCF convergence across subvolumes.
Save results to CSV for analysis.

Usage:
    python compute_convergence.py --iz 271 --max-ivols all
    python compute_convergence.py --iz 207 --max-ivols 100
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
project_root = Path(__file__).parent
if str(project_root / 'src') not in sys.path:
    sys.path.insert(0, str(project_root / 'src'))

from galform_analysis.config import get_base_dir, get_snapshot_redshift
from galform_analysis.analysis import (
    completed_galaxies,
    avg_hmf_given_redshift_and_subvolumes,
    avg_correlation_given_redshift_and_subvolumes,
)


def compute_convergence(iz_num, max_ivols=None, output_dir='convergence_results'):
    """
    Compute HMF and 2PCF convergence for a redshift across subvolumes.
    
    Parameters
    ----------
    iz_num : int
        Snapshot number (e.g., 207, 271)
    max_ivols : int or None
        Maximum number of subvolumes to use. If None, use all available.
    output_dir : str
        Directory to save results
        
    Returns
    -------
    tuple of (hmf_df, corr_df)
        DataFrames with HMF and 2PCF results
    """
    base_dir = Path(get_base_dir())
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    iz_key = f'iz{iz_num}'
    print(f"\n{'='*60}")
    print(f"Computing convergence for {iz_key}")
    print(f"{'='*60}")
    
    # Get completed subvolumes
    df_completed = completed_galaxies(str(base_dir), [str(iz_num)])
    iz_completed = df_completed[(df_completed['iz'] == iz_key) & (df_completed['completed'])]
    available_ivols = sorted(iz_completed['ivol'].unique())
    
    if len(available_ivols) == 0:
        print(f"ERROR: No completed subvolumes found for {iz_key}")
        return None, None
    
    if max_ivols is None or max_ivols == 'all':
        max_ivols = len(available_ivols)
    else:
        max_ivols = min(int(max_ivols), len(available_ivols))
    
    print(f"Available completed subvolumes: {len(available_ivols)}")
    print(f"Using up to {max_ivols} subvolumes")
    
    # Setup bins
    hmf_bins = np.arange(9.0, 15.0, 0.2)
    hmf_centers = 0.5 * (hmf_bins[:-1] + hmf_bins[1:])
    corr_rbins = np.logspace(np.log10(0.1), np.log10(50.0), 20)
    
    # Get redshift
    z = get_snapshot_redshift(iz_key)
    if z is None:
        z = np.nan
    
    # Compute HMF and 2PCF for each subvolume count
    hmf_results = []
    corr_results = []
    
    for n_ivols in range(1, max_ivols + 1):
        ivols_use = available_ivols[:n_ivols]
        
        print(f"\n  n_ivols={n_ivols}: ", end='', flush=True)
        
        # HMF
        try:
            h = avg_hmf_given_redshift_and_subvolumes(
                iz_num=iz_num,
                ivols=ivols_use,
                bins=hmf_bins,
                base_dir=str(base_dir)
            )
            if h:
                for j, (center, phi) in enumerate(zip(hmf_centers, h['phi'])):
                    hmf_results.append({
                        'iz': iz_key,
                        'z': z,
                        'n_ivols': n_ivols,
                        'log_Mhalo': center,
                        'phi': phi,
                        'bin_idx': j
                    })
                print("HMF ", end='', flush=True)
            else:
                print("HMF_failed ", end='', flush=True)
        except Exception as e:
            print(f"HMF_error({e}) ", end='', flush=True)
        
        # 2PCF
        try:
            corr = avg_correlation_given_redshift_and_subvolumes(
                iz_num=iz_num,
                ivols=ivols_use,
                rbins=corr_rbins,
                nthreads=4,
                base_dir=str(base_dir)
            )
            if corr is not None:
                for i, (r, xi) in enumerate(zip(corr['r'], corr['xi'])):
                    corr_results.append({
                        'iz': iz_key,
                        'z': z,
                        'n_ivols': n_ivols,
                        'r': r,
                        'xi': xi,
                        'bin_idx': i
                    })
                print("2PCF")
            else:
                print("2PCF_none")
        except Exception as e:
            print(f"2PCF_error({e})")
    
    # Convert to DataFrames
    hmf_df = pd.DataFrame(hmf_results) if hmf_results else None
    corr_df = pd.DataFrame(corr_results) if corr_results else None
    
    # Save to CSV
    if hmf_df is not None:
        hmf_file = output_path / f'hmf_convergence_iz{iz_num}.csv'
        hmf_df.to_csv(hmf_file, index=False)
        print(f"\n✓ Saved HMF: {hmf_file}")
    
    if corr_df is not None:
        corr_file = output_path / f'corr_convergence_iz{iz_num}.csv'
        corr_df.to_csv(corr_file, index=False)
        print(f"✓ Saved 2PCF: {corr_file}")
    
    return hmf_df, corr_df


def main():
    parser = argparse.ArgumentParser(
        description='Compute HMF and 2PCF convergence across subvolumes'
    )
    parser.add_argument('--iz', type=int, required=True, help='Snapshot number (e.g., 207, 271)')
    parser.add_argument('--max-ivols', type=str, default='all', 
                       help='Max subvolumes to use ("all" or integer)')
    parser.add_argument('--output-dir', type=str, default='convergence_results',
                       help='Output directory for CSV files')
    
    args = parser.parse_args()
    
    print(f"\nStarting convergence computation:")
    print(f"  snapshot: iz{args.iz}")
    print(f"  max_ivols: {args.max_ivols}")
    print(f"  output_dir: {args.output_dir}")
    
    hmf_df, corr_df = compute_convergence(
        iz_num=args.iz,
        max_ivols=args.max_ivols,
        output_dir=args.output_dir
    )
    
    if hmf_df is not None:
        print(f"\nHMF DataFrame shape: {hmf_df.shape}")
        print(f"  Snapshots: {hmf_df['iz'].unique()}")
        print(f"  N_ivols range: {hmf_df['n_ivols'].min()}-{hmf_df['n_ivols'].max()}")
    
    if corr_df is not None:
        print(f"\n2PCF DataFrame shape: {corr_df.shape}")
        print(f"  Snapshots: {corr_df['iz'].unique()}")
        print(f"  N_ivols range: {corr_df['n_ivols'].min()}-{corr_df['n_ivols'].max()}")
    
    print("\n✓ Done!")


if __name__ == '__main__':
    main()
