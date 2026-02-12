#!/usr/bin/env python3
"""
Compute pair counts (DD) for multiple subvolume combinations.

For each n_ivols value, combines galaxies from n subvolumes and computes
raw pair counts across multiple radial bins. Saves results to CSV files
for both iz271 and iz207.

Usage:
    python compute_pair_counts.py [--output-dir OUTDIR]
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src to path
project_root = Path(__file__).parent
if str(project_root / 'src') not in sys.path:
    sys.path.insert(0, str(project_root / 'src'))

from galform_analysis.config import get_base_dir
from galform_analysis.analysis.correlation.correlation import _load_positions_from_hdf5
from Corrfunc.theory.xi import xi as corrfunc_xi


def compute_pair_counts_for_n(iz_num, n_ivols, base_dir_path, boxsize=542.158):
    """
    Compute pair counts for a given redshift and number of subvolumes.
    
    Args:
        iz_num: Redshift snapshot number (e.g., 207, 271)
        n_ivols: Number of subvolumes to combine
        base_dir_path: Base directory path
        boxsize: Box size in Mpc/h
    
    Returns:
        dict with keys: r_bin_centers, pair_counts, rbins_edges, ngal
    """
    # Load and combine positions
    all_pos = []
    iz_path = str(base_dir_path / f"iz{iz_num}")
    
    for ivol in range(n_ivols):
        try:
            pos, _ = _load_positions_from_hdf5(iz_path, ivol, centrals_only=True)
            all_pos.append(pos)
        except (FileNotFoundError, KeyError):
            print(f"Warning: Could not load iz{iz_num} ivol={ivol}")
            break
    
    if not all_pos:
        return None
    
    combined_pos = np.vstack(all_pos)
    ngal = len(combined_pos)
    
    # Define radial bins
    rbins = np.logspace(np.log10(0.5), np.log10(100), 35)
    
    # Call Corrfunc
    try:
        results = corrfunc_xi(
            boxsize=boxsize,
            nthreads=4,
            binfile=rbins,
            X=combined_pos[:, 0],
            Y=combined_pos[:, 1],
            Z=combined_pos[:, 2],
            output_ravg=True,
        )
    except Exception as e:
        print(f"Error calling Corrfunc for iz{iz_num} n={n_ivols}: {e}")
        return None
    
    # Extract results
    r_centers = np.array([res['ravg'] for res in results])
    pair_counts = np.array([res['npairs'] for res in results], dtype=np.int64)
    
    return {
        'r_centers': r_centers,
        'pair_counts': pair_counts,
        'rbins': rbins,
        'ngal': ngal,
        'n_ivols': n_ivols
    }


def compute_all_pair_counts(output_dir=None, iz_nums=None):
    """
    Compute pair counts for all redshifts and subvolume combinations.
    
    Args:
        output_dir: Output directory for CSV files (default: convergence_results)
    """
    if output_dir is None:
        output_dir = Path('convergence_results')
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_dir_path = Path(get_base_dir())
    
    # Subvolume numbers to test
    n_ivols_list = [1, 2, 4, 8, 20, 50, 100, 300, 600, 1000, 1024]
    if not iz_nums:
        iz_nums = [271, 207]
    
    # Compute for each redshift
    for iz_num in iz_nums:
        print(f"\n{'='*70}")
        print(f"Computing pair counts for iz{iz_num}")
        print(f"{'='*70}")
        
        all_results = []
        
        # Radial bins (computed once)
        rbins = np.logspace(np.log10(0.5), np.log10(100), 35)
        r_centers = 0.5 * (rbins[:-1] + rbins[1:])
        
        # Compute for each n
        for n in tqdm(n_ivols_list, desc=f"iz{iz_num}"):
            result = compute_pair_counts_for_n(iz_num, n, base_dir_path)
            
            if result is not None:
                # Create a row for each radial bin
                for i, (r, dd) in enumerate(zip(result['r_centers'], result['pair_counts'])):
                    all_results.append({
                        'iz': iz_num,
                        'n_ivols': n,
                        'n_gal': result['ngal'],
                        'r': r,
                        'r_min': rbins[i],
                        'r_max': rbins[i+1],
                        'dd': dd,
                    })
        
        # Save to CSV
        if all_results:
            df = pd.DataFrame(all_results)
            output_path = output_dir / f'pair_counts_iz{iz_num}.csv'
            df.to_csv(output_path, index=False)
            print(f"\nSaved {len(df)} records to {output_path}")
            print(f"  Unique n_ivols: {sorted(df['n_ivols'].unique())}")
            print(f"  Radial bins: {len(df) // len(df['n_ivols'].unique())}")
        else:
            print(f"ERROR: No results computed for iz{iz_num}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute pair counts for convergence analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python compute_pair_counts.py
  python compute_pair_counts.py --output-dir ./pair_counts_results
        '''
    )
    parser.add_argument(
        '--output-dir',
        default='convergence_results',
        help='Output directory for CSV files (default: convergence_results)'
    )
    parser.add_argument(
        '--iz',
        type=int,
        nargs='*',
        choices=[207, 271],
        help='Snapshot(s) to process (e.g. --iz 271 or --iz 207 271). Default: both.'
    )
    
    args = parser.parse_args()
    
    compute_all_pair_counts(output_dir=args.output_dir, iz_nums=args.iz)
    
    print(f"\n{'='*70}")
    print("Pair count computation complete!")
    print(f"{'='*70}")
