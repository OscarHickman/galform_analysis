#!/usr/bin/env python
# ruff: noqa: E402
"""
Compute weighted (bias-corrected) 2PCF convergence for specific subvolume counts and save to CSV.

This computes the 2PCF corrected for incomplete group/halo sampling using marked pair counting.

Usage:
  python compute_weighted_convergence_specific.py --iz 271 --subvols 1,2,4,8,20,50,100,300,600,1000
  python compute_weighted_convergence_specific.py --iz 207 --subvols 1,5,10,20 --sampling-fraction 0.5
"""

import random
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure src on path
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from galform_analysis.config import get_base_dir, get_snapshot_redshift
from galform_analysis.analysis import completed_galaxies
from galform_analysis.analysis.correlation.mass_weighted_correlation import (
    avg_weighted_correlation_given_redshift_and_subvolumes,
)


def compute_weighted_convergence_specific(
    iz_num,
    subvol_counts,
    output_dir="data/convergence/convergence_results_weighted",
    mhalo_min=None,
    centrals_only=True,
    sampling_fraction=None,
):
    """Compute weighted 2PCF for a snapshot using specific subvolume counts.
    
    Args:
        iz_num: Snapshot number (e.g., 271)
        subvol_counts: List of subvolume counts to compute convergence for
        output_dir: Output directory for CSV files
        mhalo_min: Minimum halo mass filter in Msun
        centrals_only: If True, only use central galaxies
        sampling_fraction: Fraction of groups sampled in [0,1].
                          If None, infers from subvolume coverage
    """
    base_dir = Path(get_base_dir())
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    iz_key = f"iz{iz_num}"
    df_completed = completed_galaxies(str(base_dir), [str(iz_num)])
    iz_completed = df_completed[(df_completed["iz"] == iz_key) & (df_completed["completed"])]
    available_ivols = sorted(iz_completed["ivol"].unique())

    if len(available_ivols) == 0:
        raise RuntimeError(f"No completed subvolumes found for {iz_key}")

    corr_rbins = np.logspace(np.log10(0.1), np.log10(50.0), 20)

    z = get_snapshot_redshift(iz_key)
    if z is None:
        z = np.nan

    corr_results = []
    processed_n_ivols = []

    for n_ivols in subvol_counts:
        if n_ivols < 1:
            raise ValueError(f"Invalid n_ivols={n_ivols}")

        # Randomly sample subvolumes (range 0-1023 for 1024 total subvolumes)
        ivols_use = random.sample(range(1024), n_ivols)
        processed_n_ivols.append(n_ivols)

        print(f"Computing weighted 2PCF for n_ivols={n_ivols}...", flush=True)
        
        try:
            corr = avg_weighted_correlation_given_redshift_and_subvolumes(
                iz_num=iz_num,
                ivols=ivols_use,
                rbins=corr_rbins,
                nthreads=16,
                base_dir=str(base_dir),
                mhalo_min=mhalo_min,
                centrals_only=centrals_only,
                sampling_fraction=sampling_fraction,
            )
            if corr is not None:
                for i, (r_val, xi_val) in enumerate(zip(corr["r"], corr["xi"])):
                    corr_results.append(
                        {
                            "iz": iz_key,
                            "z": z,
                            "n_ivols": n_ivols,
                            "r": float(r_val),
                            "xi": float(xi_val),
                            "bin_idx": i,
                        }
                    )
                print(f"  ✓ Completed n_ivols={n_ivols}", flush=True)
            else:
                print(f"  ✗ Failed for n_ivols={n_ivols}: returned None", flush=True)
        except Exception as e:
            print(f"  ✗ Error for n_ivols={n_ivols}: {e}", flush=True)
            raise RuntimeError(f"Weighted 2PCF computation failed for n_ivols={n_ivols}: {e}") from e

    corr_df = pd.DataFrame(corr_results) if corr_results else pd.DataFrame()

    # Save results
    output_path.mkdir(parents=True, exist_ok=True)
    corr_csv = output_path / f"weighted_corr_convergence_iz{iz_num}.csv"

    # If existing results exist, append/merge (replace any n_ivols we just computed)
    replace_n = sorted(set(processed_n_ivols))

    if corr_csv.exists():
        try:
            existing_corr = pd.read_csv(corr_csv)
            if not existing_corr.empty and "n_ivols" in existing_corr.columns:
                existing_corr = existing_corr[~existing_corr["n_ivols"].isin(replace_n)]
                corr_df = pd.concat([existing_corr, corr_df], ignore_index=True)
        except Exception as e:
            raise RuntimeError(f"Failed to read existing weighted 2PCF CSV for append: {e}") from e

    if not corr_df.empty:
        corr_df = corr_df.sort_values(["n_ivols", "bin_idx"]).reset_index(drop=True)
        corr_df.to_csv(corr_csv, index=False)
        print(f"\n✓ Saved weighted 2PCF: {corr_csv}")
        print(f"  Shape: {corr_df.shape}")
        print(f"  Snapshots: {corr_df['iz'].unique()}")
        print(f"  N_ivols range: {corr_df['n_ivols'].min()}-{corr_df['n_ivols'].max()}")
    else:
        print("\n⚠ Warning: No 2PCF results generated")

    return corr_df


def main():
    parser = argparse.ArgumentParser(
        description="Compute weighted (bias-corrected) 2PCF convergence at specific subvolume counts",
    )
    parser.add_argument("--iz", type=int, required=True, help="Snapshot number (e.g., 271, 207)")
    parser.add_argument(
        "--subvols",
        type=str,
        default="1,2,4,8,10,15,20,30,50,100,200,300,600,1024",
        help="Comma-separated subvolume counts",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/convergence/convergence_results_weighted",
        help="Output directory for CSV files",
    )
    parser.add_argument(
        "--mhalo-min",
        type=float,
        default=None,
        help="Minimum halo mass (Msun) for filtering (e.g., 1e11)",
    )
    parser.add_argument(
        "--sampling-fraction",
        type=float,
        default=None,
        help="Fraction of groups sampled in [0,1]. If None, infers from subvolume coverage.",
    )
    parser.add_argument(
        "--include-satellites",
        action="store_true",
        help="Include satellites in 2PCF (centrals+satellites). Default: centrals only.",
    )

    args = parser.parse_args()
    subvol_counts = [int(x.strip()) for x in args.subvols.split(",") if x.strip()]
    subvol_counts.sort()

    compute_weighted_convergence_specific(
        iz_num=args.iz,
        subvol_counts=subvol_counts,
        output_dir=args.output_dir,
        mhalo_min=args.mhalo_min,
        centrals_only=not args.include_satellites,
        sampling_fraction=args.sampling_fraction,
    )


if __name__ == "__main__":
    main()
