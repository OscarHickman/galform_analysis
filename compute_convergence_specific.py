#!/usr/bin/env python
"""
Compute HMF and 2PCF convergence for specific subvolume counts and save to CSV.

Usage:
  python compute_convergence_specific.py --iz 271 --subvols 1,2,4,8,20,50,100,300,600,1000
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure src on path
project_root = Path(__file__).parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from galform_analysis.config import get_base_dir, get_snapshot_redshift
from galform_analysis.analysis import (
    completed_galaxies,
    avg_hmf_given_redshift_and_subvolumes,
    avg_correlation_given_redshift_and_subvolumes,
)


def compute_convergence_specific(iz_num, subvol_counts, output_dir="convergence_results", mhalo_min=None):
    """Compute HMF and 2PCF for a snapshot using specific subvolume counts."""
    base_dir = Path(get_base_dir())
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    iz_key = f"iz{iz_num}"
    print("\n" + "=" * 60)
    print(f"Computing convergence for {iz_key}")
    print(f"Subvolume counts: {subvol_counts}")
    if mhalo_min is not None:
        print(f"Halo mass cut: {mhalo_min:.2e}")
    print("=" * 60)

    df_completed = completed_galaxies(str(base_dir), [str(iz_num)])
    iz_completed = df_completed[(df_completed["iz"] == iz_key) & (df_completed["completed"])]
    available_ivols = sorted(iz_completed["ivol"].unique())

    if len(available_ivols) == 0:
        print(f"ERROR: No completed subvolumes found for {iz_key}")
        return None, None

    print(f"Available completed subvolumes: {len(available_ivols)}")

    hmf_bins = np.arange(9.0, 15.0, 0.2)
    hmf_centers = 0.5 * (hmf_bins[:-1] + hmf_bins[1:])
    corr_rbins = np.logspace(np.log10(0.1), np.log10(50.0), 20)

    z = get_snapshot_redshift(iz_key)
    if z is None:
        z = np.nan

    hmf_results = []
    corr_results = []
    processed_n_ivols = []

    for n_ivols in subvol_counts:
        if n_ivols < 1:
            print(f"  Skipping invalid n_ivols={n_ivols}")
            continue

        ivols_use = available_ivols[: min(n_ivols, len(available_ivols))]

        print(f"\n  n_ivols={n_ivols}: ", end="", flush=True)

        processed_n_ivols.append(n_ivols)

        try:
            h = avg_hmf_given_redshift_and_subvolumes(
                iz_num=iz_num,
                ivols=ivols_use,
                bins=hmf_bins,
                base_dir=str(base_dir),
                halo_mass_lower_limit=mhalo_min,
            )
            if h:
                for j, (center, phi) in enumerate(zip(hmf_centers, h["phi"])):
                    hmf_results.append(
                        {
                            "iz": iz_key,
                            "z": z,
                            "n_ivols": n_ivols,
                            "log_Mhalo": float(center),
                            "phi": float(phi),
                            "bin_idx": j,
                        }
                    )
                print("HMF ", end="", flush=True)
            else:
                print("HMF_none ", end="", flush=True)
        except Exception as e:
            print(f"HMF_error({e}) ", end="", flush=True)

        try:
            corr = avg_correlation_given_redshift_and_subvolumes(
                iz_num=iz_num,
                ivols=ivols_use,
                rbins=corr_rbins,
                nthreads=4,
                base_dir=str(base_dir),
                mhalo_min=mhalo_min,
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
                print("2PCF", flush=True)
            else:
                print("2PCF_none", flush=True)
        except Exception as e:
            print(f"2PCF_error({e})", flush=True)

    hmf_df = pd.DataFrame(hmf_results) if hmf_results else pd.DataFrame()
    corr_df = pd.DataFrame(corr_results) if corr_results else pd.DataFrame()

    # Save results (always save even if partial)
    output_path.mkdir(parents=True, exist_ok=True)
    hmf_csv = output_path / f"hmf_convergence_iz{iz_num}.csv"
    corr_csv = output_path / f"corr_convergence_iz{iz_num}.csv"

    # If existing results exist, append/merge (replace any n_ivols we just computed)
    replace_n = sorted(set(processed_n_ivols))

    if hmf_csv.exists():
        try:
            existing_hmf = pd.read_csv(hmf_csv)
            if not existing_hmf.empty and "n_ivols" in existing_hmf.columns:
                existing_hmf = existing_hmf[~existing_hmf["n_ivols"].isin(replace_n)]
                hmf_df = pd.concat([existing_hmf, hmf_df], ignore_index=True)
        except Exception as e:
            print(f"Warning: could not read existing HMF CSV for append ({e}); overwriting.")

    if corr_csv.exists():
        try:
            existing_corr = pd.read_csv(corr_csv)
            if not existing_corr.empty and "n_ivols" in existing_corr.columns:
                existing_corr = existing_corr[~existing_corr["n_ivols"].isin(replace_n)]
                corr_df = pd.concat([existing_corr, corr_df], ignore_index=True)
        except Exception as e:
            print(f"Warning: could not read existing 2PCF CSV for append ({e}); overwriting.")

    if not hmf_df.empty:
        hmf_df = hmf_df.sort_values(["n_ivols", "bin_idx"]).reset_index(drop=True)
    if not corr_df.empty:
        corr_df = corr_df.sort_values(["n_ivols", "bin_idx"]).reset_index(drop=True)

    hmf_df.to_csv(hmf_csv, index=False)
    corr_df.to_csv(corr_csv, index=False)

    print(f"\n\n✓ Saved HMF: {hmf_csv}")
    print(f"✓ Saved 2PCF: {corr_csv}")

    print(f"\nHMF DataFrame shape: {hmf_df.shape}")
    if len(hmf_df) > 0:
        print(f"  Snapshots: {hmf_df['iz'].unique().tolist()}")
        print(f"  N_ivols range: {hmf_df['n_ivols'].min()}-{hmf_df['n_ivols'].max()}")

    print(f"\n2PCF DataFrame shape: {corr_df.shape}")
    if len(corr_df) > 0:
        print(f"  Snapshots: {corr_df['iz'].unique().tolist()}")
        print(f"  N_ivols range: {corr_df['n_ivols'].min()}-{corr_df['n_ivols'].max()}")

    print("\n✓ Done!")
    return hmf_df, corr_df


def main():
    parser = argparse.ArgumentParser(
        description="Compute HMF and 2PCF convergence at specific subvolume counts",
    )
    parser.add_argument("--iz", type=int, required=True, help="Snapshot number (e.g., 271, 207)")
    parser.add_argument(
        "--subvols",
        type=str,
        default="1,2,4,8,10,15,20,30,50,100,200,300,600,1024",
        help="Comma-separated subvolume counts",
    )
    parser.add_argument("--output-dir", type=str, default="convergence_results", help="Output directory")
    parser.add_argument("--iteration", type=int, default=1, help="Iteration number (for averaging multiple runs)")
    parser.add_argument("--mhalo-min", type=float, default=None, help="Minimum halo mass (Msun) for filtering (e.g., 1e11)")

    args = parser.parse_args()
    subvol_counts = [int(x.strip()) for x in args.subvols.split(",") if x.strip()]
    subvol_counts.sort()

    print("\nStarting convergence computation:")
    print(f"  snapshot: iz{args.iz}")
    print(f"  iteration: {args.iteration}")
    print(f"  subvol_counts: {subvol_counts}")
    if args.mhalo_min is not None:
        print(f"  mhalo_min: {args.mhalo_min:.2e}")
    print(f"  output_dir: {args.output_dir}")
    print()

    compute_convergence_specific(
        iz_num=args.iz,
        subvol_counts=subvol_counts,
        output_dir=args.output_dir,
        mhalo_min=args.mhalo_min,
    )


if __name__ == "__main__":
    main()
