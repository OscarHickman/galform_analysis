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


def compute_convergence_specific(iz_num, subvol_counts, output_dir="convergence_results"):
    """Compute HMF and 2PCF for a snapshot using specific subvolume counts."""
    base_dir = Path(get_base_dir())
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    iz_key = f"iz{iz_num}"
    print("\n" + "=" * 60)
    print(f"Computing convergence for {iz_key}")
    print(f"Subvolume counts: {subvol_counts}")
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

    for n_ivols in subvol_counts:
        if n_ivols < 1:
            print(f"  Skipping invalid n_ivols={n_ivols}")
            continue

        ivols_use = available_ivols[: min(n_ivols, len(available_ivols))]

        print(f"\n  n_ivols={n_ivols}: ", end="", flush=True)

        try:
            h = avg_hmf_given_redshift_and_subvolumes(
                iz_num=iz_num,
                ivols=ivols_use,
                bins=hmf_bins,
                base_dir=str(base_dir),
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
            continue

        try:
            corr = avg_correlation_given_redshift_and_subvolumes(
                iz_num=iz_num,
                ivols=ivols_use,
                rbins=corr_rbins,
                nthreads=4,
                base_dir=str(base_dir),
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

    hmf_csv = output_path / f"hmf_convergence_iz{iz_num}.csv"
    corr_csv = output_path / f"corr_convergence_iz{iz_num}.csv"

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
        default="1,2,4,8,20,50,100,300,600,1000",
        help="Comma-separated subvolume counts",
    )
    parser.add_argument("--output-dir", type=str, default="convergence_results", help="Output directory")

    args = parser.parse_args()
    subvol_counts = [int(x.strip()) for x in args.subvols.split(",") if x.strip()]
    subvol_counts.sort()

    print("\nStarting convergence computation:")
    print(f"  snapshot: iz{args.iz}")
    print(f"  subvol_counts: {subvol_counts}")
    print(f"  output_dir: {args.output_dir}")
    print()

    compute_convergence_specific(
        iz_num=args.iz,
        subvol_counts=subvol_counts,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
