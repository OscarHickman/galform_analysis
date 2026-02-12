#!/usr/bin/env python
"""
Compute HMF convergence for specific subvolume counts and save to CSV.
This is HMF-only (no 2PCF), useful after fixing HMF volume scaling.
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
from galform_analysis.analysis import completed_galaxies, avg_hmf_given_redshift_and_subvolumes


def compute_hmf_convergence_specific(iz_num, subvol_counts, output_dir="convergence_results_hmf_only", mhalo_min=None):
    base_dir = Path(get_base_dir())
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    iz_key = f"iz{iz_num}"
    print("\n" + "=" * 60)
    print(f"Computing HMF convergence for {iz_key}")
    print(f"Subvolume counts: {subvol_counts}")
    if mhalo_min is not None:
        print(f"Halo mass cut: {mhalo_min:.2e}")
    print("=" * 60)

    df_completed = completed_galaxies(str(base_dir), [str(iz_num)])
    iz_completed = df_completed[(df_completed["iz"] == iz_key) & (df_completed["completed"])]
    available_ivols = sorted(iz_completed["ivol"].unique())

    if len(available_ivols) == 0:
        print(f"ERROR: No completed subvolumes found for {iz_key}")
        return pd.DataFrame()

    print(f"Available completed subvolumes: {len(available_ivols)}")

    hmf_bins = np.arange(9.0, 15.0, 0.2)
    hmf_centers = 0.5 * (hmf_bins[:-1] + hmf_bins[1:])
    z = get_snapshot_redshift(iz_key)
    if z is None:
        z = np.nan

    hmf_results = []

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
                halo_mass_lower_limit=mhalo_min,
            )
            if h:
                # DEBUG: Check what we got
                if n_ivols in [1, 2]:
                    print(f"DEBUG n={n_ivols}: counts[20]={h['counts'][20]}, V_ivol={h.get('V_ivol', 'MISSING')}, V_total={h.get('V_total', 'MISSING')}", flush=True)
                for j, (center, phi) in enumerate(zip(hmf_centers, h["phi"])):
                    hmf_results.append(
                        {
                            "iz": iz_key,
                            "z": z,
                            "n_ivols": n_ivols,
                            "log_Mhalo": float(center),
                            "phi": float(phi),
                            "counts": int(h["counts"][j]),
                            "bin_idx": j,
                        }
                    )
                print("HMF ✓", flush=True)
            else:
                print("HMF_none", flush=True)
        except Exception as e:
            print(f"HMF_error({e})", flush=True)

    hmf_df = pd.DataFrame(hmf_results) if hmf_results else pd.DataFrame()

    # Save results (always save even if partial)
    output_path.mkdir(parents=True, exist_ok=True)
    hmf_csv = output_path / f"hmf_convergence_iz{iz_num}.csv"

    hmf_df.to_csv(hmf_csv, index=False)

    print(f"\n\n✓ Saved HMF: {hmf_csv}")
    print(f"HMF DataFrame shape: {hmf_df.shape}")
    if len(hmf_df) > 0:
        print(f"  Snapshots: {hmf_df['iz'].unique().tolist()}")
        print(f"  N_ivols range: {hmf_df['n_ivols'].min()}-{hmf_df['n_ivols'].max()}")

    print("\n✓ Done!")
    return hmf_df


def main():
    parser = argparse.ArgumentParser(
        description="Compute HMF convergence at specific subvolume counts (HMF only)",
    )
    parser.add_argument("--iz", type=int, required=True, help="Snapshot number (e.g., 271, 207)")
    parser.add_argument(
        "--subvols",
        type=str,
        default="1,2,3,4,5,8,10,15,20,25,30,40,50,80,100,150,200,300,500,750,1024",
        help="Comma-separated subvolume counts",
    )
    parser.add_argument("--output-dir", type=str, default="convergence_results_hmf_only", help="Output directory")
    parser.add_argument("--iteration", type=int, default=1, help="Iteration number (informational)")
    parser.add_argument("--mhalo-min", type=float, default=None, help="Minimum halo mass (Msun) for filtering (e.g., 1e11)")

    args = parser.parse_args()
    subvol_counts = [int(x.strip()) for x in args.subvols.split(",") if x.strip()]
    subvol_counts.sort()

    print("\nStarting HMF-only convergence computation:")
    print(f"  snapshot: iz{args.iz}")
    print(f"  iteration: {args.iteration}")
    print(f"  subvol_counts: {subvol_counts}")
    if args.mhalo_min is not None:
        print(f"  mhalo_min: {args.mhalo_min:.2e}")
    print(f"  output_dir: {args.output_dir}")
    print()

    compute_hmf_convergence_specific(
        iz_num=args.iz,
        subvol_counts=subvol_counts,
        output_dir=args.output_dir,
        mhalo_min=args.mhalo_min,
    )


if __name__ == "__main__":
    main()
