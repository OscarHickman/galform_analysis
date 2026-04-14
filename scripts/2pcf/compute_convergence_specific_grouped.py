#!/usr/bin/env python
# ruff: noqa: E402
"""
Compute HMF and 2PCF convergence for specific subvolume counts and save to CSV.

Usage:
  python compute_convergence_specific.py --iz 271 --subvols 1,2,4,8,20,50,100,300,600,1000
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
from galform_analysis.analysis import (
    completed_galaxies,
    avg_hmf_given_redshift_and_subvolumes,
    avg_correlation_given_redshift_and_subvolumes,
)


def compute_convergence_specific(
    iz_num,
    subvol_counts,
    output_dir="data/convergence/convergence_results",
    mhalo_min=None,
    mode="both",
    corr_centrals_only=True,
    n_groups=4,
):
    """Compute HMF and/or 2PCF for a snapshot using specific subvolume counts."""
    if mode not in {"hmf", "corr", "both"}:
        raise ValueError(f"Invalid mode '{mode}'. Expected 'hmf', 'corr', or 'both'.")
    do_hmf = mode in {"hmf", "both"}
    do_corr = mode in {"corr", "both"}
    base_dir = Path(get_base_dir())
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    iz_key = f"iz{iz_num}"
    df_completed = completed_galaxies(str(base_dir), [str(iz_num)])
    iz_completed = df_completed[(df_completed["iz"] == iz_key) & (df_completed["completed"])]
    available_ivols = sorted(iz_completed["ivol"].unique())

    if len(available_ivols) == 0:
        raise RuntimeError(f"No completed subvolumes found for {iz_key}")

    hmf_bins = np.arange(9.0, 15.0, 0.2)
    hmf_centers = 0.5 * (hmf_bins[:-1] + hmf_bins[1:])
    corr_rbins = np.logspace(np.log10(0.1), np.log10(50.0), 20)

    z = get_snapshot_redshift(iz_key)
    if z is None:
        z = np.nan

    hmf_results = []
    corr_results = []
    processed_n_ivols = []
    rng = random.Random(42)

    for n_ivols in subvol_counts:
        if n_ivols < 1:
            raise ValueError(f"Invalid n_ivols={n_ivols}")

        processed_n_ivols.append(n_ivols)

        groups = n_groups if n_ivols >= 4 else 1
        for group_id in range(groups):
            ivols_use = rng.sample(
                available_ivols,
                k=min(n_ivols, len(available_ivols)),
            )
            ivols_use.sort()

            if do_hmf:
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
                                    "group_id": group_id,
                                    "log_Mhalo": float(center),
                                    "phi": float(phi),
                                    "bin_idx": j,
                                }
                            )
                except Exception as e:
                    raise RuntimeError(
                        f"HMF computation failed for n_ivols={n_ivols}, group={group_id}: {e}"
                    ) from e

            if do_corr:
                try:
                    corr = avg_correlation_given_redshift_and_subvolumes(
                        iz_num=iz_num,
                        ivols=ivols_use,
                        rbins=corr_rbins,
                        nthreads=16,
                        base_dir=str(base_dir),
                        mhalo_min=mhalo_min,
                        centrals_only=corr_centrals_only,
                    )
                    if corr is not None:
                        for i, (r_val, xi_val) in enumerate(zip(corr["r"], corr["xi"])):
                            corr_results.append(
                                {
                                    "iz": iz_key,
                                    "z": z,
                                    "n_ivols": n_ivols,
                                    "group_id": group_id,
                                    "r": float(r_val),
                                    "xi": float(xi_val),
                                    "bin_idx": i,
                                }
                            )
                except Exception as e:
                    raise RuntimeError(
                        f"2PCF computation failed for n_ivols={n_ivols}, group={group_id}: {e}"
                    ) from e

    hmf_df = pd.DataFrame(hmf_results) if hmf_results else pd.DataFrame()
    corr_df = pd.DataFrame(corr_results) if corr_results else pd.DataFrame()

    # Save results (always save even if partial)
    output_path.mkdir(parents=True, exist_ok=True)
    hmf_csv = output_path / f"hmf_convergence_iz{iz_num}.csv"
    corr_csv = output_path / f"corr_convergence_iz{iz_num}.csv"

    # If existing results exist, append/merge (replace any n_ivols we just computed)
    replace_n = sorted(set(processed_n_ivols))

    if do_hmf and hmf_csv.exists():
        try:
            existing_hmf = pd.read_csv(hmf_csv)
            if not existing_hmf.empty and "n_ivols" in existing_hmf.columns:
                existing_hmf = existing_hmf[~existing_hmf["n_ivols"].isin(replace_n)]
                hmf_df = pd.concat([existing_hmf, hmf_df], ignore_index=True)
        except Exception as e:
            raise RuntimeError(f"Failed to read existing HMF CSV for append: {e}") from e

    if do_corr and corr_csv.exists():
        try:
            existing_corr = pd.read_csv(corr_csv)
            if not existing_corr.empty and "n_ivols" in existing_corr.columns:
                existing_corr = existing_corr[~existing_corr["n_ivols"].isin(replace_n)]
                corr_df = pd.concat([existing_corr, corr_df], ignore_index=True)
        except Exception as e:
            raise RuntimeError(f"Failed to read existing 2PCF CSV for append: {e}") from e

    if do_hmf and not hmf_df.empty:
        hmf_df = hmf_df.sort_values(["n_ivols", "bin_idx"]).reset_index(drop=True)
        hmf_df.to_csv(hmf_csv, index=False)
    if do_corr and not corr_df.empty:
        corr_df = corr_df.sort_values(["n_ivols", "bin_idx"]).reset_index(drop=True)
        corr_df.to_csv(corr_csv, index=False)

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
    parser.add_argument("--output-dir", type=str, default="data/convergence/convergence_results", help="Output directory")
    parser.add_argument("--iteration", type=int, default=1, help="Iteration number (for averaging multiple runs)")
    parser.add_argument("--mhalo-min", type=float, default=None, help="Minimum halo mass (Msun) for filtering (e.g., 1e11)")
    parser.add_argument(
        "--mode",
        choices=["hmf", "corr", "both"],
        default="both",
        help="Which outputs to compute and write",
    )
    parser.add_argument(
        "--corr-include-satellites",
        action="store_true",
        help="Include satellites in 2PCF (centrals+satellites).",
    )

    args = parser.parse_args()
    subvol_counts = [int(x.strip()) for x in args.subvols.split(",") if x.strip()]
    subvol_counts.sort()

    compute_convergence_specific(
        iz_num=args.iz,
        subvol_counts=subvol_counts,
        output_dir=args.output_dir,
        mhalo_min=args.mhalo_min,
        mode=args.mode,
        corr_centrals_only=not args.corr_include_satellites,
    )


if __name__ == "__main__":
    main()
