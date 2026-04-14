#!/usr/bin/env python
"""Compute subvolume-weighted 2PCF over a subvolume grid.

This script uses the new estimator in
``galform_analysis.analysis.correlation.subvol_weighted_correction`` and
writes one CSV per job in the same directory layout as prior halo-sampling
grid runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src on path when running from repository root or via SLURM scripts.
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from galform_analysis.analysis.correlation.subvol_weighted_correction import (  # noqa: E402
    compute_weighted_xi_for_n_list,
)


def _parse_subvols(subvols_arg: str, nmax: int) -> list[int]:
    """Parse subvolume specification: comma list or range like 1-64."""
    subvols_arg = subvols_arg.strip()
    if "-" in subvols_arg and "," not in subvols_arg:
        start_s, end_s = subvols_arg.split("-", 1)
        start = int(start_s)
        end = int(end_s)
        if start < 1 or end < start:
            raise ValueError(f"Invalid subvolume range: {subvols_arg}")
        vals = list(range(start, end + 1))
    else:
        vals = [int(x.strip()) for x in subvols_arg.split(",") if x.strip()]

    vals = sorted(set(vals))
    if not vals:
        raise ValueError("No valid subvolume values were provided")
    if vals[-1] > nmax:
        raise ValueError(f"Requested n_subvol={vals[-1]} exceeds nmax={nmax}")
    if vals[0] < 1:
        raise ValueError("Subvolume counts must be >= 1")
    return vals


def _discover_max_subvol(base_dir: Path, iz: int) -> int:
    iz_dir = base_dir / f"iz{iz}"
    if not iz_dir.is_dir():
        raise FileNotFoundError(f"Snapshot directory not found: {iz_dir}")
    return len([p for p in iz_dir.glob("ivol*") if p.is_dir()])


def _pick_output_xi(df: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "standard":
        return df["xi_standard"]
    if mode == "weighted":
        return df["xi_corrected"]
    raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute subvolume-weighted 2PCF for many subvolume counts."
    )
    parser.add_argument("--base-dir", required=True, help="Directory containing izXX folders")
    parser.add_argument("--sim-name", required=True, help="Simulation label for CSV metadata (e.g. L800)")
    parser.add_argument("--iz", type=int, required=True, help="Snapshot number, e.g. 155")
    parser.add_argument("--subvols", default="all", help="Subvolume spec: 'all', '1-64', or '1,2,4,...'")
    parser.add_argument("--nmax", type=int, default=None, help="Override maximum subvolumes")
    parser.add_argument("--k-total", type=int, default=None, help="Total partitions k in correction coefficients")
    parser.add_argument(
        "--output-dir",
        default="data/convergence/subvol_weighted_jobs",
        help="Output directory",
    )
    parser.add_argument("--boxsize", type=float, default=542.16)
    parser.add_argument(
        "--mstar-min-log10",
        type=float,
        default=None,
        help="Minimum log10 stellar mass cut. If omitted, no stellar-mass cut is applied.",
    )
    parser.add_argument("--mhalo-min", type=float, default=1e11)
    parser.add_argument(
        "--centrals-only",
        action="store_true",
        help="Use central galaxies only (is_central == 1)",
    )
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--random-multiplier", type=float, default=2.0)
    parser.add_argument("--random-seed", type=int, default=12345)
    parser.add_argument("--ivol-start", type=int, default=0)
    parser.add_argument(
        "--partition-scheme",
        choices=["ivol", "halo_id_hash"],
        default="ivol",
        help="Partition label source for auto/cross decomposition",
    )
    parser.add_argument(
        "--mode",
        choices=["standard", "weighted"],
        default="weighted",
        help="standard=xi_standard, weighted=xi_corrected",
    )

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    discovered_nmax = _discover_max_subvol(base_dir, args.iz)
    nmax = args.nmax if args.nmax is not None else discovered_nmax
    nmax = min(nmax, discovered_nmax)

    if args.subvols.lower() == "all":
        nvolumes = list(range(1, nmax + 1))
    else:
        nvolumes = _parse_subvols(args.subvols, nmax=nmax)

    k_total = int(args.k_total) if args.k_total is not None else int(nmax)
    if k_total < max(nvolumes):
        raise ValueError("k_total must be >= max requested n_subvol")

    rbins = np.logspace(-1, 1.5, 21)

    mstar_msg = "none" if args.mstar_min_log10 is None else f"{args.mstar_min_log10}"
    print(
        f"Running {args.sim_name} iz{args.iz} mode={args.mode} partition={args.partition_scheme} "
        f"for {len(nvolumes)} subvolume values, base_dir={base_dir}, "
        f"centrals_only={args.centrals_only}, mstar_min_log10={mstar_msg}",
        flush=True,
    )

    result_df = compute_weighted_xi_for_n_list(
        base_dir=str(base_dir),
        iz_num=args.iz,
        n_subvol_list=nvolumes,
        k_total=k_total,
        rbins=rbins,
        boxsize=args.boxsize,
        centrals_only=args.centrals_only,
        mhalo_min=args.mhalo_min,
        mstar_min_log10=args.mstar_min_log10,
        random_multiplier=args.random_multiplier,
        random_seed=args.random_seed,
        nthreads=args.num_threads,
        ivol_start=args.ivol_start,
        load_n_subvolumes=max(nvolumes),
        partition_scheme=args.partition_scheme,
    )

    if result_df.empty:
        raise RuntimeError("Estimator returned an empty dataframe")

    result_df = result_df.copy()
    result_df["xi"] = _pick_output_xi(result_df, args.mode)

    rows: list[dict[str, float | int | str]] = []
    for _, rec in result_df.iterrows():
        rows.append(
            {
                "sim": args.sim_name,
                "iz": int(args.iz),
                "mode": args.mode,
                "partition_scheme": args.partition_scheme,
                "centrals_only": int(args.centrals_only),
                "mstar_min_log10": np.nan if args.mstar_min_log10 is None else float(args.mstar_min_log10),
                "n_subvol": int(rec["n_subvol"]),
                "bin_idx": int(rec["bin_idx"]),
                "r": float(rec["r"]),
                "xi": float(rec["xi"]),
                "xi_standard": float(rec["xi_standard"]),
                "xi_corrected": float(rec["xi_corrected"]),
                "alpha": float(rec["alpha"]),
                "beta": float(rec["beta"]),
                "ngal": int(rec["ngal"]),
                "nrandom": int(rec["nrandom"]),
                "k_total": int(k_total),
            }
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"halo_sampling_convergence_{args.mode}_{args.sim_name}_iz{args.iz}.csv"

    new_df = pd.DataFrame(rows)
    if out_csv.exists():
        old_df = pd.read_csv(out_csv)
        if not old_df.empty and "n_subvol" in old_df.columns:
            if "mstar_min_log10" not in old_df.columns:
                old_df["mstar_min_log10"] = np.nan
            if "partition_scheme" not in old_df.columns:
                old_df["partition_scheme"] = "ivol"

            nmask = old_df["n_subvol"].isin(new_df["n_subvol"].unique())
            pmask = old_df["partition_scheme"].astype(str) == str(args.partition_scheme)
            if args.mstar_min_log10 is None:
                smask = old_df["mstar_min_log10"].isna()
            else:
                smask = np.isclose(
                    old_df["mstar_min_log10"],
                    args.mstar_min_log10,
                    rtol=0.0,
                    atol=1e-12,
                    equal_nan=False,
                )

            old_df = old_df[~(nmask & pmask & smask)]
            new_df = pd.concat([old_df, new_df], ignore_index=True)

    new_df = new_df.sort_values(["n_subvol", "bin_idx"]).reset_index(drop=True)
    new_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}", flush=True)
    print(f"Rows: {len(new_df)}", flush=True)


if __name__ == "__main__":
    main()
