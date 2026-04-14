#!/usr/bin/env python
"""Compute subvolume-weighted 2PCF with random subvolume selection.

This script mirrors ``compute_subvol_weighted_grid.py`` except for one behavior:
subvolumes are selected from a seeded random permutation (without replacement)
instead of deterministic ``ivol_start..ivol_start+n-1`` ordering.
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
    compute_weighted_xi_from_catalogue,
    load_subvolume_galaxies,
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


def _discover_available_ivols(base_dir: Path, iz: int) -> list[int]:
    """Return sorted ivol IDs available under base_dir/izXX."""
    iz_dir = base_dir / f"iz{iz}"
    if not iz_dir.is_dir():
        raise FileNotFoundError(f"Snapshot directory not found: {iz_dir}")

    ivols: list[int] = []
    for p in iz_dir.glob("ivol*"):
        if not p.is_dir():
            continue
        suffix = p.name.replace("ivol", "", 1)
        if suffix.isdigit():
            ivols.append(int(suffix))

    ivols = sorted(set(ivols))
    if not ivols:
        raise RuntimeError(f"No ivol directories found under {iz_dir}")
    return ivols


def _pick_output_xi(df: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "standard":
        return df["xi_standard"]
    if mode == "weighted":
        return df["xi_corrected"]
    raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute subvolume-weighted 2PCF for many subvolume counts with random ivol selection."
    )
    parser.add_argument("--base-dir", required=True, help="Directory containing izXX folders")
    parser.add_argument("--sim-name", required=True, help="Simulation label for CSV metadata (e.g. L800)")
    parser.add_argument("--iz", type=int, required=True, help="Snapshot number, e.g. 155")
    parser.add_argument("--subvols", default="all", help="Subvolume spec: 'all', '1-64', or '1,2,4,...'")
    parser.add_argument("--nmax", type=int, default=None, help="Override maximum subvolumes")
    parser.add_argument("--k-total", type=int, default=None, help="Total partitions k in correction coefficients")
    parser.add_argument(
        "--output-dir",
        default="data/convergence/subvol_weighted_jobs_random",
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
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=314159,
        help="Seed for random ivol permutation without replacement.",
    )

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    discovered_ivols = _discover_available_ivols(base_dir, args.iz)
    discovered_nmax = len(discovered_ivols)
    nmax = args.nmax if args.nmax is not None else discovered_nmax
    nmax = min(int(nmax), discovered_nmax)

    if args.subvols.lower() == "all":
        nvolumes = list(range(1, nmax + 1))
    else:
        nvolumes = _parse_subvols(args.subvols, nmax=nmax)

    k_total = int(args.k_total) if args.k_total is not None else int(nmax)
    if k_total < max(nvolumes):
        raise ValueError("k_total must be >= max requested n_subvol")

    rbins = np.logspace(-1, 1.5, 21)

    rng = np.random.default_rng(args.selection_seed)
    permuted_ivols = rng.permutation(np.array(discovered_ivols, dtype=np.int64)).tolist()
    ivol_order = permuted_ivols[:nmax]
    ivols_to_load = ivol_order[: max(nvolumes)]

    mstar_msg = "none" if args.mstar_min_log10 is None else f"{args.mstar_min_log10}"
    print(
        f"Running {args.sim_name} iz{args.iz} mode={args.mode} partition={args.partition_scheme} "
        f"for {len(nvolumes)} subvolume values, base_dir={base_dir}, "
        f"centrals_only={args.centrals_only}, mstar_min_log10={mstar_msg}",
        flush=True,
    )
    print(
        f"Random ivol selection without replacement | selection_seed={args.selection_seed} | "
        f"nmax={nmax} | loaded={len(ivols_to_load)}",
        flush=True,
    )
    print(f"First 10 selected ivols: {ivols_to_load[:10]}", flush=True)

    full_cat = load_subvolume_galaxies(
        base_dir=str(base_dir),
        iz_num=args.iz,
        ivols=ivols_to_load,
        centrals_only=args.centrals_only,
        mhalo_min=args.mhalo_min,
        mstar_min_log10=args.mstar_min_log10,
        partition_scheme=args.partition_scheme,
        k_total=k_total,
    )

    rows: list[dict[str, float | int | str]] = []
    label_col = "partition_label" if "partition_label" in full_cat.columns else "subvol_rank"
    for n in nvolumes:
        sub_cat = full_cat[full_cat[label_col] < int(n)].copy()
        result = compute_weighted_xi_from_catalogue(
            catalogue=sub_cat,
            m_selected=int(n),
            k_total=k_total,
            rbins=rbins,
            boxsize=args.boxsize,
            random_multiplier=args.random_multiplier,
            random_seed=args.random_seed + int(n),
            nthreads=args.num_threads,
        )

        out = pd.DataFrame(
            {
                "r": np.asarray(result["r"], dtype=np.float64),
                "xi_standard": np.asarray(result["xi_standard"], dtype=np.float64),
                "xi_corrected": np.asarray(result["xi_corrected"], dtype=np.float64),
            }
        )
        out["xi"] = _pick_output_xi(out, args.mode)

        for bidx, rec in out.reset_index(drop=True).iterrows():
            rows.append(
                {
                    "sim": args.sim_name,
                    "iz": int(args.iz),
                    "mode": args.mode,
                    "partition_scheme": args.partition_scheme,
                    "selection_mode": "random_without_replacement",
                    "selection_seed": int(args.selection_seed),
                    "centrals_only": int(args.centrals_only),
                    "mstar_min_log10": np.nan if args.mstar_min_log10 is None else float(args.mstar_min_log10),
                    "n_subvol": int(n),
                    "bin_idx": int(bidx),
                    "r": float(rec["r"]),
                    "xi": float(rec["xi"]),
                    "xi_standard": float(rec["xi_standard"]),
                    "xi_corrected": float(rec["xi_corrected"]),
                    "alpha": float(result["alpha"]),
                    "beta": float(result["beta"]),
                    "ngal": int(result["ngal"]),
                    "nrandom": int(result["nrandom"]),
                    "k_total": int(k_total),
                }
            )

    new_df = pd.DataFrame(rows)
    if new_df.empty:
        raise RuntimeError("Estimator returned an empty dataframe")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"halo_sampling_convergence_{args.mode}_{args.sim_name}_iz{args.iz}.csv"

    if out_csv.exists():
        old_df = pd.read_csv(out_csv)
        if not old_df.empty and "n_subvol" in old_df.columns:
            if "mstar_min_log10" not in old_df.columns:
                old_df["mstar_min_log10"] = np.nan
            if "partition_scheme" not in old_df.columns:
                old_df["partition_scheme"] = "ivol"
            if "selection_mode" not in old_df.columns:
                old_df["selection_mode"] = "deterministic"
            if "selection_seed" not in old_df.columns:
                old_df["selection_seed"] = np.nan

            nmask = old_df["n_subvol"].isin(new_df["n_subvol"].unique())
            pmask = old_df["partition_scheme"].astype(str) == str(args.partition_scheme)
            sel_mode_mask = old_df["selection_mode"].astype(str) == "random_without_replacement"
            sel_seed_mask = old_df["selection_seed"].fillna(-1).astype(int) == int(args.selection_seed)

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

            old_df = old_df[~(nmask & pmask & smask & sel_mode_mask & sel_seed_mask)]
            new_df = pd.concat([old_df, new_df], ignore_index=True)

    new_df = new_df.sort_values(["n_subvol", "bin_idx"]).reset_index(drop=True)
    new_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}", flush=True)
    print(f"Rows: {len(new_df)}", flush=True)


if __name__ == "__main__":
    main()