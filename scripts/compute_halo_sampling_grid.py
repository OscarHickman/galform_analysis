#!/usr/bin/env python
"""Compute standard or halo-corrected 2PCF over a subvolume grid.

This script mirrors ``compute_group_sampling_grid.py`` but uses the halo-ID
based correction in ``halo_sampling_correction``. It is intended for GALFORM
outputs that contain either valid ``DHaloID`` or ``TreeID`` fields.
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

from galform_analysis.analysis.correlation.halo_sampling_correction import (  # noqa: E402
    compute_halo_sampling_correlations_for_nvolumes,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute standard or halo-corrected 2PCF for many subvolume counts."
    )
    parser.add_argument("--base-dir", required=True, help="Directory containing izXX folders")
    parser.add_argument("--sim-name", required=True, help="Simulation label for CSV metadata (e.g. L800, Mill1)")
    parser.add_argument("--iz", type=int, required=True, help="Snapshot number, e.g. 155")
    parser.add_argument("--subvols", default="all", help="Subvolume spec: 'all', '1-64', or '1,2,4,...'")
    parser.add_argument("--nmax", type=int, default=None, help="Override maximum subvolumes (defaults to auto-discovery)")
    parser.add_argument("--output-dir", default="data/convergence/halo_sampling_jobs", help="Output directory")
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
    parser.add_argument(
        "--mode",
        choices=["normal", "halo"],
        default="halo",
        help="normal=standard xi, halo=halo-sampling-corrected xi",
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

    rbins = np.logspace(-1, 1.5, 21)
    rmids = 0.5 * (rbins[:-1] + rbins[1:])

    rows: list[dict[str, float | int | str]] = []
    mstar_msg = "none" if args.mstar_min_log10 is None else f"{args.mstar_min_log10}"
    print(
        f"Running {args.sim_name} iz{args.iz} mode={args.mode} for {len(nvolumes)} subvolume values "
        f"(1..{nvolumes[-1]}), base_dir={base_dir}, centrals_only={args.centrals_only}, "
        f"mstar_min_log10={mstar_msg}",
        flush=True,
    )

    for idx, nvol in enumerate(nvolumes, start=1):
        print(f"[{idx}/{len(nvolumes)}] n_subvol={nvol}", flush=True)
        result = compute_halo_sampling_correlations_for_nvolumes(
            base_dir=str(base_dir),
            iz_num=args.iz,
            nvolumes_list=[nvol],
            rbins=rbins,
            boxsize=args.boxsize,
            mstar_min_log10=args.mstar_min_log10,
            mhalo_min=args.mhalo_min,
            centrals_only=args.centrals_only,
            n_total_subvolumes=nmax,
            num_threads=args.num_threads,
        )

        xi_standard = np.asarray(result[nvol]["xi_standard"])
        xi_corrected = np.asarray(result[nvol]["xi_corrected"])
        xi_out = xi_standard if args.mode == "normal" else xi_corrected
        ngal = int(result[nvol]["ngal"])
        nhalo = int(result[nvol]["nhalo"])

        for bidx, (r, xi_val) in enumerate(zip(rmids, xi_out)):
            rows.append(
                {
                    "sim": args.sim_name,
                    "iz": int(args.iz),
                    "mode": args.mode,
                    "centrals_only": int(args.centrals_only),
                    "mstar_min_log10": np.nan if args.mstar_min_log10 is None else float(args.mstar_min_log10),
                    "n_subvol": int(nvol),
                    "bin_idx": int(bidx),
                    "r": float(r),
                    "xi": float(xi_val),
                    "ngal": int(ngal),
                    "nhalo": int(nhalo),
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

            nmask = old_df["n_subvol"].isin(new_df["n_subvol"].unique())
            if args.mstar_min_log10 is None:
                smask = old_df["mstar_min_log10"].isna()
            else:
                smask = np.isclose(old_df["mstar_min_log10"], args.mstar_min_log10, rtol=0.0, atol=1e-12, equal_nan=False)

            old_df = old_df[~(nmask & smask)]
            new_df = pd.concat([old_df, new_df], ignore_index=True)

    new_df = new_df.sort_values(["n_subvol", "bin_idx"]).reset_index(drop=True)
    new_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}", flush=True)
    print(f"Rows: {len(new_df)}", flush=True)


if __name__ == "__main__":
    main()