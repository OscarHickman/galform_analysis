#!/usr/bin/env python
"""Compute Corrfunc full-box reference xi(r) for a selected number of subvolumes.

This script is intended for N=1024 reference runs and writes one CSV per job in
the same directory pattern used by previous convergence campaigns.
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

from galform_analysis.analysis.correlation.correlation import (  # noqa: E402
    avg_correlation_given_redshift_and_subvolumes,
)
from galform_analysis.io.loaders import get_output_group, open_galaxies_hdf5  # noqa: E402


def _selected_output_group_name(base_dir: Path, iz_num: int, ivol: int) -> str:
    """Return the OutputNNN group chosen by the centralized loader."""
    iz_path = str(base_dir / f"iz{iz_num}")
    f = open_galaxies_hdf5(iz_path, ivol=ivol)
    if f is None:
        raise FileNotFoundError(f"Could not open galaxies.hdf5 for iz{iz_num}/ivol{ivol}")
    try:
        g = get_output_group(f)
        if g is None:
            raise RuntimeError(f"No OutputNNN group found in iz{iz_num}/ivol{ivol}")
        return str(g.name).split("/")[-1]
    finally:
        try:
            f.close()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Corrfunc full-box reference xi(r) for N=1024-style runs."
    )
    parser.add_argument("--base-dir", required=True, help="Directory containing izXX folders")
    parser.add_argument("--sim-name", required=True, help="Simulation label for CSV metadata (e.g. L800)")
    parser.add_argument("--iz", type=int, required=True, help="Snapshot number, e.g. 207")
    parser.add_argument("--n-subvol", type=int, default=1024, help="Number of ivols to combine (default: 1024)")
    parser.add_argument("--ivol-start", type=int, default=0, help="Starting ivol index (default: 0)")
    parser.add_argument("--output-dir", required=True, help="Output directory for CSV")
    parser.add_argument("--mhalo-min", type=float, default=1e11)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--centrals-only", action="store_true")

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    if args.n_subvol < 1:
        raise ValueError("n-subvol must be >= 1")

    ivols = list(range(int(args.ivol_start), int(args.ivol_start) + int(args.n_subvol)))
    output_group = _selected_output_group_name(base_dir=base_dir, iz_num=args.iz, ivol=ivols[0])

    print(
        f"Running Corrfunc reference: sim={args.sim_name} iz={args.iz} n_subvol={args.n_subvol} "
        f"centrals_only={args.centrals_only} mhalo_min={args.mhalo_min} "
        f"output_group={output_group}",
        flush=True,
    )

    result = avg_correlation_given_redshift_and_subvolumes(
        iz_num=int(args.iz),
        ivols=ivols,
        rbins=None,
        nthreads=int(args.num_threads),
        base_dir=str(base_dir),
        centrals_only=bool(args.centrals_only),
        mhalo_min=float(args.mhalo_min),
    )

    if result is None or result.empty:
        raise RuntimeError("Corrfunc full-box reference computation returned no data")

    out_rows: list[dict[str, float | int | str]] = []
    ngal = int(result.attrs.get("total_galaxies", np.nan)) if "total_galaxies" in result.attrs else int(np.nan)
    for bidx, rec in result.reset_index(drop=True).iterrows():
        out_rows.append(
            {
                "sim": args.sim_name,
                "iz": int(args.iz),
                "mode": "corrfunc",
                "centrals_only": int(args.centrals_only),
                "mstar_min_log10": np.nan,
                "n_subvol": int(args.n_subvol),
                "bin_idx": int(bidx),
                "r": float(rec["r"]),
                "xi": float(rec["xi"]),
                "ngal": ngal,
                "output_group": output_group,
            }
        )

    new_df = pd.DataFrame(out_rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"halo_sampling_convergence_corrfunc_{args.sim_name}_iz{args.iz}.csv"

    if out_csv.exists():
        old_df = pd.read_csv(out_csv)
        if not old_df.empty and "n_subvol" in old_df.columns:
            old_df = old_df[~old_df["n_subvol"].isin(new_df["n_subvol"].unique())]
            new_df = pd.concat([old_df, new_df], ignore_index=True)

    new_df = new_df.sort_values(["n_subvol", "bin_idx"]).reset_index(drop=True)
    new_df.to_csv(out_csv, index=False)

    print(f"Saved: {out_csv}", flush=True)
    print(f"Rows: {len(new_df)}", flush=True)


if __name__ == "__main__":
    main()
