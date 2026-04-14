#!/usr/bin/env python3
"""Compute full-box RSD multipoles (standard estimator) and save to CSV.

This script loads all requested subvolumes, constructs redshift-space positions
from (x, y, z, vzgal), computes standard Landy-Szalay xi(s, mu), and projects
to monopole/quadrupole (xi0, xi2).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from galform_analysis.analysis.redshift_space_distortions.subvol_weighted_multipoles import (  # noqa: E402
    compute_standard_rsd_multipoles,
)
from galform_analysis.config import Cosmology, get_snapshot_redshift  # noqa: E402
from galform_analysis.utils.read_galaxies import read_galaxy_arrays  # noqa: E402


def _iter_existing_ivols(iz_path: Path, nmax: int) -> list[int]:
    ivols: list[int] = []
    for iv in range(nmax):
        if (iz_path / f"ivol{iv}" / "galaxies.hdf5").exists():
            ivols.append(iv)
    return ivols


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute full-box standard RSD multipoles and save CSV")
    parser.add_argument("--base-dir", required=True, help="Base simulation path containing izXX directories")
    parser.add_argument("--sim-name", required=True, help="Simulation label for metadata, e.g. L800")
    parser.add_argument("--model-name", default="lc16", help="Model label for metadata")
    parser.add_argument("--iz", type=int, required=True, help="Snapshot number, e.g. 155")
    parser.add_argument("--nmax", type=int, default=1024, help="Maximum subvolume index count to scan")
    parser.add_argument("--boxsize", type=float, default=542.16, help="Simulation box size [Mpc/h]")
    parser.add_argument("--mhalo-min", type=float, default=1e10, help="Halo-mass cut")
    parser.add_argument("--centrals-only", action="store_true", help="Use centrals only")
    parser.add_argument("--max-galaxies-per-ivol", type=int, default=0, help="Optional cap per subvolume (0 means no cap)")
    parser.add_argument("--n-random", type=int, default=200000, help="Random catalogue size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--s-min", type=float, default=0.1)
    parser.add_argument("--s-max", type=float, default=25.0)
    parser.add_argument("--n-s-bins", type=int, default=20)
    parser.add_argument("--mu-max", type=float, default=1.0)
    parser.add_argument("--n-mu-bins", type=int, default=24)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--output-dir", default="data/redshift_space_distortions/fullbox")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    iz_path = Path(args.base_dir) / f"iz{args.iz}"
    if not iz_path.is_dir():
        raise FileNotFoundError(f"Snapshot directory not found: {iz_path}")

    ivols = _iter_existing_ivols(iz_path, args.nmax)
    if not ivols:
        raise RuntimeError(f"No ivol*/galaxies.hdf5 files found under {iz_path}")

    z_snap = get_snapshot_redshift(f"iz{args.iz}")
    if z_snap is None:
        z_snap = 0.0

    h = Cosmology.h
    ez = np.sqrt(Cosmology.OMEGA_M * (1.0 + z_snap) ** 3 + Cosmology.OMEGA_L)
    hz = 100.0 * h * ez  # km/s/Mpc

    pos_chunks: list[np.ndarray] = []
    n_loaded = 0
    for ivol in ivols:
        arrays, _ = read_galaxy_arrays(
            iz_path=str(iz_path),
            ivol=ivol,
            fields=["vzgal"],
            include_positions=True,
            include_derived=True,
            centrals_only=args.centrals_only,
            mhalo_min=args.mhalo_min,
        )

        x = np.asarray(arrays["x"], dtype=np.float64)
        y = np.asarray(arrays["y"], dtype=np.float64)
        z = np.asarray(arrays["z"], dtype=np.float64)
        vz = np.asarray(arrays["vzgal"], dtype=np.float64)

        if args.max_galaxies_per_ivol > 0 and x.size > args.max_galaxies_per_ivol:
            idx = rng.choice(x.size, size=args.max_galaxies_per_ivol, replace=False)
            x, y, z, vz = x[idx], y[idx], z[idx], vz[idx]

        ds_par = (vz / hz) * h
        z_rsd = np.mod(z + ds_par, args.boxsize)

        pos_chunks.append(np.column_stack([x, y, z_rsd]))
        n_loaded += x.size

        if ivol % 64 == 0:
            print(f"Loaded ivol{ivol}: cumulative ngal={n_loaded}", flush=True)

    galaxy_pos = np.ascontiguousarray(np.vstack(pos_chunks), dtype=np.float64)
    random_pos = np.ascontiguousarray(
        rng.uniform(0.0, args.boxsize, size=(args.n_random, 3)),
        dtype=np.float64,
    )

    s_bins = np.logspace(np.log10(args.s_min), np.log10(args.s_max), args.n_s_bins + 1)

    res = compute_standard_rsd_multipoles(
        galaxy_pos=galaxy_pos,
        random_pos=random_pos,
        s_bins=s_bins,
        mu_max=args.mu_max,
        n_mu_bins=args.n_mu_bins,
        boxsize=args.boxsize,
        nthreads=args.num_threads,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"rsd_fullbox_standard_{args.sim_name}_{args.model_name}_iz{args.iz}.csv"

    df = pd.DataFrame(
        {
            "sim": args.sim_name,
            "model": args.model_name,
            "iz": int(args.iz),
            "z": float(z_snap),
            "centrals_only": int(args.centrals_only),
            "mhalo_min": float(args.mhalo_min),
            "n_subvol_loaded": int(len(ivols)),
            "ngal": int(res["ngal"]),
            "nrandom": int(res["nrandom"]),
            "s": res["s"],
            "xi0": res["xi0"],
            "xi2": res["xi2"],
        }
    )
    df.to_csv(out_csv, index=False)

    print(f"Saved full-box standard RSD multipoles: {out_csv}", flush=True)
    print(f"Rows={len(df)} n_subvol_loaded={len(ivols)} ngal={res['ngal']}", flush=True)


if __name__ == "__main__":
    main()
