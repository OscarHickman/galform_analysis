#!/usr/bin/env python3
"""Compute RSD multipoles for a selected number of subvolumes and save CSV."""

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
    compute_weighted_rsd_multipoles,
)
from galform_analysis.config import Cosmology, get_snapshot_redshift  # noqa: E402
from galform_analysis.utils.read_galaxies import read_galaxy_arrays  # noqa: E402


def _existing_ivols(iz_path: Path, nmax: int) -> list[int]:
    out: list[int] = []
    for iv in range(nmax):
        if (iz_path / f"ivol{iv}" / "galaxies.hdf5").exists():
            out.append(iv)
    return out


def _select_ivols(
    available_ivols: list[int], n_subvol: int, selection: str, rng: np.random.Generator
) -> list[int]:
    if len(available_ivols) < n_subvol:
        raise RuntimeError(
            f"Requested n_subvol={n_subvol} but only {len(available_ivols)} subvolumes found"
        )

    if selection == "first":
        return available_ivols[:n_subvol]
    if selection == "random":
        chosen = rng.choice(np.asarray(available_ivols), size=n_subvol, replace=False)
        return [int(v) for v in np.sort(chosen)]

    raise ValueError(f"Unknown ivol selection mode: {selection}")


def main() -> None:
    p = argparse.ArgumentParser(description="Compute RSD multipoles for n selected subvolumes")
    p.add_argument("--base-dir", required=True)
    p.add_argument("--sim-name", required=True)
    p.add_argument("--model-name", default="lc16")
    p.add_argument("--iz", type=int, required=True)
    p.add_argument("--n-subvol", type=int, required=True)
    p.add_argument("--nmax", type=int, default=1024)
    p.add_argument("--mode", choices=["normal", "corrected"], default="normal")
    p.add_argument("--ivol-selection", choices=["first", "random"], default="first")
    p.add_argument("--boxsize", type=float, default=542.16)
    p.add_argument("--mhalo-min", type=float, default=1e10)
    p.add_argument("--centrals-only", action="store_true")
    p.add_argument("--n-random", type=int, default=120000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--s-min", type=float, default=0.1)
    p.add_argument("--s-max", type=float, default=25.0)
    p.add_argument("--n-s-bins", type=int, default=20)
    p.add_argument("--mu-max", type=float, default=1.0)
    p.add_argument("--n-mu-bins", type=int, default=24)
    p.add_argument("--num-threads", type=int, default=16)
    p.add_argument("--output-dir", default="data/redshift_space_distortions/subvol_jobs")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    iz_path = Path(args.base_dir) / f"iz{args.iz}"
    if not iz_path.is_dir():
        raise FileNotFoundError(f"Missing snapshot dir: {iz_path}")

    ivols = _select_ivols(
        available_ivols=_existing_ivols(iz_path, args.nmax),
        n_subvol=args.n_subvol,
        selection=args.ivol_selection,
        rng=rng,
    )

    z_snap = get_snapshot_redshift(f"iz{args.iz}")
    if z_snap is None:
        z_snap = 0.0

    h = Cosmology.h
    ez = np.sqrt(Cosmology.OMEGA_M * (1.0 + z_snap) ** 3 + Cosmology.OMEGA_L)
    hz = 100.0 * h * ez

    pos_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []

    for label, iv in enumerate(ivols):
        arr, _ = read_galaxy_arrays(
            iz_path=str(iz_path),
            ivol=iv,
            fields=["vzgal"],
            include_positions=True,
            include_derived=True,
            centrals_only=args.centrals_only,
            mhalo_min=args.mhalo_min,
        )

        x = np.asarray(arr["x"], dtype=np.float64)
        y = np.asarray(arr["y"], dtype=np.float64)
        z = np.asarray(arr["z"], dtype=np.float64)
        vz = np.asarray(arr["vzgal"], dtype=np.float64)

        ds_par = (vz / hz) * h
        z_rsd = np.mod(z + ds_par, args.boxsize)
        pos = np.column_stack([x, y, z_rsd])

        pos_chunks.append(pos)
        label_chunks.append(np.full(pos.shape[0], label, dtype=np.int64))

    galaxy_pos = np.ascontiguousarray(np.vstack(pos_chunks), dtype=np.float64)
    galaxy_labels = np.ascontiguousarray(np.concatenate(label_chunks), dtype=np.int64)
    random_pos = np.ascontiguousarray(
        rng.uniform(0.0, args.boxsize, size=(args.n_random, 3)), dtype=np.float64
    )
    s_bins = np.logspace(np.log10(args.s_min), np.log10(args.s_max), args.n_s_bins + 1)

    if args.mode == "normal":
        res = compute_standard_rsd_multipoles(
            galaxy_pos=galaxy_pos,
            random_pos=random_pos,
            s_bins=s_bins,
            mu_max=args.mu_max,
            n_mu_bins=args.n_mu_bins,
            boxsize=args.boxsize,
            nthreads=args.num_threads,
        )
        xi0 = res["xi0"]
        xi2 = res["xi2"]
    else:
        res = compute_weighted_rsd_multipoles(
            galaxy_pos=galaxy_pos,
            galaxy_labels=galaxy_labels,
            random_pos=random_pos,
            s_bins=s_bins,
            mu_max=args.mu_max,
            n_mu_bins=args.n_mu_bins,
            k_total=args.nmax,
            boxsize=args.boxsize,
            nthreads=args.num_threads,
        )
        xi0 = res["xi0_corrected"]
        xi2 = res["xi2_corrected"]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selection_suffix = "" if args.ivol_selection == "first" else f"_ivsel{args.ivol_selection}"
    out_csv = out_dir / (
        f"rsd_subvol_{args.mode}_{args.sim_name}_{args.model_name}_iz{args.iz}_n{args.n_subvol}"
        f"{selection_suffix}.csv"
    )

    pd.DataFrame(
        {
            "sim": args.sim_name,
            "model": args.model_name,
            "iz": int(args.iz),
            "z": float(z_snap),
            "mode": args.mode,
            "n_subvol": int(args.n_subvol),
            "ivol_selection": args.ivol_selection,
            "centrals_only": int(args.centrals_only),
            "mhalo_min": float(args.mhalo_min),
            "ngal": int(galaxy_pos.shape[0]),
            "nrandom": int(random_pos.shape[0]),
            "s": res["s"],
            "xi0": xi0,
            "xi2": xi2,
        }
    ).to_csv(out_csv, index=False)

    print(f"Saved: {out_csv}", flush=True)


if __name__ == "__main__":
    main()
