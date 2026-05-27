import time

import numpy as np
import polars as pl
from scipy.spatial import cKDTree


def compute_3pcf_counts_with_scope(
    positions, labels, rbins, m_selected, k_total=1024, boxsize=542.16
):
    """
    Compute triplet counts decomposed by subvolume origin.

    positions: (N, 3)
    labels: (N,) subvolume IDs
    rbins: (nbins,) distance bins for the sides of the triangle
           (simplified to equilateral/isosceles for this test)
    """
    N = len(positions)
    nbins = len(rbins) - 1

    # Triplets decomposed
    t_sss = np.zeros(nbins)
    t_ssd = np.zeros(nbins)
    t_ddd = np.zeros(nbins)

    tree = cKDTree(positions, boxsize=boxsize)
    rmax = rbins[-1]

    print(f"Starting triplet counting for {N} galaxies...")
    start_time = time.time()

    # We'll compute counts for "triplets with two sides < r"
    # To keep it simple for this test: we count triplets (i,j,k)
    # where dist(i,j) and dist(i,k) are in the same bin
    for i in range(N):
        if i % 1000 == 0:
            print(f"Progress: {i}/{N}")

        idx = tree.query_ball_point(positions[i], rmax)
        idx = [j for j in idx if j > i]  # Avoid double counting and self-pairs

        for j_idx, j in enumerate(idx):
            r_ij = np.linalg.norm(positions[i] - positions[j])
            # Check periodic wrap manually if not handled by norm?
            # cKDTree handles wrap in query.
            # But dist needs to be periodic.
            diff_ij = np.abs(positions[i] - positions[j])
            diff_ij = np.where(diff_ij > 0.5 * boxsize, boxsize - diff_ij, diff_ij)
            r_ij = np.sqrt(np.sum(diff_ij**2))

            bin_j = np.searchsorted(rbins, r_ij) - 1
            if bin_j < 0 or bin_j >= nbins:
                continue

            for k in idx[j_idx + 1 :]:
                diff_ik = np.abs(positions[i] - positions[k])
                diff_ik = np.where(diff_ik > 0.5 * boxsize, boxsize - diff_ik, diff_ik)
                r_ik = np.sqrt(np.sum(diff_ik**2))

                bin_k = np.searchsorted(rbins, r_ik) - 1
                if bin_k < 0 or bin_k >= nbins:
                    continue

                # For this test, we only look at "v-configurations"
                # where both arms are in the same bin
                if bin_j == bin_k:
                    L_i, L_j, L_k = labels[i], labels[j], labels[k]

                    if L_i == L_j == L_k:
                        t_sss[bin_j] += 1
                    elif L_i == L_j or L_j == L_k or L_i == L_k:
                        t_ssd[bin_j] += 1
                    else:
                        t_ddd[bin_j] += 1

    duration = time.time() - start_time
    print(f"Triplets counted in {duration:.2f} seconds.")

    # Apply SCOPE weights
    m = float(m_selected)
    k = float(k_total)

    w_sss = (m / k) ** 2
    w_ssd = (m**2 * (k - 1)) / (k**2 * (m - 1))
    w_ddd = (m**2 * (k - 1) * (k - 2)) / (k**2 * (m - 1) * (m - 2))

    t_corr = w_sss * t_sss + w_ssd * t_ssd + w_ddd * t_ddd

    return {
        "r": 0.5 * (rbins[:-1] + rbins[1:]),
        "t_sss": t_sss,
        "t_ssd": t_ssd,
        "t_ddd": t_ddd,
        "t_corr": t_corr,
        "t_total": t_sss + t_ssd + t_ddd,
        "weights": (w_sss, w_ssd, w_ddd),
    }


if __name__ == "__main__":
    import sys

    from galform_analysis.analysis.correlation.subvol_weighted_correction import (
        load_subvolume_galaxies,
    )

    # Test Parameters
    m = 10
    k = 1024
    iz = 207
    mstar_cut = 10.0

    import os

    _USER = os.environ.get("USER", "<USER>")
    base_dir = f"/cosma5/data/durham/{_USER}/Galform_Out/L800/lc16"

    print(f"Loading {m} subvolumes...")
    ivols = list(range(m))
    df = load_subvolume_galaxies(
        base_dir, iz, ivols, k_total=k, mstar_min_log10=mstar_cut
    )

    if df.is_empty():
        print("No galaxies found.")
        sys.exit(1)

    # Take a small subset for speed in this test
    if len(df) > 5000:
        print(f"Subsampling {len(df)} to 5000 for quick test.")
        df = df.sample(n=5000, seed=42)

    pos = df.select(["x", "y", "z"]).to_numpy()
    labels = df["partition_label"].to_numpy()

    rbins = np.logspace(0, 1.2, 6)  # 5 bins from 1 to 15 Mpc/h

    results = compute_3pcf_counts_with_scope(
        pos, labels, rbins, m_selected=m, k_total=k
    )

    # Save results
    out_df = pl.DataFrame(
        {
            "r": results["r"],
            "t_sss": results["t_sss"],
            "t_ssd": results["t_ssd"],
            "t_ddd": results["t_ddd"],
            "t_corr": results["t_corr"],
            "t_total": results["t_total"],
        }
    )

    output_path = f"science/SCOPE/3pcf_scope_m{m}_iz{iz}.csv"
    out_df.write_csv(output_path)
    print(f"Results saved to {output_path}")
