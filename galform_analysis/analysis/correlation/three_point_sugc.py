import time

import numpy as np
import polars as pl

try:
    import sugc
except ImportError:
    sugc = None


def compute_3pcf_counts_with_sugc(
    positions, labels, rbins, m_selected, k_total=1024, boxsize=542.16
):
    """
    Compute triplet counts decomposed by subvolume origin using SUGC.

    positions: (N, 3)
    labels: (N,) subvolume IDs (0 to k-1)
    rbins: (nbins+1,) distance bins for the max side of the triangle
    """
    if sugc is None:
        raise ImportError("sugc package is not installed.")

    print(f"Starting SUGC triplet counting for {len(positions)} galaxies...")
    start_time = time.time()

    # SUGC count_npoint returns counts by 's' (number of distinct labels)
    # T_by_s[s-1, bin]
    T_by_s = sugc.count_npoint(
        positions.astype(np.float32),
        labels.astype(np.int32),
        rbins.astype(np.float32),
        float(boxsize),
        3,  # N=3 for 3PCF
    )

    duration = time.time() - start_time
    print(f"Triplets counted in {duration:.2f} seconds.")

    # Apply SUGC weights
    m = float(m_selected)
    k = float(k_total)

    # Weights for s=1, 2, 3
    w_sss = (m / k) ** 2
    w_ssd = (m**2 * (k - 1)) / (k**2 * (m - 1))
    w_ddd = (m**2 * (k - 1) * (k - 2)) / (k**2 * (m - 1) * (m - 2))

    t_sss = T_by_s[0]
    t_ssd = T_by_s[1]
    t_ddd = T_by_s[2]

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


# Legacy alias
def compute_3pcf_counts_with_scope(*args, **kwargs):
    return compute_3pcf_counts_with_sugc(*args, **kwargs)


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

    pos = df.select(["x", "y", "z"]).to_numpy()
    labels = df["partition_label"].to_numpy()

    rbins = np.logspace(0, 1.2, 6)  # 5 bins from 1 to 15 Mpc/h

    results = compute_3pcf_counts_with_sugc(pos, labels, rbins, m_selected=m, k_total=k)

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

    output_path = f"data/SUGC/3pcf_sugc_m{m}_iz{iz}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.write_csv(output_path)
    print(f"Results saved to {output_path}")
