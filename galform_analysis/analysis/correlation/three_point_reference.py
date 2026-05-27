import numpy as np
import polars as pl
import os
from pathlib import Path
from scipy.spatial import cKDTree
import time

def compute_3pcf_counts_reference(
    positions, 
    rbins, 
    boxsize=542.16
):
    N = len(positions)
    nbins = len(rbins) - 1
    t_total = np.zeros(nbins)
    
    tree = cKDTree(positions, boxsize=boxsize)
    rmax = rbins[-1]
    
    print(f"Starting reference triplet counting for {N} galaxies...")
    start_time = time.time()
    
    for i in range(N):
        if i % 1000 == 0:
            print(f"Progress: {i}/{N}")
            
        idx = tree.query_ball_point(positions[i], rmax)
        idx = [j for j in idx if j > i]
        
        for j_idx, j in enumerate(idx):
            diff_ij = np.abs(positions[i] - positions[j])
            diff_ij = np.where(diff_ij > 0.5 * boxsize, boxsize - diff_ij, diff_ij)
            r_ij = np.sqrt(np.sum(diff_ij**2))
            
            bin_j = np.searchsorted(rbins, r_ij) - 1
            if bin_j < 0 or bin_j >= nbins: continue
            
            for k in idx[j_idx + 1:]:
                diff_ik = np.abs(positions[i] - positions[k])
                diff_ik = np.where(diff_ik > 0.5 * boxsize, boxsize - diff_ik, diff_ik)
                r_ik = np.sqrt(np.sum(diff_ik**2))
                
                bin_k = np.searchsorted(rbins, r_ik) - 1
                if bin_k < 0 or bin_k >= nbins: continue
                
                if bin_j == bin_k:
                    t_total[bin_j] += 1
                        
    duration = time.time() - start_time
    print(f"Reference triplets counted in {duration:.2f} seconds.")
    
    return {
        'r': 0.5 * (rbins[:-1] + rbins[1:]),
        't_total': t_total
    }

if __name__ == "__main__":
    import sys
    from src.analysis.correlation.subvol_weighted_correction import load_subvolume_galaxies
    
    # Reference Parameters
    k = 1024
    iz = 207
    mstar_cut = 10.0
    base_dir = "/cosma5/data/durham/dc-hick2/Galform_Out/L800/lc16"
    
    print(f"Loading reference sample (all subvolumes)...")
    # To get a "Full Box" reference with the same number of galaxies for direct comparison:
    # We load all k subvolumes and sample 5000.
    # This represents the "Full Box" density scaled down to the same N as the SCOPE test.
    # Wait, if I use the same N, the counts should match exactly if SCOPE is correct.
    
    ivols = list(range(k))
    # Note: Loading all 1024 ivols into memory might be heavy. Let's do it in chunks or just sample.
    # Better: just load a few hundred to get 5000 galaxies.
    df = load_subvolume_galaxies(base_dir, iz, ivols[:100], k_total=k, mstar_min_log10=mstar_cut)
    
    if len(df) > 5000:
        df = df.sample(n=5000, seed=123) # Different seed for reference
        
    pos = df.select(["x", "y", "z"]).to_numpy()
    rbins = np.logspace(0, 1.2, 6)
    
    results = compute_3pcf_counts_reference(pos, rbins)
    
    out_df = pl.DataFrame({
        "r": results['r'],
        "t_total": results['t_total']
    })
    
    output_path = f"science/SCOPE/3pcf_reference_iz{iz}.csv"
    out_df.write_csv(output_path)
    print(f"Reference results saved to {output_path}")
