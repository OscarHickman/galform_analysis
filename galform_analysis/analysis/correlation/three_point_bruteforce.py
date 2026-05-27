"""Vectorised brute-force 3PCF triplet counter.

For each i, queries all j within rmax via cKDTree, then enumerates all (j, k)
pairs vectorised over numpy.  Bins by max(r_ij, r_ik, r_jk).  If subvolume
labels are passed, decomposes the count into SSS / SSD / DDD by number of
distinct parent realisations.

Algorithm is identical to the loop in run_3pcf_reference_v3.py — just executes
the inner (j, k) enumeration in numpy rather than Python.
"""

import numpy as np
from scipy.spatial import cKDTree


def _periodic_dist(d, L):
    d = np.abs(d)
    return np.where(d > 0.5 * L, L - d, d)


def compute_triplet_counts(positions, rbins, boxsize, labels=None, log_every=10000):
    N = len(positions)
    nbins = len(rbins) - 1
    rmax = rbins[-1]

    t_sss = np.zeros(nbins, dtype=np.int64)
    t_ssd = np.zeros(nbins, dtype=np.int64)
    t_ddd = np.zeros(nbins, dtype=np.int64)
    t_total = np.zeros(nbins, dtype=np.int64)

    tree = cKDTree(positions, boxsize=boxsize)

    for i in range(N):
        if log_every and i % log_every == 0:
            print(f"  i={i}/{N}", flush=True)

        idx = tree.query_ball_point(positions[i], rmax)
        idx = np.fromiter((j for j in idx if j > i), dtype=np.int64)
        n_idx = len(idx)
        if n_idx < 2:
            continue

        pos_n = positions[idx]
        d_ij = _periodic_dist(positions[i] - pos_n, boxsize)
        r_ij = np.sqrt(np.einsum('ij,ij->i', d_ij, d_ij))

        diff = pos_n[:, None, :] - pos_n[None, :, :]
        diff = _periodic_dist(diff, boxsize)
        r_pair = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))

        jj, kk = np.triu_indices(n_idx, k=1)
        r_jk = r_pair[jj, kk]
        r_max_side = np.maximum(np.maximum(r_ij[jj], r_ij[kk]), r_jk)
        valid = r_max_side < rmax
        if not valid.any():
            continue

        bin_idx = np.searchsorted(rbins, r_max_side[valid]) - 1
        keep = bin_idx >= 0
        if not keep.any():
            continue
        bin_idx = bin_idx[keep]

        if labels is None:
            np.add.at(t_total, bin_idx, 1)
        else:
            jj_v = jj[valid][keep]
            kk_v = kk[valid][keep]
            L_i = labels[i]
            L_j = labels[idx[jj_v]]
            L_k = labels[idx[kk_v]]
            is_sss = (L_i == L_j) & (L_j == L_k)
            shared_any = (L_i == L_j) | (L_j == L_k) | (L_i == L_k)
            is_ssd = shared_any & ~is_sss
            is_ddd = ~shared_any
            np.add.at(t_sss, bin_idx[is_sss], 1)
            np.add.at(t_ssd, bin_idx[is_ssd], 1)
            np.add.at(t_ddd, bin_idx[is_ddd], 1)

    if labels is None:
        return None, None, None, t_total
    t_total = t_sss + t_ssd + t_ddd
    return t_sss, t_ssd, t_ddd, t_total


def scope_weights(m, k):
    m = float(m)
    k = float(k)
    w_sss = (m / k) ** 2
    w_ssd = (m ** 2 * (k - 1)) / (k ** 2 * (m - 1)) if m > 1 else 0.0
    w_ddd = (m ** 2 * (k - 1) * (k - 2)) / (k ** 2 * (m - 1) * (m - 2)) if m > 2 else 0.0
    return w_sss, w_ssd, w_ddd
