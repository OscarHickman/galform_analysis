"""General N-point brute-force counter and SUGC correction weights.

For each starting galaxy i, finds all neighbours within rmax via cKDTree, then
enumerates every (N-1)-subset of those neighbours. For each subset, computes all
C(N, 2) pairwise periodic distances; if max < rmax, bins by max pairwise distance.

If labels are passed, decomposes each kept tuple by number of distinct labels s
in {1, ..., N}, returning per-s counts T_by_s[s-1].

sugc_weights_npcf: scale-down weight for s distinct labels,
    w_s = (m/k)^N * (k)_s / (m)_s
"""

import itertools

import numpy as np
from scipy.spatial import cKDTree


def _periodic_dist_matrix(pos_set, boxsize):
    diff = np.abs(pos_set[:, None, :] - pos_set[None, :, :])
    diff = np.where(diff > 0.5 * boxsize, boxsize - diff, diff)
    return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))


def compute_npoint_counts(
    positions, rbins, boxsize, N, labels=None, log_every=10000, chunk_subsets=1_000_000
):
    if N < 2:
        raise ValueError("N must be >= 2")

    n_gal = len(positions)
    nbins = len(rbins) - 1
    rmax = float(rbins[-1])

    T_by_s = np.zeros((N, nbins), dtype=np.int64) if labels is not None else None
    T_total = np.zeros(nbins, dtype=np.int64)

    tree = cKDTree(positions, boxsize=boxsize)
    pair_idx = np.array(list(itertools.combinations(range(N), 2)), dtype=np.int64)

    for i in range(n_gal):
        if log_every and i % log_every == 0:
            print(f"  i={i}/{n_gal}", flush=True)

        raw = tree.query_ball_point(positions[i], rmax)
        nbrs = np.fromiter((j for j in raw if j > i), dtype=np.int64)
        M = len(nbrs)
        if M < N - 1:
            continue

        pos_set = np.empty((M + 1, 3))
        pos_set[0] = positions[i]
        pos_set[1:] = positions[nbrs]
        D = _periodic_dist_matrix(pos_set, boxsize)

        comb_iter = itertools.combinations(range(1, M + 1), N - 1)
        while True:
            chunk = list(itertools.islice(comb_iter, chunk_subsets))
            if not chunk:
                break
            nbr_sub = np.asarray(chunk, dtype=np.int64)
            n_sub = nbr_sub.shape[0]

            subsets = np.zeros((n_sub, N), dtype=np.int64)
            subsets[:, 1:] = nbr_sub
            a = subsets[:, pair_idx[:, 0]]
            b = subsets[:, pair_idx[:, 1]]
            d = D[a, b]
            max_d = d.max(axis=1)
            keep = max_d < rmax
            if not keep.any():
                continue
            max_d = max_d[keep]
            subsets = subsets[keep]

            bins = np.searchsorted(rbins, max_d) - 1
            valid = (bins >= 0) & (bins < nbins)
            if not valid.any():
                continue
            bins = bins[valid]
            subsets = subsets[valid]
            np.add.at(T_total, bins, 1)

            if labels is None:
                continue

            idx_map = np.concatenate(([i], nbrs))
            actual_labels = labels[idx_map[subsets]]
            sorted_labels = np.sort(actual_labels, axis=1)
            n_distinct = 1 + (np.diff(sorted_labels, axis=1) != 0).sum(axis=1)
            for s in range(1, N + 1):
                sub = n_distinct == s
                if sub.any():
                    np.add.at(T_by_s[s - 1], bins[sub], 1)

    return T_by_s, T_total


def sugc_weights_npcf(N, m, k):
    """SUGC scale-down weights for N-point functions: w[s-1] = (m/k)^N * (k)_s / (m)_s."""
    if N < 2:
        raise ValueError("N must be >= 2")
    m, k = float(m), float(k)
    weights = np.zeros(N)
    ratio = 1.0
    for s in range(1, N + 1):
        ratio *= (m - s + 1) / (k - s + 1)
        weights[s - 1] = (m / k) ** N / ratio if ratio != 0 else 0.0
    return weights
