"""Corrfunc-backed RSD multipole estimators checked against brute force.

The line of sight is the z axis and mu = |Delta z| / s. Corrfunc ``DDsmu``
auto counts are ordered (every pair counted twice) and are normalised by
n (n - 1); cross counts are normalised by n_d n_r. Multipoles are checked
against an independent Legendre projection built from ``numpy.polynomial``.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from numpy.polynomial import legendre

from galform_analysis.analysis.redshift_space_distortions.subvol_weighted_multipoles import (  # noqa: E501
    _paircounts_smu_auto,
    _paircounts_smu_cross,
    _project_rsd_multipoles,
    compute_direct_rsd_multipoles,
    compute_standard_rsd_multipoles,
    compute_weighted_direct_rsd_multipoles,
    compute_weighted_rsd_multipoles,
)

requires_corrfunc = pytest.mark.skipif(
    importlib.util.find_spec("Corrfunc") is None,
    reason="needs Corrfunc (optional 'clustering' extra)",
)

BOX = 100.0
S_BINS = np.array([4.0, 10.0, 20.0, 30.0])
N_MU = 5
MU_MAX = 1.0


# ── independent references ───────────────────────────────────────────────────


def _points(n, seed):
    return np.random.default_rng(seed).uniform(0.0, BOX, size=(n, 3))


def _deltas(a, b=None):
    """Minimum-image separations: unique pairs if b is None, else all (a, b)."""
    if b is None:
        i, j = np.triu_indices(len(a), k=1)
        d = a[i] - a[j]
    else:
        d = (a[:, None, :] - b[None, :, :]).reshape(-1, 3)
    return d - BOX * np.round(d / BOX)


def _hist_smu(d):
    s = np.sqrt(np.sum(d**2, axis=1))
    mu = np.abs(d[:, 2]) / s
    mu_bins = np.linspace(0.0, MU_MAX, N_MU + 1)
    return np.histogram2d(s, mu, bins=[S_BINS, mu_bins])[0]


def _legendre_multipole(xi_grid, ell):
    mu_edges = np.linspace(0.0, MU_MAX, N_MU + 1)
    mu = 0.5 * (mu_edges[1:] + mu_edges[:-1])
    coeffs = np.zeros(ell + 1)
    coeffs[ell] = 1.0
    weights = legendre.legval(mu, coeffs) * (MU_MAX / N_MU)
    return (2 * ell + 1) * np.nansum(xi_grid * weights, axis=1)


def _ls(dd_unique, dr, rr_unique, nd, nr):
    with np.errstate(divide="ignore", invalid="ignore"):
        rr_n = rr_unique / (0.5 * nr * (nr - 1))
        xi = (dd_unique / (0.5 * nd * (nd - 1)) - 2.0 * dr / (nd * nr) + rr_n) / rr_n
    xi[~np.isfinite(xi)] = np.nan
    return xi


def _natural_analytic(dd_unique, nd):
    shell = 4.0 / 3.0 * np.pi * (S_BINS[1:] ** 3 - S_BINS[:-1] ** 3)
    rr = (0.5 * nd * (nd - 1)) * shell[:, None] / BOX**3 / N_MU
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = dd_unique / rr - 1.0
    xi[~np.isfinite(xi)] = np.nan
    return xi


def _auto_cross_split(pos, labels):
    auto = sum(
        _hist_smu(_deltas(pos[labels == lab]))
        for lab in np.unique(labels)
        if np.count_nonzero(labels == lab) > 1
    )
    total = _hist_smu(_deltas(pos))
    return auto, total - auto


def _alpha_beta(m, k):
    return m / k, m * (k - 1) / (k * (m - 1))


# ── projection (no Corrfunc) ────────────────────────────────────────────────


class TestProjectionRecoversLegendreModes:
    @pytest.mark.parametrize("ell", [0, 2, 4])
    def test_pure_legendre_mode(self, ell):
        n_mu = 2000
        mu_edges = np.linspace(0.0, 1.0, n_mu + 1)
        mu = 0.5 * (mu_edges[1:] + mu_edges[:-1])
        coeffs = np.zeros(ell + 1)
        coeffs[ell] = 1.0
        grid = np.tile(legendre.legval(mu, coeffs), (2, 1))

        _, xi0, xi2, xi4 = _project_rsd_multipoles(grid, 1.0, n_mu, np.arange(3.0))

        expected = {0: (1, 0, 0), 2: (0, 1, 0), 4: (0, 0, 1)}[ell]
        np.testing.assert_allclose(xi0, expected[0], atol=1e-5)
        np.testing.assert_allclose(xi2, expected[1], atol=1e-5)
        np.testing.assert_allclose(xi4, expected[2], atol=1e-5)

    def test_kaiser_linear_rsd_multipoles(self):
        # xi(s, mu) = xi0 + xi2 P2(mu) + xi4 P4(mu) round-trips exactly
        # (to midpoint-rule accuracy) for any amplitudes.
        n_mu = 4000
        mu_edges = np.linspace(0.0, 1.0, n_mu + 1)
        mu = 0.5 * (mu_edges[1:] + mu_edges[:-1])
        amps = (0.7, -0.4, 0.05)
        grid = legendre.legval(mu, [amps[0], 0.0, amps[1], 0.0, amps[2]])[None, :]
        _, xi0, xi2, xi4 = _project_rsd_multipoles(grid, 1.0, n_mu, np.array([1, 2]))
        np.testing.assert_allclose([xi0[0], xi2[0], xi4[0]], amps, atol=1e-5)


# ── pair counts ──────────────────────────────────────────────────────────────


@requires_corrfunc
class TestSmuPairCounts:
    def test_auto_counts_every_pair_twice(self):
        pos = _points(150, 1)
        out = _paircounts_smu_auto(pos, S_BINS, MU_MAX, N_MU, BOX, 1)
        ref = _hist_smu(_deltas(pos))
        assert ref.sum() > 0
        np.testing.assert_array_equal(out, 2.0 * ref)

    def test_cross_counts_every_pair_once(self):
        a, b = _points(80, 2), _points(70, 3)
        out = _paircounts_smu_cross(a, b, S_BINS, MU_MAX, N_MU, BOX, 1)
        np.testing.assert_array_equal(out, _hist_smu(_deltas(a, b)))

    def test_early_returns(self):
        shape = (len(S_BINS) - 1, N_MU)
        auto = _paircounts_smu_auto(np.zeros((1, 3)), S_BINS, MU_MAX, N_MU, BOX, 1)
        cross = _paircounts_smu_cross(
            np.zeros((0, 3)), _points(3, 0), S_BINS, MU_MAX, N_MU, BOX, 1
        )
        np.testing.assert_array_equal(auto, np.zeros(shape))
        np.testing.assert_array_equal(cross, np.zeros(shape))


# ── estimators ───────────────────────────────────────────────────────────────


@requires_corrfunc
class TestStandardRsdMultipoles:
    def test_matches_brute_force(self):
        gal, rnd = _points(120, 4), _points(200, 5)
        out = compute_standard_rsd_multipoles(
            gal, rnd, S_BINS, mu_max=MU_MAX, n_mu_bins=N_MU, boxsize=BOX, nthreads=1
        )
        xi = _ls(
            _hist_smu(_deltas(gal)),
            _hist_smu(_deltas(gal, rnd)),
            _hist_smu(_deltas(rnd)),
            len(gal),
            len(rnd),
        )
        np.testing.assert_allclose(out["xi_grid"], xi, rtol=1e-12)
        for ell in (0, 2, 4):
            np.testing.assert_allclose(
                out[f"xi{ell}"], _legendre_multipole(xi, ell), rtol=1e-10, atol=1e-12
            )
        np.testing.assert_allclose(out["s"], 0.5 * (S_BINS[1:] + S_BINS[:-1]))
        assert (out["ngal"], out["nrandom"]) == (120, 200)


@requires_corrfunc
class TestDirectRsdMultipoles:
    def test_matches_brute_force_natural_estimator(self):
        gal = _points(150, 6)
        out = compute_direct_rsd_multipoles(
            gal, S_BINS, mu_max=MU_MAX, n_mu_bins=N_MU, boxsize=BOX, nthreads=1
        )
        xi = _natural_analytic(_hist_smu(_deltas(gal)), len(gal))
        np.testing.assert_allclose(out["xi_grid"], xi, rtol=1e-12)
        for ell in (0, 2, 4):
            np.testing.assert_allclose(
                out[f"xi{ell}"], _legendre_multipole(xi, ell), rtol=1e-10, atol=1e-12
            )
        assert out["nrandom"] == 0

    def test_uniform_field_has_vanishing_multipoles(self):
        out = compute_direct_rsd_multipoles(
            _points(3000, 7), S_BINS, n_mu_bins=N_MU, boxsize=BOX, nthreads=1
        )
        np.testing.assert_allclose(out["xi0"], 0.0, atol=0.03)
        np.testing.assert_allclose(out["xi2"], 0.0, atol=0.1)

    def test_uniform_field_with_partial_mu_range(self):
        out = compute_direct_rsd_multipoles(
            _points(3000, 7),
            S_BINS,
            mu_max=0.5,
            n_mu_bins=N_MU,
            boxsize=BOX,
            nthreads=1,
        )
        np.testing.assert_allclose(out["xi_grid"], 0.0, atol=0.15)


@requires_corrfunc
class TestWeightedRsdMultipoles:
    m, k = 3, 10

    def _catalogue(self, seed):
        pos = _points(40 * self.m, seed)
        labels = np.repeat(np.arange(self.m), 40)
        return pos, labels

    def test_weighted_ls_matches_brute_force(self):
        pos, labels = self._catalogue(8)
        rnd = _points(150, 9)
        out = compute_weighted_rsd_multipoles(
            pos,
            labels,
            rnd,
            S_BINS,
            mu_max=MU_MAX,
            n_mu_bins=N_MU,
            k_total=self.k,
            boxsize=BOX,
            nthreads=1,
        )
        auto, cross = _auto_cross_split(pos, labels)
        dr, rr = _hist_smu(_deltas(pos, rnd)), _hist_smu(_deltas(rnd))
        alpha, beta = _alpha_beta(self.m, self.k)
        nd, nr = len(pos), len(rnd)
        xi_std = _ls(auto + cross, dr, rr, nd, nr)
        xi_corr = _ls(alpha * auto + beta * cross, dr, rr, nd, nr)

        np.testing.assert_allclose(out["xi_standard_grid"], xi_std, rtol=1e-12)
        np.testing.assert_allclose(out["xi_corrected_grid"], xi_corr, rtol=1e-12)
        for ell in (0, 2, 4):
            np.testing.assert_allclose(
                out[f"xi{ell}_standard"], _legendre_multipole(xi_std, ell), atol=1e-12
            )
            np.testing.assert_allclose(
                out[f"xi{ell}_corrected"], _legendre_multipole(xi_corr, ell), atol=1e-12
            )
        assert (out["alpha"], out["beta"]) == pytest.approx((alpha, beta))
        assert (out["m_selected"], out["k_total"]) == (self.m, self.k)

    def test_weighted_direct_matches_brute_force(self):
        pos, labels = self._catalogue(10)
        out = compute_weighted_direct_rsd_multipoles(
            pos,
            labels,
            S_BINS,
            mu_max=MU_MAX,
            n_mu_bins=N_MU,
            k_total=self.k,
            boxsize=BOX,
            nthreads=1,
        )
        auto, cross = _auto_cross_split(pos, labels)
        alpha, beta = _alpha_beta(self.m, self.k)
        xi_naive = _natural_analytic(auto + cross, len(pos))
        xi_corr = _natural_analytic(alpha * auto + beta * cross, len(pos))

        np.testing.assert_allclose(out["xi_grid"], xi_corr, rtol=1e-12)
        for ell in (0, 2, 4):
            np.testing.assert_allclose(
                out[f"xi{ell}"], _legendre_multipole(xi_corr, ell), atol=1e-12
            )
            np.testing.assert_allclose(
                out[f"xi{ell}_naive"], _legendre_multipole(xi_naive, ell), atol=1e-12
            )
        assert (out["ngal"], out["nrandom"]) == (len(pos), 0)

    def test_all_subvolumes_selected_needs_no_correction(self):
        pos, labels = self._catalogue(11)
        out = compute_weighted_direct_rsd_multipoles(
            pos, labels, S_BINS, n_mu_bins=N_MU, k_total=self.m, boxsize=BOX, nthreads=1
        )
        assert out["alpha"] == out["beta"] == 1.0
        np.testing.assert_allclose(out["xi0"], out["xi0_naive"])
        np.testing.assert_allclose(out["xi2"], out["xi2_naive"])

    @pytest.mark.parametrize("direct", [True, False])
    def test_single_label_gives_nan_correction(self, direct):
        pos = _points(80, 12)
        labels = np.zeros(80, dtype=int)
        if direct:
            out = compute_weighted_direct_rsd_multipoles(
                pos, labels, S_BINS, n_mu_bins=N_MU, k_total=8, boxsize=BOX, nthreads=1
            )
            grid = out["xi_grid"]
        else:
            out = compute_weighted_rsd_multipoles(
                pos,
                labels,
                _points(100, 13),
                S_BINS,
                n_mu_bins=N_MU,
                k_total=8,
                boxsize=BOX,
                nthreads=1,
            )
            grid = out["xi_corrected_grid"]
            assert np.all(np.isfinite(out["xi_standard_grid"]))
        assert out["m_selected"] == 1
        assert out["alpha"] == pytest.approx(1 / 8)
        assert np.isnan(out["beta"])
        assert np.all(np.isnan(grid))
