"""Tests for pure-numpy functions in subvol_weighted_multipoles.

These functions have no external dependencies (no Corrfunc) and can be tested
with synthetic inputs directly.
"""

import numpy as np
import pytest

from analysis.redshift_space_distortions.subvol_weighted_multipoles import (
    _analytic_rr_smu,
    _counts_to_grid_smu,
    _project_rsd_multipoles,
)


# ---------------------------------------------------------------------------
# _analytic_rr_smu
# ---------------------------------------------------------------------------

class TestAnalyticRRSmu:
    def test_output_shape(self):
        s_bins = np.linspace(0.0, 50.0, 11)   # 10 s-bins
        rr = _analytic_rr_smu(s_bins, mu_max=1.0, n_mu_bins=5, boxsize=100.0, n_points=1000)
        assert rr.shape == (10, 5)

    def test_integrates_to_shell_fraction(self):
        """Summing over all mu bins should give the analytic shell volume fraction."""
        s_bins = np.array([1.0, 2.0, 3.0])
        boxsize = 100.0
        n_mu = 4
        rr = _analytic_rr_smu(s_bins, mu_max=1.0, n_mu_bins=n_mu, boxsize=boxsize, n_points=1)
        shell_vols = (4.0 / 3.0) * np.pi * (s_bins[1:] ** 3 - s_bins[:-1] ** 3)
        expected_shell_fractions = shell_vols / boxsize ** 3
        np.testing.assert_allclose(rr.sum(axis=1), expected_shell_fractions, rtol=1e-10)

    def test_uniform_mu_bins(self):
        """All mu bins in a shell should carry equal weight."""
        s_bins = np.array([1.0, 2.0])
        rr = _analytic_rr_smu(s_bins, mu_max=1.0, n_mu_bins=4, boxsize=100.0, n_points=1)
        # Each mu bin = rr[0, :] / 4; all equal
        np.testing.assert_allclose(rr[0], rr[0, 0], rtol=1e-10)

    def test_invalid_mu_max_raises(self):
        s_bins = np.array([0.0, 1.0])
        with pytest.raises(ValueError):
            _analytic_rr_smu(s_bins, mu_max=0.0, n_mu_bins=4, boxsize=100.0, n_points=1)

    def test_invalid_s_bins_raises(self):
        with pytest.raises(ValueError):
            _analytic_rr_smu(np.array([1.0]), mu_max=1.0, n_mu_bins=4, boxsize=100.0, n_points=1)


# ---------------------------------------------------------------------------
# _counts_to_grid_smu
# ---------------------------------------------------------------------------

class TestCountsToGridSmu:
    def _make_fake_result(self, n_s, n_mu):
        counts = np.arange(n_s * n_mu, dtype=np.float64)
        return {"npairs": counts}

    def test_correct_shape(self):
        n_s, n_mu = 5, 4
        result = self._make_fake_result(n_s, n_mu)
        grid = _counts_to_grid_smu(result, n_s, n_mu)
        assert grid.shape == (n_s, n_mu)

    def test_values_preserved(self):
        n_s, n_mu = 3, 2
        counts = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        grid = _counts_to_grid_smu({"npairs": counts}, n_s, n_mu)
        np.testing.assert_array_equal(grid.ravel(), counts)

    def test_wrong_size_raises(self):
        result = {"npairs": np.ones(5)}
        with pytest.raises(RuntimeError):
            _counts_to_grid_smu(result, n_s_bins=3, n_mu_bins=4)


# ---------------------------------------------------------------------------
# _project_rsd_multipoles
# ---------------------------------------------------------------------------

class TestProjectRsdMultipoles:
    def test_output_shapes(self):
        n_s, n_mu = 5, 10
        xi_grid = np.ones((n_s, n_mu))
        s_bins = np.linspace(0.0, 50.0, n_s + 1)
        s_mid, xi0, xi2 = _project_rsd_multipoles(xi_grid, mu_max=1.0, n_mu_bins=n_mu, s_bins=s_bins)
        assert s_mid.shape == (n_s,)
        assert xi0.shape == (n_s,)
        assert xi2.shape == (n_s,)

    def test_monopole_of_isotropic_field(self):
        """For xi(s,mu)=C, xi0 = C * integral(1 * dmu, 0..1) = C."""
        n_s, n_mu = 4, 100
        C = 2.5
        xi_grid = np.full((n_s, n_mu), C)
        s_bins = np.linspace(1.0, 5.0, n_s + 1)
        _, xi0, _ = _project_rsd_multipoles(xi_grid, mu_max=1.0, n_mu_bins=n_mu, s_bins=s_bins)
        # xi0 = sum(C * 1 * dmu) = C * sum(dmu) = C * 1.0
        np.testing.assert_allclose(xi0, C, rtol=1e-2)

    def test_quadrupole_of_isotropic_field(self):
        """For xi(s,mu)=C (isotropic), xi2 must vanish (P2 integrates to zero)."""
        n_s, n_mu = 4, 200
        xi_grid = np.full((n_s, n_mu), 3.0)
        s_bins = np.linspace(1.0, 5.0, n_s + 1)
        _, _, xi2 = _project_rsd_multipoles(xi_grid, mu_max=1.0, n_mu_bins=n_mu, s_bins=s_bins)
        np.testing.assert_allclose(xi2, 0.0, atol=1e-2)

    def test_s_mid_is_bin_centers(self):
        n_s, n_mu = 3, 5
        s_bins = np.array([0.0, 1.0, 2.0, 3.0])
        xi_grid = np.ones((n_s, n_mu))
        s_mid, _, _ = _project_rsd_multipoles(xi_grid, mu_max=1.0, n_mu_bins=n_mu, s_bins=s_bins)
        expected = np.array([0.5, 1.5, 2.5])
        np.testing.assert_allclose(s_mid, expected)
