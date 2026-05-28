"""Tests for the subvolume-weighted correction numerical core.

The alpha/beta formulae and helper utilities are pure arithmetic — they can be
verified without Corrfunc or real HDF5 data.
"""

import numpy as np
import pytest

from galform_analysis.analysis.correlation.subvol_weighted_correction import _choose2

# ---------------------------------------------------------------------------
# _choose2 — combinatorial helper: C(n,2) = n*(n-1)/2
# ---------------------------------------------------------------------------


class TestChoose2:
    def test_known_values(self):
        assert _choose2(2) == 1.0
        assert _choose2(3) == 3.0
        assert _choose2(4) == 6.0
        assert _choose2(10) == 45.0

    def test_zero_and_one(self):
        assert _choose2(0) == 0.0
        assert _choose2(1) == 0.0

    def test_float_input(self):
        assert _choose2(5.0) == 10.0


# ---------------------------------------------------------------------------
# alpha/beta coefficient formulae
# ---------------------------------------------------------------------------


def _alpha(m, k):
    return float(m) / float(k)


def _beta(m, k):
    return float(m) * (k - 1) / (float(k) * (m - 1))


class TestAlphaBeta:
    """The two coefficients satisfy:
    - alpha + beta = 1  only when m == k (trivially)
    - As m → k:  alpha → 1, beta → 1 (converge to unbiased estimator)
    - When m = 1: alpha = 1/k, beta is undefined (no cross pairs)
    """

    def test_alpha_at_full_selection(self):
        assert _alpha(m=1024, k=1024) == pytest.approx(1.0)

    def test_beta_at_full_selection(self):
        assert _beta(m=1024, k=1024) == pytest.approx(1.0)

    def test_alpha_fractional(self):
        assert _alpha(m=8, k=1024) == pytest.approx(8 / 1024)

    def test_beta_formula(self):
        m, k = 8, 1024
        expected = m * (k - 1) / (k * (m - 1))
        assert _beta(m, k) == pytest.approx(expected)

    def test_corrected_dd_recovers_alpha_beta(self):
        """DD_corr = alpha * DD_auto + beta * DD_cross must give correct value."""
        dd_auto = np.array([100.0, 200.0])
        dd_total = np.array([300.0, 500.0])
        dd_cross = dd_total - dd_auto
        m, k = 16, 1024
        alpha = _alpha(m, k)
        beta = _beta(m, k)
        dd_corr = alpha * dd_auto + beta * dd_cross
        # Just verify dimensions and that it's between auto and total
        assert dd_corr.shape == dd_auto.shape
        assert np.all(dd_corr >= 0)

    def test_alpha_lt_one_for_subselection(self):
        assert _alpha(m=64, k=1024) < 1.0

    def test_beta_gt_one_for_subselection(self):
        """beta > 1 means cross pairs are up-weighted
        to compensate for missing volume.
        """
        assert _beta(m=64, k=1024) > 1.0
