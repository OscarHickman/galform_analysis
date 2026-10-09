"""Tests for galaxy bias b(r) = sqrt(xi_gal / xi_m).

These use synthetic xi tables, so they need no optional dependency.
"""

import numpy as np
import polars as pl
import pytest

from galform_analysis.analysis.correlation.galaxy_bias import (
    avg_galaxy_bias_over_subvolumes,
    compute_galaxy_bias,
)

RBINS = np.logspace(0, 1.5, 6)
R_CENTRES = 0.5 * (RBINS[1:] + RBINS[:-1])
XI_M = 2.0 * R_CENTRES**-1.8  # power-law stand-in for the matter xi


def make_xi(r, xi, rbins=None):
    df = pl.DataFrame({"r": np.asarray(r, float), "xi": np.asarray(xi, float)})
    df.attrs = {} if rbins is None else {"rbins": rbins}
    return df


@pytest.fixture
def xi_matter():
    return make_xi(R_CENTRES, XI_M, rbins=RBINS)


# ── compute_galaxy_bias ──────────────────────────────────────────────────────


def test_bias_equals_one_when_galaxy_equals_matter(xi_matter):
    bias = compute_galaxy_bias(xi_matter, xi_matter)
    np.testing.assert_allclose(bias["bias"].to_numpy(), 1.0)
    np.testing.assert_array_equal(bias["r"].to_numpy(), R_CENTRES)


@pytest.mark.parametrize("b", [0.8, 1.5, 2.0, 3.3])
def test_bias_recovers_a_known_linear_bias(xi_matter, b):
    xi_gal = make_xi(R_CENTRES, b**2 * XI_M)
    bias = compute_galaxy_bias(xi_gal, xi_matter)
    np.testing.assert_allclose(bias["bias"].to_numpy(), b, rtol=1e-12)


def test_scale_dependent_bias_is_recovered_bin_by_bin(xi_matter):
    b_r = np.linspace(1.0, 2.0, len(R_CENTRES))
    bias = compute_galaxy_bias(make_xi(R_CENTRES, b_r**2 * XI_M), xi_matter)
    np.testing.assert_allclose(bias["bias"].to_numpy(), b_r, rtol=1e-12)


def test_negative_ratio_gives_magnitude(xi_matter):
    bias = compute_galaxy_bias(make_xi(R_CENTRES, -4.0 * XI_M), xi_matter)
    np.testing.assert_allclose(bias["bias"].to_numpy(), 2.0)


def test_raises_on_mismatched_r_without_rbins(xi_matter):
    xi_matter.attrs = {}
    with pytest.raises(ValueError, match="Radial bins"):
        compute_galaxy_bias(make_xi(R_CENTRES * 1.01, XI_M), xi_matter)


def test_raises_on_different_number_of_bins(xi_matter):
    xi_gal = make_xi(R_CENTRES[:-1], XI_M[:-1], rbins=RBINS[:-1])
    with pytest.raises(ValueError, match="Radial bins"):
        compute_galaxy_bias(xi_gal, xi_matter)


def test_raises_on_different_number_of_r_without_rbins():
    with pytest.raises(ValueError, match="Radial bins"):
        compute_galaxy_bias(
            make_xi(R_CENTRES[:-1], XI_M[:-1]), make_xi(R_CENTRES, XI_M)
        )


def test_raises_on_mismatched_rbins_attrs(xi_matter):
    xi_gal = make_xi(R_CENTRES, XI_M, rbins=RBINS * 1.1)
    with pytest.raises(ValueError, match="Radial bins"):
        compute_galaxy_bias(xi_gal, xi_matter)


def test_matching_rbins_with_shifted_r_interpolates_matter(xi_matter):
    """Galaxy r is a pair-weighted mean (ravg), matter r is the bin centre."""
    r_avg = R_CENTRES * 1.02
    xi_gal = make_xi(r_avg, 4.0 * XI_M, rbins=RBINS)

    bias = compute_galaxy_bias(xi_gal, xi_matter)

    xi_m_at_ravg = np.interp(r_avg, R_CENTRES, XI_M)
    np.testing.assert_allclose(
        bias["bias"].to_numpy(), np.sqrt(4.0 * XI_M / xi_m_at_ravg), rtol=1e-12
    )
    np.testing.assert_array_equal(bias["r"].to_numpy(), r_avg)


def test_attrs_record_inputs(xi_matter):
    xi_gal = make_xi(R_CENTRES, XI_M, rbins=RBINS)
    xi_gal.attrs["ngal"] = 123
    bias = compute_galaxy_bias(xi_gal, xi_matter)
    assert "xi_matter_linear" in bias.attrs["method"]
    assert bias.attrs["xi_galaxy_metadata"]["ngal"] == 123
    assert bias.attrs["xi_matter_metadata"] is xi_matter.attrs


# ── avg_galaxy_bias_over_subvolumes ──────────────────────────────────────────


def test_avg_bias_mean_and_std(xi_matter):
    xi_list = [make_xi(R_CENTRES, b**2 * XI_M) for b in (1.0, 2.0, 4.0)]

    avg = avg_galaxy_bias_over_subvolumes(xi_list, xi_matter)

    assert avg.columns == ["r", "bias", "bias_std"]
    np.testing.assert_allclose(avg["bias"].to_numpy(), 7.0 / 3.0)
    np.testing.assert_allclose(avg["bias_std"].to_numpy(), np.std([1.0, 2.0, 4.0]))
    np.testing.assert_array_equal(avg["r"].to_numpy(), R_CENTRES)


def test_avg_bias_rejects_empty_list(xi_matter):
    with pytest.raises(ValueError, match="empty"):
        avg_galaxy_bias_over_subvolumes([], xi_matter)
