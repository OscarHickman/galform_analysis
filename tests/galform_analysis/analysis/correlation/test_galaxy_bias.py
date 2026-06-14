"""Tests for galaxy bias and matter xi computation."""

import numpy as np
import polars as pl
import pytest

from galform_analysis.analysis.correlation.galaxy_bias import (
    avg_galaxy_bias_over_subvolumes,
    compute_galaxy_bias,
)
from galform_analysis.analysis.correlation.matter_xi import compute_matter_xi
from galform_analysis.config import SimulationConfig

pytest.importorskip("camb", reason="camb required for matter xi tests")


@pytest.fixture
def sim():
    return SimulationConfig("L800")


@pytest.fixture
def rbins():
    return np.logspace(0, 1.5, 6)  # coarse bins — fast for tests


@pytest.fixture
def xi_matter(sim, rbins):
    return compute_matter_xi(sim, z=0.0, rbins=rbins)


# ── compute_matter_xi ────────────────────────────────────────────────────────


def test_matter_xi_returns_dataframe(xi_matter):
    assert isinstance(xi_matter, pl.DataFrame)
    assert "r" in xi_matter.columns
    assert "xi" in xi_matter.columns


def test_matter_xi_positive_on_small_scales(xi_matter):
    small = xi_matter.filter(pl.col("r") < 20.0)
    assert (small["xi"] > 0).all()


def test_matter_xi_attrs(xi_matter):
    assert xi_matter.attrs["linear"] is True
    assert xi_matter.attrs["z"] == pytest.approx(0.0)


def test_matter_xi_z_evolution(sim, rbins):
    xi_z0 = compute_matter_xi(sim, z=0.0, rbins=rbins)
    xi_z1 = compute_matter_xi(sim, z=1.0, rbins=rbins)
    # xi_m should be larger at z=0 (growth factor suppression at high z)
    assert (xi_z0["xi"].to_numpy() > xi_z1["xi"].to_numpy()).all()


def test_matter_xi_sigma8_scaling(sim, rbins):
    """Doubling sigma8 should quadruple xi_m (xi proportional to sigma8^2)."""
    sim2 = SimulationConfig("L800")
    sim2.sigma_8 = sim.sigma_8 * 2.0

    xi_ref = compute_matter_xi(sim, z=0.0, rbins=rbins)
    xi_2s8 = compute_matter_xi(sim2, z=0.0, rbins=rbins)

    ratio = xi_2s8["xi"].to_numpy() / xi_ref["xi"].to_numpy()
    np.testing.assert_allclose(ratio, 4.0, rtol=0.02)


# ── compute_galaxy_bias ──────────────────────────────────────────────────────


def _make_xi(r_centers, xi_vals):
    df = pl.DataFrame({"r": r_centers, "xi": xi_vals})
    df.attrs = {}
    return df


def test_bias_equals_one_when_galaxy_equals_matter(xi_matter):
    bias = compute_galaxy_bias(xi_matter, xi_matter)
    np.testing.assert_allclose(bias["bias"].to_numpy(), 1.0, rtol=1e-6)


def test_bias_equals_two_when_galaxy_is_four_times_matter(xi_matter):
    r = xi_matter["r"].to_numpy()
    xi_gal = _make_xi(r, xi_matter["xi"].to_numpy() * 4.0)
    bias = compute_galaxy_bias(xi_gal, xi_matter)
    np.testing.assert_allclose(bias["bias"].to_numpy(), 2.0, rtol=1e-6)


def test_bias_raises_on_mismatched_bins(xi_matter):
    r_other = xi_matter["r"].to_numpy() * 1.01
    xi_other = _make_xi(r_other, xi_matter["xi"].to_numpy())
    with pytest.raises(ValueError, match="Radial bins"):
        compute_galaxy_bias(xi_other, xi_matter)


def test_bias_attrs_method(xi_matter):
    bias = compute_galaxy_bias(xi_matter, xi_matter)
    assert "xi_matter_linear" in bias.attrs["method"]


def test_bias_interpolates_on_matching_bin_boundaries_mismatched_r(xi_matter, rbins):
    r_centers = xi_matter["r"].to_numpy()
    # Create slightly shifted r points simulating ravg
    r_shifted = r_centers * 1.005
    xi_galaxy = _make_xi(r_shifted, xi_matter["xi"].to_numpy() * 4.0)

    # Store the same rbins in attrs
    xi_galaxy.attrs["rbins"] = rbins
    xi_matter.attrs["rbins"] = rbins

    bias = compute_galaxy_bias(xi_galaxy, xi_matter)

    expected_xi_matter_interp = np.interp(
        r_shifted, r_centers, xi_matter["xi"].to_numpy()
    )
    expected_bias = np.sqrt(
        np.abs(xi_galaxy["xi"].to_numpy() / expected_xi_matter_interp)
    )
    np.testing.assert_allclose(bias["bias"].to_numpy(), expected_bias, rtol=1e-6)


# ── avg_galaxy_bias_over_subvolumes ──────────────────────────────────────────


def test_avg_bias_mean_and_std(xi_matter):
    r = xi_matter["r"].to_numpy()
    xi_b1 = _make_xi(r, xi_matter["xi"].to_numpy())  # b=1
    xi_b2 = _make_xi(r, xi_matter["xi"].to_numpy() * 4.0)  # b=2

    avg = avg_galaxy_bias_over_subvolumes([xi_b1, xi_b2], xi_matter)

    np.testing.assert_allclose(avg["bias"].to_numpy(), 1.5, rtol=1e-6)
    assert "bias_std" in avg.columns
