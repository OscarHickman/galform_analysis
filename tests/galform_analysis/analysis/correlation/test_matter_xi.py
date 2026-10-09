"""Tests for the CAMB linear matter correlation function.

CAMB's xi_m(r) is checked against colossus, which uses an independent
(Eisenstein & Hu) transfer function and Hankel transform, so agreement at the
few-percent level is a genuine cross-check of the normalisation and transform.
"""

import numpy as np
import polars as pl
import pytest

from galform_analysis.analysis.correlation.matter_xi import compute_matter_xi
from galform_analysis.config import SimulationConfig

pytest.importorskip("camb", reason="camb required for matter xi tests")
cosmology = pytest.importorskip("colossus.cosmology.cosmology")

RBINS = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])


@pytest.fixture(scope="module")
def sim():
    return SimulationConfig("L800")


@pytest.fixture(scope="module")
def colossus_cosmo(sim):
    params = dict(
        flat=True,
        H0=sim.h0,
        Om0=sim.omega_m,
        Ob0=sim.omega_b,
        sigma8=sim.sigma_8,
        ns=0.961,
    )
    return cosmology.setCosmology("galform_analysis_test", params=params)


@pytest.fixture(scope="module")
def xi_z0(sim):
    """CAMB is slow (~10 s per call), so compute each redshift once."""
    return compute_matter_xi(sim, z=0.0, rbins=RBINS)


@pytest.fixture(scope="module")
def xi_z1(sim):
    return compute_matter_xi(sim, z=1.0, rbins=list(RBINS))  # list input


def test_returns_bin_centres_and_metadata(xi_z0, sim):
    assert isinstance(xi_z0, pl.DataFrame)
    assert xi_z0.columns == ["r", "xi"]
    np.testing.assert_allclose(xi_z0["r"].to_numpy(), 0.5 * (RBINS[1:] + RBINS[:-1]))
    assert xi_z0.attrs["linear"] is True
    assert xi_z0.attrs["z"] == pytest.approx(0.0)
    assert xi_z0.attrs["sim"] == sim.name
    assert xi_z0.attrs["ns"] == pytest.approx(0.961)
    np.testing.assert_array_equal(xi_z0.attrs["rbins"], RBINS)


@pytest.mark.parametrize("fixture,z", [("xi_z0", 0.0), ("xi_z1", 1.0)])
def test_agrees_with_colossus(request, fixture, z, colossus_cosmo):
    df = request.getfixturevalue(fixture)
    r = df["r"].to_numpy()
    expected = colossus_cosmo.correlationFunction(r, z)
    np.testing.assert_allclose(df["xi"].to_numpy(), expected, rtol=0.04)


def test_redshift_scaling_is_linear_growth_squared(xi_z0, xi_z1, colossus_cosmo):
    ratio = xi_z1["xi"].to_numpy() / xi_z0["xi"].to_numpy()
    growth2 = (colossus_cosmo.growthFactor(1.0) / colossus_cosmo.growthFactor(0.0)) ** 2
    np.testing.assert_allclose(ratio, growth2, rtol=0.01)


def test_sigma8_scaling(sim, xi_z0):
    """Doubling sigma8 should quadruple xi_m (xi proportional to sigma8^2)."""
    sim2 = SimulationConfig("L800")
    sim2.sigma_8 = sim.sigma_8 * 2.0

    xi_2s8 = compute_matter_xi(sim2, z=0.0, rbins=RBINS)

    ratio = xi_2s8["xi"].to_numpy() / xi_z0["xi"].to_numpy()
    np.testing.assert_allclose(ratio, 4.0, rtol=1e-3)
