"""Tests for theoretical halo mass functions and mass-definition conversions.

Pure-numpy helpers are tested everywhere. Tests that call hmf, CAMB, colossus
or SciPy are skipped when the ``science`` extra is not installed.
"""

import importlib.util
import sys

import numpy as np
import pytest

from galform_analysis.analysis.mass_functions import theoretical_hmf as th

requires_hmf = pytest.mark.skipif(
    importlib.util.find_spec("hmf") is None, reason="needs hmf"
)
requires_gps = pytest.mark.skipif(
    any(importlib.util.find_spec(m) is None for m in ("colossus", "scipy", "camb")),
    reason="needs colossus, scipy and camb",
)

# Coarse theory grid keeps the hmf/CAMB calls fast.
GRID = dict(mmin=11.0, mmax=15.0, dlog10m=0.1)


def nfw_mu(x):
    return np.log1p(x) - x / (1.0 + x)


def mean_overdensity(c, y):
    """Mean enclosed density at radius y * r_ref, in units of that at r_ref."""
    return nfw_mu(c * y) / nfw_mu(c) / y**3


def delta_vir_crit(z):
    """Bryan & Norman (1998) virial overdensity relative to critical density."""
    om_z = th._OMEGA_M * (1 + z) ** 3 / (th._OMEGA_M * (1 + z) ** 3 + th._OMEGA_L)
    x = om_z - 1.0
    return 18 * np.pi**2 + 82 * x - 39 * x**2


# ── missing optional dependency ──────────────────────────────────────────────


class TestMissingDependency:
    def test_create_theoretical_hmf_raises_import_error_with_hint(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "hmf", None)  # makes `import hmf` fail
        with pytest.raises(ImportError, match=r"galform_analysis\[science\]"):
            th.create_theoretical_hmf(z=0.0)

    def test_compute_theoretical_hmfs_does_not_hide_missing_hmf(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "hmf", None)
        with pytest.raises(ImportError, match=r"galform_analysis\[science\]"):
            th.compute_theoretical_hmfs(0.0, np.linspace(12, 14, 5))

    def test_gps_plus_raises_import_error_with_hint(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "colossus", None)
        monkeypatch.setitem(sys.modules, "colossus.cosmology", None)
        monkeypatch.setitem(sys.modules, "colossus.cosmology.cosmology", None)
        with pytest.raises(ImportError, match=r"galform_analysis\[science\]"):
            th.create_press_schechter_plus(z=0.0)


# ── concentration and mass-definition conversion (numpy only) ────────────────


class TestConcentration:
    def test_matches_duffy08_m200c_relation(self):
        m = np.array([1e11, 2e12, 1e14])
        expected = 5.71 * (m / 2e12) ** -0.084 * 2.0**-0.47
        np.testing.assert_allclose(th.get_concentration(m, 1.0), expected)

    def test_scalar_in_scalar_out(self):
        c = th.get_concentration(2e12, 0.0)
        assert isinstance(c, float)
        assert c == pytest.approx(5.71)

    def test_clipped_to_realistic_range(self):
        c = th.get_concentration(np.array([1e-10, 1e30]), 0.0)
        np.testing.assert_allclose(c, [20.0, 2.0])


class TestMvirToM200cRatio:
    MASSES = np.logspace(11, 15, 9)

    def test_requires_mass(self):
        with pytest.raises(ValueError, match="mass"):
            th.get_mvir_to_m200c_ratio(0.0)

    @pytest.mark.parametrize("z", [0.0, 0.5, 1.0, 3.0])
    def test_enclosed_density_definitions_are_satisfied(self, z):
        """Independent check: radii implied by the ratio enclose the right density.

        With c = r_200c / r_s, M_vir / M_200c = mu(c y) / mu(c) where
        y = r_vir / r_200c, and the mean density inside r_vir must be
        Delta_vir rho_c. Recover y from the ratio and check that condition.
        """
        ratio = th.get_mvir_to_m200c_ratio(z, self.MASSES)
        c = th.get_concentration(self.MASSES, z)
        ys = np.linspace(0.5, 3.0, 200001)
        for ci, ri in zip(c, ratio):
            y = ys[np.argmin(np.abs(nfw_mu(ci * ys) / nfw_mu(ci) - ri))]
            mean_density = 200.0 * mean_overdensity(ci, y)
            assert mean_density == pytest.approx(delta_vir_crit(z), rel=1e-3)

    @pytest.mark.parametrize("z", [0.0, 1.0, 5.0])
    def test_mvir_exceeds_m200c(self, z):
        # Delta_vir < 200 rho_c at every z, so r_vir > r_200c and M_vir > M_200c.
        ratio = th.get_mvir_to_m200c_ratio(z, self.MASSES)
        assert np.all(ratio > 1.0)

    def test_literature_values_at_z0(self):
        # Mvir/M200c ~ 1.2-1.3 for c200c ~ 4-7 (e.g. White 2001; Hu & Kravtsov 2003)
        ratio = th.get_mvir_to_m200c_ratio(0.0, self.MASSES)
        assert np.all((ratio > 1.15) & (ratio < 1.35))

    def test_ratio_approaches_one_at_high_redshift(self):
        lo_z = th.get_mvir_to_m200c_ratio(0.0, self.MASSES)
        hi_z = th.get_mvir_to_m200c_ratio(4.0, self.MASSES)
        assert np.all(hi_z < lo_z)
        np.testing.assert_allclose(hi_z, 1.0, atol=0.1)

    def test_scalar_in_scalar_out(self):
        assert isinstance(th.get_mvir_to_m200c_ratio(0.0, 1e12), float)


class TestNfwMassRatio:
    def test_same_overdensity_is_identity(self):
        np.testing.assert_allclose(
            th._nfw_mass_ratio(np.array([3.0, 10.0]), 200.0, 200.0), 1.0, rtol=1e-10
        )

    def test_inverse_conversion_round_trips(self):
        """Converting 200 -> 100 and back recovers the original mass."""
        c200 = np.array([4.0, 8.0])
        up = th._nfw_mass_ratio(c200, 200.0, 100.0)
        # Concentration at the new overdensity: c100 = c200 * r100 / r200.
        ys = np.linspace(1.0, 2.0, 100001)
        c100 = []
        for ci, ri in zip(c200, up):
            c100.append(ci * ys[np.argmin(np.abs(nfw_mu(ci * ys) / nfw_mu(ci) - ri))])
        down = th._nfw_mass_ratio(np.array(c100), 100.0, 200.0)
        np.testing.assert_allclose(up * down, 1.0, rtol=1e-4)


class TestConvertMassDefinition:
    def test_constant_ratio_preserves_dn_dlogm(self):
        log10m = np.linspace(10, 15, 51)
        dn = 10 ** (-log10m + 9)
        new_m, new_dn = th._convert_mass_definition(log10m, dn, np.full(51, 1.3))
        np.testing.assert_allclose(new_m, log10m + np.log10(1.3))
        np.testing.assert_allclose(new_dn, dn)

    def test_jacobian_conserves_number_density(self):
        """Integral of dn/dlog10M over the mapped range is unchanged."""
        log10m = np.linspace(10, 15, 2001)
        dn = np.exp(-((log10m - 12.5) ** 2))
        ratio = 10 ** (0.1 * (log10m - 12.5))  # mass-dependent ratio
        new_m, new_dn = th._convert_mass_definition(log10m, dn, ratio)
        assert th._trapezoid(new_dn, new_m) == pytest.approx(
            th._trapezoid(dn, log10m), rel=1e-6
        )

    def test_single_point_grid(self):
        new_m, new_dn = th._convert_mass_definition(
            np.array([12.0]), np.array([1e-3]), np.array([2.0])
        )
        np.testing.assert_allclose(new_m, [12.0 + np.log10(2.0)])
        np.testing.assert_allclose(new_dn, [1e-3])


# ── interpolation ────────────────────────────────────────────────────────────


class TestInterpolateHmfToBins:
    def test_exact_for_power_law(self):
        log10m = np.linspace(10, 15, 101)
        theory = {"log10M": log10m, "dndlog10m": 10 ** (2.0 - 0.9 * log10m)}
        bins = np.array([11.0, 12.0, 13.5, 14.0])
        centers = 0.5 * (bins[1:] + bins[:-1])
        np.testing.assert_allclose(
            th.interpolate_hmf_to_bins(theory, bins), 10 ** (2.0 - 0.9 * centers)
        )

    def test_ignores_non_positive_and_non_finite_points(self):
        log10m = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        dn = np.array([1e-1, np.nan, 1e-3, 0.0, 1e-5])
        out = th.interpolate_hmf_to_bins(
            {"log10M": log10m, "dndlog10m": dn}, bins=np.array([11.5, 12.5])
        )
        # The bin centre 12.0 is a valid grid point; NaN and zero are skipped.
        np.testing.assert_allclose(out, [1e-3])

    def test_too_few_valid_points_gives_nan(self):
        theory = {"log10M": np.array([10.0, 11.0]), "dndlog10m": np.array([1.0, 0.0])}
        out = th.interpolate_hmf_to_bins(theory, np.array([10.0, 10.5, 11.0]))
        assert out.shape == (2,)
        assert np.all(np.isnan(out))


def test_mass_definition_info_documents_galform_mvir():
    info = th.get_mass_definition_info()
    assert "Mvir" in info["GALFORM"]
    assert "NFW" in info["Conversion_Method"]


# ── hmf-backed predictions ───────────────────────────────────────────────────


@requires_hmf
class TestCreateTheoreticalHmf:
    @pytest.fixture(scope="class")
    def m200c(self):
        return th.create_theoretical_hmf(z=0.0, use_mvir=False, **GRID)

    @pytest.fixture(scope="class")
    def mvir(self):
        return th.create_theoretical_hmf(z=0.0, use_mvir=True, **GRID)

    @pytest.fixture(scope="class")
    def native(self):
        """Tinker08 straight from hmf, in its native M200m, L800 cosmology."""
        from astropy.cosmology import FlatLambdaCDM
        from hmf import MassFunction

        return MassFunction(
            z=0.0,
            Mmin=GRID["mmin"],
            Mmax=GRID["mmax"],
            dlog10m=GRID["dlog10m"],
            hmf_model="Tinker08",
            cosmo_model=FlatLambdaCDM(
                H0=100 * th._HUBBLE_H, Om0=th._OMEGA_M, Ob0=th._OMEGA_B, Tcmb0=2.7255
            ),
            sigma_8=th._SIGMA_8,
            n=th._N_S,
            transfer_params={"extrapolate_with_eh": True},
        )

    def test_uses_l800_cosmology(self, native):
        assert native.sigma_8 == pytest.approx(0.8288)
        assert native.cosmo.Om0 == pytest.approx(0.307)
        assert native.cosmo.h == pytest.approx(0.6777)

    def test_m200c_is_native_m200m_shifted(self, m200c, native):
        # Tinker08 is calibrated on M200m; M200c < M200m for every halo.
        assert m200c["native_mass_definition"] == "M200m"
        assert m200c["mass_definition"] == "M200c"
        assert m200c["model"] == "Tinker08"
        assert m200c["h_hubble"] == pytest.approx(0.6777)
        c200m = th._duffy_concentration(native.m, 0.0, th._DUFFY_200M)
        ratio = th._nfw_mass_ratio(c200m, 200.0 * th._omega_m_z(0.0), 200.0)
        assert np.all((ratio > 0.6) & (ratio < 0.9))
        np.testing.assert_allclose(m200c["mass_ratio"], ratio)
        np.testing.assert_allclose(m200c["log10M"], np.log10(native.m * ratio))

    def test_mass_function_decreases_at_high_mass(self, m200c):
        hi = m200c["log10M"] > 12.0
        assert np.all(np.diff(m200c["dndlog10m"][hi]) < 0)

    def test_mvir_lies_between_m200c_and_m200m(self, native, mvir):
        # Delta_vir(z=0) ~ 102 rho_c sits between 200 rho_c and 200 Omega_m rho_c.
        assert mvir["mass_definition"] == "Mvir"
        assert np.all(mvir["mass_ratio"] < 1.0)
        assert np.all(mvir["mass_ratio"] > 0.8)
        np.testing.assert_allclose(
            mvir["log10M"], np.log10(native.m) + np.log10(mvir["mass_ratio"])
        )

    def test_smt_is_native_mvir(self):
        out = th.create_theoretical_hmf(z=0.0, use_mvir=True, model="SMT", **GRID)
        assert out["native_mass_definition"] == "Mvir"
        np.testing.assert_allclose(out["mass_ratio"], 1.0, rtol=1e-6)

    def test_mvir_conversion_conserves_cumulative_abundance(self, m200c, mvir):
        """n(>M200c) at a grid point equals n(>Mvir) at the mapped mass."""

        def n_above(log10m, dn):
            seg = 0.5 * (dn[1:] + dn[:-1]) * np.diff(log10m)
            return np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])

        n_200c = n_above(m200c["log10M"], m200c["dndlog10m"])
        n_vir = n_above(mvir["log10M"], mvir["dndlog10m"])
        np.testing.assert_allclose(n_vir[:-5], n_200c[:-5], rtol=0.02)

    def test_mvir_abundance_exceeds_m200c_at_fixed_mass(self, m200c, mvir):
        # Same haloes, larger masses -> more haloes above any fixed mass.
        bins = np.array([12.0, 13.0, 14.0])
        a = th.interpolate_hmf_to_bins(mvir, bins)
        b = th.interpolate_hmf_to_bins(m200c, bins)
        assert np.all(a > b)

    def test_invalid_parameters_raise_value_error(self):
        with pytest.raises(ValueError, match="Failed to create MassFunction"):
            th.create_theoretical_hmf(z=0.0, mmin=12.0, mmax=13.0, dlog10m="bad")


@requires_hmf
@requires_gps
class TestComputeTheoreticalHmfs:
    def test_all_models_finite_and_ordered(self):
        bins = np.linspace(12.0, 14.5, 6)
        models = th.compute_theoretical_hmfs(0.0, bins, use_mvir=True)
        assert set(models) == {"PS", "SMT", "Tinker08", "GPS+"}
        for name, dn in models.items():
            assert dn.shape == (5,), name
            assert np.all(np.isfinite(dn) & (dn > 0)), name
            assert np.all(np.diff(dn) < 0), name
        # All models agree to within a factor of a few at 1e13 M_sun/h.
        mid = np.array([m[2] for m in models.values()])
        assert mid.max() / mid.min() < 3.0

    def test_failed_model_gives_nan(self, monkeypatch):
        def broken(**kwargs):
            raise ValueError("boom")

        monkeypatch.setattr(th, "create_theoretical_hmf", broken)
        bins = np.linspace(12.0, 14.0, 4)
        models = th.compute_theoretical_hmfs(0.0, bins, include_ps_plus=False)
        assert set(models) == {"PS", "SMT", "Tinker08"}
        assert all(np.all(np.isnan(v)) for v in models.values())

    def test_failed_gps_plus_gives_nan(self, monkeypatch):
        def broken(**kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(th, "create_press_schechter_plus", broken)
        monkeypatch.setattr(
            th,
            "create_theoretical_hmf",
            lambda **kw: {
                "log10M": np.array([10.0, 16.0]),
                "dndlog10m": np.array([1.0, 1e-6]),
            },
        )
        models = th.compute_theoretical_hmfs(0.0, np.array([12.0, 13.0]))
        assert np.all(np.isnan(models["GPS+"]))
        assert np.all(np.isfinite(models["PS"]))


@requires_gps
class TestPressSchechterPlus:
    @pytest.fixture(scope="class")
    def m200b(self):
        return th.create_press_schechter_plus(
            z=0.0, use_mvir=False, mmin=11.0, mmax=15.0, dlog10m=0.25
        )

    def test_m200b_grid_and_units(self, m200b):
        np.testing.assert_allclose(m200b["log10M"], np.arange(11.0, 15.01, 0.25))
        assert m200b["mass_definition"] == "m200b"
        assert m200b["model"] == "GPS+"
        dn = m200b["dndlog10m"]
        assert np.all(np.isfinite(dn) & (dn > 0))
        assert np.all(np.diff(dn) < 0)

    @requires_hmf
    def test_close_to_tinker08_at_intermediate_mass(self, m200b):
        """GPS+ (M200m) agrees with Tinker08 (M200m) to tens of per cent."""
        from hmf import MassFunction

        ref = MassFunction(
            z=0.0,
            Mmin=11.0,
            Mmax=15.0,
            dlog10m=0.25,
            hmf_model="Tinker08",
            mdef_model="SOMean",
            mdef_params={"overdensity": 200},
            transfer_params={"extrapolate_with_eh": True},
        )
        mid = (m200b["log10M"] >= 12.0) & (m200b["log10M"] <= 14.0)
        tinker = 10 ** np.interp(
            m200b["log10M"][mid], np.log10(ref.m), np.log10(ref.dndlog10m)
        )
        ratio = m200b["dndlog10m"][mid] / tinker
        assert np.all((ratio > 0.6) & (ratio < 1.6)), ratio

    def test_mvir_conversion(self, m200b):
        mvir = th.create_press_schechter_plus(
            z=0.0, use_mvir=True, mmin=11.0, mmax=15.0, dlog10m=0.25
        )
        assert mvir["mass_definition"] == "Mvir"
        # Delta_vir rho_c ~ 102 rho_c > 200 rho_m ~ 61 rho_c at z = 0: Mvir < M200m.
        ratio = mvir["ratio_mvir_to_m200b"]
        assert np.all((ratio > 0.8) & (ratio < 1.0))
        np.testing.assert_allclose(mvir["log10M"], m200b["log10M"] + np.log10(ratio))

    def test_unsupported_mass_definition(self):
        with pytest.raises(ValueError, match="mdef"):
            th.create_press_schechter_plus(z=0.0, mdef="mvir", dlog10m=1.0)


@pytest.mark.skipif(
    importlib.util.find_spec("camb") is None, reason="needs camb (science extra)"
)
class TestCambSigma:
    """sigma(R) for GPS+ is computed from CAMB directly, not via colossus."""

    @pytest.fixture(scope="class")
    def sigma(self):
        return th._camb_sigma_function(th._OMEGA_M)

    def test_normalised_to_sigma8(self, sigma):
        assert sigma(np.array([8.0]), 1.0)[0] == pytest.approx(th._SIGMA_8)

    def test_scales_with_growth_and_decreases_with_radius(self, sigma):
        R = np.array([0.5, 2.0, 8.0, 30.0])
        s0 = sigma(R, 1.0)
        assert np.all(np.diff(s0) < 0)
        np.testing.assert_allclose(sigma(R, 0.5), 0.5 * s0)
