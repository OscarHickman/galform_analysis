"""Tests for the satellite–central cross-correlation.

Corrfunc's cross pair counts are checked against brute-force minimum-image
counts, normalised with the analytic random expectation
n_A n_B V_shell / L^3.
"""

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

from galform_analysis.analysis.correlation.satellite_cross_correlation import (
    _load_galaxy_positions,
    compute_xi_cross_corrfunc,
    satellite_central_cross_correlation,
)
from galform_analysis.config import DEFAULT_RBINS
from tests.conftest import write_galaxy_hdf5

requires_corrfunc = pytest.mark.skipif(
    importlib.util.find_spec("Corrfunc") is None, reason="needs Corrfunc"
)

TRUE_BOXSIZE = (155626.09375 * 1024) ** (1.0 / 3.0)  # (V_ivol * 1024)^(1/3)
N_GALS = 400  # 200 centrals + 200 satellites
COARSE_RBINS = np.array([20.0, 60.0, 120.0, 200.0, 250.0])


def uniform_points(n, boxsize, seed):
    return np.random.default_rng(seed).uniform(0.0, boxsize, (n, 3))


def reference_xi_cross(pos_a, pos_b, boxsize, rbins):
    d = pos_a[:, None, :] - pos_b[None, :, :]
    d -= boxsize * np.round(d / boxsize)
    r = np.sqrt((d**2).sum(axis=-1)).ravel()
    dd = np.histogram(r, rbins)[0]
    shell = 4.0 / 3.0 * np.pi * (rbins[1:] ** 3 - rbins[:-1] ** 3)
    return dd / (len(pos_a) * len(pos_b) * shell / boxsize**3) - 1.0


def read_output(path):
    with h5py.File(path, "r") as f:
        return {k: np.asarray(v) for k, v in f["Output001"].items()}


def positions_of(cat, mask):
    pos = np.column_stack([cat["xgal"], cat["ygal"], cat["zgal"]])
    return pos[mask].astype(np.float64)


def galaxies_file(iz_dir, ivol=0):
    return Path(iz_dir) / f"ivol{ivol}" / "galaxies.hdf5"


@pytest.fixture
def iz_dir(tmp_path):
    iz = tmp_path / "iz155"
    (iz / "ivol0").mkdir(parents=True)
    write_galaxy_hdf5(galaxies_file(iz), n_gals=N_GALS, seed=7)
    return iz


def delete_datasets(path, *names):
    with h5py.File(path, "r+") as f:
        for name in names:
            del f[name]


# ── compute_xi_cross_corrfunc: paths that never reach Corrfunc ───────────────


class TestCrossValidation:
    def test_raises_when_no_bins_below_half_box(self):
        with pytest.raises(ValueError, match="No valid rbins"):
            compute_xi_cross_corrfunc(np.zeros((2, 3)), np.zeros((2, 3)), 0.15)

    @pytest.mark.parametrize("n_a,n_b", [(0, 5), (5, 0)])
    def test_empty_sample_gives_nan_xi_and_zero_pairs(self, n_a, n_b):
        rbins = np.array([1.0, 5.0, 10.0])
        df = compute_xi_cross_corrfunc(
            np.ones((n_a, 3)), np.ones((n_b, 3)), 100.0, rbins=rbins
        )
        assert df.columns == ["r", "xi", "npairs"]
        np.testing.assert_allclose(df["r"].to_numpy(), [3.0, 7.5])
        assert np.all(np.isnan(df["xi"].to_numpy()))
        np.testing.assert_array_equal(df["npairs"].to_numpy(), 0.0)
        assert (df.attrs["n1"], df.attrs["n2"]) == (n_a, n_b)

    def test_edge_at_exactly_half_box_is_dropped(self):
        rbins = np.array([5.0, 10.0, 50.0])
        df = compute_xi_cross_corrfunc(np.ones((0, 3)), np.ones((1, 3)), 100.0, rbins)
        np.testing.assert_array_equal(df.attrs["rbins"], [5.0, 10.0])

    def test_default_rbins(self):
        df = compute_xi_cross_corrfunc(np.ones((0, 3)), np.ones((1, 3)), 1000.0)
        np.testing.assert_array_equal(df.attrs["rbins"], DEFAULT_RBINS)


@requires_corrfunc
class TestCrossValues:
    def test_matches_brute_force_reference(self):
        a = uniform_points(200, 100.0, seed=1)
        b = uniform_points(150, 100.0, seed=2)
        rbins = np.array([2.0, 5.0, 10.0, 20.0, 45.0])

        df = compute_xi_cross_corrfunc(a, b, 100.0, rbins=rbins, nthreads=1)

        np.testing.assert_allclose(
            df["xi"].to_numpy(), reference_xi_cross(a, b, 100.0, rbins), rtol=1e-10
        )
        assert df.attrs["boxsize"] == 100.0
        r = df["r"].to_numpy()
        assert np.all((r > rbins[:-1]) & (r < rbins[1:]))

    def test_is_symmetric_in_the_two_samples(self):
        a = uniform_points(120, 50.0, seed=3)
        b = uniform_points(80, 50.0, seed=4)
        rbins = np.array([1.0, 5.0, 10.0, 20.0])
        ab = compute_xi_cross_corrfunc(a, b, 50.0, rbins=rbins, nthreads=1)
        ba = compute_xi_cross_corrfunc(b, a, 50.0, rbins=rbins, nthreads=1)
        np.testing.assert_allclose(ab["xi"].to_numpy(), ba["xi"].to_numpy())
        np.testing.assert_array_equal(ab["npairs"], ba["npairs"])

    def test_positions_outside_the_box_are_wrapped(self):
        a = uniform_points(100, 50.0, seed=5)
        b = uniform_points(100, 50.0, seed=6)
        rbins = np.array([1.0, 5.0, 10.0, 20.0])
        ref = compute_xi_cross_corrfunc(a, b, 50.0, rbins=rbins, nthreads=1)
        shifted = compute_xi_cross_corrfunc(
            a - 50.0, b + 100.0, 50.0, rbins=rbins, nthreads=1
        )
        np.testing.assert_allclose(
            shifted["xi"].to_numpy(), ref["xi"].to_numpy(), rtol=1e-10
        )

    def test_satellites_around_their_centrals_cluster_strongly(self):
        rng = np.random.default_rng(8)
        centrals = rng.uniform(0.0, 100.0, (100, 3))
        sats = np.mod(centrals + rng.normal(scale=0.3, size=(100, 3)), 100.0)
        rbins = np.array([0.05, 2.0, 20.0, 40.0])

        xi = compute_xi_cross_corrfunc(sats, centrals, 100.0, rbins, nthreads=1)["xi"]

        assert xi[0] > 100.0
        assert abs(xi[2]) < 0.2


# ── _load_galaxy_positions ───────────────────────────────────────────────────


class TestLoadGalaxyPositions:
    def test_selects_centrals_and_satellites(self, iz_dir):
        cat = read_output(galaxies_file(iz_dir))
        for centrals, flag in ((True, 1), (False, 0)):
            pos, z = _load_galaxy_positions(str(iz_dir), 0, select_centrals=centrals)
            np.testing.assert_array_equal(
                pos, positions_of(cat, cat["is_central"] == flag)
            )
            assert z == pytest.approx(0.0)

    def test_mass_cuts(self, iz_dir):
        cat = read_output(galaxies_file(iz_dir))
        mstar = cat["mstars_disk"] + cat["mstars_bulge"]
        mask = (cat["is_central"] == 0) & (mstar >= 4e10) & (cat["mhhalo"] >= 1e11)
        assert 0 < mask.sum() < N_GALS // 2

        pos, _ = _load_galaxy_positions(
            str(iz_dir),
            0,
            select_centrals=False,
            stellar_mass_min=4e10,
            host_halo_mass_min=1e11,
        )

        np.testing.assert_array_equal(pos, positions_of(cat, mask))

    def test_falls_back_to_total_stellar_mass_field(self, iz_dir):
        path = galaxies_file(iz_dir)
        cat = read_output(path)
        mstar = cat["mstars_disk"] + cat["mstars_bulge"]
        delete_datasets(path, "Output001/mstars_disk", "Output001/mstars_bulge")
        with h5py.File(path, "r+") as f:
            f["Output001"].create_dataset("mstars", data=mstar)

        pos, _ = _load_galaxy_positions(
            str(iz_dir), 0, select_centrals=True, stellar_mass_min=4e10
        )

        mask = (cat["is_central"] == 1) & (mstar >= 4e10)
        np.testing.assert_array_equal(pos, positions_of(cat, mask))

    def test_non_finite_positions_are_dropped(self, iz_dir):
        path = galaxies_file(iz_dir)
        x = read_output(path)["xgal"]
        x[0] = np.nan  # galaxy 0 is a central
        with h5py.File(path, "r+") as f:
            f["Output001/xgal"][...] = x
        pos, _ = _load_galaxy_positions(str(iz_dir), 0, select_centrals=True)
        assert len(pos) == N_GALS // 2 - 1

    @pytest.mark.parametrize(
        "deleted,kwargs,error",
        [
            (["Output001/xgal"], {}, KeyError),
            (["Output001/is_central"], {}, KeyError),
            (
                ["Output001/mstars_disk", "Output001/mstars_bulge"],
                {"stellar_mass_min": 1e9},
                KeyError,
            ),
            (["Output001/mhhalo"], {"host_halo_mass_min": 1e11}, KeyError),
            (["Output001"], {}, RuntimeError),
        ],
    )
    def test_missing_fields_raise(self, iz_dir, deleted, kwargs, error):
        delete_datasets(galaxies_file(iz_dir), *deleted)
        with pytest.raises(error):
            _load_galaxy_positions(str(iz_dir), 0, select_centrals=True, **kwargs)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _load_galaxy_positions(str(tmp_path), 0, select_centrals=True)


# ── satellite_central_cross_correlation ──────────────────────────────────────


class TestSatelliteCentralFailures:
    def test_missing_file_returns_none(self, tmp_path):
        assert satellite_central_cross_correlation(str(tmp_path), 0) is None

    def test_missing_is_central_returns_none(self, iz_dir):
        delete_datasets(galaxies_file(iz_dir), "Output001/is_central")
        assert satellite_central_cross_correlation(str(iz_dir), 0) is None

    def test_empty_satellite_sample_returns_none(self, iz_dir):
        res = satellite_central_cross_correlation(
            str(iz_dir), 0, satellite_stellar_mass_min=1e20
        )
        assert res is None


@requires_corrfunc
class TestSatelliteCentralValues:
    def test_matches_reference_in_true_box(self, iz_dir):
        cat = read_output(galaxies_file(iz_dir))
        sats = positions_of(cat, cat["is_central"] == 0)
        cens = positions_of(cat, cat["is_central"] == 1)

        res = satellite_central_cross_correlation(
            str(iz_dir), 0, rbins=COARSE_RBINS, nthreads=1
        )

        np.testing.assert_allclose(
            res["xi"].to_numpy(),
            reference_xi_cross(sats, cens, TRUE_BOXSIZE, COARSE_RBINS),
            rtol=1e-6,
        )
        assert res.attrs["boxsize"] == pytest.approx(TRUE_BOXSIZE, rel=1e-6)
        assert res.attrs["n_sat"] == res.attrs["n1"] == len(sats)
        assert res.attrs["n_cen"] == res.attrs["n2"] == len(cens)
        assert (res.attrs["iz"], res.attrs["ivol"]) == ("iz155", 0)
        assert res.attrs["z"] == pytest.approx(0.0)

    def test_separate_stellar_mass_cuts(self, iz_dir):
        cat = read_output(galaxies_file(iz_dir))
        mstar = cat["mstars_disk"] + cat["mstars_bulge"]
        sats = positions_of(cat, (cat["is_central"] == 0) & (mstar >= 2e10))
        cens = positions_of(cat, (cat["is_central"] == 1) & (mstar >= 5e10))

        res = satellite_central_cross_correlation(
            str(iz_dir),
            0,
            rbins=COARSE_RBINS,
            nthreads=1,
            satellite_stellar_mass_min=2e10,
            central_stellar_mass_min=5e10,
        )

        assert (res.attrs["n_sat"], res.attrs["n_cen"]) == (len(sats), len(cens))
        np.testing.assert_allclose(
            res["xi"].to_numpy(),
            reference_xi_cross(sats, cens, TRUE_BOXSIZE, COARSE_RBINS),
            rtol=1e-6,
        )

    def test_boxsize_override(self, iz_dir):
        cat = read_output(galaxies_file(iz_dir))
        sats = positions_of(cat, cat["is_central"] == 0)
        cens = positions_of(cat, cat["is_central"] == 1)

        res = satellite_central_cross_correlation(
            str(iz_dir), 0, rbins=COARSE_RBINS, nthreads=1, boxsize_override=600.0
        )

        assert res.attrs["boxsize"] == 600.0
        np.testing.assert_allclose(
            res["xi"].to_numpy(),
            reference_xi_cross(sats, cens, 600.0, COARSE_RBINS),
            rtol=1e-10,
        )

    def test_without_volume_falls_back_to_extent_with_warning(self, iz_dir):
        delete_datasets(galaxies_file(iz_dir), "Parameters/volume")
        with pytest.warns(RuntimeWarning, match="position extent"):
            res = satellite_central_cross_correlation(
                str(iz_dir), 0, rbins=COARSE_RBINS, nthreads=1
            )
        assert res.attrs["boxsize"] < TRUE_BOXSIZE
