"""Tests for the periodic-box galaxy/halo 2PCF helpers in correlation.py.

Every xi(r) produced by Corrfunc is checked against an independent numpy
reference: brute-force minimum-image pair counts and the analytic random
pair count RR = N(N-1)/2 * V_shell / L^3.
"""

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import polars as pl
import pytest

import galform_analysis.config as config
from galform_analysis.analysis.correlation.correlation import (
    _wrap_into_box,
    avg_correlation_given_redshift_and_subvolumes,
    avg_correlation_given_subvolume_and_redshifts,
    compute_xi_corrfunc,
    correlation_given_redshift_and_subvolume,
    correlations_given_redshifts_and_subvolume,
    halo_correlation_given_redshift_and_subvolume,
)
from galform_analysis.config import DEFAULT_RBINS
from tests.conftest import write_galaxy_hdf5

requires_corrfunc = pytest.mark.skipif(
    importlib.util.find_spec("Corrfunc") is None, reason="needs Corrfunc"
)

# Full L800 box encoded in the mock files: (V_ivol * 1024)^(1/3) ~ 542.16 Mpc/h
TRUE_BOXSIZE = (155626.09375 * 1024) ** (1.0 / 3.0)
N_GALS = 400  # 200 centrals + 200 satellites per mock subvolume
COARSE_RBINS = np.array([20.0, 60.0, 120.0, 200.0, 250.0])  # all < L/2


# ── independent reference implementation ─────────────────────────────────────


def pair_separations(pos_a, pos_b, boxsize, auto):
    """Minimum-image separations between two point sets in a periodic cube."""
    d = pos_a[:, None, :] - pos_b[None, :, :]
    d -= boxsize * np.round(d / boxsize)
    r = np.sqrt((d**2).sum(axis=-1))
    if auto:
        return r[np.triu_indices(len(pos_a), k=1)]
    return r.ravel()


def shell_volumes(rbins):
    return 4.0 / 3.0 * np.pi * (rbins[1:] ** 3 - rbins[:-1] ** 3)


def reference_xi_auto(pos, boxsize, rbins):
    """Natural estimator DD/RR - 1 with analytic RR for unordered pairs."""
    n = len(pos)
    dd = np.histogram(pair_separations(pos, pos, boxsize, auto=True), rbins)[0]
    rr = 0.5 * n * (n - 1) * shell_volumes(rbins) / boxsize**3
    return dd / rr - 1.0


# ── mock catalogue helpers ───────────────────────────────────────────────────


def read_output(path):
    with h5py.File(path, "r") as f:
        g = f["Output001"]
        return {k: np.asarray(g[k]) for k in g}


def positions_of(cat, mask):
    return np.column_stack([cat["xgal"], cat["ygal"], cat["zgal"]])[mask].astype(
        np.float64
    )


def extent_boxsize(pos):
    """Fallback box size used when a file has no Parameters/volume."""
    return float(np.max(np.ptp(pos, axis=0)))


def overwrite_output(path, **datasets):
    with h5py.File(path, "r+") as f:
        g = f["Output001"]
        for name, data in datasets.items():
            if name in g:
                del g[name]
            g.create_dataset(name, data=data)


def make_snapshot(base, iz_num, ivols, n_gals=N_GALS):
    iz_dir = Path(base) / f"iz{iz_num}"
    for ivol in ivols:
        ivol_dir = iz_dir / f"ivol{ivol}"
        ivol_dir.mkdir(parents=True)
        write_galaxy_hdf5(ivol_dir / "galaxies.hdf5", n_gals=n_gals, seed=iz_num + ivol)
    return iz_dir


def galaxies_file(iz_dir, ivol):
    return Path(iz_dir) / f"ivol{ivol}" / "galaxies.hdf5"


@pytest.fixture
def iz_dir(tmp_path):
    return make_snapshot(tmp_path, 155, ivols=(0, 1))


@pytest.fixture
def base_dir(tmp_path, monkeypatch):
    """Two snapshots x two ivols; also installed as the configured base dir."""
    make_snapshot(tmp_path, 155, ivols=(0, 1))
    make_snapshot(tmp_path, 207, ivols=(0, 1))
    monkeypatch.setattr(config, "BASE_DIR", config.BASE_DIR)  # restored on teardown
    config.set_base_dir(str(tmp_path))
    return tmp_path


def uniform_points(n, boxsize, seed):
    return np.random.default_rng(seed).uniform(0.0, boxsize, (n, 3))


# ── compute_xi_corrfunc: paths that never reach Corrfunc ─────────────────────


class TestComputeXiCorrfuncValidation:
    def test_raises_when_no_bin_edge_within_half_box(self):
        with pytest.raises(ValueError, match="No valid rbins"):
            compute_xi_corrfunc(uniform_points(10, 0.15, 0), boxsize=0.15)

    def test_raises_when_only_one_edge_survives(self):
        rbins = np.array([1.0, 60.0, 80.0])
        with pytest.raises(ValueError, match="No valid rbins"):
            compute_xi_corrfunc(uniform_points(10, 100.0, 0), 100.0, rbins=rbins)

    @pytest.mark.parametrize("n", [0, 1])
    def test_fewer_than_two_points_gives_nan_xi(self, n):
        rbins = np.array([1.0, 5.0, 10.0])
        df = compute_xi_corrfunc(uniform_points(n, 100.0, 0), 100.0, rbins=rbins)
        np.testing.assert_allclose(df["r"].to_numpy(), [3.0, 7.5])
        assert np.all(np.isnan(df["xi"].to_numpy()))
        assert df.attrs["ngal"] == n

    def test_default_rbins_used_when_none(self):
        df = compute_xi_corrfunc(np.zeros((1, 3)), boxsize=TRUE_BOXSIZE)
        assert len(df) == len(DEFAULT_RBINS) - 1
        np.testing.assert_array_equal(df.attrs["rbins"], DEFAULT_RBINS)

    def test_bins_beyond_half_box_are_trimmed(self):
        rbins = np.array([1.0, 10.0, 40.0, 60.0, 80.0])
        df = compute_xi_corrfunc(np.zeros((1, 3)), boxsize=100.0, rbins=rbins)
        np.testing.assert_array_equal(df.attrs["rbins"], [1.0, 10.0, 40.0])
        assert len(df) == 2


# ── compute_xi_corrfunc: numerical correctness ───────────────────────────────


@requires_corrfunc
class TestComputeXiCorrfuncValues:
    def test_matches_brute_force_reference(self):
        pos = uniform_points(300, 100.0, seed=1)
        rbins = np.array([2.0, 5.0, 10.0, 20.0, 30.0, 45.0])

        df = compute_xi_corrfunc(pos, boxsize=100.0, rbins=rbins, nthreads=1)

        expected = reference_xi_auto(pos, 100.0, rbins)
        np.testing.assert_allclose(df["xi"].to_numpy(), expected, rtol=1e-10)
        assert df.columns == ["r", "xi"]
        assert df.attrs["ngal"] == 300

    def test_ravg_lies_inside_each_bin(self):
        pos = uniform_points(300, 100.0, seed=2)
        rbins = np.array([5.0, 10.0, 20.0, 30.0])
        r = compute_xi_corrfunc(pos, 100.0, rbins=rbins, nthreads=1)["r"].to_numpy()
        assert np.all((r > rbins[:-1]) & (r < rbins[1:]))

    def test_uniform_random_points_give_xi_near_zero(self):
        pos = uniform_points(3000, 100.0, seed=3)
        rbins = np.array([5.0, 10.0, 20.0, 30.0])
        df = compute_xi_corrfunc(pos, boxsize=100.0, rbins=rbins, nthreads=1)
        np.testing.assert_allclose(df["xi"].to_numpy(), 0.0, atol=0.05)

    def test_close_pairs_give_strong_small_scale_clustering(self):
        rng = np.random.default_rng(4)
        parents = rng.uniform(0.0, 100.0, (200, 3))
        offsets = rng.normal(size=(200, 3))
        offsets *= 0.05 / np.linalg.norm(offsets, axis=1, keepdims=True)
        pos = np.mod(np.vstack([parents, parents + offsets]), 100.0)
        rbins = np.array([0.01, 0.1, 10.0, 30.0])

        xi = compute_xi_corrfunc(pos, 100.0, rbins=rbins, nthreads=1)["xi"]

        np.testing.assert_allclose(
            xi.to_numpy(), reference_xi_auto(pos, 100.0, rbins), rtol=1e-10
        )
        assert xi[0] > 1e3  # 200 companion pairs vs ~1e-4 expected
        assert abs(xi[2]) < 0.2  # large scales stay unclustered

    def test_empty_bins_fall_back_to_bin_centres(self):
        pos = uniform_points(20, 100.0, seed=5)
        rbins = np.array([1e-3, 2e-3, 30.0])  # first bin has no pairs
        df = compute_xi_corrfunc(pos, 100.0, rbins=rbins, nthreads=1)
        np.testing.assert_allclose(df["r"].to_numpy(), 0.5 * (rbins[:-1] + rbins[1:]))
        assert df["xi"][0] == -1.0

    def test_bin_edge_exactly_at_half_box_is_handled(self):
        pos = uniform_points(300, 100.0, seed=6)
        rbins = np.array([5.0, 10.0, 20.0, 50.0])

        df = compute_xi_corrfunc(pos, boxsize=100.0, rbins=rbins, nthreads=1)

        kept = df.attrs["rbins"]
        assert kept[-1] < 50.0 or len(df) == 3
        np.testing.assert_allclose(
            df["xi"].to_numpy(), reference_xi_auto(pos, 100.0, kept), rtol=1e-10
        )


# ── correlation_given_redshift_and_subvolume ─────────────────────────────────


class TestCorrelationGivenRedshiftAndSubvolumeFailures:
    def test_missing_file_returns_none(self, tmp_path):
        assert correlation_given_redshift_and_subvolume(str(tmp_path), 0) is None

    def test_missing_output_group_returns_none(self, iz_dir):
        with h5py.File(galaxies_file(iz_dir, 0), "r+") as f:
            del f["Output001"]
        assert correlation_given_redshift_and_subvolume(str(iz_dir), 0) is None

    def test_missing_is_central_returns_none(self, iz_dir):
        with h5py.File(galaxies_file(iz_dir, 0), "r+") as f:
            del f["Output001/is_central"]
        assert correlation_given_redshift_and_subvolume(str(iz_dir), 0) is None

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_single_galaxy_gives_no_correlation(self, iz_dir):
        is_central = np.zeros(N_GALS, dtype=np.int32)
        is_central[0] = 1
        overwrite_output(galaxies_file(iz_dir, 0), is_central=is_central)

        res = correlation_given_redshift_and_subvolume(str(iz_dir), 0)

        assert res is None or np.all(np.isnan(res["xi"].to_numpy()))

    def test_empty_selection_does_not_crash(self, iz_dir):
        res = correlation_given_redshift_and_subvolume(
            str(iz_dir), 0, rbins=COARSE_RBINS, mhalo_min=1e20
        )
        assert res is None or np.all(np.isnan(res["xi"].to_numpy()))


@requires_corrfunc
class TestCorrelationGivenRedshiftAndSubvolumeValues:
    def test_centrals_xi_matches_reference(self, iz_dir):
        cat = read_output(galaxies_file(iz_dir, 0))
        pos = positions_of(cat, cat["is_central"] == 1)
        boxsize = TRUE_BOXSIZE

        res = correlation_given_redshift_and_subvolume(
            str(iz_dir), 0, rbins=COARSE_RBINS, nthreads=1
        )

        np.testing.assert_allclose(
            res["xi"].to_numpy(),
            reference_xi_auto(pos, boxsize, COARSE_RBINS),
            rtol=1e-6,
        )
        assert res.attrs["ngal"] == N_GALS // 2
        assert res.attrs["boxsize"] == pytest.approx(boxsize, rel=1e-6)
        assert res.attrs["ivol"] == 0
        assert res.attrs["z"] == pytest.approx(0.0)
        assert res.attrs["V_ivol"] == pytest.approx(155626.09375)
        np.testing.assert_array_equal(res.attrs["rbins"], COARSE_RBINS)

    def test_all_galaxies_used_when_centrals_only_false(self, iz_dir):
        cat = read_output(galaxies_file(iz_dir, 0))
        pos = positions_of(cat, np.ones(N_GALS, dtype=bool))

        res = correlation_given_redshift_and_subvolume(
            str(iz_dir), 0, rbins=COARSE_RBINS, nthreads=1, centrals_only=False
        )

        assert res.attrs["ngal"] == N_GALS
        np.testing.assert_allclose(
            res["xi"].to_numpy(),
            reference_xi_auto(pos, TRUE_BOXSIZE, COARSE_RBINS),
            rtol=1e-10,
        )

    def test_mhalo_cut_selects_massive_centrals(self, iz_dir):
        cat = read_output(galaxies_file(iz_dir, 0))
        mask = (cat["is_central"] == 1) & (cat["mhalo"] >= 1e11)
        pos = positions_of(cat, mask)
        assert 20 < mask.sum() < N_GALS // 2  # the cut really removes objects

        res = correlation_given_redshift_and_subvolume(
            str(iz_dir), 0, rbins=COARSE_RBINS, nthreads=1, mhalo_min=1e11
        )

        assert res.attrs["ngal"] == mask.sum()
        np.testing.assert_allclose(
            res["xi"].to_numpy(),
            reference_xi_auto(pos, TRUE_BOXSIZE, COARSE_RBINS),
            rtol=1e-10,
        )

    def test_negative_coordinates_are_shifted_into_the_box(self, iz_dir):
        path = galaxies_file(iz_dir, 0)
        ref = correlation_given_redshift_and_subvolume(
            str(iz_dir), 0, rbins=COARSE_RBINS, nthreads=1
        )
        cat = read_output(path)
        overwrite_output(
            path,
            xgal=cat["xgal"] - np.float32(300.0),
            ygal=cat["ygal"] - np.float32(300.0),
            zgal=cat["zgal"] - np.float32(300.0),
        )

        shifted = correlation_given_redshift_and_subvolume(
            str(iz_dir), 0, rbins=COARSE_RBINS, nthreads=1
        )

        # A rigid translation cannot change a periodic-box correlation function.
        np.testing.assert_allclose(
            shifted["xi"].to_numpy(), ref["xi"].to_numpy(), rtol=1e-4, atol=1e-6
        )

    def test_uses_true_box_size(self, iz_dir):
        # V_total = V_ivol * 1024 = 542.16^3 is available from Parameters/volume.
        cat = read_output(galaxies_file(iz_dir, 0))
        pos = positions_of(cat, cat["is_central"] == 1)

        res = correlation_given_redshift_and_subvolume(
            str(iz_dir), 0, rbins=COARSE_RBINS, nthreads=1
        )

        assert res.attrs["boxsize"] == pytest.approx(TRUE_BOXSIZE, rel=1e-4)
        np.testing.assert_allclose(
            res["xi"].to_numpy(),
            reference_xi_auto(pos, TRUE_BOXSIZE, COARSE_RBINS),
            rtol=1e-6,
        )


# ── halo_correlation_given_redshift_and_subvolume ────────────────────────────


class TestHaloCorrelationFailures:
    def test_missing_file_returns_none(self, tmp_path):
        assert halo_correlation_given_redshift_and_subvolume(str(tmp_path), 0) is None

    def test_missing_is_central_returns_none(self, iz_dir):
        with h5py.File(galaxies_file(iz_dir, 0), "r+") as f:
            del f["Output001/is_central"]
        assert halo_correlation_given_redshift_and_subvolume(str(iz_dir), 0) is None

    def test_empty_selection_does_not_crash(self, iz_dir):
        res = halo_correlation_given_redshift_and_subvolume(
            str(iz_dir), 0, rbins=COARSE_RBINS, mhhalo_min=1e20
        )
        assert res is None or np.all(np.isnan(res["xi"].to_numpy()))


@requires_corrfunc
class TestHaloCorrelationValues:
    def test_mhhalo_cut_matches_reference(self, iz_dir):
        cat = read_output(galaxies_file(iz_dir, 0))
        mask = (cat["is_central"] == 1) & (cat["mhhalo"] >= 3e11)
        pos = positions_of(cat, mask)
        assert 20 < mask.sum() < N_GALS // 2

        res = halo_correlation_given_redshift_and_subvolume(
            str(iz_dir), 0, rbins=COARSE_RBINS, nthreads=1, mhhalo_min=3e11
        )

        assert res.attrs["nhalo"] == mask.sum()
        assert res.attrs["boxsize"] == pytest.approx(TRUE_BOXSIZE, rel=1e-6)
        assert res.attrs["V_ivol"] == pytest.approx(155626.09375)
        np.testing.assert_allclose(
            res["xi"].to_numpy(),
            reference_xi_auto(pos, TRUE_BOXSIZE, COARSE_RBINS),
            rtol=1e-10,
        )

    def test_uncut_halos_equal_central_galaxies(self, iz_dir):
        halos = halo_correlation_given_redshift_and_subvolume(
            str(iz_dir), 1, rbins=COARSE_RBINS, nthreads=1
        )
        centrals = correlation_given_redshift_and_subvolume(
            str(iz_dir), 1, rbins=COARSE_RBINS, nthreads=1
        )
        np.testing.assert_allclose(
            halos["xi"].to_numpy(), centrals["xi"].to_numpy(), rtol=1e-12
        )


# ── avg_correlation_given_redshift_and_subvolumes ────────────────────────────


class TestAvgCorrelationOverSubvolumesFailures:
    def test_missing_snapshot_dir_returns_none(self, tmp_path):
        res = avg_correlation_given_redshift_and_subvolumes(
            155, [0, 1], base_dir=str(tmp_path)
        )
        assert res is None

    def test_no_readable_subvolume_returns_none(self, iz_dir):
        res = avg_correlation_given_redshift_and_subvolumes(
            155, [7, 8], base_dir=str(iz_dir.parent)
        )
        assert res is None

    @requires_corrfunc
    def test_empty_subvolume_is_skipped(self, iz_dir):
        overwrite_output(
            galaxies_file(iz_dir, 1), is_central=np.zeros(N_GALS, dtype=np.int32)
        )
        res = avg_correlation_given_redshift_and_subvolumes(
            155, [0, 1], rbins=COARSE_RBINS, nthreads=1, base_dir=str(iz_dir.parent)
        )
        assert res is not None
        assert res.attrs["total_galaxies"] == N_GALS // 2


@requires_corrfunc
class TestAvgCorrelationOverSubvolumesValues:
    def test_stacked_xi_matches_reference(self, iz_dir):
        cats = [read_output(galaxies_file(iz_dir, iv)) for iv in (0, 1)]
        pos_each = [positions_of(c, c["is_central"] == 1) for c in cats]
        boxsize = TRUE_BOXSIZE  # from Parameters/volume
        stacked = np.vstack(pos_each)

        res = avg_correlation_given_redshift_and_subvolumes(
            155, [0, 1], rbins=COARSE_RBINS, nthreads=1, base_dir=str(iz_dir.parent)
        )

        np.testing.assert_allclose(
            res["xi"].to_numpy(),
            reference_xi_auto(stacked, boxsize, COARSE_RBINS),
            rtol=1e-10,
        )
        assert res.attrs["total_galaxies"] == len(stacked)
        assert res.attrs["n_used"] == 2
        assert res.attrs["iz"] == "iz155"
        assert res.attrs["boxsize"] == pytest.approx(boxsize, rel=1e-6)
        assert res.attrs["method"] == "combined_overlapping_subvolumes"

    def test_missing_subvolumes_are_skipped(self, iz_dir):
        res = avg_correlation_given_redshift_and_subvolumes(
            155, [0, 1, 9], rbins=COARSE_RBINS, nthreads=1, base_dir=str(iz_dir.parent)
        )
        assert res.attrs["n_used"] == 2
        assert res.attrs["total_galaxies"] == N_GALS

    def test_defaults_to_configured_base_dir(self, base_dir):
        res = avg_correlation_given_redshift_and_subvolumes(207, [0], nthreads=1)
        assert res is not None
        assert len(res) == len(DEFAULT_RBINS) - 1
        assert res.attrs["iz"] == "iz207"

    def test_rbins_attr_describes_returned_bins(self, iz_dir):
        rbins = np.append(COARSE_RBINS, 400.0)  # last edge beyond L/2
        res = avg_correlation_given_redshift_and_subvolumes(
            155, [0], rbins=rbins, nthreads=1, base_dir=str(iz_dir.parent)
        )
        assert len(res.attrs["rbins"]) == len(res) + 1


# ── correlations_given_redshifts_and_subvolume ───────────────────────────────


def test_correlations_over_redshifts_empty_when_no_snapshots(tmp_path):
    res = correlations_given_redshifts_and_subvolume(
        [100, 120], 0, base_dir=str(tmp_path)
    )
    assert res == []


@requires_corrfunc
def test_correlations_over_redshifts_match_single_calls(base_dir):
    results = correlations_given_redshifts_and_subvolume(
        [155, 999, 207], 1, rbins=COARSE_RBINS, nthreads=1
    )

    assert [r.attrs["iz"] for r in results] == ["iz155", "iz207"]
    for res, iz in zip(results, ("iz155", "iz207")):
        single = correlation_given_redshift_and_subvolume(
            str(base_dir / iz), 1, rbins=COARSE_RBINS, nthreads=1
        )
        assert isinstance(res, pl.DataFrame)
        np.testing.assert_array_equal(res["xi"].to_numpy(), single["xi"].to_numpy())


# ── avg_correlation_given_subvolume_and_redshifts ────────────────────────────


def test_avg_over_redshifts_none_when_nothing_valid(tmp_path):
    make_snapshot(tmp_path, 155, ivols=(0,))
    res = avg_correlation_given_subvolume_and_redshifts(
        [155, 999], ivol=3, base_dir=str(tmp_path)
    )
    assert res is None


@requires_corrfunc
def test_avg_over_redshifts_is_mean_and_std_of_snapshots(base_dir):
    singles = [
        correlation_given_redshift_and_subvolume(
            str(base_dir / iz), 0, rbins=COARSE_RBINS, nthreads=1
        )
        for iz in ("iz155", "iz207")
    ]
    xi_stack = np.vstack([s["xi"].to_numpy() for s in singles])

    res = avg_correlation_given_subvolume_and_redshifts(
        [155, 207, 999], 0, rbins=COARSE_RBINS, nthreads=1
    )

    np.testing.assert_allclose(res["xi"].to_numpy(), xi_stack.mean(axis=0))
    np.testing.assert_allclose(res["xi_std"].to_numpy(), xi_stack.std(axis=0))
    np.testing.assert_array_equal(res["r"].to_numpy(), singles[0]["r"].to_numpy())
    assert res.attrs["n_used"] == 2
    assert res.attrs["used_iz"] == ["iz155", "iz207"]
    assert res.attrs["used_z"] == [pytest.approx(0.0), pytest.approx(0.0)]
    assert res.attrs["ivol"] == 0


@requires_corrfunc
def test_avg_over_redshifts_skips_empty_snapshots_and_trims_rbins(base_dir):
    overwrite_output(
        galaxies_file(base_dir / "iz207", 0), is_central=np.zeros(N_GALS, np.int32)
    )
    rbins = np.append(COARSE_RBINS, 400.0)  # last edge beyond L/2

    res = avg_correlation_given_subvolume_and_redshifts(
        [155, 207], 0, rbins=rbins, nthreads=1
    )

    single = correlation_given_redshift_and_subvolume(
        str(base_dir / "iz155"), 0, rbins=rbins, nthreads=1
    )
    assert res.attrs["used_iz"] == ["iz155"]
    np.testing.assert_array_equal(res["xi"].to_numpy(), single["xi"].to_numpy())
    np.testing.assert_array_equal(res["xi_std"].to_numpy(), 0.0)
    np.testing.assert_array_equal(res.attrs["rbins"], COARSE_RBINS)


# ── box size handling ────────────────────────────────────────────────────────


def test_wrap_into_box_maps_into_half_open_interval():
    pos = np.array([[-1e-17, 100.0, 250.0], [-30.0, 99.5, 0.0]])
    wrapped = _wrap_into_box(pos, 100.0)
    np.testing.assert_allclose(wrapped, [[0.0, 0.0, 50.0], [70.0, 99.5, 0.0]])
    assert np.all((wrapped >= 0.0) & (wrapped < 100.0))


@requires_corrfunc
class TestBoxSize:
    def test_explicit_boxsize_overrides_file(self, iz_dir):
        cat = read_output(galaxies_file(iz_dir, 0))
        pos = positions_of(cat, cat["is_central"] == 1)

        res = correlation_given_redshift_and_subvolume(
            str(iz_dir), 0, rbins=COARSE_RBINS, nthreads=1, boxsize=600.0
        )

        assert res.attrs["boxsize"] == 600.0
        np.testing.assert_allclose(
            res["xi"].to_numpy(),
            reference_xi_auto(pos, 600.0, COARSE_RBINS),
            rtol=1e-10,
        )

    def test_without_volume_falls_back_to_extent_with_warning(self, iz_dir):
        with h5py.File(galaxies_file(iz_dir, 0), "r+") as f:
            del f["Parameters/volume"]
        cat = read_output(galaxies_file(iz_dir, 0))
        pos = positions_of(cat, cat["is_central"] == 1)

        with pytest.warns(RuntimeWarning, match="position extent"):
            res = correlation_given_redshift_and_subvolume(
                str(iz_dir), 0, rbins=COARSE_RBINS, nthreads=1
            )

        assert res.attrs["boxsize"] == pytest.approx(extent_boxsize(pos))
        assert res.attrs["V_ivol"] is None

    def test_empty_selection_without_volume_returns_none(self, iz_dir):
        with h5py.File(galaxies_file(iz_dir, 0), "r+") as f:
            del f["Parameters/volume"]
        res = correlation_given_redshift_and_subvolume(
            str(iz_dir), 0, rbins=COARSE_RBINS, mhalo_min=1e20
        )
        assert res is None

    def test_halo_boxsize_override(self, iz_dir):
        res = halo_correlation_given_redshift_and_subvolume(
            str(iz_dir), 0, rbins=COARSE_RBINS, nthreads=1, boxsize=700.0
        )
        assert res.attrs["boxsize"] == 700.0

    def test_avg_boxsize_override(self, iz_dir):
        cats = [read_output(galaxies_file(iz_dir, iv)) for iv in (0, 1)]
        stacked = np.vstack([positions_of(c, c["is_central"] == 1) for c in cats])

        res = avg_correlation_given_redshift_and_subvolumes(
            155,
            [0, 1],
            rbins=COARSE_RBINS,
            nthreads=1,
            base_dir=str(iz_dir.parent),
            boxsize=650.0,
        )

        assert res.attrs["boxsize"] == 650.0
        np.testing.assert_allclose(
            res["xi"].to_numpy(),
            reference_xi_auto(stacked, 650.0, COARSE_RBINS),
            rtol=1e-10,
        )

    def test_invalid_boxsize_returns_none(self, iz_dir):
        res = correlation_given_redshift_and_subvolume(
            str(iz_dir), 0, rbins=COARSE_RBINS, boxsize=-1.0
        )
        assert res is None


# ── box size resolution ──────────────────────────────────────────────────────


class TestResolveBoxsize:
    def test_explicit_boxsize_wins(self):
        from galform_analysis.analysis.correlation.correlation import _resolve_boxsize

        pos = uniform_points(10, 100.0, 0)
        assert _resolve_boxsize(pos, 250.0, 100.0, "test") == 250.0

    def test_file_box_too_small_for_positions_raises(self):
        from galform_analysis.analysis.correlation.correlation import _resolve_boxsize

        # e.g. n_subvolumes missing from the file and the 1024 fallback wrong
        pos = uniform_points(50, 500.0, 0)
        with pytest.raises(ValueError, match="Pass boxsize="):
            _resolve_boxsize(pos, None, 100.0, "test")

    def test_file_box_accepted_when_positions_fit(self):
        from galform_analysis.analysis.correlation.correlation import _resolve_boxsize

        pos = uniform_points(50, 100.0, 0)
        assert _resolve_boxsize(pos, None, 100.0, "test") == 100.0
