"""Tests for the subvolume-weighted (auto/cross) 2PCF correction.

Conventions verified here against brute-force numpy pair counts:

- Corrfunc ``autocorr=1`` returns *ordered* pair counts (every unordered pair
  counted twice); ``autocorr=0`` cross counts count each pair once.
- ``DD_cross = DD_total - sum_tag DD_auto(tag)`` therefore lives in the same
  (ordered-pair) units as ``DD_total``.
- alpha = m / k and beta = m (k - 1) / [k (m - 1)] reweight auto and cross
  pairs so that, when each ``ivol`` is an independent full-box realisation,
  the expected corrected pair counts equal those of the full k-subvolume sample.

Pure-numpy helpers, input validation and HDF5 loading run on a base install;
tests that call Corrfunc are skipped when it is absent.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import polars as pl
import pytest

from galform_analysis.analysis.correlation import subvol_weighted_correction as swc
from galform_analysis.analysis.correlation.subvol_weighted_correction import (
    _choose2,
    _counts_to_grid,
    _paircounts_r_auto,
    _paircounts_r_cross,
    _paircounts_rppi_auto,
    _paircounts_rppi_cross,
    _pick_partition_labels,
    _select_halo_id_array,
    compute_weighted_wp_for_n_list,
    compute_weighted_wp_from_catalogue,
    compute_weighted_xi_for_n_list,
    compute_weighted_xi_from_catalogue,
    load_subvolume_galaxies,
)
from tests.conftest import write_galaxy_hdf5

requires_corrfunc = pytest.mark.skipif(
    importlib.util.find_spec("Corrfunc") is None,
    reason="needs Corrfunc (optional 'clustering' extra)",
)

IZ = 155
N_GALS = 40
N_IVOLS = 4
BOX = 542.16


# ---------------------------------------------------------------------------
# Independent brute-force references
# ---------------------------------------------------------------------------


def _alpha_ref(m: int, k: int) -> float:
    return m / k


def _beta_ref(m: int, k: int) -> float:
    return m * (k - 1) / (k * (m - 1))


def _n_pairs(n: float) -> float:
    """Unique unordered pairs from n points (written out independently)."""
    return n * (n - 1) / 2.0


def _deltas_auto(pos: np.ndarray, boxsize: float) -> np.ndarray:
    """Minimum-image separation vectors of every unique pair (i < j)."""
    i, j = np.triu_indices(len(pos), k=1)
    d = pos[i] - pos[j]
    return d - boxsize * np.round(d / boxsize)


def _deltas_cross(pos_a: np.ndarray, pos_b: np.ndarray, boxsize: float) -> np.ndarray:
    """Minimum-image separation vectors of every (a, b) pair."""
    d = (pos_a[:, None, :] - pos_b[None, :, :]).reshape(-1, 3)
    return d - boxsize * np.round(d / boxsize)


def _hist_r(d: np.ndarray, rbins: np.ndarray) -> np.ndarray:
    r = np.sqrt(np.sum(d**2, axis=1))
    return np.histogram(r, bins=rbins)[0].astype(np.float64)


def _hist_rppi(d: np.ndarray, rp_bins: np.ndarray, pimax: int) -> np.ndarray:
    """2D histogram in (r_p, |pi|) with the z-axis as line of sight, dpi = 1."""
    rp = np.hypot(d[:, 0], d[:, 1])
    pi = np.abs(d[:, 2])
    pi_bins = np.arange(pimax + 1, dtype=np.float64)
    return np.histogram2d(rp, pi, bins=[rp_bins, pi_bins])[0]


def _landy_szalay(dd_u, dr, rr_u, nd, nr):
    """LS estimator from unique DD/RR pair counts and DR cross counts."""
    with np.errstate(divide="ignore", invalid="ignore"):
        rr_n = rr_u / _n_pairs(nr)
        xi = (dd_u / _n_pairs(nd) - 2.0 * dr / (nd * nr) + rr_n) / rr_n
    xi[~np.isfinite(xi)] = np.nan
    return xi


def _randoms_like_package(nd: int, multiplier: float, seed: int, boxsize: float):
    """The random catalogue documented by ``random_multiplier``/``random_seed``."""
    nr = max(2, int(np.ceil(multiplier * nd)))
    return np.random.default_rng(seed).uniform(0.0, boxsize, size=(nr, 3))


def _reference_counts(pos, tags, m, rnd, hist, boxsize):
    """Unique-pair auto/cross/total counts, DR and unique RR via brute force."""
    auto = sum(
        hist(_deltas_auto(pos[tags == t], boxsize)) for t in range(m) if (tags == t).sum() > 1
    )
    cross = sum(
        hist(_deltas_cross(pos[tags == a], pos[tags == b], boxsize))
        for a in range(m)
        for b in range(a + 1, m)
    )
    total = hist(_deltas_auto(pos, boxsize))
    dr = hist(_deltas_cross(pos, rnd, boxsize))
    rr = hist(_deltas_auto(rnd, boxsize))
    return auto, cross, total, dr, rr


def _uniform_catalogue(n_per_tag: int, n_tags: int, boxsize: float, seed: int):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, boxsize, size=(n_per_tag * n_tags, 3))
    tags = np.repeat(np.arange(n_tags), n_per_tag)
    cat = pl.DataFrame(
        {"x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2], "subvol_rank": tags}
    )
    return cat, pos, tags


# ---------------------------------------------------------------------------
# Mock GALFORM directories
# ---------------------------------------------------------------------------


@pytest.fixture
def subvol_base_dir(tmp_path: Path) -> str:
    """base/iz155/ivol{0..3}/galaxies.hdf5 with 40 galaxies each."""
    for ivol in range(N_IVOLS):
        ivol_dir = tmp_path / f"iz{IZ}" / f"ivol{ivol}"
        ivol_dir.mkdir(parents=True)
        write_galaxy_hdf5(ivol_dir / "galaxies.hdf5", n_gals=N_GALS, seed=100 + ivol)
    return str(tmp_path)


def _read_truth(base_dir: str, ivol: int) -> dict[str, np.ndarray]:
    path = Path(base_dir) / f"iz{IZ}" / f"ivol{ivol}" / "galaxies.hdf5"
    with h5py.File(path, "r") as f:
        g = f["Output001"]
        return {
            "x": np.asarray(g["xgal"], dtype=np.float64),
            "y": np.asarray(g["ygal"], dtype=np.float64),
            "z": np.asarray(g["zgal"], dtype=np.float64),
            "mstar": np.asarray(g["mstars_disk"]) + np.asarray(g["mstars_bulge"]),
            "mhalo": np.asarray(g["mhalo"]),
            "is_central": np.asarray(g["is_central"]),
            "DHaloID": np.asarray(g["DHaloID"]),
        }


# ---------------------------------------------------------------------------
# _choose2
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
# _pick_partition_labels / _select_halo_id_array
# ---------------------------------------------------------------------------


class TestPickPartitionLabels:
    def test_prefers_partition_label_over_subvol_rank(self):
        cat = pl.DataFrame({"subvol_rank": [0, 0, 1], "partition_label": [5, 6, 7]})
        labels = _pick_partition_labels(cat)
        np.testing.assert_array_equal(labels, [5, 6, 7])
        assert labels.dtype == np.int64

    def test_falls_back_to_subvol_rank(self):
        cat = pl.DataFrame({"subvol_rank": [2, 1, 0]})
        np.testing.assert_array_equal(_pick_partition_labels(cat), [2, 1, 0])

    def test_missing_both_columns_raises(self):
        with pytest.raises(KeyError, match="partition_label"):
            _pick_partition_labels(pl.DataFrame({"x": [0.0]}))


class TestSelectHaloIdArray:
    def test_respects_priority_order(self):
        arrays = {"TreeID": np.array([7, 8]), "DHaloID": np.array([1, 2, 3])}
        arr, key = _select_halo_id_array(arrays)
        assert key == "DHaloID"
        np.testing.assert_array_equal(arr, [1, 2, 3])

    def test_skips_constant_and_empty_fields(self):
        arrays = {
            "ihalof": np.array([], dtype=np.int64),
            "ihhalo": np.array([4, 4, 4]),
            "TreeID": np.array([3, 9, 3]),
        }
        arr, key = _select_halo_id_array(arrays)
        assert key == "TreeID"
        np.testing.assert_array_equal(arr, [3, 9, 3])

    def test_negative_sentinels_are_not_informative(self):
        arrays = {"ihalof": np.array([-1, -1, 5]), "DHaloID": np.array([1, 2])}
        _, key = _select_halo_id_array(arrays)
        assert key == "DHaloID"

    def test_keeps_sentinels_in_returned_array(self):
        arrays = {"ihalof": np.array([-1, 4, 5])}
        arr, key = _select_halo_id_array(arrays)
        assert key == "ihalof"
        np.testing.assert_array_equal(arr, [-1, 4, 5])

    def test_float_ids_with_nan_and_cast_to_int64(self):
        arrays = {
            "ihalof": np.array([np.nan, np.nan]),
            "ihhalo": np.array([1.0, 2.0, np.nan]),
        }
        arr, key = _select_halo_id_array(arrays)
        assert key == "ihhalo"
        assert arr.dtype == np.int64
        np.testing.assert_array_equal(arr[:2], [1, 2])

    def test_ignores_non_halo_fields_and_returns_none(self):
        arrays = {"mhalo": np.array([1.0, 2.0]), "SubhaloID": np.array([3, 3])}
        assert _select_halo_id_array(arrays) == (None, None)


# ---------------------------------------------------------------------------
# _counts_to_grid and pair-count early returns (no Corrfunc call needed)
# ---------------------------------------------------------------------------


class TestCountsToGrid:
    def test_reshapes_rp_major(self):
        grid = _counts_to_grid({"npairs": np.arange(6)}, n_rp_bins=2, n_pi_bins=3)
        np.testing.assert_array_equal(grid, [[0, 1, 2], [3, 4, 5]])
        assert grid.dtype == np.float64

    def test_size_mismatch_raises(self):
        with pytest.raises(RuntimeError, match="DDrppi"):
            _counts_to_grid({"npairs": np.ones(5)}, n_rp_bins=2, n_pi_bins=3)


class TestPairCountEarlyReturns:
    rbins = np.array([1.0, 2.0, 4.0])

    def test_rppi_auto_fewer_than_two_points(self):
        out = _paircounts_rppi_auto(np.zeros((1, 3)), self.rbins, 5, 10.0, 1)
        np.testing.assert_array_equal(out, np.zeros((2, 5)))

    def test_rppi_cross_empty_sample(self):
        out = _paircounts_rppi_cross(
            np.zeros((0, 3)), np.ones((4, 3)), self.rbins, 3, 10.0, 1
        )
        np.testing.assert_array_equal(out, np.zeros((2, 3)))

    def test_r_auto_fewer_than_two_points(self):
        out = _paircounts_r_auto(np.zeros((0, 3)), self.rbins, 10.0, 1)
        np.testing.assert_array_equal(out, np.zeros(2))

    def test_r_cross_empty_sample(self):
        out = _paircounts_r_cross(np.ones((3, 3)), np.zeros((0, 3)), self.rbins, 10.0, 1)
        np.testing.assert_array_equal(out, np.zeros(2))


class _FakeCorrfuncModule:
    """Stand-in Corrfunc module whose pair counter returns the wrong size."""

    def __init__(self, n_out: int):
        self._n_out = n_out

    def _count(self, *args, **kwargs):
        return {"npairs": np.ones(self._n_out)}

    @property
    def DD(self):
        return self._count


class TestCorrfuncOutputValidation:
    @pytest.mark.parametrize(
        ("func", "args"),
        [
            (_paircounts_r_auto, (np.ones((3, 3)),)),
            (_paircounts_r_cross, (np.ones((3, 3)), np.ones((2, 3)))),
        ],
    )
    def test_unexpected_dd_output_size_raises(self, monkeypatch, func, args):
        monkeypatch.setattr(swc, "import_optional", lambda name: _FakeCorrfuncModule(7))
        with pytest.raises(RuntimeError, match="Unexpected DD"):
            func(*args, np.array([1.0, 2.0, 3.0]), 10.0, 1)

    def test_missing_corrfunc_gives_install_hint(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "Corrfunc", None)
        monkeypatch.setitem(sys.modules, "Corrfunc.theory", None)
        monkeypatch.setitem(sys.modules, "Corrfunc.theory.DD", None)
        with pytest.raises(ImportError, match=r"galform_analysis\[clustering\]"):
            _paircounts_r_auto(np.ones((3, 3)), np.array([1.0, 2.0]), 10.0, 1)


# ---------------------------------------------------------------------------
# Input validation and empty catalogues (no Corrfunc call needed)
# ---------------------------------------------------------------------------


_EMPTY_CAT = pl.DataFrame(
    schema={"x": pl.Float64, "y": pl.Float64, "z": pl.Float64, "subvol_rank": pl.Int64}
)


class TestValidationAndEmpty:
    @pytest.mark.parametrize("rp_bins", [np.array([1.0]), np.ones((2, 2))])
    def test_wp_rejects_bad_rp_bins(self, rp_bins):
        with pytest.raises(ValueError, match="rp_bins"):
            compute_weighted_wp_from_catalogue(_EMPTY_CAT, 2, 4, rp_bins)

    @pytest.mark.parametrize("pimax", [0, 2.5])
    def test_wp_rejects_bad_pimax(self, pimax):
        with pytest.raises(ValueError, match="pimax"):
            compute_weighted_wp_from_catalogue(
                _EMPTY_CAT, 2, 4, np.array([1.0, 2.0]), pimax=pimax
            )

    def test_wp_empty_catalogue_returns_nan(self):
        out = compute_weighted_wp_from_catalogue(
            _EMPTY_CAT, 3, 8, np.array([1.0, 3.0, 5.0]), pimax=4
        )
        np.testing.assert_allclose(out["rp"], [2.0, 4.0])
        assert np.all(np.isnan(out["wp_standard"]))
        assert np.all(np.isnan(out["wp_corrected"]))
        assert out["xi_standard_grid"].shape == (2, 4)
        assert np.isnan(out["alpha"]) and np.isnan(out["beta"])
        assert (out["ngal"], out["nrandom"], out["m_selected"], out["k_total"]) == (
            0,
            0,
            3,
            8,
        )

    def test_xi_rejects_bad_rbins(self):
        with pytest.raises(ValueError, match="rbins"):
            compute_weighted_xi_from_catalogue(_EMPTY_CAT, 2, 4, np.array([1.0]))

    def test_xi_empty_catalogue_returns_nan(self):
        out = compute_weighted_xi_from_catalogue(
            _EMPTY_CAT, 2, 4, np.array([1.0, 2.0, 4.0])
        )
        np.testing.assert_allclose(out["r"], [1.5, 3.0])
        assert np.all(np.isnan(out["xi_standard"]))
        assert np.all(np.isnan(out["xi_corrected"]))
        assert out["ngal"] == 0

    @pytest.mark.parametrize(
        "func", [compute_weighted_xi_for_n_list, compute_weighted_wp_for_n_list]
    )
    @pytest.mark.parametrize("n_list", [[], [0, 2], [-1]])
    def test_n_list_must_be_positive(self, func, n_list, tmp_path):
        with pytest.raises(ValueError, match="positive integers"):
            func(str(tmp_path), IZ, n_list, k_total=4)

    @pytest.mark.parametrize(
        "func", [compute_weighted_xi_for_n_list, compute_weighted_wp_for_n_list]
    )
    def test_load_n_must_cover_max_n(self, func, tmp_path):
        with pytest.raises(ValueError, match="load_n_subvolumes"):
            func(str(tmp_path), IZ, [1, 3], k_total=4, load_n_subvolumes=2)


# ---------------------------------------------------------------------------
# load_subvolume_galaxies (HDF5 only)
# ---------------------------------------------------------------------------


class TestLoadSubvolumeGalaxies:
    def test_invalid_partition_scheme(self, tmp_path):
        with pytest.raises(ValueError, match="partition_scheme"):
            load_subvolume_galaxies(str(tmp_path), IZ, [0], partition_scheme="bad")

    def test_k_total_too_small(self, tmp_path):
        with pytest.raises(ValueError, match="k_total"):
            load_subvolume_galaxies(str(tmp_path), IZ, [0], k_total=1)

    def test_ivol_tags_follow_selection_order(self, subvol_base_dir):
        cat = load_subvolume_galaxies(subvol_base_dir, IZ, ivols=[2, 0])

        truth2 = _read_truth(subvol_base_dir, 2)
        truth0 = _read_truth(subvol_base_dir, 0)
        assert cat.height == 2 * N_GALS
        np.testing.assert_allclose(
            cat["x"].to_numpy(), np.concatenate([truth2["x"], truth0["x"]])
        )
        np.testing.assert_allclose(
            cat["z"].to_numpy(), np.concatenate([truth2["z"], truth0["z"]])
        )
        np.testing.assert_array_equal(
            cat["ivol"].to_numpy(), np.repeat([2, 0], N_GALS)
        )
        np.testing.assert_array_equal(
            cat["subvol_rank"].to_numpy(), np.repeat([0, 1], N_GALS)
        )
        np.testing.assert_array_equal(
            cat["partition_label"].to_numpy(), cat["subvol_rank"].to_numpy()
        )
        assert set(cat["partition_scheme"].unique().to_list()) == {"ivol"}

    def test_centrals_and_mhalo_cuts(self, subvol_base_dir):
        mhalo_min = 1e11
        cat = load_subvolume_galaxies(
            subvol_base_dir, IZ, ivols=[1], centrals_only=True, mhalo_min=mhalo_min
        )

        t = _read_truth(subvol_base_dir, 1)
        keep = (t["is_central"] == 1) & (t["mhalo"] >= mhalo_min)
        assert 0 < keep.sum() < N_GALS
        np.testing.assert_allclose(cat["x"].to_numpy(), t["x"][keep])

    def test_log10_stellar_mass_cut(self, subvol_base_dir):
        threshold = 10.5
        cat = load_subvolume_galaxies(
            subvol_base_dir, IZ, ivols=[0, 3], mstar_min_log10=threshold
        )

        expected_y = []
        for ivol in (0, 3):
            t = _read_truth(subvol_base_dir, ivol)
            expected_y.append(t["y"][np.log10(t["mstar"].astype(np.float64)) >= threshold])
        expected = np.concatenate(expected_y)
        assert 0 < expected.size < 2 * N_GALS
        np.testing.assert_allclose(cat["y"].to_numpy(), expected)

    def test_everything_cut_returns_typed_empty_frame(self, subvol_base_dir):
        cat = load_subvolume_galaxies(subvol_base_dir, IZ, [0, 1], mstar_min_log10=20.0)

        assert cat.is_empty()
        assert cat.schema["x"] == pl.Float64
        assert cat.schema["partition_label"] == pl.Int64
        assert set(cat.columns) == {
            "x",
            "y",
            "z",
            "subvol_rank",
            "partition_label",
            "ivol",
            "partition_scheme",
        }

    def test_halo_id_hash_labels(self, subvol_base_dir):
        k_total = 3
        cat = load_subvolume_galaxies(
            subvol_base_dir,
            IZ,
            ivols=[0, 1],
            partition_scheme="halo_id_hash",
            k_total=k_total,
        )

        ids = np.concatenate(
            [_read_truth(subvol_base_dir, v)["DHaloID"] for v in (0, 1)]
        )
        np.testing.assert_array_equal(
            cat["partition_label"].to_numpy(), np.abs(ids) % k_total
        )
        np.testing.assert_array_equal(
            cat["subvol_rank"].to_numpy(), np.repeat([0, 1], N_GALS)
        )
        assert set(cat["partition_scheme"].unique().to_list()) == {"halo_id_hash"}

    def test_halo_id_hash_without_ids_raises(self, subvol_base_dir):
        path = Path(subvol_base_dir) / f"iz{IZ}" / "ivol0" / "galaxies.hdf5"
        with h5py.File(path, "a") as f:
            del f["Output001/DHaloID"]
            del f["Output001/TreeID"]

        with pytest.raises(RuntimeError, match="halo ID"):
            load_subvolume_galaxies(
                subvol_base_dir, IZ, [0], partition_scheme="halo_id_hash", k_total=4
            )

    def test_missing_ivol_raises(self, subvol_base_dir):
        with pytest.raises(FileNotFoundError):
            load_subvolume_galaxies(subvol_base_dir, IZ, [0, 99])
