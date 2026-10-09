"""HOD tests on hand-built catalogues with known occupation numbers.

Catalogue (BINS = [11, 12, 13, 14] in log10 M_host):

  bin [11, 12): 1 tree halo, no galaxies                -> <N> = 0
  bin [12, 13): 4 tree haloes, 4 FOF centrals, 2 sats   -> <N> = 1.5, cen 1, sat 0.5
  bin [13, 14): 2 tree haloes, 2 FOF centrals, 6 sats,
                1 subhalo central (is_central=1 but mhalo/mhhalo = 0.2,
                so it counts as a satellite)            -> <N> = 4.5, cen 1, sat 3.5

Centrals have M* = 1e11 and satellites M* = 1e9, so a 1e10 stellar-mass cut
keeps exactly the FOF centrals and the subhalo central.
"""

import numpy as np
import polars as pl
import pytest
from mf_catalogue import masses_at, write_catalogue

import galform_analysis.config as config
from galform_analysis.analysis.mass_functions import (
    avg_hod_given_redshift_and_subvolumes,
    avg_hod_given_redshifts_and_subvolume,
    hod_given_redshift_and_subvolume,
    hods_given_redshifts_and_subvolume,
)
from galform_analysis.analysis.mass_functions.hod import _compute_hod_two_histogram
from galform_analysis.config import DEFAULT_HALO_MASS_BINS

BINS = np.array([11.0, 12.0, 13.0, 14.0])
M12, M13 = 10**12.5, 10**13.5


def galaxies():
    """Per-galaxy arrays for the catalogue in the module docstring."""
    mhhalo = np.array([M12] * 6 + [M13] * 9)
    is_central = np.array([1] * 4 + [0] * 2 + [1] * 2 + [0] * 6 + [1])
    sub_ratio = np.array([1.0] * 4 + [0.1] * 2 + [1.0] * 2 + [0.1] * 6 + [0.2])
    mstar = np.where(is_central == 1, 1e11, 1e9)
    return dict(
        mhhalo=mhhalo, mhalo=mhhalo * sub_ratio, is_central=is_central, mstar=mstar
    )


TREES = masses_at([(11.5, 1), (12.5, 4), (13.5, 2)])
EXPECTED = dict(
    mean_occupation=[0.0, 1.5, 4.5],
    mean_central=[0.0, 1.0, 1.0],
    mean_satellite=[0.0, 0.5, 3.5],
    counts_halos=[1, 4, 2],
    counts_galaxies=[0, 6, 9],
)


def write(base, iz=155, ivol=0, scale=1, redshift=1.0, **overrides):
    """Write the catalogue; ``scale`` replicates it to multiply all counts."""
    cat = {k: np.tile(v, scale) for k, v in galaxies().items()}
    cat.update(overrides)
    trees = overrides.pop("tree_mphalo", np.tile(TREES, scale))
    cat.pop("tree_mphalo", None)
    return write_catalogue(base, iz, ivol, tree_mphalo=trees, redshift=redshift, **cat)


def assert_matches(res, expected=EXPECTED):
    for key, value in expected.items():
        np.testing.assert_allclose(res[key], value, err_msg=key)


@pytest.fixture
def configured_base(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BASE_DIR", config.BASE_DIR)
    config.set_base_dir(str(tmp_path))
    return tmp_path


# ── single subvolume ─────────────────────────────────────────────────────────


class TestSingleSubvolume:
    def test_known_occupation(self, tmp_path):
        iz_dir = write(tmp_path, redshift=0.7)
        res = hod_given_redshift_and_subvolume(str(iz_dir), 0, bins=BINS)
        assert_matches(res)
        np.testing.assert_allclose(res["centers"], [11.5, 12.5, 13.5])
        assert (res["iz"], res["ivol"]) == ("iz155", 0)
        assert res["z"] == pytest.approx(0.7)
        assert (res["n_halos"], res["n_galaxies"]) == (7, 15)
        assert res["halo_mass_field"] == "mhhalo"
        assert res["selection"] == {
            "galaxy_stellar_mass_min": None,
            "halo_mass_lower_limit": None,
        }

    def test_stellar_mass_cut_keeps_centrals_only(self, tmp_path):
        iz_dir = write(tmp_path)
        res = hod_given_redshift_and_subvolume(
            str(iz_dir), 0, bins=BINS, galaxy_stellar_mass_min=1e10
        )
        assert_matches(
            res,
            dict(
                mean_occupation=[0.0, 1.0, 1.5],
                mean_central=[0.0, 1.0, 1.0],
                mean_satellite=[0.0, 0.0, 0.5],
                counts_halos=[1, 4, 2],
                counts_galaxies=[0, 4, 3],
            ),
        )
        assert res["n_galaxies"] == 7

    def test_halo_mass_lower_limit_removes_haloes_and_galaxies(self, tmp_path):
        iz_dir = write(tmp_path)
        res = hod_given_redshift_and_subvolume(
            str(iz_dir), 0, bins=BINS, halo_mass_lower_limit=1e13
        )
        assert_matches(
            res,
            dict(
                mean_occupation=[0.0, 0.0, 4.5],
                mean_central=[0.0, 0.0, 1.0],
                counts_halos=[0, 0, 2],
                counts_galaxies=[0, 0, 9],
            ),
        )

    def test_default_bins(self, tmp_path):
        iz_dir = write(tmp_path)
        res = hod_given_redshift_and_subvolume(str(iz_dir), 0)
        assert len(res["centers"]) == len(DEFAULT_HALO_MASS_BINS) - 1
        assert res["counts_halos"].sum() == 7
        i13 = np.searchsorted(DEFAULT_HALO_MASS_BINS, 13.5) - 1
        assert res["mean_occupation"][i13] == pytest.approx(4.5)

    def test_falls_back_to_mhalo_when_mhhalo_missing(self, tmp_path):
        cat = galaxies()
        iz_dir = write_catalogue(
            tmp_path,
            1,
            0,
            mhalo=cat["mhhalo"],
            is_central=cat["is_central"],
            mstar=cat["mstar"],
            tree_mphalo=TREES,
        )
        res = hod_given_redshift_and_subvolume(str(iz_dir), 0, bins=BINS)
        np.testing.assert_allclose(res["mean_occupation"], [0.0, 1.5, 4.5])

    def test_missing_subhalo_mass_still_gives_total_occupation(self, tmp_path):
        cat = galaxies()
        iz_dir = write_catalogue(
            tmp_path,
            1,
            0,
            mhhalo=cat["mhhalo"],
            is_central=cat["is_central"],
            tree_mphalo=TREES,
        )
        res = hod_given_redshift_and_subvolume(str(iz_dir), 0, bins=BINS)
        np.testing.assert_allclose(res["mean_occupation"], [0.0, 1.5, 4.5])
        assert res["mean_central"] is None
        assert res["mean_satellite"] is None

    def test_missing_central_flag_gives_no_decomposition(self, tmp_path):
        cat = galaxies()
        iz_dir = write_catalogue(
            tmp_path, 1, 0, mhhalo=cat["mhhalo"], mhalo=cat["mhalo"], tree_mphalo=TREES
        )
        res = hod_given_redshift_and_subvolume(str(iz_dir), 0, bins=BINS)
        np.testing.assert_allclose(res["mean_occupation"], [0.0, 1.5, 4.5])
        assert res["mean_central"] is None

    def test_invalid_host_masses_are_dropped(self, tmp_path):
        cat = galaxies()
        bad = np.array([0.0, np.nan, -1.0])
        iz_dir = write_catalogue(
            tmp_path,
            1,
            0,
            mhhalo=np.concatenate([cat["mhhalo"], bad]),
            tree_mphalo=TREES,
        )
        res = hod_given_redshift_and_subvolume(str(iz_dir), 0, bins=BINS)
        np.testing.assert_array_equal(res["counts_galaxies"], [0, 6, 9])
        assert res["n_galaxies"] == 15

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"tree_mphalo": None},  # no Trees/mphalo
            {"mhhalo": None, "mhalo": None},  # no host-halo mass
        ],
        ids=["no-trees", "no-halo-mass"],
    )
    def test_unusable_file_returns_none(self, tmp_path, kwargs):
        cat = galaxies()
        cat.update(kwargs)
        trees = cat.pop("tree_mphalo", TREES)
        iz_dir = write_catalogue(tmp_path, 1, 0, tree_mphalo=trees, **cat)
        assert hod_given_redshift_and_subvolume(str(iz_dir), 0, bins=BINS) is None

    def test_stellar_mass_cut_without_stellar_mass_returns_none(self, tmp_path):
        cat = galaxies()
        iz_dir = write_catalogue(
            tmp_path, 1, 0, mhhalo=cat["mhhalo"], tree_mphalo=TREES
        )
        res = hod_given_redshift_and_subvolume(
            str(iz_dir), 0, bins=BINS, galaxy_stellar_mass_min=1e10
        )
        assert res is None

    def test_no_output_group_returns_none(self, tmp_path):
        import h5py

        ivol_dir = tmp_path / "iz1" / "ivol0"
        ivol_dir.mkdir(parents=True)
        with h5py.File(ivol_dir / "galaxies.hdf5", "w") as f:
            f.create_group("Trees").create_dataset("mphalo", data=TREES)
        assert hod_given_redshift_and_subvolume(str(tmp_path / "iz1"), 0) is None

    def test_missing_ivol_returns_none(self, tmp_path):
        iz_dir = write(tmp_path)
        assert hod_given_redshift_and_subvolume(str(iz_dir), 3) is None


class TestTwoHistogramCore:
    def test_no_tree_haloes_gives_zeros(self):
        res = _compute_hod_two_histogram(
            galaxy_mhhalo=np.array([M12]), tree_mphalo=np.array([]), bins=BINS
        )
        np.testing.assert_array_equal(res["mean_occupation"], np.zeros(3))
        np.testing.assert_array_equal(res["counts_halos"], np.zeros(3))
        assert res["mean_central"] is None

    def test_empty_bins_have_zero_occupation_not_nan(self):
        res = _compute_hod_two_histogram(
            galaxy_mhhalo=np.array([M12, M12]),
            tree_mphalo=np.array([M12]),
            bins=BINS,
        )
        np.testing.assert_array_equal(res["mean_occupation"], [0.0, 2.0, 0.0])


# ── combining subvolumes ─────────────────────────────────────────────────────


class TestCombineSubvolumes:
    def test_identical_subvolumes_give_same_hod(self, tmp_path):
        write(tmp_path, ivol=0)
        write(tmp_path, ivol=1)
        res = avg_hod_given_redshift_and_subvolumes(
            155, [0, 1, 9], bins=BINS, base_dir=str(tmp_path)
        )
        assert_matches(
            res,
            dict(EXPECTED, counts_halos=[2, 8, 4], counts_galaxies=[0, 12, 18]),
        )
        assert (res["n_used"], res["n_requested"]) == (2, 3)
        assert (res["n_halos"], res["n_galaxies"]) == (14, 30)
        assert res["iz"] == "iz155"
        assert res["z"] == pytest.approx(1.0)

    def test_pools_counts_before_dividing(self, tmp_path):
        # ivol0: 4 haloes, 6 galaxies in bin 2. ivol1: 1 halo, 4 galaxies.
        write_catalogue(
            tmp_path,
            1,
            0,
            mhhalo=masses_at([(12.5, 6)]),
            tree_mphalo=masses_at([(12.5, 4)]),
        )
        write_catalogue(
            tmp_path,
            1,
            1,
            mhhalo=masses_at([(12.5, 4)]),
            tree_mphalo=masses_at([(12.5, 1)]),
        )
        res = avg_hod_given_redshift_and_subvolumes(
            1, [0, 1], bins=BINS, base_dir=str(tmp_path)
        )
        # (6 + 4) / (4 + 1) = 2, not the mean of ratios (1.5 + 4) / 2.
        np.testing.assert_allclose(res["mean_occupation"], [0.0, 2.0, 0.0])

    def test_stellar_mass_cut_and_mass_limit(self, tmp_path):
        write(tmp_path, ivol=0)
        write(tmp_path, ivol=1)
        res = avg_hod_given_redshift_and_subvolumes(
            155,
            [0, 1],
            bins=BINS,
            base_dir=str(tmp_path),
            galaxy_stellar_mass_min=1e10,
            halo_mass_lower_limit=1e13,
        )
        np.testing.assert_allclose(res["mean_occupation"], [0.0, 0.0, 1.5])
        assert res["n_galaxies"] == 14  # centrals in both bins, both ivols
        assert res["selection"]["halo_mass_lower_limit"] == 1e13

    def test_mixed_central_flags_drop_decomposition(self, tmp_path):
        write(tmp_path, ivol=0)
        cat = galaxies()
        write_catalogue(
            tmp_path,
            155,
            1,
            mhhalo=cat["mhhalo"],
            mhalo=cat["mhalo"],
            tree_mphalo=TREES,
        )
        res = avg_hod_given_redshift_and_subvolumes(
            155, [0, 1], bins=BINS, base_dir=str(tmp_path)
        )
        np.testing.assert_allclose(res["mean_occupation"], [0.0, 1.5, 4.5])
        assert res["mean_central"] is None

    def test_missing_snapshot_or_no_usable_ivol(self, tmp_path):
        write(tmp_path)
        assert (
            avg_hod_given_redshift_and_subvolumes(999, [0], base_dir=str(tmp_path))
            is None
        )
        assert (
            avg_hod_given_redshift_and_subvolumes(155, [4], base_dir=str(tmp_path))
            is None
        )

    def test_uses_configured_base_dir(self, configured_base):
        write(configured_base)
        res = avg_hod_given_redshift_and_subvolumes(155, [0])
        assert res["counts_halos"].sum() == 7


# ── across snapshots ─────────────────────────────────────────────────────────


class TestAcrossSnapshots:
    @pytest.fixture
    def base(self, tmp_path):
        write(tmp_path, iz=100, redshift=1.0)  # <N> = [0, 1.5, 4.5]
        # Duplicating the is_central=0 galaxies: <N> = [0, 2, 7.5], cen 1,
        # sat [0, 1, 6.5].
        cat = galaxies()
        sat = cat["is_central"] == 0
        extra = {k: v[sat] for k, v in cat.items()}
        doubled = {k: np.concatenate([cat[k], extra[k]]) for k in cat}
        write_catalogue(tmp_path, 207, 0, tree_mphalo=TREES, redshift=0.5, **doubled)
        (tmp_path / "iz300").mkdir()  # snapshot dir without ivol0
        return tmp_path

    def test_mean_and_std(self, base):
        res = avg_hod_given_redshifts_and_subvolume(
            0, [100, 207, 300, 999], bins=BINS, base_dir=str(base)
        )
        np.testing.assert_allclose(res["mean_occupation"], [0.0, 1.75, 6.0])
        np.testing.assert_allclose(res["mean_occupation_std"], [0.0, 0.25, 1.5])
        np.testing.assert_allclose(res["mean_central"], [0.0, 1.0, 1.0])
        np.testing.assert_allclose(res["mean_satellite"], [0.0, 0.75, 5.0])
        assert res["iz_list"] == ["iz100", "iz207"]
        np.testing.assert_allclose(res["z_list"], [1.0, 0.5])
        assert (res["n_used"], res["n_requested"]) == (2, 4)

    def test_no_central_flags(self, tmp_path):
        cat = galaxies()
        write_catalogue(tmp_path, 1, 0, mhhalo=cat["mhhalo"], tree_mphalo=TREES)
        res = avg_hod_given_redshifts_and_subvolume(
            0, [1], bins=BINS, base_dir=str(tmp_path)
        )
        assert res["mean_central"] is None and res["mean_satellite"] is None

    def test_nothing_usable_returns_none(self, base):
        assert (
            avg_hod_given_redshifts_and_subvolume(0, [300, 999], base_dir=str(base))
            is None
        )

    def test_long_form_dataframe(self, base):
        df, per_z = hods_given_redshifts_and_subvolume(
            0, [100, 207, 300], base_dir=str(base), galaxy_stellar_mass_min=1e10
        )
        nbins = len(DEFAULT_HALO_MASS_BINS) - 1
        assert isinstance(df, pl.DataFrame)
        assert df.height == 2 * nbins
        assert {
            "iz",
            "iz_num",
            "z",
            "log_M",
            "mean_occupation",
            "counts_halos",
            "mean_central",
            "mean_satellite",
        } <= set(df.columns)
        assert [r["iz"] for r in per_z] == ["iz100", "iz207"]
        # With the cut, both snapshots have <N> = 1.5 in the 10^13.5 bin.
        i13 = np.searchsorted(DEFAULT_HALO_MASS_BINS, 13.5) - 1
        for r in per_z:
            assert r["mean_occupation"][i13] == pytest.approx(1.5)

    def test_long_form_without_decomposition(self, tmp_path):
        cat = galaxies()
        write_catalogue(tmp_path, 1, 0, mhhalo=cat["mhhalo"], tree_mphalo=TREES)
        df, _ = hods_given_redshifts_and_subvolume(0, [1], base_dir=str(tmp_path))
        assert "mean_central" not in df.columns

    def test_long_form_nothing_usable_returns_none(self, base):
        assert hods_given_redshifts_and_subvolume(0, [999], base_dir=str(base)) is None

    def test_uses_configured_base_dir(self, configured_base):
        write(configured_base, iz=3)
        assert avg_hod_given_redshifts_and_subvolume(0, [3])["iz_list"] == ["iz3"]
        df, _ = hods_given_redshifts_and_subvolume(0, [3])
        assert df["iz_num"].unique().to_list() == [3]
