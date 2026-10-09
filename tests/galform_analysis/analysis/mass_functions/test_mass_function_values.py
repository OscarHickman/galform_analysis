"""Value tests for the SMF and HMF helpers on catalogues with known content.

Each catalogue puts a known number of objects in each mass bin, so phi is
known exactly: phi = counts / (dlog10M * volume).
"""

import numpy as np
import polars as pl
import pytest
from mf_catalogue import masses_at, write_catalogue

import galform_analysis.config as config
from galform_analysis.analysis.mass_functions import (
    avg_hmf_given_redshift_and_subvolumes,
    avg_hmf_given_redshifts_and_subvolume,
    avg_smf_given_redshift_and_subvolumes,
    avg_smf_given_redshifts_and_subvolume,
    compute_hmf_from_aggregated,
    compute_smf_from_aggregated,
    hmf_given_redshift_and_subvolume,
    hmfs_given_redshifts_and_subvolume,
    smf_given_redshift_and_subvolume,
    smfs_given_redshifts_and_subvolume,
)
from galform_analysis.config import DEFAULT_HALO_MASS_BINS, DEFAULT_STELLAR_MASS_BINS

# Unequal widths so a wrong dlogM normalisation cannot cancel out.
BINS = np.array([9.0, 10.0, 10.5, 12.0])
WIDTHS = np.diff(BINS)
JUNK = np.array([0.0, -1e10, np.nan, np.inf, 1e8, 1e13])  # dropped or out of range


def catalogue(counts):
    """Masses with counts[i] objects at the centre of BINS[i]."""
    centres = 0.5 * (BINS[1:] + BINS[:-1])
    return np.concatenate([masses_at(zip(centres, counts)), JUNK])


def expected_phi(counts, volume):
    return np.asarray(counts, dtype=float) / (WIDTHS * volume)


@pytest.fixture
def configured_base(tmp_path, monkeypatch):
    """Install tmp_path as the configured base directory for the test."""
    monkeypatch.setattr(config, "BASE_DIR", config.BASE_DIR)
    config.set_base_dir(str(tmp_path))
    return tmp_path


# ── single subvolume ─────────────────────────────────────────────────────────


class TestSingleSubvolume:
    @pytest.mark.parametrize(
        "fn, field",
        [
            (smf_given_redshift_and_subvolume, "mstar"),
            (hmf_given_redshift_and_subvolume, "mhalo"),
        ],
    )
    def test_exact_phi_counts_and_centres(self, tmp_path, fn, field):
        counts = [3, 5, 2]
        iz_dir = write_catalogue(
            tmp_path, 155, 0, volume=250.0, redshift=1.5, **{field: catalogue(counts)}
        )
        res = fn(str(iz_dir), 0, bins=BINS)

        np.testing.assert_array_equal(res["counts"], counts)
        np.testing.assert_allclose(res["phi"], expected_phi(counts, 250.0))
        np.testing.assert_allclose(res["centers"], [9.5, 10.25, 11.25])
        assert res["iz"] == "iz155"
        assert res["ivol"] == 0
        assert res["z"] == pytest.approx(1.5)
        assert res["V_ivol"] == pytest.approx(250.0)

    def test_smf_sums_disk_and_bulge(self, tmp_path):
        # Disk and bulge are each below 10^10.5; only their sum is in bin 3.
        iz_dir = write_catalogue(tmp_path, 1, 0, mstar=np.full(4, 10**11.0))
        res = smf_given_redshift_and_subvolume(str(iz_dir), 0, bins=BINS)
        np.testing.assert_array_equal(res["counts"], [0, 0, 4])

    def test_smf_falls_back_to_total_stellar_mass(self, tmp_path):
        iz_dir = write_catalogue(
            tmp_path, 1, 0, mstar=catalogue([1, 2, 3]), split_mstar=False
        )
        res = smf_given_redshift_and_subvolume(str(iz_dir), 0, bins=BINS)
        np.testing.assert_array_equal(res["counts"], [1, 2, 3])

    def test_hmf_lower_mass_limit(self, tmp_path):
        iz_dir = write_catalogue(tmp_path, 1, 0, mhalo=catalogue([3, 5, 2]))
        res = hmf_given_redshift_and_subvolume(
            str(iz_dir), 0, bins=BINS, halo_mass_lower_limit=10**10.1
        )
        np.testing.assert_array_equal(res["counts"], [0, 5, 2])

    def test_default_bins(self, tmp_path):
        iz_dir = write_catalogue(
            tmp_path, 1, 0, mstar=catalogue([1, 1, 1]), mhalo=catalogue([1, 1, 1])
        )
        smf = smf_given_redshift_and_subvolume(str(iz_dir), 0)
        hmf = hmf_given_redshift_and_subvolume(str(iz_dir), 0)
        assert len(smf["phi"]) == len(DEFAULT_STELLAR_MASS_BINS) - 1
        assert len(hmf["phi"]) == len(DEFAULT_HALO_MASS_BINS) - 1

    @pytest.mark.parametrize(
        "fn, field",
        [
            (smf_given_redshift_and_subvolume, "mstar"),
            (hmf_given_redshift_and_subvolume, "mhalo"),
        ],
    )
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"volume": None},  # no Parameters/volume
            {"volume": 0.0},
            {"masses": np.array([0.0, -1.0, np.nan])},  # nothing valid
            {"masses": None},  # field missing entirely
        ],
        ids=["no-volume", "zero-volume", "no-valid-masses", "missing-field"],
    )
    def test_unusable_subvolume_returns_none(self, tmp_path, fn, field, kwargs):
        kwargs = dict(kwargs)
        masses = kwargs.pop("masses", catalogue([1, 1, 1]))
        iz_dir = write_catalogue(tmp_path, 1, 0, **{field: masses}, **kwargs)
        assert fn(str(iz_dir), 0, bins=BINS) is None

    @pytest.mark.parametrize(
        "fn", [smf_given_redshift_and_subvolume, hmf_given_redshift_and_subvolume]
    )
    def test_missing_ivol_returns_none(self, tmp_path, fn):
        iz_dir = write_catalogue(tmp_path, 1, 0, mstar=catalogue([1, 1, 1]))
        assert fn(str(iz_dir), 7, bins=BINS) is None


# ── averaging over subvolumes ────────────────────────────────────────────────


class TestAverageOverSubvolumes:
    COUNTS = {0: [2, 4, 6], 1: [4, 0, 2]}

    @pytest.fixture
    def base(self, tmp_path):
        for ivol, counts in self.COUNTS.items():
            write_catalogue(
                tmp_path,
                207,
                ivol,
                mstar=catalogue(counts),
                mhalo=catalogue(counts),
                volume=100.0,
                redshift=0.5,
            )
        return tmp_path

    def test_smf_mean_and_std(self, base):
        res = avg_smf_given_redshift_and_subvolumes(
            207, [0, 1, 9], bins=BINS, base_dir=str(base)
        )
        per = np.array([expected_phi(c, 100.0) for c in self.COUNTS.values()])
        np.testing.assert_allclose(res["phi"], per.mean(axis=0))
        np.testing.assert_allclose(res["phi_std"], per.std(axis=0))
        np.testing.assert_allclose(res["centers"], [9.5, 10.25, 11.25])
        assert (res["n_used"], res["n_requested"]) == (2, 3)
        assert res["iz"] == "iz207"
        assert res["z"] == pytest.approx(0.5)

    def test_hmf_pools_counts_over_summed_volume(self, base):
        res = avg_hmf_given_redshift_and_subvolumes(
            207, [0, 1, 9], bins=BINS, base_dir=str(base)
        )
        total = np.add(*self.COUNTS.values())
        np.testing.assert_array_equal(res["counts"], total)
        np.testing.assert_allclose(res["phi"], expected_phi(total, 200.0))
        assert res["V_total"] == pytest.approx(200.0)
        assert res["V_ivol"] == pytest.approx(100.0)
        assert (res["n_used"], res["n_requested"]) == (2, 3)
        assert res["z"] == pytest.approx(0.5)

    def test_hmf_volume_counts_only_usable_subvolumes(self, tmp_path):
        write_catalogue(tmp_path, 1, 0, mhalo=catalogue([1, 2, 3]), volume=100.0)
        # ivol1 has a volume but no valid halo masses, so must add no volume.
        write_catalogue(tmp_path, 1, 1, mhalo=np.zeros(3), volume=100.0)
        res = avg_hmf_given_redshift_and_subvolumes(
            1, [0, 1], bins=BINS, base_dir=str(tmp_path)
        )
        np.testing.assert_allclose(res["phi"], expected_phi([1, 2, 3], 100.0))
        assert res["V_total"] == pytest.approx(100.0)
        assert res["n_used"] == 1

    def test_hmf_mass_limit_applies_to_every_subvolume(self, base):
        res = avg_hmf_given_redshift_and_subvolumes(
            207, [0, 1], bins=BINS, base_dir=str(base), halo_mass_lower_limit=1e10
        )
        np.testing.assert_array_equal(res["counts"], [0, 4, 8])

    @pytest.mark.parametrize(
        "fn",
        [avg_smf_given_redshift_and_subvolumes, avg_hmf_given_redshift_and_subvolumes],
    )
    def test_missing_snapshot_or_no_valid_ivol_returns_none(self, base, fn):
        assert fn(999, [0, 1], bins=BINS, base_dir=str(base)) is None
        assert fn(207, [5, 6], bins=BINS, base_dir=str(base)) is None

    @pytest.mark.parametrize(
        "fn",
        [avg_smf_given_redshift_and_subvolumes, avg_hmf_given_redshift_and_subvolumes],
    )
    def test_uses_configured_base_dir_and_default_bins(self, configured_base, fn):
        write_catalogue(
            configured_base,
            3,
            0,
            mstar=catalogue([1, 1, 1]),
            mhalo=catalogue([1, 1, 1]),
        )
        res = fn(3, [0])
        assert res is not None
        assert res["n_used"] == 1


# ── averaging over snapshots ─────────────────────────────────────────────────


class TestAverageOverSnapshots:
    COUNTS = {100: [1, 2, 3], 155: [3, 2, 1], 207: [5, 5, 5]}

    @pytest.fixture
    def base(self, tmp_path):
        for iz, counts in self.COUNTS.items():
            write_catalogue(
                tmp_path,
                iz,
                0,
                mstar=catalogue(counts),
                mhalo=catalogue(counts),
                volume=50.0,
                redshift=iz / 100.0,
            )
        # A snapshot directory whose ivol0 is unusable.
        write_catalogue(tmp_path, 300, 0, mstar=np.zeros(2), mhalo=np.zeros(2))
        return tmp_path

    @pytest.mark.parametrize(
        "fn",
        [avg_smf_given_redshifts_and_subvolume, avg_hmf_given_redshifts_and_subvolume],
    )
    def test_mean_and_std_over_snapshots(self, base, fn):
        res = fn(0, [100, 155, 207, 300, 999], bins=BINS, base_dir=str(base))
        per = np.array([expected_phi(c, 50.0) for c in self.COUNTS.values()])
        np.testing.assert_allclose(res["phi"], per.mean(axis=0))
        np.testing.assert_allclose(res["phi_std"], per.std(axis=0))
        np.testing.assert_allclose(res["centers"], [9.5, 10.25, 11.25])
        assert res["iz_list"] == ["iz100", "iz155", "iz207"]
        np.testing.assert_allclose(res["z_list"], [1.0, 1.55, 2.07])
        assert (res["ivol"], res["n_used"], res["n_requested"]) == (0, 3, 5)

    def test_hmf_mass_limit_is_passed_through(self, base):
        res = avg_hmf_given_redshifts_and_subvolume(
            0, [100], bins=BINS, base_dir=str(base), halo_mass_lower_limit=1e10
        )
        np.testing.assert_allclose(res["phi"], expected_phi([0, 2, 3], 50.0))

    @pytest.mark.parametrize(
        "fn",
        [avg_smf_given_redshifts_and_subvolume, avg_hmf_given_redshifts_and_subvolume],
    )
    def test_nothing_usable_returns_none(self, base, fn):
        assert fn(0, [300, 999], bins=BINS, base_dir=str(base)) is None

    @pytest.mark.parametrize(
        "fn",
        [avg_smf_given_redshifts_and_subvolume, avg_hmf_given_redshifts_and_subvolume],
    )
    def test_uses_configured_base_dir(self, configured_base, fn):
        write_catalogue(
            configured_base,
            3,
            0,
            mstar=catalogue([1, 1, 1]),
            mhalo=catalogue([1, 1, 1]),
        )
        res = fn(0, [3])
        assert res["iz_list"] == ["iz3"]


# ── long-form DataFrames over snapshots ──────────────────────────────────────


class TestLongFormOverSnapshots:
    @pytest.fixture
    def base(self, tmp_path):
        for iz, z in ((100, 1.0), (155, 0.5)):
            write_catalogue(
                tmp_path,
                iz,
                0,
                mstar=masses_at([(10.5, 4)]),
                mhalo=masses_at([(10.5, 2), (12.5, 6)]),
                volume=10.0,
                redshift=z,
            )
        return tmp_path

    def test_smf_rows(self, base):
        df = smfs_given_redshifts_and_subvolume(0, [100, 155, 999], base_dir=str(base))
        assert isinstance(df, pl.DataFrame)
        assert df.columns == ["iz", "iz_num", "z", "log_M", "phi", "counts"]
        nbins = len(DEFAULT_STELLAR_MASS_BINS) - 1
        assert df.height == 2 * nbins
        one = df.filter(pl.col("iz") == "iz100")
        assert one["counts"].sum() == 4
        hit = one.filter(pl.col("counts") > 0)
        width = np.diff(DEFAULT_STELLAR_MASS_BINS)[0]
        assert hit["phi"][0] == pytest.approx(4 / (width * 10.0))
        assert set(df["z"].to_list()) == {1.0, 0.5}

    def test_hmf_rows_and_mass_limit(self, base):
        df = hmfs_given_redshifts_and_subvolume(
            0, [100, 155], base_dir=str(base), halo_mass_lower_limit=1e11
        )
        assert df.height == 2 * (len(DEFAULT_HALO_MASS_BINS) - 1)
        per_iz = df.group_by("iz").agg(pl.col("counts").sum()).sort("iz")
        assert per_iz["counts"].to_list() == [6, 6]

    @pytest.mark.parametrize(
        "fn", [smfs_given_redshifts_and_subvolume, hmfs_given_redshifts_and_subvolume]
    )
    def test_nothing_usable_returns_none(self, base, fn):
        assert fn(0, [999], base_dir=str(base)) is None

    @pytest.mark.parametrize(
        "fn", [smfs_given_redshifts_and_subvolume, hmfs_given_redshifts_and_subvolume]
    )
    def test_uses_configured_base_dir(self, configured_base, fn):
        write_catalogue(
            configured_base,
            3,
            0,
            mstar=masses_at([(10.5, 1)]),
            mhalo=masses_at([(12.5, 1)]),
        )
        assert fn(0, [3])["iz_num"].unique().to_list() == [3]


# ── from aggregated arrays ───────────────────────────────────────────────────


class TestFromAggregated:
    @pytest.mark.parametrize(
        "fn, key",
        [
            (compute_smf_from_aggregated, "mstar"),
            (compute_hmf_from_aggregated, "mhalo"),
        ],
    )
    def test_exact_phi(self, fn, key):
        agg = {key: catalogue([2, 0, 7]), "volume": 400.0, "iz": "iz155", "z": 1.2}
        res = fn(agg, bins=BINS)
        np.testing.assert_array_equal(res["counts"], [2, 0, 7])
        np.testing.assert_allclose(res["phi"], expected_phi([2, 0, 7], 400.0))
        np.testing.assert_allclose(res["centers"], [9.5, 10.25, 11.25])
        assert (res["iz"], res["z"]) == ("iz155", 1.2)

    def test_hmf_lower_mass_limit(self):
        agg = {"mhalo": catalogue([2, 3, 7]), "volume": 1.0, "iz": "iz1", "z": 0.0}
        res = compute_hmf_from_aggregated(agg, bins=BINS, halo_mass_lower_limit=1e11)
        np.testing.assert_array_equal(res["counts"], [0, 0, 7])

    @pytest.mark.parametrize(
        "fn, key",
        [
            (compute_smf_from_aggregated, "mstar"),
            (compute_hmf_from_aggregated, "mhalo"),
        ],
    )
    def test_default_bins(self, fn, key):
        res = fn({key: np.array([1e10, 1e12]), "volume": 1.0, "iz": "iz1", "z": 0.0})
        assert res["counts"].sum() == 2

    @pytest.mark.parametrize(
        "fn, key",
        [
            (compute_smf_from_aggregated, "mstar"),
            (compute_hmf_from_aggregated, "mhalo"),
        ],
    )
    @pytest.mark.parametrize(
        "agg",
        [
            None,
            {"volume": 1.0},
            {"masses": np.array([1e10]), "volume": 0.0},
            {"masses": np.array([1e10])},
            {"masses": np.array([0.0, np.nan]), "volume": 1.0},
        ],
        ids=["none", "no-masses", "zero-volume", "no-volume", "no-valid-masses"],
    )
    def test_insufficient_data_returns_none(self, fn, key, agg):
        if agg is not None:
            agg = dict(agg, iz="iz1", z=0.0)
            if "masses" in agg:
                agg[key] = agg.pop("masses")
        assert fn(agg, bins=BINS) is None
