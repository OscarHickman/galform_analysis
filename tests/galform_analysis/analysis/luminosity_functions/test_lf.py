"""Value tests for the luminosity function helpers on known catalogues.

Each catalogue puts a known number of galaxies at the centre of each magnitude
bin, so phi is known exactly: phi = counts / (dM * volume).
"""

import h5py
import numpy as np
import polars as pl
import pytest

import galform_analysis
import galform_analysis.config as config
from galform_analysis.analysis.luminosity_functions import (
    avg_lf_given_redshift_and_subvolumes,
    avg_lf_given_redshifts_and_subvolume,
    lf_given_redshift_and_subvolume,
    lfs_given_redshifts_and_subvolume,
    schechter_magnitude,
)
from galform_analysis.config import DEFAULT_MAGNITUDE_BINS
from tests.band_catalogue import (
    magnitudes_to_luminosity,
    mags_at,
    write_band_catalogue,
)

# Unequal widths so a wrong dM normalisation cannot cancel out.
BINS = np.array([-24.0, -22.0, -21.5, -20.0])
WIDTHS = np.diff(BINS)
CENTRES = 0.5 * (BINS[1:] + BINS[:-1])
K_REST, R_REST, K_OBS = 1, 2, 4  # indices in tests.band_catalogue.DEFAULT_BANDS


def catalogue_lum(counts, extra=()):
    """Luminosities with counts[i] galaxies at the centre of BINS[i]."""
    mags = mags_at(zip(CENTRES, counts))
    lum = magnitudes_to_luminosity(mags)
    return np.concatenate([lum, np.asarray(extra, dtype=np.float64)])


# Dropped (zero, negative, NaN) or outside the bins (too faint / too bright).
JUNK = [0.0, -5.0, np.nan, 1.0, 1e12]


def expected_phi(counts, volume):
    return np.asarray(counts, dtype=float) / (WIDTHS * volume)


@pytest.fixture
def configured_base(tmp_path, monkeypatch):
    """Install tmp_path as the configured base directory for the test."""
    monkeypatch.setattr(config, "BASE_DIR", config.BASE_DIR)
    config.set_base_dir(str(tmp_path))
    return tmp_path


class TestSingleSubvolume:
    def test_known_counts(self, tmp_path):
        counts = [3, 5, 2]
        iz_dir = write_band_catalogue(
            tmp_path,
            271,
            0,
            lum_disk={K_REST: catalogue_lum(counts, JUNK)},
            volume=500.0,
            redshift=0.0,
        )
        res = lf_given_redshift_and_subvolume(str(iz_dir), 0, "UKIRT-K", bins=BINS)
        np.testing.assert_array_equal(res["counts"], counts)
        np.testing.assert_allclose(res["phi"], expected_phi(counts, 500.0))
        np.testing.assert_allclose(res["centers"], CENTRES)
        assert res["iz"] == "iz271"
        assert res["ivol"] == 0
        assert res["z"] == 0.0
        assert res["V_ivol"] == 500.0
        assert res["band"] == "UKIRT-K"
        assert res["band_index"] == K_REST
        assert res["frame"] == "rest"
        assert res["component"] == "total"
        assert res["system"] == "AB"

    def test_phi_recovers_counts(self, tmp_path):
        iz_dir = write_band_catalogue(
            tmp_path, 271, 0, lum_disk={R_REST: catalogue_lum([4, 1, 7])}
        )
        res = lf_given_redshift_and_subvolume(str(iz_dir), 0, "SDSS-r", bins=BINS)
        np.testing.assert_allclose(
            res["phi"] * WIDTHS * res["V_ivol"], res["counts"].astype(float)
        )

    def test_disk_and_bulge_are_summed(self, tmp_path):
        # Each galaxy splits L equally, so disk or bulge alone is 2.5log10(2)
        # = 0.753 mag fainter and moves one bin; the total is in the right bin.
        lum = catalogue_lum([0, 4, 0])
        iz_dir = write_band_catalogue(
            tmp_path,
            271,
            0,
            lum_disk={K_REST: 0.5 * lum},
            lum_bulge={K_REST: 0.5 * lum},
        )
        total = lf_given_redshift_and_subvolume(str(iz_dir), 0, "UKIRT-K", bins=BINS)
        disk = lf_given_redshift_and_subvolume(
            str(iz_dir), 0, "UKIRT-K", bins=BINS, component="disk"
        )
        bulge = lf_given_redshift_and_subvolume(
            str(iz_dir), 0, "UKIRT-K", bins=BINS, component="bulge"
        )
        np.testing.assert_array_equal(total["counts"], [0, 4, 0])
        np.testing.assert_array_equal(disk["counts"], [0, 0, 4])
        np.testing.assert_array_equal(bulge["counts"], [0, 0, 4])
        assert disk["component"] == "disk"

    def test_observer_frame_reads_its_own_band(self, tmp_path):
        iz_dir = write_band_catalogue(
            tmp_path,
            207,
            0,
            lum_disk={
                K_REST: catalogue_lum([1, 0, 0]),
                K_OBS: catalogue_lum([0, 0, 6]),
            },
        )
        res = lf_given_redshift_and_subvolume(
            str(iz_dir), 0, "UKIRT-K", bins=BINS, frame="observer"
        )
        np.testing.assert_array_equal(res["counts"], [0, 0, 6])
        assert res["band_index"] == K_OBS
        assert res["frame"] == "observer"

    def test_vega_system_shifts_by_a0_ab(self, tmp_path):
        # K-band a0_ab = -1.87 in the catalogue: Vega magnitudes are 1.87 mag
        # brighter, which moves galaxies from the faint bin to the bright bin.
        mags = np.full(5, -20.5)
        iz_dir = write_band_catalogue(
            tmp_path, 271, 0, lum_disk={K_REST: magnitudes_to_luminosity(mags)}
        )
        ab = lf_given_redshift_and_subvolume(str(iz_dir), 0, "UKIRT-K", bins=BINS)
        vega = lf_given_redshift_and_subvolume(
            str(iz_dir), 0, "UKIRT-K", bins=BINS, system="Vega"
        )
        np.testing.assert_array_equal(ab["counts"], [0, 0, 5])
        np.testing.assert_array_equal(vega["counts"], [5, 0, 0])
        assert vega["system"] == "Vega"

    def test_bad_system_raises(self, tmp_path):
        with pytest.raises(ValueError, match="system"):
            lf_given_redshift_and_subvolume(
                str(tmp_path / "iz1"), 0, "UKIRT-K", system="ST"
            )

    def test_integer_band_index(self, tmp_path):
        iz_dir = write_band_catalogue(
            tmp_path, 271, 0, lum_disk={K_OBS: catalogue_lum([2, 2, 2])}
        )
        res = lf_given_redshift_and_subvolume(str(iz_dir), 0, K_OBS, bins=BINS)
        np.testing.assert_array_equal(res["counts"], [2, 2, 2])
        assert res["band"] == "UKIRT-K"
        assert res["frame"] == "observer"

    def test_default_bins(self, tmp_path):
        iz_dir = write_band_catalogue(
            tmp_path, 271, 0, lum_disk={K_REST: catalogue_lum([1, 1, 1])}
        )
        res = lf_given_redshift_and_subvolume(str(iz_dir), 0, "UKIRT-K")
        assert len(res["phi"]) == len(DEFAULT_MAGNITUDE_BINS) - 1
        assert res["counts"].sum() == 3

    def test_default_bins_span_bright_to_faint(self):
        assert np.all(np.diff(DEFAULT_MAGNITUDE_BINS) > 0)
        assert DEFAULT_MAGNITUDE_BINS[0] <= -25.0
        assert DEFAULT_MAGNITUDE_BINS[-1] >= -14.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"volume": None},
            {"volume": 0.0},
            {"write_band_table": False},
            {"write_band_lums": False},
        ],
    )
    def test_unusable_subvolume_returns_none(self, tmp_path, kwargs):
        iz_dir = write_band_catalogue(
            tmp_path, 271, 0, lum_disk={K_REST: catalogue_lum([1, 1, 1])}, **kwargs
        )
        assert (
            lf_given_redshift_and_subvolume(str(iz_dir), 0, "UKIRT-K", bins=BINS)
            is None
        )

    def test_band_group_without_bandname_returns_none(self, tmp_path):
        iz_dir = write_band_catalogue(
            tmp_path, 271, 0, lum_disk={K_REST: catalogue_lum([1, 1, 1])}
        )
        with h5py.File(iz_dir / "ivol0" / "galaxies.hdf5", "a") as f:
            del f["Bands/bandname"]
        assert (
            lf_given_redshift_and_subvolume(str(iz_dir), 0, "UKIRT-K", bins=BINS)
            is None
        )

    def test_no_luminous_galaxies_returns_none(self, tmp_path):
        iz_dir = write_band_catalogue(tmp_path, 271, 0, lum_disk={K_REST: np.zeros(5)})
        assert (
            lf_given_redshift_and_subvolume(str(iz_dir), 0, "UKIRT-K", bins=BINS)
            is None
        )

    def test_missing_file_returns_none(self, tmp_path):
        assert (
            lf_given_redshift_and_subvolume(str(tmp_path / "iz1"), 0, "UKIRT-K") is None
        )

    def test_unknown_band_raises(self, tmp_path):
        iz_dir = write_band_catalogue(
            tmp_path, 271, 0, lum_disk={K_REST: catalogue_lum([1, 1, 1])}
        )
        with pytest.raises(ValueError, match="Bj"):
            lf_given_redshift_and_subvolume(str(iz_dir), 0, "Bj")

    def test_bad_component_raises_even_without_data(self, tmp_path):
        with pytest.raises(ValueError, match="component"):
            lf_given_redshift_and_subvolume(
                str(tmp_path / "iz1"), 0, "UKIRT-K", component="burst"
            )


class TestAverages:
    def test_over_subvolumes(self, tmp_path):
        write_band_catalogue(
            tmp_path, 271, 0, lum_disk={K_REST: catalogue_lum([2, 0, 4])}
        )
        write_band_catalogue(
            tmp_path, 271, 1, lum_disk={K_REST: catalogue_lum([4, 2, 0])}
        )
        res = avg_lf_given_redshift_and_subvolumes(
            271, [0, 1, 7], "UKIRT-K", bins=BINS, base_dir=str(tmp_path)
        )
        phi0, phi1 = expected_phi([2, 0, 4], 1000.0), expected_phi([4, 2, 0], 1000.0)
        np.testing.assert_allclose(res["phi"], 0.5 * (phi0 + phi1))
        np.testing.assert_allclose(res["phi_std"], 0.5 * np.abs(phi0 - phi1))
        np.testing.assert_allclose(res["centers"], CENTRES)
        assert res["iz"] == "iz271"
        assert res["n_used"] == 2
        assert res["n_requested"] == 3
        assert res["band"] == "UKIRT-K"
        assert res["frame"] == "rest"
        assert res["system"] == "AB"

    def test_over_subvolumes_uses_configured_base(self, configured_base):
        write_band_catalogue(
            configured_base, 271, 0, lum_disk={K_REST: catalogue_lum([1, 2, 3])}
        )
        res = avg_lf_given_redshift_and_subvolumes(271, [0], "UKIRT-K", bins=BINS)
        np.testing.assert_allclose(res["phi"], expected_phi([1, 2, 3], 1000.0))

    def test_over_subvolumes_missing_snapshot(self, tmp_path):
        assert (
            avg_lf_given_redshift_and_subvolumes(
                1, [0], "UKIRT-K", base_dir=str(tmp_path)
            )
            is None
        )

    def test_over_subvolumes_none_usable(self, tmp_path):
        write_band_catalogue(tmp_path, 271, 0, lum_disk={K_REST: np.zeros(3)})
        assert (
            avg_lf_given_redshift_and_subvolumes(
                271, [0, 1], "UKIRT-K", base_dir=str(tmp_path)
            )
            is None
        )

    def test_over_snapshots(self, tmp_path):
        write_band_catalogue(
            tmp_path,
            207,
            0,
            lum_disk={K_REST: catalogue_lum([1, 1, 1])},
            redshift=0.5,
        )
        write_band_catalogue(
            tmp_path,
            271,
            0,
            lum_disk={K_REST: catalogue_lum([3, 1, 5])},
            redshift=0.0,
        )
        res = avg_lf_given_redshifts_and_subvolume(
            0, [207, 271, 999], "UKIRT-K", bins=BINS, base_dir=str(tmp_path)
        )
        phi_a, phi_b = expected_phi([1, 1, 1], 1000.0), expected_phi([3, 1, 5], 1000.0)
        np.testing.assert_allclose(res["phi"], 0.5 * (phi_a + phi_b))
        assert res["iz_list"] == ["iz207", "iz271"]
        assert res["z_list"] == [0.5, 0.0]
        assert res["n_used"] == 2
        assert res["n_requested"] == 3
        assert res["band"] == "UKIRT-K"
        assert res["band_index"] == K_REST
        assert res["system"] == "AB"

    def test_over_snapshots_reports_resolved_band(self, tmp_path):
        write_band_catalogue(
            tmp_path, 271, 0, lum_disk={K_OBS: catalogue_lum([1, 1, 1])}
        )
        res = avg_lf_given_redshifts_and_subvolume(
            0, [271], K_OBS, bins=BINS, base_dir=str(tmp_path)
        )
        assert res["band"] == "UKIRT-K"
        assert res["band_index"] == K_OBS
        assert res["frame"] == "observer"

    def test_over_snapshots_none_usable(self, tmp_path):
        assert (
            avg_lf_given_redshifts_and_subvolume(
                0, [271], "UKIRT-K", base_dir=str(tmp_path)
            )
            is None
        )

    def test_over_snapshots_uses_configured_base(self, configured_base):
        write_band_catalogue(
            configured_base, 271, 0, lum_disk={K_REST: catalogue_lum([1, 0, 0])}
        )
        res = avg_lf_given_redshifts_and_subvolume(0, [271], "UKIRT-K", bins=BINS)
        np.testing.assert_allclose(res["phi"], expected_phi([1, 0, 0], 1000.0))


class TestLongForm:
    def test_dataframe(self, tmp_path):
        for iz_num, z, counts in ((207, 0.5, [1, 2, 3]), (271, 0.0, [0, 4, 1])):
            write_band_catalogue(
                tmp_path,
                iz_num,
                0,
                lum_disk={K_REST: catalogue_lum(counts)},
                redshift=z,
            )
        df = lfs_given_redshifts_and_subvolume(
            0, [207, 271, 999], "UKIRT-K", bins=BINS, base_dir=str(tmp_path)
        )
        assert isinstance(df, pl.DataFrame)
        assert df.columns == ["iz", "iz_num", "z", "mag", "phi", "counts"]
        assert df.height == 6
        late = df.filter(pl.col("iz_num") == 271)
        np.testing.assert_allclose(late["mag"].to_numpy(), CENTRES)
        np.testing.assert_array_equal(late["counts"].to_numpy(), [0, 4, 1])
        np.testing.assert_allclose(
            late["phi"].to_numpy(), expected_phi([0, 4, 1], 1000.0)
        )
        assert late["z"].unique().to_list() == [0.0]

    def test_no_data_returns_none(self, tmp_path):
        assert (
            lfs_given_redshifts_and_subvolume(
                0, [271], "UKIRT-K", base_dir=str(tmp_path)
            )
            is None
        )

    def test_uses_configured_base(self, configured_base):
        write_band_catalogue(
            configured_base, 271, 0, lum_disk={K_REST: catalogue_lum([1, 1, 1])}
        )
        df = lfs_given_redshifts_and_subvolume(0, [271], "UKIRT-K", bins=BINS)
        assert df["counts"].sum() == 3


class TestSchechter:
    def test_value_at_knee(self):
        # At M = M*, phi = 0.4 ln10 phi* exp(-1) for any alpha.
        for alpha in (-1.5, -1.0, -0.5):
            val = schechter_magnitude(-20.0, phi_star=0.01, M_star=-20.0, alpha=alpha)
            np.testing.assert_allclose(val, 0.4 * np.log(10) * 0.01 * np.exp(-1))

    def test_integral_matches_gamma_function(self):
        # Total density integrates to phi* Gamma(alpha + 1) for alpha > -1.
        mags = np.linspace(-30.0, 10.0, 400001)
        phi = schechter_magnitude(mags, phi_star=0.02, M_star=-21.0, alpha=-0.5)
        np.testing.assert_allclose(
            np.trapezoid(phi, mags), 0.02 * np.sqrt(np.pi), rtol=1e-6
        )

    def test_faint_end_slope(self):
        # Well faintwards of M*, log10 phi rises by -0.4 (alpha + 1) per mag.
        lo, hi = schechter_magnitude(
            np.array([-2.0, -1.0]), phi_star=1.0, M_star=-21.0, alpha=-1.3
        )
        np.testing.assert_allclose(np.log10(hi / lo), -0.4 * (-1.3 + 1), rtol=1e-6)


def test_top_level_exports():
    for name in (
        "lf_given_redshift_and_subvolume",
        "lfs_given_redshifts_and_subvolume",
        "avg_lf_given_redshift_and_subvolumes",
        "avg_lf_given_redshifts_and_subvolume",
        "list_bands",
    ):
        assert name in galform_analysis.__all__
        assert callable(getattr(galform_analysis, name))
