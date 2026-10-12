"""Tests for the GALFORM band-table reader."""

import h5py
import numpy as np
import pytest

from galform_analysis.readers.bands import (
    ab_to_system_offset,
    band_luminosity,
    find_band,
    list_bands,
    luminosity_to_magnitude,
)
from tests.band_catalogue import (
    DEFAULT_BANDS,
    magnitudes_to_luminosity,
    write_band_catalogue,
)


@pytest.fixture
def band_file(tmp_path):
    lum = {i: np.full(3, 10.0 * i) for i in range(1, len(DEFAULT_BANDS) + 1)}
    bulge = {i: np.full(3, 1.0 * i) for i in range(1, len(DEFAULT_BANDS) + 1)}
    iz_dir = write_band_catalogue(tmp_path, 271, 0, lum_disk=lum, lum_bulge=bulge)
    with h5py.File(iz_dir / "ivol0" / "galaxies.hdf5", "r") as f:
        yield f


class TestListBands:
    def test_table_columns_and_frames(self, tmp_path):
        iz_dir = write_band_catalogue(tmp_path, 271, 0, lum_disk={1: np.ones(2)})
        table = list_bands(str(iz_dir), ivol=0)
        assert table.columns == [
            "index",
            "label",
            "frame",
            "lambda_eff",
            "a0_ab",
            "name",
        ]
        np.testing.assert_allclose(table["a0_ab"][0], -1.87, rtol=1e-6)
        assert table["index"].to_list() == [1, 2, 3, 4, 5]
        assert table["label"].to_list() == [
            "UKIRT-K",
            "SDSS-r",
            "V",
            "UKIRT-K",
            "SDSS-r",
        ]
        assert table["frame"].to_list() == [
            "rest",
            "rest",
            "other",
            "observer",
            "observer",
        ]
        assert table["name"][1].startswith("SDSS-r  SDSS r standard")

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            list_bands(str(tmp_path / "iz1"), ivol=0)

    def test_file_without_band_table_raises(self, tmp_path):
        iz_dir = write_band_catalogue(
            tmp_path, 271, 0, lum_disk={1: np.ones(2)}, write_band_table=False
        )
        with pytest.raises(KeyError, match="no Bands table"):
            list_bands(str(iz_dir), ivol=0)


class TestFindBand:
    @pytest.mark.parametrize(
        "band, frame, expected",
        [
            ("UKIRT-K", "rest", 1),
            ("SDSS-r", "rest", 2),
            ("UKIRT-K", "observer", 4),
            ("SDSS-r", "observer", 5),
            (3, "rest", 3),  # an integer index is used as-is, whatever the frame
        ],
    )
    def test_resolves_label_and_frame(self, band_file, band, frame, expected):
        assert find_band(band_file, band, frame=frame) == expected

    def test_label_is_case_sensitive(self, band_file):
        # "r" (SDSS) and "R" (Cousins) are different filters in GALFORM tables.
        with pytest.raises(ValueError, match="sdss-r"):
            find_band(band_file, "sdss-r")

    def test_unknown_label_lists_available(self, band_file):
        with pytest.raises(ValueError, match="UKIRT-K, SDSS-r"):
            find_band(band_file, "Bj")

    def test_special_bands_need_an_index(self, band_file):
        with pytest.raises(ValueError, match="'V'"):
            find_band(band_file, "V")

    def test_bad_frame(self, band_file):
        with pytest.raises(ValueError, match="frame"):
            find_band(band_file, "SDSS-r", frame="lab")

    @pytest.mark.parametrize("index", [0, 6, -1])
    def test_index_out_of_range(self, band_file, index):
        with pytest.raises(ValueError, match="out of range"):
            find_band(band_file, index)

    def test_bool_is_not_an_index(self, band_file):
        with pytest.raises(TypeError):
            find_band(band_file, True)

    def test_duplicate_identical_bands_take_first(self, tmp_path):
        bands = [("TH1500 top hat", 1, 200), ("TH1500 top hat", 1, 200)]
        iz_dir = write_band_catalogue(
            tmp_path, 1, 0, lum_disk={1: np.ones(2)}, bands=bands, a0_ab=None
        )
        with h5py.File(iz_dir / "ivol0" / "galaxies.hdf5", "r") as f:
            assert find_band(f, "TH1500", frame="observer") == 1

    def test_duplicate_different_bands_are_ambiguous(self, tmp_path):
        bands = [("X filter one", 0, 1), ("X filter two", 0, 2)]
        iz_dir = write_band_catalogue(
            tmp_path, 1, 0, lum_disk={1: np.ones(2)}, bands=bands, a0_ab=None
        )
        with h5py.File(iz_dir / "ivol0" / "galaxies.hdf5", "r") as f:
            with pytest.raises(ValueError, match="ambiguous"):
                find_band(f, "X")

    def test_duplicates_without_idband_are_ambiguous(self, tmp_path):
        bands = [("X filter one", 0, 1), ("X filter one", 0, 1)]
        iz_dir = write_band_catalogue(
            tmp_path, 1, 0, lum_disk={1: np.ones(2)}, bands=bands, a0_ab=None
        )
        with h5py.File(iz_dir / "ivol0" / "galaxies.hdf5", "a") as f:
            del f["Bands/idband"]
            with pytest.raises(ValueError, match="ambiguous"):
                find_band(f, "X")

    def test_column_length_mismatch_raises(self, tmp_path):
        iz_dir = write_band_catalogue(
            tmp_path, 1, 0, lum_disk={1: np.ones(2)}, a0_ab=[0.0, 0.0]
        )
        with h5py.File(iz_dir / "ivol0" / "galaxies.hdf5", "r") as f:
            with pytest.raises(ValueError, match="a0_ab has 2 entries"):
                find_band(f, "SDSS-r")


class TestBandLuminosity:
    @pytest.mark.parametrize(
        "component, expected", [("total", 22.0), ("disk", 20.0), ("bulge", 2.0)]
    )
    def test_components(self, band_file, component, expected):
        lum = band_luminosity(band_file["Output001"], 2, component)
        np.testing.assert_allclose(lum, expected)

    def test_bad_component(self, band_file):
        with pytest.raises(ValueError, match="component"):
            band_luminosity(band_file["Output001"], 2, "burst")

    def test_missing_dataset_returns_none(self, band_file):
        assert band_luminosity(band_file["Output001"], 99, "total") is None

    def test_group_without_bands_returns_none(self, tmp_path):
        iz_dir = write_band_catalogue(
            tmp_path, 1, 0, lum_disk={1: np.ones(2)}, write_band_lums=False
        )
        with h5py.File(iz_dir / "ivol0" / "galaxies.hdf5", "r") as f:
            assert band_luminosity(f["Output001"], 1, "total") is None


class TestAbToSystemOffset:
    def test_ab_is_zero(self, band_file):
        assert ab_to_system_offset(band_file, 1, "AB") == 0.0

    def test_vega_reads_a0_ab(self, band_file):
        # GALFORM: M_Vega = M_AB + a0_ab.
        np.testing.assert_allclose(
            ab_to_system_offset(band_file, 1, "Vega"), -1.87, rtol=1e-6
        )
        np.testing.assert_allclose(
            ab_to_system_offset(band_file, 2, "Vega"), 0.12, rtol=1e-6
        )

    def test_bad_system(self, band_file):
        with pytest.raises(ValueError, match="system"):
            ab_to_system_offset(band_file, 1, "ST")

    def test_vega_without_a0_ab_raises(self, tmp_path):
        iz_dir = write_band_catalogue(
            tmp_path, 1, 0, lum_disk={1: np.ones(2)}, a0_ab=None
        )
        with h5py.File(iz_dir / "ivol0" / "galaxies.hdf5", "r") as f:
            assert ab_to_system_offset(f, 1, "AB") == 0.0
            with pytest.raises(KeyError, match="a0_ab"):
                ab_to_system_offset(f, 1, "Vega")


class TestLuminosityToMagnitude:
    def test_galform_convention(self):
        mags = luminosity_to_magnitude(np.array([1.0, 100.0, 1e10]))
        np.testing.assert_allclose(mags, [0.0, -5.0, -25.0])

    def test_round_trip(self):
        mags = np.array([-24.3, -18.0, -12.5])
        np.testing.assert_allclose(
            luminosity_to_magnitude(magnitudes_to_luminosity(mags)), mags
        )

    def test_non_positive_and_non_finite_are_nan(self):
        mags = luminosity_to_magnitude(np.array([0.0, -1.0, np.nan, np.inf]))
        assert np.all(np.isnan(mags))
