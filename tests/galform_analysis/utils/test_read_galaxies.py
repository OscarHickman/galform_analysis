import numpy as np
import polars as pl
import pytest

from galform_analysis.utils.read_galaxies import (
    read_galaxies_dataframe,
    read_galaxy_arrays,
    read_galaxy_positions,
    read_halo_arrays,
    read_halo_positions,
)

_BOXSIZE = 542.16


def test_arrays_contains_expected_keys(galform_iz_dir):
    arrays, _ = read_galaxy_arrays(galform_iz_dir, ivol=0, centrals_only=False)
    for key in ("x", "y", "z", "mstar", "mhalo", "is_central"):
        assert key in arrays, f"Missing key: {key}"


def test_centrals_only_filters_correctly(galform_iz_dir):
    arrays, _ = read_galaxy_arrays(galform_iz_dir, ivol=0, centrals_only=True)
    assert np.all(arrays["is_central"] == 1)


def test_all_galaxies_larger_than_centrals(galform_iz_dir):
    all_arr, _ = read_galaxy_arrays(galform_iz_dir, ivol=0, centrals_only=False)
    cen_arr, _ = read_galaxy_arrays(galform_iz_dir, ivol=0, centrals_only=True)
    assert len(all_arr["x"]) > len(cen_arr["x"])


def test_mhalo_cut_respected(galform_iz_dir):
    cut = 1e11
    arrays, _ = read_galaxy_arrays(
        galform_iz_dir, ivol=0, centrals_only=True, mhalo_min=cut
    )
    assert np.all(arrays["mhalo"] >= cut)


def test_mstar_all_positive(galform_iz_dir):
    arrays, _ = read_galaxy_arrays(galform_iz_dir, ivol=0, centrals_only=False)
    assert np.all(arrays["mstar"] > 0)


def test_positions_within_box(galform_iz_dir):
    arrays, _ = read_galaxy_arrays(galform_iz_dir, ivol=0, centrals_only=False)
    for ax in ("x", "y", "z"):
        assert arrays[ax].min() >= 0.0
        assert arrays[ax].max() <= _BOXSIZE


def test_metadata_keys(galform_iz_dir):
    _, meta = read_galaxy_arrays(galform_iz_dir, ivol=0)
    assert meta["iz"] == "iz155"
    assert meta["ivol"] == 0
    assert meta["V_ivol"] is not None


def test_dataframe_type_and_columns(galform_iz_dir):
    df = read_galaxies_dataframe(galform_iz_dir, ivol=0)
    assert isinstance(df, pl.DataFrame)
    assert "mstar" in df.columns


def test_dataframe_with_metadata(galform_iz_dir):
    df, meta = read_galaxies_dataframe(galform_iz_dir, ivol=0, return_metadata=True)
    assert isinstance(df, pl.DataFrame)
    assert "iz" in meta


def test_read_halo_arrays_centrals_only(galform_iz_dir):
    arrays, _ = read_halo_arrays(galform_iz_dir, ivol=0)
    assert np.all(arrays["is_central"] == 1)
    assert "mhhalo" in arrays


def test_halo_mhhalo_cut(galform_iz_dir):
    cut = 1e11
    arrays, _ = read_halo_arrays(galform_iz_dir, ivol=0, mhhalo_min=cut)
    assert np.all(arrays["mhhalo"] >= cut)


def test_galaxy_positions_shape(galform_iz_dir):
    pos, _ = read_galaxy_positions(galform_iz_dir, ivol=0, centrals_only=False)
    assert pos.ndim == 2
    assert pos.shape[1] == 3


def test_halo_positions_shape(galform_iz_dir):
    pos, _ = read_halo_positions(galform_iz_dir, ivol=0)
    assert pos.ndim == 2
    assert pos.shape[1] == 3


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_galaxy_arrays(str(tmp_path / "iz_bad"), ivol=0)
