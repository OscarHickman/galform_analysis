import numpy as np
import pytest

from galform_analysis.analysis.mass_functions import hod_given_redshift_and_subvolume
from galform_analysis.config import DEFAULT_HALO_MASS_BINS


def test_returns_expected_keys(galform_iz_dir):
    result = hod_given_redshift_and_subvolume(galform_iz_dir, ivol=0)
    assert result is not None
    for key in ("iz", "ivol", "centers", "mean_occupation"):
        assert key in result, f"Missing key: {key}"


def test_mean_occupation_non_negative(galform_iz_dir):
    result = hod_given_redshift_and_subvolume(galform_iz_dir, ivol=0)
    assert np.all(result["mean_occupation"] >= 0)


def test_bin_centers_shape(galform_iz_dir):
    result = hod_given_redshift_and_subvolume(galform_iz_dir, ivol=0)
    n = len(DEFAULT_HALO_MASS_BINS) - 1
    assert len(result["centers"]) == n
    assert len(result["mean_occupation"]) == n


def test_central_satellite_decomposition(galform_iz_dir):
    """When centrals are present, mean_central + mean_satellite ≈ mean_occupation."""
    result = hod_given_redshift_and_subvolume(galform_iz_dir, ivol=0)
    if result["mean_central"] is not None and result["mean_satellite"] is not None:
        total = result["mean_central"] + result["mean_satellite"]
        np.testing.assert_allclose(total, result["mean_occupation"], rtol=1e-5)


def test_stellar_mass_cut_reduces_occupation(galform_iz_dir):
    """Applying a high stellar mass cut should lower or equal mean occupation."""
    result_all = hod_given_redshift_and_subvolume(galform_iz_dir, ivol=0)
    result_cut = hod_given_redshift_and_subvolume(
        galform_iz_dir, ivol=0, galaxy_stellar_mass_min=1e10
    )
    assert result_cut is not None
    assert np.all(result_cut["mean_occupation"] <= result_all["mean_occupation"] + 1e-12)


def test_missing_file_returns_none(tmp_path):
    result = hod_given_redshift_and_subvolume(str(tmp_path / "iz_bad"), ivol=0)
    assert result is None
