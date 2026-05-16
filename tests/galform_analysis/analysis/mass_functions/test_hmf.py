import numpy as np
import pytest

from analysis.mass_functions import hmf_given_redshift_and_subvolume
from config import DEFAULT_HALO_MASS_BINS


def test_returns_expected_keys(galform_iz_dir):
    result = hmf_given_redshift_and_subvolume(galform_iz_dir, ivol=0)
    assert result is not None
    for key in ("iz", "ivol", "centers", "phi", "counts", "V_ivol"):
        assert key in result, f"Missing key: {key}"


def test_iz_and_ivol_metadata(galform_iz_dir):
    result = hmf_given_redshift_and_subvolume(galform_iz_dir, ivol=0)
    assert result["iz"] == "iz155"
    assert result["ivol"] == 0


def test_phi_normalisation(galform_iz_dir):
    """phi * dlogM * V_ivol must exactly recover raw counts."""
    result = hmf_given_redshift_and_subvolume(galform_iz_dir, ivol=0)
    dlogM = np.diff(DEFAULT_HALO_MASS_BINS)
    recovered = result["phi"] * dlogM * result["V_ivol"]
    np.testing.assert_allclose(recovered, result["counts"].astype(float), rtol=1e-5)


def test_counts_are_non_negative(galform_iz_dir):
    result = hmf_given_redshift_and_subvolume(galform_iz_dir, ivol=0)
    assert np.all(result["counts"] >= 0)


def test_halo_mass_cut(galform_iz_dir):
    """Bins entirely below the cut must have zero counts."""
    cut = 1e12
    result = hmf_given_redshift_and_subvolume(
        galform_iz_dir, ivol=0, halo_mass_lower_limit=cut
    )
    below = result["centers"] < np.log10(cut)
    assert np.all(result["counts"][below] == 0)


def test_bin_centers_shape(galform_iz_dir):
    result = hmf_given_redshift_and_subvolume(galform_iz_dir, ivol=0)
    expected_n = len(DEFAULT_HALO_MASS_BINS) - 1
    assert len(result["centers"]) == expected_n
    assert len(result["phi"]) == expected_n
    assert len(result["counts"]) == expected_n


def test_missing_file_returns_none(tmp_path):
    result = hmf_given_redshift_and_subvolume(str(tmp_path / "iz_bad"), ivol=0)
    assert result is None
