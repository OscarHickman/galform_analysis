"""Tests for the GALFORM dust-table reader."""

import numpy as np
import pytest

from galform_analysis.dust.table import find_dust_table, read_attenuation_table
from tests.dust_table import (
    INC,
    LAMBDA,
    RE_H,
    TAUV,
    bulge_atten,
    disk_atten,
    write_dust_table,
)


@pytest.fixture
def table(tmp_path):
    return read_attenuation_table(write_dust_table(tmp_path / "dust.dat"))


def test_grids_are_padded_as_galform_does(table):
    np.testing.assert_allclose(table.inc, [0.0] + INC)
    np.testing.assert_allclose(table.tauv0, [0.0] + TAUV)
    np.testing.assert_allclose(1.0 / table.invlam, [100.0] + LAMBDA + [1.0e5])
    np.testing.assert_allclose(table.re_h, RE_H)
    assert table.taueff.shape == (4, 5, 4, 2, 2)
    assert table.taueff_ave.shape == (4, 5, 2, 2)
    assert table.label == "synthetic dust"


def test_tabulated_values(table):
    # taueff[tau, lam, inc, reh, comp]; values are rounded to 3 decimals.
    atten = np.exp(-table.taueff)
    np.testing.assert_allclose(
        atten[2, 2, 2, 0, 0], disk_atten(TAUV[1], LAMBDA[1], INC[1]), atol=6e-4
    )
    np.testing.assert_allclose(
        atten[3, 1, 3, 1, 1],
        bulge_atten(TAUV[2], LAMBDA[0], INC[2], RE_H[1]),
        atol=6e-4,
    )
    # The disk table is copied to every re/h slot.
    np.testing.assert_array_equal(table.taueff[..., 0, 0], table.taueff[..., 1, 0])


def test_padding(table):
    atten = np.exp(-table.taueff)
    lam = 1.0 / table.invlam
    # No attenuation at tau_V0 = 0.
    np.testing.assert_allclose(table.taueff[0, 1:-1, 1:], 0.0, atol=1e-12)
    # ln(atten) scales as 1/lambda beyond the tabulated wavelengths.
    np.testing.assert_allclose(
        table.taueff[:, 0, 1:], table.taueff[:, 1, 1:] * lam[1] / lam[0]
    )
    np.testing.assert_allclose(
        table.taueff[:, -1, 1:], table.taueff[:, -2, 1:] * lam[-2] / lam[-1]
    )
    # Inclinations below the first tabulated value use that value.
    np.testing.assert_array_equal(atten[:, :, 0], atten[:, :, 1])


def test_arrays_are_read_only(table):
    with pytest.raises(ValueError):
        table.taueff[0, 0, 0, 0, 0] = 1.0


def test_angle_average_lies_between_extremes(table):
    face_on = table.taueff[:, :, 0]
    edge_on = table.taueff[:, :, -1]
    ave = table.taueff_ave
    assert np.all(ave >= np.minimum(face_on, edge_on) - 1e-9)
    assert np.all(ave <= np.maximum(face_on, edge_on) + 1e-9)
    assert np.all(ave >= 0.0)  # attenuations are capped at 1


def test_angle_average_of_constant_is_constant(tmp_path, monkeypatch):
    import tests.dust_table as dt

    monkeypatch.setattr(dt, "disk_atten", lambda tau, lam, inc: 0.5)
    table = read_attenuation_table(write_dust_table(tmp_path / "flat.dat"))
    # d(cos i) weights from 0 to 90 deg sum to 1.
    np.testing.assert_allclose(
        np.exp(-table.taueff_ave[1:, 1:-1, :, 0]), 0.5, rtol=1e-6
    )


class TestFindDustTable:
    def test_explicit_path(self, tmp_path):
        path = write_dust_table(tmp_path / "dust.dat")
        assert find_dust_table(str(path)) == path

    def test_data_dir(self, tmp_path):
        (tmp_path / "dust").mkdir()
        path = write_dust_table(tmp_path / "dust" / "x.dat")
        assert find_dust_table("x.dat", data_dir=tmp_path) == path

    def test_environment(self, tmp_path, monkeypatch):
        (tmp_path / "dust").mkdir()
        path = write_dust_table(tmp_path / "dust" / "x.dat")
        monkeypatch.setenv("GALFORM_DATA_DIR", str(tmp_path))
        assert find_dust_table("x.dat") == path

    def test_bare_name_is_not_taken_from_cwd(self, tmp_path, monkeypatch):
        write_dust_table(tmp_path / "x.dat")
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("GALFORM_DATA_DIR", raising=False)
        with pytest.raises(FileNotFoundError, match="no data directory"):
            find_dust_table("x.dat")

    def test_missing(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GALFORM_DATA_DIR", raising=False)
        with pytest.raises(FileNotFoundError, match="GALFORM_DATA_DIR"):
            find_dust_table("nope.dat", data_dir=tmp_path)
