import h5py
import numpy as np
import pytest

from io.loaders import (
    close_snapshot,
    get_completed_subvolumes,
    get_output_group,
    open_galaxies_hdf5,
    read_snapshot_data,
)


def test_open_returns_file(galform_iz_dir):
    f = open_galaxies_hdf5(galform_iz_dir, ivol=0)
    assert f is not None
    assert isinstance(f, h5py.File)
    f.close()


def test_open_missing_returns_none(tmp_path):
    assert open_galaxies_hdf5(str(tmp_path / "iz_missing"), ivol=0) is None


def test_get_output_group(galform_iz_dir):
    f = open_galaxies_hdf5(galform_iz_dir, ivol=0)
    g = get_output_group(f)
    assert g is not None
    assert "mstars_disk" in g
    f.close()


def test_get_output_group_empty_file(tmp_path):
    p = tmp_path / "empty.hdf5"
    with h5py.File(str(p), "w"):
        pass
    with h5py.File(str(p), "r") as f:
        assert get_output_group(f) is None


def test_read_snapshot_data_keys(galform_iz_dir):
    d = read_snapshot_data(galform_iz_dir, ivol=0)
    for key in ("mstar", "mhalo", "V_ivol", "file", "group"):
        assert key in d, f"Missing key: {key}"
    close_snapshot(d)


def test_read_snapshot_mstar_is_disk_plus_bulge(galform_iz_dir):
    d = read_snapshot_data(galform_iz_dir, ivol=0)
    expected = (
        np.array(d["group"]["mstars_disk"]) + np.array(d["group"]["mstars_bulge"])
    )
    np.testing.assert_allclose(d["mstar"], expected)
    close_snapshot(d)


def test_read_snapshot_volume_positive(galform_iz_dir):
    d = read_snapshot_data(galform_iz_dir, ivol=0)
    assert d["V_ivol"] is not None
    assert d["V_ivol"] > 0
    close_snapshot(d)


def test_read_snapshot_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_snapshot_data(str(tmp_path / "iz_bad"), ivol=0)


def test_close_snapshot_idempotent(galform_iz_dir):
    d = read_snapshot_data(galform_iz_dir, ivol=0)
    close_snapshot(d)
    close_snapshot(d)  # second call must not raise


def test_get_completed_subvolumes_finds_both(galform_iz_dir):
    completed = get_completed_subvolumes(galform_iz_dir)
    assert 0 in completed
    assert 1 in completed


def test_get_completed_subvolumes_empty_dir(tmp_path):
    completed = get_completed_subvolumes(str(tmp_path / "iz_empty"))
    assert completed == []
