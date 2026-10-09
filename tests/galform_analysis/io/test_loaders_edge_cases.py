"""Edge cases for the low-level HDF5 readers in readers/loaders.py.

Each test builds a synthetic galaxies.hdf5 (or zsnap.dat) whose contents are
known exactly, so expected values are written down by hand.
"""

import h5py
import numpy as np
import pytest

from galform_analysis.readers.loaders import (
    _get_redshift_from_file,
    _get_redshift_from_zsnap,
    close_snapshot,
    get_completed_subvolumes,
    get_output_group,
    open_galaxies_hdf5,
    read_snapshot_data,
    resolve_redshift,
)
from tests.conftest import write_galaxy_hdf5


def _ivol_file(iz_dir, ivol):
    d = iz_dir / f"ivol{ivol}"
    d.mkdir(parents=True, exist_ok=True)
    return d / "galaxies.hdf5"


def _write_minimal(path, n=4):
    """Completed galaxies file with known masses and no Parameters group."""
    with h5py.File(path, "w") as f:
        f.create_dataset("CompletionFlag", data=np.int32(1))
        g = f.create_group("Output001")
        g.create_dataset("mstars_disk", data=np.arange(1, n + 1, dtype=np.float64))
        g.create_dataset("mstars_bulge", data=np.full(n, 10.0))
        g.create_dataset("mhalo", data=np.full(n, 1e12))


# ── zsnap.dat parsing ─────────────────────────────────────────────────────────


class TestZsnapRedshift:
    @pytest.mark.parametrize(
        "line, expected",
        [
            ("1.496\n", 1.496),
            ("iz= 155 z= 1.496\n", 1.496),  # "iz=" must not be read as "z="
            ("iz=155 z=0.0\n", 0.0),
            ("z = 2.5e-1\n", 0.25),
        ],
    )
    def test_parses_snapshot_level_file(self, tmp_path, line, expected):
        (tmp_path / "zsnap.dat").write_text(line)
        assert _get_redshift_from_zsnap(str(tmp_path), 0) == pytest.approx(expected)

    def test_falls_back_to_subvolume_level_file(self, tmp_path):
        ivol_dir = tmp_path / "ivol3"
        ivol_dir.mkdir()
        (ivol_dir / "zsnap.dat").write_text("iz= 155 z= 3.0\n")
        assert _get_redshift_from_zsnap(str(tmp_path), 3) == pytest.approx(3.0)

    def test_unparseable_or_missing_returns_none(self, tmp_path):
        assert _get_redshift_from_zsnap(str(tmp_path), 0) is None
        (tmp_path / "zsnap.dat").write_text("no redshift here\n")
        assert _get_redshift_from_zsnap(str(tmp_path), 0) is None


# ── redshift from the HDF5 file ──────────────────────────────────────────────


class TestRedshiftFromFile:
    def test_output_group_redshift_takes_priority(self, tmp_path):
        p = tmp_path / "g.hdf5"
        with h5py.File(p, "w") as f:
            f.create_group("Output001").create_dataset("redshift", data=0.5)
            f.create_group("Output002").create_dataset("redshift", data=1.25)
            f.create_group("Redshifts").create_dataset("3.0", data=0)
        with h5py.File(p, "r") as f:
            assert _get_redshift_from_file(f) == pytest.approx(1.25)

    def test_redshifts_dataset_with_bytes(self, tmp_path):
        p = tmp_path / "g.hdf5"
        with h5py.File(p, "w") as f:
            f.create_dataset("Redshifts", data=np.array([b"2.0", b"3.0"]))
        with h5py.File(p, "r") as f:
            assert _get_redshift_from_file(f) == pytest.approx(2.0)

    def test_redshifts_group_returns_smallest_numeric_key(self, tmp_path):
        p = tmp_path / "g.hdf5"
        with h5py.File(p, "w") as f:
            r = f.create_group("Redshifts")
            for key in ("2.0000", "0.5000", "notanumber"):
                r.create_dataset(key, data=0)
        with h5py.File(p, "r") as f:
            assert _get_redshift_from_file(f) == pytest.approx(0.5)

    def test_output_times_skips_non_numeric_entries(self, tmp_path):
        p = tmp_path / "g.hdf5"
        with h5py.File(p, "w") as f:
            f.create_dataset("Output_Times", data=np.array([b"aout", b"0.75"]))
        with h5py.File(p, "r") as f:
            assert _get_redshift_from_file(f) == pytest.approx(0.75)

    def test_none_and_empty_file(self, tmp_path):
        assert _get_redshift_from_file(None) is None
        p = tmp_path / "g.hdf5"
        h5py.File(p, "w").close()
        with h5py.File(p, "r") as f:
            assert _get_redshift_from_file(f) is None

    def test_resolve_redshift_keeps_zero(self, tmp_path):
        """z = 0.0 from the file must not fall through to zsnap.dat."""
        (tmp_path / "zsnap.dat").write_text("4.0\n")
        p = tmp_path / "g.hdf5"
        with h5py.File(p, "w") as f:
            f.create_group("Output001").create_dataset("redshift", data=0.0)
        with h5py.File(p, "r") as f:
            assert resolve_redshift(f, str(tmp_path), 0) == 0.0
        assert resolve_redshift(None, str(tmp_path), 0) == pytest.approx(4.0)


# ── subvolume discovery ──────────────────────────────────────────────────────


class TestGetCompletedSubvolumes:
    def test_numeric_order_and_flag_filtering(self, tmp_path):
        for ivol, flag in ((10, 1), (2, 1), (0, 1), (5, 0)):
            write_galaxy_hdf5(
                _ivol_file(tmp_path, ivol), n_gals=4, completion_flag=flag
            )
        assert get_completed_subvolumes(str(tmp_path)) == [0, 2, 10]

    def test_skips_non_numeric_ivol_entries(self, tmp_path):
        write_galaxy_hdf5(_ivol_file(tmp_path, 1), n_gals=4)
        (tmp_path / "ivol_old").mkdir()
        (tmp_path / "ivol_old" / "galaxies.hdf5").write_bytes(b"junk")
        (tmp_path / "ivol7.tar").write_bytes(b"")
        assert get_completed_subvolumes(str(tmp_path)) == [1]

    def test_skips_non_hdf5_and_flagless_files(self, tmp_path):
        _ivol_file(tmp_path, 0).write_bytes(b"not an hdf5 file")
        with h5py.File(_ivol_file(tmp_path, 1), "w") as f:
            f.create_group("Output001")
        _ivol_file(tmp_path, 2)  # directory without galaxies.hdf5
        assert get_completed_subvolumes(str(tmp_path)) == []


# ── open / output group ──────────────────────────────────────────────────────


def test_open_unreadable_file_returns_none(tmp_path):
    _ivol_file(tmp_path, 0).write_bytes(b"not an hdf5 file")
    assert open_galaxies_hdf5(str(tmp_path), 0) is None


def test_get_output_group_picks_highest_index_numerically(tmp_path):
    p = tmp_path / "g.hdf5"
    with h5py.File(p, "w") as f:
        for name in ("Output002", "Output010", "Output009", "OutputX"):
            f.create_group(name)
    with h5py.File(p, "r") as f:
        assert get_output_group(f).name == "/Output010"
    assert get_output_group(None) is None


# ── read_snapshot_data ───────────────────────────────────────────────────────


class TestReadSnapshotData:
    def test_no_output_group_raises_and_closes(self, tmp_path):
        p = _ivol_file(tmp_path, 0)
        with h5py.File(p, "w") as f:
            f.create_dataset("CompletionFlag", data=np.int32(1))
        with pytest.raises(RuntimeError, match="No OutputNNN"):
            read_snapshot_data(str(tmp_path), 0)
        # The file handle was released: it can be reopened for writing.
        with h5py.File(p, "a"):
            pass

    def test_mstar_fallback_when_split_masses_absent(self, tmp_path):
        p = _ivol_file(tmp_path, 0)
        with h5py.File(p, "w") as f:
            g = f.create_group("Output001")
            g.create_dataset("mstars", data=np.array([1.0, 2.0]))
            g.create_dataset("mchalo", data=np.array([5.0, 6.0]))
            g.create_dataset("Sfr", data=np.array([0.1, 0.2]))
        d = read_snapshot_data(str(tmp_path), 0)
        try:
            np.testing.assert_array_equal(d["mstar"], [1.0, 2.0])
            np.testing.assert_array_equal(d["mhalo"], [5.0, 6.0])
            np.testing.assert_array_equal(d["sfr"], [0.1, 0.2])
            assert d["Lg"] is None and d["Lr"] is None
            assert d["V_ivol"] is None and d["V_total"] is None
            assert d["n_subvolumes"] is None
            assert d["iz"] == tmp_path.name and d["ivol"] == 0
        finally:
            close_snapshot(d)

    def test_volume_uses_file_n_subvolumes(self, tmp_path):
        p = _ivol_file(tmp_path, 0)
        _write_minimal(p)
        with h5py.File(p, "a") as f:
            params = f.create_group("Parameters")
            params.create_dataset("volume", data=100.0)
            params.create_dataset("n_subvolumes", data=8)
        d = read_snapshot_data(str(tmp_path), 0)
        close_snapshot(d)
        assert d["V_ivol"] == 100.0
        assert d["V_total"] == 800.0
        assert d["n_subvolumes"] == 8

    def test_volume_falls_back_to_default_n_subvolumes(self, galform_iz_dir):
        d = read_snapshot_data(galform_iz_dir, 0)
        close_snapshot(d)
        assert d["n_subvolumes"] == 1024
        assert d["V_total"] == pytest.approx(d["V_ivol"] * 1024)

    @pytest.mark.parametrize(
        "bandnames, expect_g, expect_r",
        [
            # Generic "_r" inside "SDSS_g_r" must not beat the exact "SDSS_r".
            ([b"SDSS_g_r", b"SDSS_r", b"SDSS_g"], 3, 2),
            ([b"SDSS-g", b"SDSS-r"], 1, 2),
            ([b"r_SDSS", b"g_SDSS"], 2, 1),
            (["UKIDSS-K", "sdss r"], None, 2),
        ],
    )
    def test_band_luminosities(self, tmp_path, bandnames, expect_g, expect_r):
        p = _ivol_file(tmp_path, 0)
        n = 3
        with h5py.File(p, "w") as f:
            f.create_group("Bands").create_dataset("bandname", data=bandnames)
            g = f.create_group("Output001")
            g.create_dataset("mstars_disk", data=np.ones(n))
            g.create_dataset("mstars_bulge", data=np.ones(n))
            bands = g.create_group("Bands")
            for i in range(1, len(bandnames) + 1):
                bands.create_dataset(f"Band{i:03d}_Lum_Disk", data=np.full(n, 10.0 * i))
                bands.create_dataset(f"Band{i:03d}_Lum_Bulge", data=np.full(n, 1.0 * i))
        d = read_snapshot_data(str(tmp_path), 0)
        close_snapshot(d)
        for key, idx in (("Lg", expect_g), ("Lr", expect_r)):
            if idx is None:
                assert d[key] is None
            else:
                np.testing.assert_array_equal(d[key], np.full(n, 11.0 * idx))


def test_close_snapshot_tolerates_missing_or_none_file():
    close_snapshot({})
    close_snapshot({"file": None})


def test_empty_completion_flag_counts_as_incomplete(tmp_path):
    from galform_analysis.readers.loaders import read_completion_flag

    path = tmp_path / "galaxies.hdf5"
    with h5py.File(path, "w") as f:
        f.create_dataset("CompletionFlag", data=np.array([], dtype=np.int32))
    with h5py.File(path, "r") as f:
        assert read_completion_flag(f) is None
