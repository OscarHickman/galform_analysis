import h5py
import numpy as np
import pytest

import galform_analysis.config as config
from galform_analysis.analysis.aggregation import (
    aggregate_snapshot,
    completed_galaxies,
    incomplete_subvolumes,
)
from tests.conftest import write_galaxy_hdf5


def test_finds_files_in_both_snapshots(galform_base_dir):
    df = completed_galaxies(basedir=galform_base_dir)
    assert not df.is_empty()
    assert set(df["iz"].unique()) == {"iz155", "iz207"}


def test_all_mock_files_are_completed(galform_base_dir):
    df = completed_galaxies(basedir=galform_base_dir)
    assert df["completed"].all()


def test_snapshot_filter(galform_base_dir):
    df = completed_galaxies(basedir=galform_base_dir, iz_snapshots=[155])
    assert not df.is_empty()
    assert (df["iz"] == "iz155").all()


def test_empty_basedir_returns_empty(tmp_path):
    df = completed_galaxies(basedir=str(tmp_path))
    assert df.is_empty() or len(df) == 0


def test_result_columns(galform_base_dir):
    df = completed_galaxies(basedir=galform_base_dir)
    for col in ("iz", "iz_num", "ivol", "path", "completed"):
        assert col in df.columns, f"Missing column: {col}"


def test_sorted_by_iz_num_and_ivol(galform_base_dir):
    df = completed_galaxies(basedir=galform_base_dir)
    assert list(df["iz_num"]) == sorted(df["iz_num"].to_list())


# ── scanning edge cases and aggregate_snapshot ───────────────────────────────


_V_IVOL = 155626.09375  # Parameters/volume written by write_galaxy_hdf5


def _ivol_file(iz_dir, ivol):
    d = iz_dir / f"ivol{ivol}"
    d.mkdir(parents=True, exist_ok=True)
    return d / "galaxies.hdf5"


@pytest.fixture
def mixed_snapshot(tmp_path):
    """iz100 with every subvolume state the scanners must tell apart.

    ivol0, ivol2, ivol10: complete (10 galaxies each)
    ivol3: CompletionFlag = 0
    ivol4: directory without galaxies.hdf5
    ivol5: truncated (< 1000 bytes)
    ivol6: not an HDF5 file but large enough to try opening
    ivol_old: non-numeric entry that must be ignored
    """
    iz = tmp_path / "iz100"
    for ivol in (0, 2, 10):
        write_galaxy_hdf5(_ivol_file(iz, ivol), n_gals=10, seed=ivol)
    write_galaxy_hdf5(_ivol_file(iz, 3), n_gals=10, completion_flag=0)
    (iz / "ivol4").mkdir()
    _ivol_file(iz, 5).write_bytes(b"\x89HDF\r\n\x1a\n")
    _ivol_file(iz, 6).write_bytes(b"x" * 4096)
    (iz / "ivol_old").mkdir()
    write_galaxy_hdf5(iz / "ivol_old" / "galaxies.hdf5", n_gals=10)
    (tmp_path / "izfoo").mkdir()  # non-numeric snapshot directory
    return tmp_path


class TestCompletionScan:
    def test_completed_reads_completion_flag(self, mixed_snapshot):
        df = completed_galaxies(basedir=str(mixed_snapshot))
        status = dict(zip(df["ivol"].to_list(), df["completed"].to_list()))
        assert status == {0: True, 2: True, 3: False, 5: False, 6: False, 10: True}

    def test_incomplete_reasons(self, mixed_snapshot):
        df = incomplete_subvolumes(basedir=str(mixed_snapshot))
        reasons = dict(zip(df["ivol"].to_list(), df["reason"].to_list()))
        assert reasons == {
            3: "incomplete",
            4: "missing",
            5: "incomplete",
            6: "corrupted",
        }
        assert df["ivol"].to_list() == sorted(df["ivol"].to_list())

    def test_incomplete_snapshot_filter(self, mixed_snapshot):
        assert incomplete_subvolumes(str(mixed_snapshot), iz_snapshots=[999]).is_empty()
        df = incomplete_subvolumes(str(mixed_snapshot), iz_snapshots=[100])
        assert set(df["iz"].to_list()) == {"iz100"}

    @pytest.mark.parametrize(
        "scan, columns",
        [
            (completed_galaxies, ["iz", "iz_num", "ivol", "path", "completed"]),
            (incomplete_subvolumes, ["iz", "iz_num", "ivol", "path", "reason"]),
        ],
    )
    def test_empty_scan_has_documented_columns(self, tmp_path, scan, columns):
        df = scan(basedir=str(tmp_path))
        assert df.is_empty()
        assert df.columns == columns
        # Filtering on the documented columns works on an empty result.
        df.filter(df[columns[-1]] == df[columns[-1]])

    @pytest.mark.parametrize("scan", [completed_galaxies, incomplete_subvolumes])
    def test_default_basedir_follows_set_base_dir(
        self, mixed_snapshot, monkeypatch, scan
    ):
        monkeypatch.setattr(config, "BASE_DIR", config.BASE_DIR)
        config.set_base_dir(str(mixed_snapshot))
        assert not scan().is_empty()
        assert not scan(iz_snapshots=[100]).is_empty()


class TestAggregateSnapshot:
    def test_values_from_completed_subvolumes_only(self, mixed_snapshot):
        iz = mixed_snapshot / "iz100"
        expected_mstar, expected_mhalo = [], []
        for ivol in (0, 2, 10):
            with h5py.File(_ivol_file(iz, ivol), "r") as f:
                g = f["Output001"]
                expected_mstar.append(g["mstars_disk"][()] + g["mstars_bulge"][()])
                expected_mhalo.append(g["mhalo"][()])

        agg = aggregate_snapshot(str(iz))

        assert agg["iz"] == "iz100"
        assert agg["z"] == 0.0
        assert agg["ivols"] == [0, 2, 10]  # numeric order; ivol3 flag = 0
        assert agg["volume"] == pytest.approx(3 * _V_IVOL)
        np.testing.assert_array_equal(agg["mstar"], np.concatenate(expected_mstar))
        np.testing.assert_array_equal(agg["mhalo"], np.concatenate(expected_mhalo))

    def test_subvolume_missing_mhalo_keeps_arrays_aligned(self, tmp_path):
        iz = tmp_path / "iz100"
        for ivol in range(3):
            write_galaxy_hdf5(_ivol_file(iz, ivol), n_gals=10, seed=ivol)
        with h5py.File(_ivol_file(iz, 1), "r+") as f:
            del f["Output001/mhalo"]

        with pytest.warns(UserWarning, match="ivol1"):
            agg = aggregate_snapshot(str(iz))

        assert len(agg["mstar"]) == len(agg["mhalo"]) == 20
        assert agg["ivols"] == [0, 2]
        assert agg["volume"] == pytest.approx(2 * _V_IVOL)

    def test_subvolume_without_volume_is_not_counted(self, tmp_path):
        iz = tmp_path / "iz100"
        for ivol in range(2):
            write_galaxy_hdf5(_ivol_file(iz, ivol), n_gals=10, seed=ivol)
        with h5py.File(_ivol_file(iz, 1), "r+") as f:
            del f["Parameters/volume"]

        with pytest.warns(UserWarning, match="volume"):
            agg = aggregate_snapshot(str(iz))

        assert len(agg["mstar"]) == 10
        assert agg["volume"] == pytest.approx(_V_IVOL)

    def test_returns_none_without_usable_subvolumes(self, tmp_path):
        assert aggregate_snapshot(str(tmp_path / "iz1")) is None
        iz = tmp_path / "iz2"
        write_galaxy_hdf5(_ivol_file(iz, 0), n_gals=10, completion_flag=0)
        (iz / "ivol_old").mkdir()
        assert aggregate_snapshot(str(iz)) is None
