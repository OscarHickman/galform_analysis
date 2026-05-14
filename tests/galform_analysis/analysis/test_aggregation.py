import pytest

from galform_analysis.analysis.aggregation import completed_galaxies


def test_finds_files_in_both_snapshots(galform_base_dir):
    df = completed_galaxies(basedir=galform_base_dir)
    assert not df.empty
    assert set(df["iz"].unique()) == {"iz155", "iz207"}


def test_all_mock_files_are_completed(galform_base_dir):
    df = completed_galaxies(basedir=galform_base_dir)
    assert df["completed"].all()


def test_snapshot_filter(galform_base_dir):
    df = completed_galaxies(basedir=galform_base_dir, iz_snapshots=[155])
    assert not df.empty
    assert (df["iz"] == "iz155").all()


def test_empty_basedir_returns_empty(tmp_path):
    df = completed_galaxies(basedir=str(tmp_path))
    assert df.empty or len(df) == 0


def test_result_columns(galform_base_dir):
    df = completed_galaxies(basedir=galform_base_dir)
    for col in ("iz", "iz_num", "ivol", "path", "completed"):
        assert col in df.columns, f"Missing column: {col}"


def test_sorted_by_iz_num_and_ivol(galform_base_dir):
    df = completed_galaxies(basedir=galform_base_dir)
    assert list(df["iz_num"]) == sorted(df["iz_num"].tolist())
