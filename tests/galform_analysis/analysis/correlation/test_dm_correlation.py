"""Tests for the snapshot wrappers in dm_correlation.py.

The CAMB call is replaced by a stub that records its arguments, so these tests
check the snapshot/redshift plumbing without needing camb.
"""

import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import polars as pl
import pytest

import galform_analysis.analysis.correlation.dm_correlation as dm
import galform_analysis.config as config
from galform_analysis.analysis.correlation.correlation import (
    halo_correlation_given_redshift_and_subvolume,
)
from galform_analysis.config import SimulationConfig
from tests.conftest import write_galaxy_hdf5

requires_corrfunc = pytest.mark.skipif(
    importlib.util.find_spec("Corrfunc") is None, reason="needs Corrfunc"
)

COARSE_RBINS = np.array([20.0, 60.0, 120.0, 200.0, 250.0])


def make_snapshot(base, iz_num, ivols, redshift=None):
    iz_dir = Path(base) / f"iz{iz_num}"
    for ivol in ivols:
        ivol_dir = iz_dir / f"ivol{ivol}"
        ivol_dir.mkdir(parents=True)
        path = ivol_dir / "galaxies.hdf5"
        write_galaxy_hdf5(path, n_gals=400, seed=iz_num + ivol)
        if redshift is not None:
            with h5py.File(path, "r+") as f:
                del f["Redshifts"]
                f.create_group("Redshifts").create_dataset(
                    f"{redshift:.4f}", data=np.int32(0)
                )
    return iz_dir


@pytest.fixture
def sim():
    return SimulationConfig("L800")


@pytest.fixture
def camb_calls(monkeypatch):
    """Replace compute_matter_xi with a stub returning xi = z for each r."""
    calls = []

    def fake_compute_matter_xi(sim, z, rbins=None, ns=0.961):
        calls.append({"sim": sim, "z": z, "rbins": rbins, "ns": ns})
        df = pl.DataFrame({"r": [1.0, 2.0], "xi": [z, z]})
        df.attrs = {"z": z}
        return df

    monkeypatch.setattr(dm, "compute_matter_xi", fake_compute_matter_xi)
    return calls


@pytest.fixture
def base_dir(tmp_path, monkeypatch):
    make_snapshot(tmp_path, 155, ivols=(0, 1), redshift=1.25)
    make_snapshot(tmp_path, 207, ivols=(0, 1))
    monkeypatch.setattr(config, "BASE_DIR", config.BASE_DIR)  # restored on teardown
    config.set_base_dir(str(tmp_path))
    return tmp_path


# ── matter_xi_at_snapshot(s) ─────────────────────────────────────────────────


def test_matter_xi_uses_snapshot_redshift(base_dir, sim, camb_calls):
    res = dm.matter_xi_at_snapshot(
        str(base_dir / "iz155"), sim, rbins=COARSE_RBINS, ns=1.0
    )

    assert res.attrs["z"] == pytest.approx(1.25)
    (call,) = camb_calls
    assert call["z"] == pytest.approx(1.25)
    assert call["sim"] is sim
    assert call["ns"] == 1.0
    np.testing.assert_array_equal(call["rbins"], COARSE_RBINS)


def test_matter_xi_missing_snapshot_returns_none(tmp_path, sim, camb_calls):
    assert dm.matter_xi_at_snapshot(str(tmp_path / "iz1"), sim) is None
    assert camb_calls == []


def test_matter_xi_without_camb_raises_install_hint(base_dir, sim, monkeypatch):
    monkeypatch.setitem(sys.modules, "camb", None)  # makes `import camb` fail
    with pytest.raises(ImportError, match=r"galform_analysis\[science\]"):
        dm.matter_xi_at_snapshot(str(base_dir / "iz155"), sim)


def test_matter_xi_at_snapshots_keeps_order_and_gaps(base_dir, sim, camb_calls):
    res = dm.matter_xi_at_snapshots([207, 999, 155], sim)

    assert res[1] is None
    assert [r.attrs["iz"] for r in (res[0], res[2])] == ["iz207", "iz155"]
    assert [c["z"] for c in camb_calls] == [pytest.approx(0.0), pytest.approx(1.25)]


def test_matter_xi_at_snapshots_explicit_base_dir(tmp_path, sim, camb_calls):
    make_snapshot(tmp_path, 82, ivols=(0,), redshift=3.0)
    (res,) = dm.matter_xi_at_snapshots([82], sim, base_dir=str(tmp_path))
    assert res.attrs["iz"] == "iz82"
    assert camb_calls[0]["z"] == pytest.approx(3.0)


# ── DM halo correlation wrappers ─────────────────────────────────────────────


def test_dm_correlations_missing_snapshots_are_none(tmp_path):
    res = dm.dm_correlations_given_redshifts_and_subvolume(
        [1, 2], 0, base_dir=str(tmp_path)
    )
    assert res == [None, None]


def test_avg_dm_correlation_none_when_nothing_valid(tmp_path):
    assert (
        dm.avg_dm_correlation_given_subvolume_and_redshifts(
            [1, 2], 0, base_dir=str(tmp_path)
        )
        is None
    )


@requires_corrfunc
def test_dm_correlation_is_halo_correlation(base_dir):
    iz_path = str(base_dir / "iz155")
    res = dm.dm_correlation_given_redshift_and_subvolume(
        iz_path, 1, rbins=COARSE_RBINS, nthreads=1, mhhalo_min=1e11
    )
    ref = halo_correlation_given_redshift_and_subvolume(
        iz_path, 1, rbins=COARSE_RBINS, nthreads=1, mhhalo_min=1e11
    )
    np.testing.assert_array_equal(res["xi"].to_numpy(), ref["xi"].to_numpy())
    assert res.attrs["nhalo"] == ref.attrs["nhalo"]


@requires_corrfunc
def test_dm_correlations_use_configured_base_dir(base_dir):
    res = dm.dm_correlations_given_redshifts_and_subvolume(
        [155, 999, 207], 0, rbins=COARSE_RBINS, nthreads=1
    )
    assert res[1] is None
    assert res[0].attrs["iz"] == "iz155"
    assert res[2].attrs["iz"] == "iz207"
    assert res[0].attrs["z"] == pytest.approx(1.25)


@requires_corrfunc
def test_avg_dm_correlation_is_mean_and_std(base_dir):
    singles = [
        halo_correlation_given_redshift_and_subvolume(
            str(base_dir / iz), 0, rbins=COARSE_RBINS, nthreads=1
        )
        for iz in ("iz155", "iz207")
    ]
    xi = np.vstack([s["xi"].to_numpy() for s in singles])

    res = dm.avg_dm_correlation_given_subvolume_and_redshifts(
        [155, 207, 999], 0, rbins=COARSE_RBINS, nthreads=1, base_dir=str(base_dir)
    )

    np.testing.assert_allclose(res["xi_mean"], xi.mean(axis=0))
    np.testing.assert_allclose(res["xi_std"], xi.std(axis=0))
    np.testing.assert_array_equal(res["r"], singles[0]["r"].to_numpy())
    assert isinstance(res["r"], np.ndarray)
    np.testing.assert_array_equal(res["rbins"], COARSE_RBINS)
    assert res["iz_list"] == ["iz155", "iz207"]
    assert res["z_list"] == [pytest.approx(1.25), pytest.approx(0.0)]
    assert res["ngal_list"] == [200, 200]
