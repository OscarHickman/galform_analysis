import json
from pathlib import Path

import pytest

import galform_analysis.config as config
from galform_analysis.config import (
    SimulationConfig,
    find_snapshot_at_redshift,
    get_base_dir,
    get_snapshot_redshift,
    load_redshift_mapping,
    set_base_dir,
)


def test_load_redshift_mapping():
    z_map = load_redshift_mapping("L800")
    assert isinstance(z_map, dict)
    assert len(z_map) > 0
    assert 99 in z_map
    assert abs(z_map[99] - 4.377) < 0.01


def test_load_redshift_mapping_unknown_sim():
    import pytest

    with pytest.raises(FileNotFoundError, match="No redshift list for simulation"):
        load_redshift_mapping("UnknownSim")


def test_get_snapshot_redshift():
    z = get_snapshot_redshift("iz99", "L800")
    assert z is not None
    assert abs(z - 4.377) < 0.01


def test_find_snapshot_at_redshift():
    snap = find_snapshot_at_redshift(4.4, "L800", tolerance=0.1)
    assert snap == "iz99"


def test_get_base_dir_unset_raises():
    """With no base dir configured, get_base_dir explains how to set one."""
    with pytest.raises(RuntimeError, match="set_base_dir"):
        get_base_dir()


def test_set_base_dir(tmp_path):
    """set_base_dir stores an absolute, resolved path."""
    set_base_dir(str(tmp_path))
    new_base = get_base_dir()
    assert isinstance(new_base, Path)
    assert new_base == tmp_path.resolve()
    assert new_base.is_absolute()


def test_base_dir_from_environment(monkeypatch, tmp_path):
    """GALFORM_BASE_DIR is read when the package is imported."""
    import importlib

    import galform_analysis.config as cfg

    monkeypatch.setenv("GALFORM_BASE_DIR", str(tmp_path))
    try:
        importlib.reload(cfg)
        assert cfg.get_base_dir() == tmp_path
    finally:
        monkeypatch.delenv("GALFORM_BASE_DIR")
        importlib.reload(cfg)


# ── SimulationConfig parsing, redshift lists and family loading ──────────────


_NESTED = {
    "name": "Nested",
    "box_size": 100.0,
    "n_subvolumes": 8,
    "cosmology": {
        "omega_m": 0.3,
        "omega_l": 0.7,
        "omega_b": 0.045,
        "h": 0.7,
        "sigma_8": 0.8,
    },
}


def _with_config(monkeypatch, cfg):
    monkeypatch.setattr(config, "load_sim_config", lambda name: cfg)


def test_simulation_config_nested_cosmology(monkeypatch):
    _with_config(monkeypatch, _NESTED)
    cfg = SimulationConfig("anything")
    assert (cfg.name, cfg.box_size, cfg.n_subvolumes) == ("Nested", 100.0, 8)
    assert (cfg.omega_m, cfg.omega_l, cfg.omega_b) == (0.3, 0.7, 0.045)
    assert (cfg.h, cfg.sigma_8, cfg.delta_c) == (0.7, 0.8, 1.686)
    assert cfg.f_b == pytest.approx(0.15)
    assert cfg.h0 == pytest.approx(70.0)


@pytest.mark.parametrize(
    "nvol_range, expected",
    [("1-64", 64), ("0-63", 64), ("64", 64), (" 1-1024 ", 1024)],
)
def test_simulation_config_counts_nvol_range(monkeypatch, nvol_range, expected):
    _with_config(monkeypatch, {"lbox": 50.0, "nvol_range": nvol_range})
    cfg = SimulationConfig("Flat")
    assert cfg.name == "Flat"
    assert cfg.box_size == 50.0
    assert cfg.n_subvolumes == expected


def test_simulation_config_missing_fields(monkeypatch):
    _with_config(monkeypatch, {})
    cfg = SimulationConfig("Bare")
    assert cfg.n_subvolumes is None
    assert cfg.box_size is None
    assert cfg.f_b is None and cfg.h0 is None


def test_load_redshift_mapping_skips_malformed_lines(tmp_path, monkeypatch):
    (tmp_path / "Toy.txt").write_text("# header\n1 2.5\nbad line here\nx 1.0\n2 0.5\n")
    monkeypatch.setattr(config, "_REDSHIFT_LISTS_DIR", tmp_path)
    assert config.load_redshift_mapping("Toy") == {1: 2.5, 2: 0.5}
    assert config.get_snapshot_redshift("2", "Toy") == 0.5  # "iz" prefix optional
    assert config.get_snapshot_redshift("iz7", "Toy") is None
    assert config.get_snapshot_redshift("izabc", "Toy") is None
    assert config.find_snapshot_at_redshift(2.45, "Toy") == "iz1"
    assert config.find_snapshot_at_redshift(10.0, "Toy") is None


def test_load_simulation_families_merges_and_strips_metadata(tmp_path, monkeypatch):
    (tmp_path / "a_family.json").write_text(
        json.dumps({"SimA": {"lbox": 1.0, "_note": "x"}, "SimB": {"lbox": 2.0}})
    )
    (tmp_path / "b_family.json").write_text(json.dumps({"SimB": {"lbox": 3.0}}))
    monkeypatch.setattr(config, "_SIM_CONFIGS_DIR", tmp_path)
    assert config.load_simulation_families() == {
        "SimA": {"lbox": 1.0},
        "SimB": {"lbox": 3.0},  # later files (sorted by name) win
    }
    with pytest.raises(FileNotFoundError, match="Available: \\['SimA', 'SimB'\\]"):
        config.load_sim_config("SimC")


def test_load_simulation_families_missing_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_SIM_CONFIGS_DIR", tmp_path / "nope")
    assert config.load_simulation_families() == {}


def test_bundled_configs_cover_execution_configs():
    """Every bundled fallback config parses into a usable SimulationConfig."""

    bundled = Path(config.__file__).parent / "sim_configs"
    for path in sorted(bundled.glob("*.json")):
        for name in json.loads(path.read_text()):
            cfg = SimulationConfig(name)
            # A few unverified boxes (e.g. MillGas62.5) deliberately have lbox null.
            assert cfg.box_size is None or cfg.box_size > 0, name
            assert cfg.n_subvolumes and cfg.n_subvolumes > 0, name
            assert 0 < cfg.h < 1.5, name
