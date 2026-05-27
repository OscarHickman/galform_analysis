from galform_analysis.config import SimulationConfig, load_sim_config
import pytest

def test_load_sim_config_l800():
    config = load_sim_config('L800')
    # Use keys present in the family format
    assert config['h0'] == 0.6777
    assert config['lbox'] == 542.16

def test_simulation_config_class():
    cfg = SimulationConfig('L800')
    assert cfg.name == 'L800'
    assert cfg.box_size == 542.16
    assert cfg.n_subvolumes == 1024
    assert abs(cfg.h - 0.6777) < 1e-6
    assert abs(cfg.h0 - 67.77) < 1e-6
    assert "Mpc/h" in repr(cfg)

def test_simulation_config_mill1():
    cfg = SimulationConfig('Mill1')
    assert cfg.name == 'Mill1'
    assert cfg.box_size == 500.0
    assert cfg.n_subvolumes == 64
    assert cfg.h == 0.73

def test_load_nonexistent_config():
    with pytest.raises(FileNotFoundError):
        load_sim_config('NonExistentSim')
