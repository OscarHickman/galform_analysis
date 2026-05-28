import pytest

from galform_analysis.config import load_sim_config, load_simulation_families


def test_load_simulation_families():
    families = load_simulation_families()
    assert len(families) > 0
    # Check for some expected simulations
    assert "L800" in families
    assert "Mill1" in families
    assert "FLAMINGO-L1000N1800" in families

    # Check that metadata is filtered out
    for sim in families.values():
        for key in sim:
            assert not key.startswith("_")


def test_load_sim_config_fallback():
    # Test that load_sim_config falls back to family files
    config = load_sim_config("COLIBRE-L100m6")
    assert config["h0"] == 0.681
    assert config["lbox"] == 68.1


def test_load_sim_config_nonexistent():
    with pytest.raises(FileNotFoundError):
        load_sim_config("NonExistentSim")
