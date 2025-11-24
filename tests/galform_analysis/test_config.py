import sys
import os
import tempfile
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from galform_analysis.config import load_redshift_mapping, get_snapshot_redshift, find_snapshot_at_redshift, get_base_dir, set_base_dir

def test_load_redshift_mapping():
    z_map = load_redshift_mapping()
    assert isinstance(z_map, dict)
    assert len(z_map) > 0
    # Check a known mapping
    assert 99 in z_map
    assert abs(z_map[99] - 4.377) < 0.01

def test_get_snapshot_redshift():
    z = get_snapshot_redshift('iz99')
    assert z is not None
    assert abs(z - 4.377) < 0.01

def test_find_snapshot_at_redshift():
    snap = find_snapshot_at_redshift(4.4, tolerance=0.1)
    assert snap == 'iz99'

def test_get_base_dir():
    """Test that get_base_dir returns a Path object."""
    base_dir = get_base_dir()
    assert isinstance(base_dir, Path)
    # Don't check for specific path - it may not exist on CI

def test_set_base_dir():
    """Test that set_base_dir updates the base directory."""
    # Save original
    original_base = get_base_dir()
    
    # Create a temporary directory and set it
    with tempfile.TemporaryDirectory() as tmpdir:
        set_base_dir(tmpdir)
        new_base = get_base_dir()
        assert str(new_base) == tmpdir
        assert new_base.is_absolute()
    
    # Restore original
    set_base_dir(str(original_base))

