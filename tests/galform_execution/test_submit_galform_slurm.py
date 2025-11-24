"""Tests for submit_galform_slurm.py script."""

import subprocess
import tempfile
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from galform_execution.submit_galform_slurm import GalformSubmitter


def test_galform_submitter_initialization():
    """Test that GalformSubmitter can be initialized with valid inputs."""
    # Create temporary files for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.exe', delete=False) as exe_file:
        exe_path = exe_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csh', delete=False) as script_file:
        script_file.write("#!/bin/tcsh\necho 'test script'\n")
        script_path = script_file.name
    
    try:
        # Test initialization with default simulation
        submitter = GalformSubmitter(
            galform_exe=exe_path,
            run_script=script_path,
            nbody_sim='L800',
            model='gp14'
        )
        
        assert submitter.nbody_sim == 'L800'
        assert submitter.model == 'gp14'
        assert submitter.partition == 'cosma5'
        assert submitter.account == 'durham'
        assert submitter.walltime == '21:00:00'
        assert len(submitter.iz_list) > 0
        assert submitter.nvol_range == '701-900'
        
    finally:
        # Cleanup
        Path(exe_path).unlink()
        Path(script_path).unlink()


def test_galform_submitter_custom_config():
    """Test GalformSubmitter with custom configuration."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.exe', delete=False) as exe_file:
        exe_path = exe_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csh', delete=False) as script_file:
        script_file.write("#!/bin/tcsh\necho 'test'\n")
        script_path = script_file.name
    
    try:
        submitter = GalformSubmitter(
            galform_exe=exe_path,
            run_script=script_path,
            nbody_sim='MillGas',
            model='b06',
            partition='cosma7',
            account='dp004',
            walltime='48:00:00',
            iz_list=[61, 62],
            nvol_range='1-5'
        )
        
        assert submitter.nbody_sim == 'MillGas'
        assert submitter.model == 'b06'
        assert submitter.partition == 'cosma7'
        assert submitter.account == 'dp004'
        assert submitter.walltime == '48:00:00'
        assert submitter.iz_list == [61, 62]
        assert submitter.nvol_range == '1-5'
        
    finally:
        Path(exe_path).unlink()
        Path(script_path).unlink()


def test_create_slurm_script():
    """Test SLURM script generation."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.exe', delete=False) as exe_file:
        exe_path = exe_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csh', delete=False) as script_file:
        script_file.write("#!/bin/tcsh\necho 'test run script'\n")
        script_path = script_file.name
    
    try:
        submitter = GalformSubmitter(
            galform_exe=exe_path,
            run_script=script_path,
            nbody_sim='L800',
            model='gp14'
        )
        
        script_content = submitter.create_slurm_script(iz=100)
        
        # Check that script contains expected elements
        assert '#!/bin/tcsh -ef' in script_content
        assert '#SBATCH --ntasks 1' in script_content
        assert '#SBATCH -J L800.gp14' in script_content
        assert '#SBATCH -p cosma5' in script_content
        assert '#SBATCH -A durham' in script_content
        assert '#SBATCH -t 21:00:00' in script_content
        assert 'set model     = gp14' in script_content
        assert 'set Nbody_sim = L800' in script_content
        assert 'set iz        = 100' in script_content
        assert 'test run script' in script_content
        
    finally:
        Path(exe_path).unlink()
        Path(script_path).unlink()


def test_simulation_configs():
    """Test that all predefined simulation configurations are accessible."""
    configs = GalformSubmitter.SIMULATION_CONFIGS
    
    # Check that key simulations exist
    assert 'L800' in configs
    assert 'MillGas' in configs
    assert 'EagleDM' in configs
    
    # Check L800 configuration
    assert configs['L800']['iz_list'] == [271, 207, 176, 155, 142, 120, 105, 100, 82]
    assert configs['L800']['nvol_range'] == '701-900'
    
    # Check that all configs have required keys
    for sim_name, config in configs.items():
        assert 'iz_list' in config
        assert 'nvol_range' in config
        assert isinstance(config['iz_list'], list)
        assert isinstance(config['nvol_range'], str)


def test_invalid_paths():
    """Test that appropriate errors are raised for invalid paths."""
    try:
        GalformSubmitter(
            galform_exe='/nonexistent/path/galform2',
            run_script='/nonexistent/path/run_script.csh',
            nbody_sim='L800'
        )
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError as e:
        assert "GALFORM executable not found" in str(e)


def test_unknown_simulation():
    """Test handling of unknown simulation without custom config."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.exe', delete=False) as exe_file:
        exe_path = exe_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csh', delete=False) as script_file:
        script_path = script_file.name
    
    try:
        # Should raise ValueError when unknown sim without iz_list and nvol_range
        try:
            GalformSubmitter(
                galform_exe=exe_path,
                run_script=script_path,
                nbody_sim='UnknownSim'
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Unknown simulation" in str(e)
        
        # Should work with custom parameters
        submitter = GalformSubmitter(
            galform_exe=exe_path,
            run_script=script_path,
            nbody_sim='CustomSim',
            iz_list=[100],
            nvol_range='1-10'
        )
        assert submitter.nbody_sim == 'CustomSim'
        assert submitter.iz_list == [100]
        
    finally:
        Path(exe_path).unlink()
        Path(script_path).unlink()


def test_script_help_option():
    """Test that the script's help option works."""
    script_path = Path(__file__).parent.parent.parent / 'src' / 'galform_execution' / 'submit_galform_slurm.py'
    
    result = subprocess.run(
        ['python', str(script_path), '--help'],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0
    assert 'Submit GALFORM N-body runs to SLURM' in result.stdout
    assert 'galform_exe' in result.stdout
    assert 'run_script' in result.stdout
    assert '--nbody-sim' in result.stdout
    assert '--dry-run' in result.stdout


def test_script_list_simulations():
    """Test that the script can list available simulations."""
    script_path = Path(__file__).parent.parent.parent / 'src' / 'galform_execution' / 'submit_galform_slurm.py'
    
    result = subprocess.run(
        ['python', str(script_path), '--list-simulations'],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0
    assert 'Available simulation configurations' in result.stdout
    assert 'L800' in result.stdout
    assert 'MillGas' in result.stdout
    assert 'EagleDM' in result.stdout
    assert 'Subvolume Range' in result.stdout


def test_script_dry_run():
    """Test that the script's dry-run mode works."""
    script_path = Path(__file__).parent.parent.parent / 'src' / 'galform_execution' / 'submit_galform_slurm.py'
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.exe', delete=False) as exe_file:
        exe_path = exe_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csh', delete=False) as script_file:
        script_file.write("#!/bin/tcsh\necho 'test'\n")
        script_path_temp = script_file.name
    
    try:
        result = subprocess.run(
            [
                'python', str(script_path),
                exe_path,
                script_path_temp,
                '--nbody-sim', 'L800',
                '--iz-list', '100',
                '--nvol-range', '1-5',
                '--dry-run'
            ],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert 'DRY RUN' in result.stdout
        assert 'iz=100' in result.stdout
        assert 'nvol_range=1-5' in result.stdout
        assert '#SBATCH' in result.stdout
        assert 'set model' in result.stdout
        
    finally:
        Path(exe_path).unlink()
        Path(script_path_temp).unlink()


def test_log_path_creation():
    """Test that log directory is created properly."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.exe', delete=False) as exe_file:
        exe_path = exe_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csh', delete=False) as script_file:
        script_file.write("#!/bin/tcsh\necho 'test'\n")
        script_path = script_file.name
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            log_path = Path(tmpdir) / 'test_logs'
            
            submitter = GalformSubmitter(
                galform_exe=exe_path,
                run_script=script_path,
                nbody_sim='L800',
                log_path=str(log_path)
            )
            
            # Create script should create log directory
            script_content = submitter.create_slurm_script(iz=100)
            
            assert log_path.exists()
            assert (log_path / 'L800').exists()
            
        finally:
            Path(exe_path).unlink()
            Path(script_path).unlink()
