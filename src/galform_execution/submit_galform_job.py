#!/usr/bin/env python3
"""
Submit GALFORM N-body tree runs to SLURM batch queue on COSMA.

This script replaces the qsub_galform_Nbody_example.csh functionality,
submitting GALFORM jobs as SLURM array jobs.

Author: Oscar Hickman
Date: November 2025
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from typing import List, Optional


class GalformSubmitter:
    """Handle submission of GALFORM jobs to SLURM."""
    
    # Simulation configurations
    SIMULATION_CONFIGS = {
        'MilliMil': {
            'iz_list': [63],
            'nvol_range': '1-8'
        },
        'Mill1': {
            'iz_list': [63],
            'nvol_range': '1-10'
        },
        'Mill2': {
            'iz_list': [67],
            'nvol_range': '1-10'
        },
        'MillGas': {
            'iz_list': [61],
            'nvol_range': '1-10'
        },
        'L800': {
            'iz_list': [271, 207, 176, 155, 142, 120, 105, 100, 82],
            'nvol_range': '701-900'
        },
        'EagleDM': {
            'iz_list': [200],
            'nvol_range': '1-128'
        },
        'EagleDM67': {
            'iz_list': [67],
            'nvol_range': '1-128'
        },
        'EagleDM101': {
            'iz_list': [101],
            'nvol_range': '1-128'
        },
        'nifty62.5': {
            'iz_list': [61],
            'nvol_range': '1-64'
        },
        'MillGas62.5': {
            'iz_list': [61],
            'nvol_range': '1-1'
        },
        'DoveCDM': {
            'iz_list': [159],
            'nvol_range': '1-64'
        },
        'DoveWDM.clean': {
            'iz_list': [79],
            'nvol_range': '1-64'
        }
    }
    
    def __init__(
        self,
        galform_exe: str,
        run_script: str,
        nbody_sim: str = 'L800',
        model: str = 'gp14',
        log_path: Optional[str] = None,
        partition: str = 'cosma5',
        account: str = 'durham',
        walltime: str = '21:00:00',
        iz_list: Optional[List[int]] = None,
        nvol_range: Optional[str] = None
    ):
        """
        Initialize the GALFORM job submitter.
        
        Args:
            galform_exe: Path to the GALFORM executable
            run_script: Path to the run_galform_Nbody_example.csh script
            nbody_sim: N-body simulation name (e.g., 'L800')
            model: Model name (e.g., 'gp14')
            log_path: Directory for log files
            partition: SLURM partition (default: 'cosma5')
            account: SLURM account (default: 'durham')
            walltime: Job walltime (default: '21:00:00')
            iz_list: Custom list of snapshot numbers
            nvol_range: Custom subvolume range (e.g., '1-10')
        """
        self.galform_exe = Path(galform_exe)
        self.run_script = Path(run_script)
        self.nbody_sim = nbody_sim
        self.model = model
        self.partition = partition
        self.account = account
        self.walltime = walltime
        
        # Determine log path with CI-safe fallback
        # Priority: explicit arg > env var > writable COSMA path > local cwd
        def _deepest_existing_ancestor(p: Path) -> Path:
            for ancestor in [p] + list(p.parents):
                if ancestor.exists():
                    return ancestor
            # Should not happen (at least '/' exists), but be safe
            return Path('/')

        def _is_creatable(p: Path) -> bool:
            ancestor = _deepest_existing_ancestor(p)
            return os.access(ancestor, os.W_OK)

        if log_path is not None:
            self.log_path = Path(log_path)
        else:
            # Environment override (useful for CI): GALFORM_LOG_PATH
            env_log = os.environ.get('GALFORM_LOG_PATH')
            if env_log:
                self.log_path = Path(env_log)
            else:
                # If running in CI, always use local writable path
                if os.environ.get('CI', '').lower() == 'true':
                    self.log_path = Path.cwd() / 'Galform_Out' / 'logs'
                else:
                    # Prefer COSMA default if we can create it; otherwise fallback to local
                    cosma_default = Path(f'/cosma5/data/durham/{Path.home().name}/Galform_Out/logs')
                    if _is_creatable(cosma_default):
                        self.log_path = cosma_default
                    else:
                        self.log_path = Path.cwd() / 'Galform_Out' / 'logs'
        
        # Get simulation configuration
        if nbody_sim in self.SIMULATION_CONFIGS:
            config = self.SIMULATION_CONFIGS[nbody_sim]
            self.iz_list = iz_list if iz_list is not None else config['iz_list']
            self.nvol_range = nvol_range if nvol_range is not None else config['nvol_range']
        else:
            if iz_list is None or nvol_range is None:
                raise ValueError(
                    f"Unknown simulation '{nbody_sim}'. "
                    "Please provide iz_list and nvol_range explicitly."
                )
            self.iz_list = iz_list
            self.nvol_range = nvol_range
        
        # Validate inputs
        if not self.galform_exe.exists():
            raise FileNotFoundError(f"GALFORM executable not found: {self.galform_exe}")
        if not self.run_script.exists():
            raise FileNotFoundError(f"Run script not found: {self.run_script}")
    
    def create_slurm_script(self, iz: int) -> str:
        """
        Create SLURM batch script content for a given snapshot.
        
        Args:
            iz: Snapshot number
            
        Returns:
            String containing the SLURM script
        """
        jobname = f"{self.nbody_sim}.{self.model}"
        logname = self.log_path / self.nbody_sim / f"{self.model}.%A.%a.log"
        
        # Ensure log directory exists
        logname.parent.mkdir(parents=True, exist_ok=True)
        
        # Read the run script content
        with open(self.run_script, 'r') as f:
            run_script_content = f.read()
        
        # Create SLURM header
        slurm_header = f"""#!/bin/tcsh -ef
#
#SBATCH --ntasks 1
#SBATCH -J {jobname}
#SBATCH -o {logname}
#SBATCH -p {self.partition}
#SBATCH -A {self.account}
#SBATCH -t {self.walltime}
#

# Set parameters
set model     = {self.model}
set Nbody_sim = {self.nbody_sim}
set iz        = {iz}
@ ivol        = ${{SLURM_ARRAY_TASK_ID}} - 1

# Galform run script follows
"""
        
        # Combine header and run script
        full_script = slurm_header + run_script_content
        
        return full_script
    
    def submit_job(self, iz: int, dry_run: bool = False) -> Optional[str]:
        """
        Submit a SLURM job for a given snapshot.
        
        Args:
            iz: Snapshot number
            dry_run: If True, print the script but don't submit
            
        Returns:
            Job ID if submitted, None if dry_run
        """
        script_content = self.create_slurm_script(iz)
        
        if dry_run:
            print(f"\n{'='*70}")
            print(f"DRY RUN: Would submit job for iz={iz}, nvol_range={self.nvol_range}")
            print(f"{'='*70}")
            print(script_content)
            print(f"{'='*70}\n")
            return None
        
        # Submit via sbatch
        cmd = ['sbatch', f'--array={self.nvol_range}']
        
        try:
            result = subprocess.run(
                cmd,
                input=script_content.encode(),
                capture_output=True,
                check=True
            )
            
            # Parse job ID from output
            output = result.stdout.decode().strip()
            print(output)
            
            # Extract job ID (format: "Submitted batch job 12345")
            if "Submitted batch job" in output:
                job_id = output.split()[-1]
                return job_id
            
            return None
            
        except subprocess.CalledProcessError as e:
            print(f"Error submitting job for iz={iz}: {e}", file=sys.stderr)
            print(f"STDOUT: {e.stdout.decode()}", file=sys.stderr)
            print(f"STDERR: {e.stderr.decode()}", file=sys.stderr)
            return None
    
    def submit_all_jobs(self, dry_run: bool = False) -> List[str]:
        """
        Submit SLURM jobs for all snapshots.
        
        Args:
            dry_run: If True, print scripts but don't submit
            
        Returns:
            List of job IDs
        """
        print("Submitting GALFORM jobs:")
        print(f"  N-body simulation: {self.nbody_sim}")
        print(f"  Model: {self.model}")
        print(f"  Snapshots (iz): {self.iz_list}")
        print(f"  Subvolume range: {self.nvol_range}")
        print(f"  GALFORM executable: {self.galform_exe}")
        print(f"  Run script: {self.run_script}")
        print(f"  Log path: {self.log_path}")
        print()
        
        job_ids = []
        for iz in self.iz_list:
            job_id = self.submit_job(iz, dry_run=dry_run)
            if job_id:
                job_ids.append(job_id)
                print(f"Submitted job for iz={iz}: Job ID {job_id}")
        
        if not dry_run and job_ids:
            print(f"\nSuccessfully submitted {len(job_ids)} jobs")
        
        return job_ids


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Submit GALFORM N-body runs to SLURM batch queue on COSMA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit jobs for L800 simulation with gp14 model
  %(prog)s /path/to/galform2 /path/to/run_galform_Nbody_example.csh
  
  # Submit jobs with custom simulation
  %(prog)s /path/to/galform2 /path/to/run_script.csh --nbody-sim MillGas --model b06
  
  # Dry run to preview what would be submitted
  %(prog)s /path/to/galform2 /path/to/run_script.csh --dry-run
  
  # Custom snapshot list and subvolume range
  %(prog)s /path/to/galform2 /path/to/run_script.csh --iz-list 100 120 155 --nvol-range 1-50
        """
    )
    
    parser.add_argument(
        'galform_exe',
        nargs='?',
        help='Path to the GALFORM executable (e.g., galform2)'
    )
    
    parser.add_argument(
        'run_script',
        nargs='?',
        help='Path to the GALFORM run script (e.g., run_galform_Nbody_example.csh)'
    )
    
    parser.add_argument(
        '--nbody-sim',
        default='L800',
        help='N-body simulation name (default: L800)'
    )
    
    parser.add_argument(
        '--model',
        default='gp14',
        help='Model name (default: gp14)'
    )
    
    parser.add_argument(
        '--log-path',
        help='Directory for log files (default: COSMA path if writable; otherwise a local Galform_Out/logs). Can be overridden with GALFORM_LOG_PATH.'
    )
    
    parser.add_argument(
        '--partition',
        default='cosma5',
        help='SLURM partition (default: cosma5)'
    )
    
    parser.add_argument(
        '--account',
        default='durham',
        help='SLURM account (default: durham)'
    )
    
    parser.add_argument(
        '--walltime',
        default='21:00:00',
        help='Job walltime (default: 21:00:00)'
    )
    
    parser.add_argument(
        '--iz-list',
        type=int,
        nargs='+',
        help='Custom list of snapshot numbers (overrides simulation defaults)'
    )
    
    parser.add_argument(
        '--nvol-range',
        help='Custom subvolume range (e.g., "1-10", overrides simulation defaults)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print job scripts without submitting them'
    )
    
    parser.add_argument(
        '--list-simulations',
        action='store_true',
        help='List available simulation configurations and exit'
    )
    
    args = parser.parse_args()
    
    # List simulations if requested
    if args.list_simulations:
        print("Available simulation configurations:")
        print(f"{'Simulation':<20} {'Snapshots (iz)':<30} {'Subvolume Range':<15}")
        print("-" * 70)
        for sim_name, config in sorted(GalformSubmitter.SIMULATION_CONFIGS.items()):
            iz_str = str(config['iz_list'][:3]) + '...' if len(config['iz_list']) > 3 else str(config['iz_list'])
            print(f"{sim_name:<20} {iz_str:<30} {config['nvol_range']:<15}")
        return 0
    
    # Check required arguments
    if not args.galform_exe or not args.run_script:
        parser.error("galform_exe and run_script are required unless using --list-simulations")
    
    try:
        submitter = GalformSubmitter(
            galform_exe=args.galform_exe,
            run_script=args.run_script,
            nbody_sim=args.nbody_sim,
            model=args.model,
            log_path=args.log_path,
            partition=args.partition,
            account=args.account,
            walltime=args.walltime,
            iz_list=args.iz_list,
            nvol_range=args.nvol_range
        )
        
        submitter.submit_all_jobs(dry_run=args.dry_run)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
