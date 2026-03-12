#!/usr/bin/env python3
"""
Submit GALFORM N-body tree runs to SLURM batch queue on COSMA.

This script replaces the legacy qsub_galform_Nbody_example.csh +
run_galform_Nbody_example.csh workflow. It generates a complete tcsh
run script with all simulation parameters, model configurations, and
post-processing steps, then submits it as a SLURM array job.

Author: Oscar Hickman
Date: November 2025
"""

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Data classes for simulation / model / dust configuration
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """N-body simulation configuration (tree paths, cosmology, snapshots)."""
    iz_list: List[int]
    nvol_range: str
    nbody_trees_dir: str
    snapshot_file: str
    aquarius_tree_file: str
    aquarius_particle_file: str
    volume: float
    omega0: float
    lambda0: float
    omegab: float
    h0: float
    sigma8: float
    pk_file: str
    iz0: int
    # Optional fields with defaults
    lbox: Optional[float] = None
    mpart: Optional[float] = None


@dataclass
class DustParams:
    """Dust model parameters for post-processing."""
    dustfile: str = 'Data/dust/dust_MW_hz1.0.dat'
    emdustfile: str = '0'
    rfacburst: float = 1.0
    fcloud: float = 0.25
    tesc_disk: float = 0.001
    tesc_burst: float = 0.001
    lambda_break_disk: float = 1e4
    beta2_disk: float = 2.0
    lambda_break_burst: float = 100.0
    beta2_burst: float = 1.6


@dataclass
class ModelConfig:
    """GALFORM model configuration."""
    base_inputs_file: str
    dust_params: DustParams
    extra_replacements: Dict[str, str] = field(default_factory=dict)


@dataclass
class RunFlags:
    """Flags controlling which parts of the GALFORM pipeline to run."""
    compile: bool = False
    galform: bool = False
    neta: bool = True
    dust_props: bool = False
    lum_fun: bool = True
    samp_z0: bool = False
    cosmicsed: bool = False
    lum_fun_burst: bool = False
    samp2_z0: bool = False
    sedfit: bool = False
    agn: bool = False
    sed_agn: bool = False
    samp_mah: bool = False
    study_stellar_mass_function: bool = True


# ---------------------------------------------------------------------------
# Predefined configurations
# ---------------------------------------------------------------------------

DUST_BAUGH05 = DustParams(
    fcloud=0.25, tesc_disk=0.001, tesc_burst=0.001,
    lambda_break_disk=1e4, beta2_disk=2.0,
    lambda_break_burst=100.0, beta2_burst=1.6,
)

DUST_LACEY16 = DustParams(
    fcloud=0.5, tesc_disk=0.001, tesc_burst=0.001,
    lambda_break_disk=1e4, beta2_disk=2.0,
    lambda_break_burst=100.0, beta2_burst=1.5,
)

SIMULATION_CONFIGS: Dict[str, SimulationConfig] = {
    'MilliMil': SimulationConfig(
        iz_list=[63], nvol_range='1-8',
        nbody_trees_dir='/cosma5/data/jch/MilliMillennium/trees/',
        snapshot_file='/cosma5/data/jch/MilliMillennium/trees/redshift_list',
        aquarius_tree_file='/cosma5/data/jch/MilliMillennium/trees/treedir_063/tree_063',
        aquarius_particle_file='/cosma5/data/jch/MilliMillennium/trees/particle_lists/particle_list_063',
        volume=0, omega0=0.25, lambda0=0.75, omegab=0.045, h0=0.73,
        sigma8=0.9, pk_file='Power_Spec/pk_Mill.dat', iz0=63,
    ),
    'Mill1': SimulationConfig(
        iz_list=[33, 63], nvol_range='1-64',
        nbody_trees_dir='/cosma5/data/jch/Galform/Merger_Trees/Millennium/new',
        snapshot_file='/cosma5/data/jch/Galform/Merger_Trees/Millennium/new/redshift_list',
        aquarius_tree_file='/cosma5/data/jch/Galform/Merger_Trees/Millennium/new/treedir_063/tree_063',
        aquarius_particle_file='/cosma5/data/jch/Galform/Merger_Trees/Millennium/new/particle_lists/particle_list_063',
        volume=0, omega0=0.25, lambda0=0.75, omegab=0.045, h0=0.73,
        sigma8=0.9, pk_file='Power_Spec/pk_Mill.dat', iz0=63,
    ),
    'Mill2': SimulationConfig(
        iz_list=[67], nvol_range='1-10',
        nbody_trees_dir='/cosma5/data/jch/Galform/Merger_Trees/Millennium2/new',
        snapshot_file='/cosma5/data/jch/Galform/Merger_Trees/Millennium2/new/redshift_list',
        aquarius_tree_file='/cosma5/data/jch/Galform/Merger_Trees/Millennium2/new/treedir_067/tree_067',
        aquarius_particle_file='/cosma5/data/jch/Galform/Merger_Trees/Millennium2/new/particle_lists/particle_list_067',
        volume=15625.0, omega0=0.25, lambda0=0.75, omegab=0.045, h0=0.73,
        sigma8=0.9, pk_file='Power_Spec/pk_Mill.dat', iz0=67,
    ),
    'MillGas': SimulationConfig(
        iz_list=[61], nvol_range='1-10',
        nbody_trees_dir='/cosma5/data/jch/Galform/Merger_Trees/MillGas/dm/500/new/',
        snapshot_file='/cosma5/data/jch/Galform/Merger_Trees/MillGas/dm/500/new/redshift_list',
        aquarius_tree_file='/cosma5/data/jch/Galform/Merger_Trees/MillGas/dm/500/new/treedir_061/tree_061',
        aquarius_particle_file='/cosma5/data/jch/Galform/Merger_Trees/MillGas/dm/500/new/particle_lists/particle_list_061',
        volume=1953125.0, omega0=0.272, lambda0=0.728, omegab=0.0455, h0=0.704,
        sigma8=0.810, pk_file='Power_Spec/pk_MillGas_norm.dat', iz0=61,
    ),
    'L800': SimulationConfig(
        iz_list=[271, 207, 176, 155, 142, 120, 105, 100, 82],
        nvol_range='0-161',
        nbody_trees_dir='/cosma5/data/jch/Galform/Merger_Trees/P-Millennium/Updated_Trees/all_snaps',
        snapshot_file='/cosma5/data/jch/Galform/Merger_Trees/P-Millennium/Updated_Trees/all_snaps/redshift_list.txt',
        aquarius_tree_file='/cosma5/data/jch/Galform/Merger_Trees/P-Millennium/Updated_Trees/all_snaps/treedir_269/tree_269',
        aquarius_particle_file='/cosma5/data/jch/Galform/Merger_Trees/P-Millennium/Updated_Trees/all_snaps/particle_lists/particle_list_269',
        volume=155626.1, omega0=0.307, lambda0=0.693, omegab=0.0482519, h0=0.6777,
        sigma8=0.8288, pk_file='Power_Spec/pk_EAGLE_norm.dat', iz0=271,
        lbox=542.16, mpart=1.061e8,
    ),
    'EagleDM': SimulationConfig(
        iz_list=[200], nvol_range='1-128',
        nbody_trees_dir='/cosma7/data/dp004/jch/Eagle/Merger_Trees/DMONLY/L0100N1504',
        snapshot_file='/cosma7/data/dp004/jch/Eagle/Merger_Trees/DMONLY/L0100N1504/eagle_redshift_list',
        aquarius_tree_file='/cosma7/data/dp004/jch/Eagle/Merger_Trees/DMONLY/L0100N1504/trees/treedir_200/tree_200',
        aquarius_particle_file='/cosma7/data/dp004/jch/Eagle/Merger_Trees/DMONLY/L0100N1504/trees/particle_lists/particle_list_200',
        volume=2431.65796432, omega0=0.307, lambda0=0.693, omegab=0.0482519, h0=0.6777,
        sigma8=0.8288, pk_file='Power_Spec/pk_EAGLE_norm.dat', iz0=200,
    ),
    'EagleDM67': SimulationConfig(
        iz_list=[67], nvol_range='1-128',
        nbody_trees_dir='/cosma7/data/dp004/jch/Eagle/Merger_Trees/DMONLY/L0100N1504/trees_snapnums62',
        snapshot_file='/cosma7/data/dp004/jch/Eagle/Merger_Trees/DMONLY/L0100N1504/trees_snapnums62/redshift_list',
        aquarius_tree_file='/cosma7/data/dp004/jch/Eagle/Merger_Trees/DMONLY/L0100N1504/trees_snapnums62/treedir_067/tree_067',
        aquarius_particle_file='/cosma7/data/dp004/jch/Eagle/Merger_Trees/DMONLY/L0100N1504/trees_snapnums62/particle_lists/particle_list_067',
        volume=2431.65796432, omega0=0.307, lambda0=0.693, omegab=0.0482519, h0=0.6777,
        sigma8=0.8288, pk_file='Power_Spec/pk_EAGLE_norm.dat', iz0=67,
    ),
    'EagleDM101': SimulationConfig(
        iz_list=[101], nvol_range='1-128',
        nbody_trees_dir='/cosma7/data/dp004/jch/Eagle/Merger_Trees/DMONLY/L0100N1504/trees_snapnums100',
        snapshot_file='/cosma7/data/dp004/jch/Eagle/Merger_Trees/DMONLY/L0100N1504/trees_snapnums100/redshift_list',
        aquarius_tree_file='/cosma7/data/dp004/jch/Eagle/Merger_Trees/DMONLY/L0100N1504/trees_snapnums100/treedir_101/tree_101',
        aquarius_particle_file='/cosma7/data/dp004/jch/Eagle/Merger_Trees/DMONLY/L0100N1504/trees_snapnums100/particle_lists/particle_list_101',
        volume=2431.65796432, omega0=0.307, lambda0=0.693, omegab=0.0482519, h0=0.6777,
        sigma8=0.8288, pk_file='Power_Spec/pk_EAGLE_norm.dat', iz0=101,
    ),
    'DoveCDM': SimulationConfig(
        iz_list=[159], nvol_range='1-64',
        nbody_trees_dir='/cosma5/data/jch/Galform/Merger_Trees/Dove/CDM/trees',
        snapshot_file='/cosma5/data/jch/Galform/Merger_Trees/Dove/CDM/trees/redshift_list',
        aquarius_tree_file='/cosma5/data/jch/Galform/Merger_Trees/Dove/CDM/trees/treedir_159/tree_159',
        aquarius_particle_file='/cosma5/data/jch/Galform/Merger_Trees/Dove/CDM/trees/particle_lists/particle_list_159',
        volume=5451.776, omega0=0.272, lambda0=0.728, omegab=0.0455, h0=0.704,
        sigma8=0.810, pk_file='Data/Power_Spec/pk_MillGas_norm.dat', iz0=159,
        lbox=70.4, mpart=6195595.0,
    ),
    'DoveWDM.clean': SimulationConfig(
        iz_list=[79], nvol_range='1-64',
        nbody_trees_dir='/gpfs/data/dph3apc/dove/wdm/trees_cleaned',
        snapshot_file='/gpfs/data/dph3apc/dove/wdm/trees_cleaned/dovewdmclean_redshift_list',
        aquarius_tree_file='/gpfs/data/dph3apc/dove/wdm/trees_cleaned/treedir_079/tree_079',
        aquarius_particle_file='/gpfs/data/dph3apc/dove/wdm/trees_cleaned/particle_lists/particle_list_079',
        volume=5451.776, omega0=0.272, lambda0=0.728, omegab=0.0455, h0=0.704,
        sigma8=0.810, pk_file='Power_Spec/pk_WDMDove.dat', iz0=79,
        lbox=70.4, mpart=6195595.0,
    ),
    'MillGas62.5': SimulationConfig(
        iz_list=[61], nvol_range='1-1',
        nbody_trees_dir='/cosma5/data/jch/Galform/Merger_Trees/MillGas/dm/62.5/new/',
        snapshot_file='/cosma5/data/jch/Galform/Merger_Trees/MillGas/dm/62.5/new/redshift_list',
        aquarius_tree_file='/cosma5/data/jch/Galform/Merger_Trees/MillGas/dm/62.5/new/treedir_061/tree_061',
        aquarius_particle_file='/cosma5/data/jch/Galform/Merger_Trees/MillGas/dm/62.5/new/particle_lists/particle_list_061',
        volume=1953125.0, omega0=0.272, lambda0=0.728, omegab=0.0455, h0=0.704,
        sigma8=0.810, pk_file='Power_Spec/pk_MillGas_norm.dat', iz0=61,
    ),
    'nifty62.5': SimulationConfig(
        iz_list=[61], nvol_range='1-64',
        nbody_trees_dir='/cosma5/data/jch/Galform/Merger_Trees/nifty/62.5/',
        snapshot_file='/cosma5/data/jch/Galform/Merger_Trees/nifty/62.5/redshift_list',
        aquarius_tree_file='/cosma5/data/jch/Galform/Merger_Trees/nifty/62.5/treedir_061/tree_061',
        aquarius_particle_file='',  # No particle file for nifty
        volume=1953125.0, omega0=0.272, lambda0=0.728, omegab=0.0455, h0=0.704,
        sigma8=0.810, pk_file='Power_Spec/pk_MillGas_norm.dat', iz0=61,
    ),
}

MODEL_CONFIGS: Dict[str, ModelConfig] = {
    'b06': ModelConfig(
        base_inputs_file='Bower06_Nbody_MilliMil.input.ref',
        dust_params=DUST_BAUGH05,
    ),
    'gp14': ModelConfig(
        base_inputs_file='Gonzalez13_Nbody_MillGas.input.ref',
        dust_params=DUST_BAUGH05,
    ),
    'lc16': ModelConfig(
        base_inputs_file='Lacey16_Nbody_MillGas.input.ref',
        dust_params=DUST_LACEY16,
    ),
    'lc16.newSMBH': ModelConfig(
        base_inputs_file='Lacey16_Nbody_MillGas.input.ref',
        dust_params=DUST_LACEY16,
    ),
    'lc16.newmg': ModelConfig(
        base_inputs_file='Lacey16_newmg_Nbody_L800.input.ref',
        dust_params=DUST_LACEY16,
    ),
    'lc16.newmg.kenn83.vhot': ModelConfig(
        base_inputs_file='Lacey16_newmg_Nbody_L800.input.ref',
        dust_params=DUST_LACEY16,
        extra_replacements={'nmf': '1', 'vhotdisk': '290', 'vhotburst': '290'},
    ),
    'gp14.satf': ModelConfig(
        base_inputs_file='Gonzalez13_Nbody_MillGas.input.ref',
        dust_params=DUST_BAUGH05,
        extra_replacements={'Saturate_Feedback': '.true.', 'thresholdVcirc': '65.0'},
    ),
    'gp18': ModelConfig(
        base_inputs_file='Gonzalez13_Nbody_MillGas.input.ref',
        dust_params=DUST_BAUGH05,
    ),
}


# ---------------------------------------------------------------------------
# Helper: resolve log path
# ---------------------------------------------------------------------------

def _default_cosma_user_root() -> Path:
    """Return the default COSMA user root path."""
    user = os.environ.get('USER', Path.home().name)
    return Path(f'/cosma5/data/durham/{user}')


def _resolve_log_path(explicit: Optional[str], output_folder_name: str) -> Path:
    """Determine the log directory.

    By default this always lives under the COSMA user root.
    """
    if explicit is not None:
        return Path(explicit)

    env_log = os.environ.get('GALFORM_LOG_PATH')
    if env_log:
        return Path(env_log)

    return _default_cosma_user_root() / output_folder_name / 'logs'


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GalformSubmitter:
    """Handle submission of GALFORM jobs to SLURM.

    Instead of concatenating a legacy csh run script, this class generates
    a complete tcsh script that sets up the GALFORM input parameter file,
    runs the GALFORM executable, and performs post-processing — mirroring
    ``run_galform_Nbody_example.csh`` entirely from Python-controlled
    configuration.
    """

    def __init__(
        self,
        galform_dir: str,
        nbody_sim: str = 'L800',
        model: str = 'gp14',
        iz: Optional[int] = None,
        nvol: Optional[str] = None,
        output_base_dir: Optional[str] = None,
        output_folder_name: str = 'Galform_Out',
        log_path: Optional[str] = None,
        partition: str = 'cosma5',
        account: str = 'durham',
        walltime: str = '72:00:00',
        iz_list: Optional[List[int]] = None,
        nvol_range: Optional[str] = None,
        run_flags: Optional[RunFlags] = None,
        stellar_pop_dir: str = '/cosma5/data/jch/Galform/Data/stellar_pop/',
        modules: Optional[List[str]] = None,
    ):
        """
        Initialise the GALFORM job submitter.

        Args:
            galform_dir: Path to the GALFORM source directory containing
                ``build/``, ``*.input.ref``, helper csh scripts, etc.
            nbody_sim: N-body simulation label (e.g. ``'L800'``).
            model: GALFORM model label (e.g. ``'gp14'``).
            iz: Snapshot number for a single-job submission.
            nvol: Subvolume range for SLURM array submission, in the same
                format used by the legacy scripts (e.g. ``'1-10'``).
            output_base_dir: Root directory for GALFORM outputs. Defaults to
                ``/cosma5/data/durham/$USER``.
            output_folder_name: Name of the output folder created under the
                base output directory.
            log_path: Directory for SLURM log files.
            partition: SLURM partition.
            account: SLURM account.
            walltime: SLURM walltime.
            iz_list: Override default snapshot list for array-style submission.
            nvol_range: Override default subvolume range for array-style
                submission (e.g. ``'0-161'``).
            run_flags: ``RunFlags`` controlling pipeline stages.
            stellar_pop_dir: Location of stellar-population data files.
            modules: List of ``module load`` commands.  Defaults to the Intel
                2024 toolchain.
        """
        self.galform_dir = Path(galform_dir)
        self.nbody_sim = nbody_sim
        self.model = model
        self.iz = iz
        self.nvol = nvol
        self.partition = partition
        self.account = account
        self.walltime = walltime
        self.stellar_pop_dir = stellar_pop_dir
        self.run_flags = run_flags or RunFlags()
        self.output_folder_name = output_folder_name

        # Modules
        if modules is not None:
            self.modules = modules
        else:
            self.modules = [
                'intel_comp/2024.2.0',
                'compiler-rt',
                'tbb',
                'compiler',
                'mpi',
            ]

        # Log path
        self.log_path = _resolve_log_path(log_path, output_folder_name)

        # Output base directory
        if output_base_dir is not None:
            self.output_base_dir = Path(output_base_dir)
        else:
            self.output_base_dir = _default_cosma_user_root()
        self.models_dir = self.output_base_dir / output_folder_name / nbody_sim

        # Resolve simulation config
        if nbody_sim in SIMULATION_CONFIGS:
            self.sim_config = SIMULATION_CONFIGS[nbody_sim]
            default_iz_list = list(self.sim_config.iz_list)
            self.iz_list = iz_list if iz_list is not None else default_iz_list
            if nvol is not None and nvol_range is not None:
                raise ValueError('Specify only one of nvol and nvol_range')
            resolved_nvol_range = nvol if nvol is not None else nvol_range
            self.nvol_range = resolved_nvol_range if resolved_nvol_range is not None else self.sim_config.nvol_range
        else:
            if nvol is not None and nvol_range is not None:
                raise ValueError('Specify only one of nvol and nvol_range')
            resolved_nvol_range = nvol if nvol is not None else nvol_range
            if iz_list is None or resolved_nvol_range is None:
                raise ValueError(
                    f"Unknown simulation '{nbody_sim}'. "
                    "Provide iz_list and nvol explicitly."
                )
            self.sim_config = None
            self.iz_list = iz_list
            self.nvol_range = resolved_nvol_range

        # Resolve model config
        if model in MODEL_CONFIGS:
            self.model_config = MODEL_CONFIGS[model]
        else:
            self.model_config = None

        if self.iz is not None:
            self.iz_list = [self.iz]

        # Validate
        if not self.galform_dir.is_dir():
            raise FileNotFoundError(f"GALFORM directory not found: {self.galform_dir}")
        galform_exe = self.galform_dir / 'build' / 'galform2'
        if not galform_exe.exists():
            raise FileNotFoundError(
                f"GALFORM executable not found: {galform_exe}"
            )

    # ------------------------------------------------------------------
    # Script generation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bool_to_csh(value: bool) -> str:
        return 'true' if value else 'false'

    def _generate_run_flags_block(self) -> str:
        rf = self.run_flags
        lines = [
            '# ---- run flags (set by GalformSubmitter) ----',
            f'set compile     = {self._bool_to_csh(rf.compile)}',
            f'set galform     = {self._bool_to_csh(rf.galform)}',
            f'set neta        = {self._bool_to_csh(rf.neta)}',
            f'set dust_props  = {self._bool_to_csh(rf.dust_props)}',
            f'set lum_fun     = {self._bool_to_csh(rf.lum_fun)}',
            f'set samp_z0     = {self._bool_to_csh(rf.samp_z0)}',
            f'set cosmicsed   = {self._bool_to_csh(rf.cosmicsed)}',
            f'set lum_fun_burst = {self._bool_to_csh(rf.lum_fun_burst)}',
            f'set samp2_z0      = {self._bool_to_csh(rf.samp2_z0)}',
            f'set sedfit        = {self._bool_to_csh(rf.sedfit)}',
            f'set agn           = {self._bool_to_csh(rf.agn)}',
            f'set sed_agn       = {self._bool_to_csh(rf.sed_agn)}',
            f'set samp_mah      = {self._bool_to_csh(rf.samp_mah)}',
            f'set study_stellar_mass_function = {self._bool_to_csh(rf.study_stellar_mass_function)}',
        ]
        return '\n'.join(lines)

    def _generate_dust_params_block(self, dust: DustParams) -> str:
        lines = [
            '# ---- dust parameters ----',
            f'set dustfile = {dust.dustfile}',
            f'set emdustfile = {dust.emdustfile}',
            f'set rfacburst  = {dust.rfacburst}',
            f'set fcloud = {dust.fcloud}',
            f'set tesc_disk  = {dust.tesc_disk}',
            f'set tesc_burst = {dust.tesc_burst}',
            f'set lambda_break_disk = {dust.lambda_break_disk}',
            f'set beta2_disk = {dust.beta2_disk}',
            f'set lambda_break_burst = {dust.lambda_break_burst}',
            f'set beta2_burst = {dust.beta2_burst}',
        ]
        return '\n'.join(lines)

    def _generate_simulation_block(self, sim: SimulationConfig) -> str:
        lines = [
            '# ---- N-body simulation parameters ----',
            f'set snapshot_file          = {sim.snapshot_file}',
            f'set aquarius_tree_file     = {sim.aquarius_tree_file}',
            f'set aquarius_particle_file = {sim.aquarius_particle_file}',
            f'set volume     = {sim.volume}',
            f'set omega0     = {sim.omega0}',
            f'set lambda0    = {sim.lambda0}',
            f'set omegab     = {sim.omegab}',
            f'set h0         = {sim.h0}',
            f'set sigma8     = {sim.sigma8}',
            f'set PKfile     = {sim.pk_file}',
            f'set iz0        = {sim.iz0}',
        ]
        if sim.lbox is not None:
            lines.append(f'set lbox  = {sim.lbox}')
        if sim.mpart is not None:
            lines.append(f'set mpart = {sim.mpart}')
        return '\n'.join(lines)

    def _generate_model_setup_block(self) -> str:
        """Generate the block that copies the base .input.ref file and applies modifications."""
        if self.model_config is None:
            raise ValueError(
                f"Unknown model '{self.model}'. "
                "Add it to MODEL_CONFIGS or provide a custom model config."
            )
        mc = self.model_config
        lines = [
            '# ---- model parameter file setup ----',
            f'set base_inputs_file = {mc.base_inputs_file}',
            'set galform_inputs_file = ./params/${Nbody_sim}_${model}_iz${iz}_ivol${ivol}.input.temp',
            '\\mkdir -p ./params',
            'cp $base_inputs_file $galform_inputs_file',
        ]
        for name, value in mc.extra_replacements.items():
            lines.append(f'./replace_variable.csh $galform_inputs_file {name} {value}')
        return '\n'.join(lines)

    def _generate_parameter_overrides_block(self) -> str:
        """Generate the block that injects simulation/cosmology params into the input file."""
        lines = [
            '# ---- override parameters for N-body run ----',
            f'./replace_variable.csh $galform_inputs_file stellar_pop_dir {self.stellar_pop_dir}',
            './replace_variable.csh $galform_inputs_file append_ivolume .true.',
            './replace_variable.csh $galform_inputs_file aquarius_tree_file $aquarius_tree_file',
        ]
        if self.nbody_sim != 'nifty62.5':
            lines.append('./replace_variable.csh $galform_inputs_file aquarius_particle_file $aquarius_particle_file')
        else:
            lines.append('./delete_variable.csh $galform_inputs_file aquarius_particle_file')
        lines += [
            './replace_variable.csh $galform_inputs_file volume $volume',
            './replace_variable.csh $galform_inputs_file omega0 $omega0',
            './replace_variable.csh $galform_inputs_file lambda0 $lambda0',
            './replace_variable.csh $galform_inputs_file omegab $omegab',
            './replace_variable.csh $galform_inputs_file h0 $h0',
            './replace_variable.csh $galform_inputs_file sigma8 $sigma8',
            './replace_variable.csh $galform_inputs_file itrans -1',
            './replace_variable.csh $galform_inputs_file PKfile $PKfile',
            './replace_vector.csh $galform_inputs_file zout $z',
        ]
        return '\n'.join(lines)

    def _generate_bands_block(self) -> str:
        """Generate the photometric bands and emission lines configuration."""
        return r"""# ---- photometric bands ----
# Rest frame bands
set idband      = (200 201 127 51 52 53 54 47 48 49 6 202 203 204 205 206 212 213 214 215 216)
set iselect     = (0   0   0   0  0  0  0  0  0  0  0 0   0   0   0   0   0   0   0   0   0  )
# Special bands
set idband_add  = (52 52 1001 1002 1005 1005)
set iselect_add = (2  3  0    0    0    2)
set idband  = ( $idband  $idband_add )
set iselect = ( $iselect $iselect_add )
# Observer frame bands
set idband_add  = (200 201 127 51 52 53 54 47 48 49 6 202 203 204 205 206 212 213 214 215 216)
set iselect_add = (1   1   1   1  1  1  1  1  1  1  1 1   1   1   1   1   1   1   1   1   1  )
set idband  = ( $idband  $idband_add )
set iselect = ( $iselect $iselect_add )
# Top hat bands for dust emission (TH0-TH14)
set idband_add  = (185 186 187 188 189 190 191 192 193 194 195 196 197 198 199)
set iselect_add = (0   0   0   0   0   0   0   0   0   0   0   0   0   0   0)
set idband  = ( $idband  $idband_add )
set iselect = ( $iselect $iselect_add )
# Observer frame NIRCAM bands N1-N8
set idband_add  = (440 441 442 443 444 445 446 447)
set iselect_add = (1   1   1   1   1   1   1   1  )
set idband  = ( $idband  $idband_add )
set iselect = ( $iselect $iselect_add )
# Rest frame NIRCAM bands N1-N8
set idband_add  = (440 441 442 443 444 445 446 447)
set iselect_add = (0   0   0   0   0   0   0   0  )
set idband  = ( $idband  $idband_add )
set iselect = ( $iselect $iselect_add )
# Additional bands for sedfit / cosmicsed
set idband_add  = (232 233 164 165 166 167 294 295 297 294 295 297 200 201 164 165 166 167)
set iselect_add = (1   1   1   1   1   1   0   0   0   1   1   1   1   1   0   0   0   0  )
set idband  = ( $idband  $idband_add )
set iselect = ( $iselect $iselect_add )

set nband = `echo $idband | wc -w`
./replace_vector.csh $galform_inputs_file idband $idband
./replace_vector.csh $galform_inputs_file iselect $iselect

# ---- emission lines ----
./replace_variable.csh $galform_inputs_file emlines .true.
set lines = (Lyalpha Halpha Hbeta OII3727)
set nline = `echo $lines | wc -w`
./replace_variable.csh $galform_inputs_file nline $nline
./replace_vector.csh $galform_inputs_file lines $lines
"""

    def _generate_run_galform_block(self) -> str:
        """Generate the GALFORM execution and post-processing sections."""
        return r"""
############################################################################
# RUN GALFORM

if( $galform == true ) then
    echo '******************************************************************'
    echo running GALFORM
    $GALFORM2_EXE $output_dir $galform_inputs_file  -ivolume=$ivol
    if (( $status != 0 ) || ! ( -e ${output_dir}/global )) then
        echo Galform run failed, aborting script
        exit
    endif
endif

############################################################################
# CREATE ETA FILES  for extinction by dust clouds

if( $neta == true ) then
    echo '******************************************************************'
    echo running NETA_AVE
    set dustparfile = $output_dir/dustpars
    echo dustfile = $dustfile      >! $dustparfile
    echo emdustfile = $emdustfile  >> $dustparfile
    echo rfacburst = $rfacburst    >> $dustparfile
    echo fcloud = $fcloud          >> $dustparfile
    echo tesc_disk = $tesc_disk    >> $dustparfile
    echo tesc_burst = $tesc_burst  >> $dustparfile
    echo upsilon2 = $upsilon2      >> $dustparfile
    echo lambda_break_disk = $lambda_break_disk    >> $dustparfile
    echo beta2_disk = $beta2_disk      >> $dustparfile
    echo lambda_break_burst = $lambda_break_burst  >> $dustparfile
    echo beta2_burst = $beta2_burst    >> $dustparfile

    $NETA_AVE_DISK_EXE <<EOF
    $output_dir
    $tesc_disk
    1
EOF
    $NETA_AVE_BURST_EXE <<EOF
    $output_dir
    $tesc_burst
    1
EOF
endif

############################################################################
# CALCULATE LUMINOSITY FUNCTIONS

if ( $lum_fun == 'true' ) then
    echo '******************************************************************'
    echo running LUM_FUN
    set lffile = $output_dir/gal
    $SAMPLE_GALS_EXE  odir $output_dir  iseed $ISEED2  file $lffile redshift $z \
    mag_sys AB  volume 0  upsilon $upsilon2 \
    dust $dustfile $emdustfile $rfacburst $fcloud $tesc_disk $tesc_burst \
    dust_SED $lambda_break_disk $beta2_disk $lambda_break_burst $beta2_burst \
    dustem 24r 24o 60r 60o 100r 100o 160r 160o 250r 250o 350r 350o 500r 500o 850r 850o 870r 870o \
    lum_fun
    set lffile = $output_dir/gal.Vega
    $SAMPLE_GALS_EXE  odir $output_dir  iseed $ISEED2  file $lffile  redshift $z \
    mag_sys vega  volume 0  upsilon $upsilon2 \
    dust $dustfile $emdustfile $rfacburst $fcloud $tesc_disk $tesc_burst \
    lum_fun
endif

############################################################################
# STELLAR MASS FUNCTION

if ( $study_stellar_mass_function == true ) then
    echo creating smass.cat
    set file = $output_dir/smass.cat
    set vol = 0
    $SAMPLE_GALS_EXE  odir $output_dir  iseed $ISEED2  file $file  redshift $z \
    mag_sys AB  volume 0  upsilon $upsilon2 \
    dust $dustfile $emdustfile $rfacburst $fcloud $tesc_disk $tesc_burst \
    props weight mstars_tot mstars_allburst
endif

echo 'The end'
rm -f $galform_inputs_file
exit
"""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_slurm_script(self, iz: int) -> str:
        """
        Create the complete SLURM batch script for a given snapshot.

        The generated tcsh script is fully self-contained: it sets up
        the environment, changes to the GALFORM source directory,
        constructs the parameter file, runs GALFORM, and performs
        post-processing — without requiring any external csh script.

        Args:
            iz: Snapshot number.

        Returns:
            String containing the complete SLURM script.
        """
        if self.sim_config is None:
            raise ValueError(
                f"No simulation config for '{self.nbody_sim}'. "
                "Provide a SimulationConfig explicitly."
            )
        if self.model_config is None:
            raise ValueError(
                f"No model config for '{self.model}'. "
                "Add it to MODEL_CONFIGS or provide one explicitly."
            )

        jobname = f"{self.nbody_sim}.{self.model}"
        logname = self.log_path / self.nbody_sim / f"{self.model}.%A.%a.log"

        # Ensure log directory exists (best-effort)
        try:
            logname.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            pass

        # Load COSMA modules without relying on interactive tcsh startup files.
        modulecmd = '/cosma/local/Modules/default/libexec/modulecmd.tcl'
        module_lines = (
            f'eval `/usr/bin/tclsh {modulecmd} csh purge`\n'
            + '\n'.join(
                f'eval `/usr/bin/tclsh {modulecmd} csh load {m}`' for m in self.modules
            )
        )

        script = f"""#!/bin/tcsh -ef
#
#SBATCH --ntasks 1
#SBATCH -J {jobname}
#SBATCH -o {logname}
#SBATCH -p {self.partition}
#SBATCH -A {self.account}
#SBATCH -t {self.walltime}
#

# ---- environment ----
{module_lines}

unlimit stacksize
unlimit datasize

# ---- parameters from GalformSubmitter ----
set model     = {self.model}
set Nbody_sim = {self.nbody_sim}
set iz        = {iz}
@ ivol        = ${{SLURM_ARRAY_TASK_ID}} - 1

# Change to GALFORM source directory (scripts use relative paths)
cd {self.galform_dir}
set src_dir = `pwd`
set path = ( $src_dir $path )
set build_dir = ./build/

{self._generate_run_flags_block()}

set models_dir = {self.models_dir}
mkdir -p $models_dir
set upsilon2 = 1
set ISEED2 = -81027

{self._generate_simulation_block(self.sim_config)}
{self._generate_dust_params_block(self.model_config.dust_params)}

# ---- extract redshift from snapshot file ----
set z = `awk -v iz=$iz '$1==iz {{print $2}}' $snapshot_file`
set z0 = `awk -v iz=${{iz0}} '$1==iz {{print $2}}' $snapshot_file`
if ($z == '') then
    echo no redshift for snapshot $iz in file $snapshot_file
    exit
endif
echo running snapshot iz= $iz,   redshift z= $z

set zname = `echo $z | awk '{{printf( "%6.3f",$1)}}'`

set model_dir = $models_dir/$model
mkdir -p $model_dir
set output_dir = $model_dir/iz${{iz}}/ivol${{ivol}}
mkdir -p $output_dir
echo iz= $iz  z= $zname >! $model_dir/iz${{iz}}/zsnap.dat

# ---- executables ----
set GALFORM2_EXE       = ${{build_dir}}/galform2
set NETA_AVE_DISK_EXE  = ${{build_dir}}/neta_ave_disk
set NETA_AVE_BURST_EXE = ${{build_dir}}/neta_ave_burst
set SAMPLE_GALS_EXE    = ${{build_dir}}/sample_gals

# ---- construct GALFORM input parameters file ----
{self._generate_model_setup_block()}
{self._generate_parameter_overrides_block()}

# ---- photometric bands & emission lines ----
{self._generate_bands_block()}

# ---- execute GALFORM & post-processing ----
{self._generate_run_galform_block()}
"""
        return script

    def submit_job(self, iz: int, dry_run: bool = False) -> Optional[str]:
        """
        Submit a SLURM job for a given snapshot.

        Args:
            iz: Snapshot number.
            dry_run: If True, print the script but don't submit.

        Returns:
            Job ID if submitted, None if dry_run.
        """
        script_content = self.create_slurm_script(iz)

        if dry_run:
            print(f"DRY RUN: iz={iz}, nvol_range={self.nvol_range}")
            print(script_content)
            return None

        cmd = ['sbatch']
        cmd.append(f'--array={self.nvol_range}')

        try:
            result = subprocess.run(
                cmd,
                input=script_content.encode(),
                capture_output=True,
                check=True,
            )

            output = result.stdout.decode().strip()
            if "Submitted batch job" in output:
                job_id = output.split()[-1]
                return job_id
            return None

        except subprocess.CalledProcessError as e:
            stdout = e.stdout.decode() if e.stdout else ""
            stderr = e.stderr.decode() if e.stderr else ""
            raise RuntimeError(
                f"Failed to submit job for iz={iz}: {e}\n"
                f"STDOUT: {stdout}\nSTDERR: {stderr}"
            ) from e

    def submit_all_jobs(self, dry_run: bool = False) -> List[str]:
        """
        Submit SLURM jobs for all snapshots in ``iz_list``.

        Args:
            dry_run: If True, print scripts but don't submit.

        Returns:
            List of submitted job IDs.
        """
        job_ids = []
        for iz in self.iz_list:
            job_id = self.submit_job(iz, dry_run=dry_run)
            if job_id:
                job_ids.append(job_id)
        return job_ids


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Submit GALFORM N-body runs to SLURM batch queue on COSMA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit jobs for L800 simulation with gp14 model
  %(prog)s /path/to/galform

  # Submit jobs with custom simulation & model
  %(prog)s /path/to/galform --nbody-sim MillGas --model b06

  # Dry run to preview what would be submitted
    %(prog)s /path/to/galform --iz 271 --nvol 12 --dry-run

    # Custom snapshot list and subvolume range
    %(prog)s /path/to/galform --iz-list 100 120 155 --nvol 1-50

  # Enable/disable pipeline stages
  %(prog)s /path/to/galform --run-galform --no-neta --no-lum-fun
        """,
    )

    parser.add_argument(
        'galform_dir', nargs='?',
        help='Path to the GALFORM source directory (contains build/, *.input.ref, etc.)',
    )

    parser.add_argument('--nbody-sim', default='L800',
                        help='N-body simulation name (default: L800)')
    parser.add_argument('--model', default='gp14',
                        help='GALFORM model name (default: gp14)')
    parser.add_argument('--iz', type=int,
                        help='Single snapshot number to submit')
    parser.add_argument('--nvol',
                        help='Subvolume range for SLURM array submission (e.g. "1-10" or "12")')
    parser.add_argument('--output-base-dir',
                        help='Root directory for GALFORM outputs (default: /cosma5/data/durham/$USER)')
    parser.add_argument('--output-folder-name', default='Galform_Out',
                        help='Folder name under the base output directory (default: Galform_Out)')
    parser.add_argument('--log-path',
                        help='Directory for SLURM log files')
    parser.add_argument('--partition', default='cosma5',
                        help='SLURM partition (default: cosma5)')
    parser.add_argument('--account', default='durham',
                        help='SLURM account (default: durham)')
    parser.add_argument('--walltime', default='72:00:00',
                        help='Job wall-time (default: 72:00:00)')
    parser.add_argument('--iz-list', type=int, nargs='+',
                        help='Override default snapshot list')
    parser.add_argument('--nvol-range',
                        help='Deprecated alias for --nvol')

    # Run-flag toggles
    flag_group = parser.add_argument_group('pipeline stages')
    flag_group.add_argument('--run-galform', action='store_true', default=False,
                            help='Run galform2 executable (default: off)')
    flag_group.add_argument('--no-neta', action='store_true',
                            help='Disable neta_ave dust calculation')
    flag_group.add_argument('--no-lum-fun', action='store_true',
                            help='Disable luminosity function calculation')
    flag_group.add_argument('--no-study-smf', action='store_true',
                            help='Disable stellar mass function output')
    flag_group.add_argument('--run-dust-props', action='store_true',
                            help='Enable dust properties output')
    flag_group.add_argument('--run-samp-z0', action='store_true',
                            help='Enable z=0 galaxy sample output')

    parser.add_argument('--dry-run', action='store_true',
                        help='Print job scripts without submitting')
    parser.add_argument('--list-simulations', action='store_true',
                        help='List available simulation configurations and exit')
    parser.add_argument('--list-models', action='store_true',
                        help='List available model configurations and exit')

    args = parser.parse_args()

    if args.list_simulations:
        print("Available simulation configurations:")
        fmt = f"{'Simulation':<20} {'Snapshots (iz)':<40} {'Subvolumes':<15}"
        print(fmt)
        print("-" * 75)
        for name, cfg in sorted(SIMULATION_CONFIGS.items()):
            iz_str = str(cfg.iz_list)
            if len(iz_str) > 37:
                iz_str = iz_str[:34] + '...'
            print(f"{name:<20} {iz_str:<40} {cfg.nvol_range:<15}")
        return 0

    if args.list_models:
        print("Available model configurations:")
        fmt = f"{'Model':<25} {'Base Input File':<45} {'Dust'}"
        print(fmt)
        print("-" * 80)
        for name, cfg in sorted(MODEL_CONFIGS.items()):
            dust_label = f"fcloud={cfg.dust_params.fcloud}"
            print(f"{name:<25} {cfg.base_inputs_file:<45} {dust_label}")
        return 0

    if not args.galform_dir:
        parser.error("galform_dir is required unless using --list-simulations or --list-models")

    run_flags = RunFlags(
        galform=args.run_galform,
        neta=not args.no_neta,
        lum_fun=not args.no_lum_fun,
        study_stellar_mass_function=not args.no_study_smf,
        dust_props=args.run_dust_props,
        samp_z0=args.run_samp_z0,
    )

    try:
        submitter = GalformSubmitter(
            galform_dir=args.galform_dir,
            nbody_sim=args.nbody_sim,
            model=args.model,
            iz=args.iz,
            nvol=args.nvol,
            output_base_dir=args.output_base_dir,
            output_folder_name=args.output_folder_name,
            log_path=args.log_path,
            partition=args.partition,
            account=args.account,
            walltime=args.walltime,
            iz_list=args.iz_list,
            nvol_range=args.nvol_range,
            run_flags=run_flags,
        )
        submitter.submit_all_jobs(dry_run=args.dry_run)
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
