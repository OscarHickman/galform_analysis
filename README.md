# galform_analysis

A Python library for analyzing GALFORM galaxy formation simulation outputs, providing tools for reading HDF5 snapshot data, computing mass functions, and performing convergence analysis.

## Features

- **Data I/O**: Robust HDF5 snapshot readers with completion flag checking
- **Mass Functions**: Compute stellar and halo mass functions with subvolume averaging
- **Convergence Analysis**: Test how results vary with subvolume sample size
- **Analysis Tools**: Aggregate data across subvolumes with error handling
- **Utilities**: Statistical functions and plotting helpers for visualization
- **Flexible Configuration**: Environment-based BASE_DIR management for different runs
- **Comprehensive Testing**: Unit tests and CI/CD pipeline with GitHub Actions

## Installation

```bash
# Clone the repository
cd galform_analysis

# Install from requirements
pip install -r requirements.txt
```

## Quick Start

### 1. Configure Your Base Directory

Set the path to your GALFORM output directory:

```python
from galform_analysis.config import set_base_dir

# Option 1: Set in Python
set_base_dir('/cosma5/data/durham/dc-hick2/Galform_Out/L800/gp14')

# Option 2: Use environment variable
# export GALFORM_BASE_DIR=/path/to/galform/output

# Option 3: Edit src/galform_analysis/config.py directly
```

### 2. Compute a Stellar Mass Function

```python
from galform_analysis.analysis import compute_smf_avg_by_snapshot
import matplotlib.pyplot as plt

# Compute SMF for z=0 (iz99)
smf = compute_smf_avg_by_snapshot('iz99')

# Plot
plt.plot(smf['centers'], smf['phi'], 'o-')
plt.yscale('log')
plt.xlabel(r'$\log_{10}(M_*/M_\odot)$')
plt.ylabel(r'$\Phi$ [Mpc$^{-3}$ dex$^{-1}$]')
plt.show()
```

### 3. Aggregate Data from a Snapshot

```python
from galform_analysis.analysis import aggregate_snapshot

# Load all subvolumes for a snapshot
agg = aggregate_snapshot('iz99')

print(f"Redshift: {agg['z']}")
print(f"Total volume: {agg['volume']} Mpc^3")
print(f"Number of galaxies: {len(agg['mstar'])}")
```

## Examples

See the `examples/` directory for complete working Jupyter notebooks:

- **`compute_smf.ipynb`** - Basic stellar mass function computation
- **`compare_mass_functions.ipynb`** - Multi-redshift SMF and HMF comparison with observational data
- **`galaxy_efficiency.ipynb`** - Galaxy formation efficiency analysis
- **`subvolume_convergence.ipynb`** - Convergence testing with varying subvolume counts
- **`imf_replication.ipynb`** - Initial mass function analysis and replication studies
- **`test_memory_loading.ipynb`** - Memory optimization and data loading performance tests

Open a notebook:
```bash
cd examples
jupyter notebook compute_smf.ipynb
# or
jupyter lab
```

## Library Structure

```
galform_analysis/
├── src/galform_analysis/
│   ├── config.py              # Configuration (BASE_DIR, cosmology, constants)
│   ├── redshift_list.txt      # Snapshot-to-redshift mapping
│   ├── io/                    # Data loading
│   │   ├── loaders.py        # HDF5 snapshot readers with robust error handling
│   │   └── readers.py        # Luminosity function file readers
│   ├── analysis/              # Analysis functions
│   │   ├── aggregation.py    # Data aggregation across subvolumes
│   │   ├── smf.py            # Stellar mass function computation
│   │   ├── hmf.py            # Halo mass function computation
│   │   └── convergence.py    # Convergence testing utilities
│   └── utils/                 # Utilities
│       ├── statistics.py     # Statistical helper functions
│       └── plotting.py       # Plotting utilities and layout helpers
├── src/galform_execution/
│   ├── runner.py             # GALFORM execution wrapper
│   └── submit_galform_slurm.py  # Python script for submitting GALFORM jobs to SLURM
├── examples/                  # Example notebooks and plots
├── tests/                     # Unit tests
└── README.md
```

## GALFORM Job Submission

The `submit_galform_slurm.py` script provides a Python interface for submitting GALFORM N-body runs to the SLURM batch queue on COSMA, replacing the traditional bash qsub scripts.

### Basic Usage

```bash
# Submit jobs for L800 simulation with gp14 model (default)
python src/galform_execution/submit_galform_slurm.py \
    /path/to/galform2 \
    /path/to/run_galform_Nbody_example.csh

# Dry run to preview what would be submitted
python src/galform_execution/submit_galform_slurm.py \
    /path/to/galform2 \
    /path/to/run_galform_Nbody_example.csh \
    --dry-run

# List available simulation configurations
python src/galform_execution/submit_galform_slurm.py --list-simulations
```

### Advanced Options

```bash
# Submit with custom simulation and model
python src/galform_execution/submit_galform_slurm.py \
    /path/to/galform2 \
    /path/to/run_script.csh \
    --nbody-sim MillGas \
    --model b06

# Custom snapshot list and subvolume range
python src/galform_execution/submit_galform_slurm.py \
    /path/to/galform2 \
    /path/to/run_script.csh \
    --iz-list 100 120 155 \
    --nvol-range 1-50

# Custom SLURM parameters
python src/galform_execution/submit_galform_slurm.py \
    /path/to/galform2 \
    /path/to/run_script.csh \
    --partition cosma7 \
    --account dp004 \
    --walltime 48:00:00
```

### Supported Simulations

- **L800** (default): iz=[271, 207, 176, 155, 142, 120, 105, 100, 82], nvol=701-900
- **MilliMil**: iz=[63], nvol=1-8
- **MillGas**: iz=[61], nvol=1-10
- **EagleDM**: iz=[200], nvol=1-128
- And more... (use `--list-simulations` to see all)

## Key Functions

### Configuration (`galform_analysis.config`)
- `set_base_dir(path)` - Set GALFORM output directory
- `get_base_dir()` - Get current base directory as Path object
- `get_snapshot_redshift(snapshot)` - Get redshift for snapshot (e.g., 'iz99')
- `find_snapshot_at_redshift(z, tolerance)` - Find snapshot closest to target redshift
- `load_redshift_mapping()` - Load full iz→redshift mapping from file
- `Cosmology` - Class with cosmological parameters (Ω_m, Ω_Λ, H0, etc.)

### I/O (`galform_analysis.io`)
- `read_snapshot_data(iz_path, ivol)` - Read single subvolume data (mstar, mhalo, sfr, luminosities)
- `close_snapshot(data)` - Close HDF5 file safely
- `get_completed_subvolumes(iz_path)` - Find all completed subvolumes
- `open_galaxies_hdf5(iz_path, ivol)` - Open HDF5 file handle
- `get_output_group(f)` - Get Output group from HDF5 file

### Analysis (`galform_analysis.analysis`)
- `aggregate_snapshot(iz_path)` - Combine all subvolumes into single dataset
- `compute_smf_avg_by_snapshot(iz_path, bins, ivol_sample)` - Stellar mass function with averaging
- `compute_hmf_avg_by_snapshot(iz_path, bins, ivol_sample)` - Halo mass function with averaging
- `compute_smf_from_aggregated(agg_data, bins)` - SMF from pre-aggregated data
- `compute_hmf_from_aggregated(agg_data, bins)` - HMF from pre-aggregated data
- `plot_smf_convergence(iz_paths, n_samples)` - Plot SMF convergence across subvolumes
- `plot_hmf_convergence(iz_paths, n_samples)` - Plot HMF convergence across subvolumes

### Utilities (`galform_analysis.utils`)
- `count_occurrences(x, y)` - Count unique values efficiently
- `create_mask_inside_range(x, low, upp)` - Create boolean masks for filtering
- `create_random_sample_mask(n, percent)` - Random sampling masks
- `change_axes_fontsize(fs)` - Set global plot font sizes
- `set_minor_ticks(ax_obj)` - Add minor ticks to plots
- `create_residual_axes(...)` - Create plot with residual panel
- `draw_cuboid_3d(...)` - Draw 3D shapes in matplotlib

### Import Examples

```python
# Top-level imports (convenience)
from galform_analysis import (
    set_base_dir,
    get_snapshot_redshift,
    read_snapshot_data,
    aggregate_snapshot,
    compute_smf_avg_by_snapshot,
    compute_hmf_avg_by_snapshot,
)

# Or import from specific modules
from galform_analysis.config import Cosmology, load_redshift_mapping
from galform_analysis.analysis.smf import compute_smf_from_aggregated
from galform_analysis.analysis.hmf import compute_hmf_from_aggregated
from galform_analysis.analysis.convergence import plot_smf_convergence
from galform_analysis.io.loaders import get_completed_subvolumes
from galform_analysis.io.readers import LuminosityFunction
from galform_analysis.utils.statistics import count_occurrences
from galform_analysis.utils.plotting import create_residual_axes
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/galform_analysis/test_config.py
```

### Linting

```bash
# Check code quality with ruff
ruff check src/galform_analysis
ruff check src/galform_execution
```

### CI/CD

The repository includes a GitHub Actions workflow (`.github/workflows/ci.yml`) that:
- Runs on push and pull requests to main branch
- Tests with Python 3.9
- Runs ruff linting on both source packages
- Executes pytest test suite

### Code Organization

The library follows a clean, modular structure:
- All source code in `src/galform_analysis/` and `src/galform_execution/`
- Examples and plots in `examples/`
- Tests in `tests/` mirroring source structure
- Configuration centralized in `config.py`
- Package metadata in `setup.py`

## Requirements

### Core Dependencies
- Python >= 3.8
- numpy >= 1.23.0
- scipy >= 1.7.0
- matplotlib >= 3.3.0
- pandas >= 1.3.0
- h5py >= 3.0.0
- seaborn >= 0.11.0

### Development Dependencies
- pytest >= 8.0.0
- ruff >= 0.1.0

### Optional Dependencies
- astropy >= 4.0
- hmf >= 3.0 (for theoretical halo mass functions)
- jupyter (for running example notebooks)

## Contributing

Contributions are welcome! Please ensure that:
1. All tests pass (`pytest tests/`)
2. Code passes linting (`ruff check src/`)
3. New features include appropriate tests
4. Documentation is updated as needed

## Citation

If you use this library in your research, please cite:
```
Hickman, O. (2025). galform_analysis: A Python library for GALFORM simulation analysis.
GitHub repository: https://github.com/OscarHickman/galform_analysis
```

## License

TBD

## Author

Oscar Hickman (oscar.hickman@durham.ac.uk)
