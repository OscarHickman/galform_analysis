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

### Configure Your Base Directory

Set the path to your GALFORM output directory:

```python
from galform_analysis.config import set_base_dir

set_base_dir('/cosma5/data/durham/dc-hick2/Galform_Out/L800/gp14')
```


## Examples

See the `examples/` directory for complete working Jupyter notebooks:

- **`compare_mass_functions.ipynb`** - HMF comparison to Press–Schechter, Sheth–Tormen, and Tinker08 (via `hmf`)
- **`galaxy_efficiency.ipynb`** - Galaxy formation efficiency analysis
- **`subvolume_convergence.ipynb`** - HMF convergence with varying subvolume counts
- many more examples

Open a notebook:
```bash
cd examples
jupyter notebook compute_smf.ipynb
# or
jupyter lab
```

## GALFORM Job Submission

Use `submit_galform_slurm.py` to submit GALFORM runs to SLURM on COSMA.

```bash
python src/galform_execution/submit_galform_slurm.py --help
```

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

### Import Examples

```python
# Top-level imports (convenience)
from galform_analysis import (
    set_base_dir,
    get_snapshot_redshift,
    read_snapshot_data,
    aggregate_snapshot,
    hmf_given_redshift_and_subvolume,
    avg_hmf_given_redshift_and_subvolumes,
    smf_given_redshift_and_subvolume,
    avg_smf_given_redshift_and_subvolumes,
    plot_hmf_convergence_by_subvolumes,
    plot_smf_convergence_by_subvolumes,
)

# Or import from specific subpackages
from galform_analysis.analysis import plot_hmf_convergence_by_redshift
from galform_analysis.analysis import plot_smf_convergence_by_redshift
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

## Requirements
See requirements.txt


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
