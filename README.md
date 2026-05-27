# galform_analysis

[![CI](https://github.com/OscarHickman/galform_analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/OscarHickman/galform_analysis/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.FIXME.svg)](https://doi.org/10.5281/zenodo.FIXME)


A modular Python framework designed for the efficient reading and analysis of GALFORM HDF5 simulation outputs. This library provides standardized tools for processing large-scale galaxy formation data, from low-level HDF5 I/O to high-level astronomical probes.

## Core Features

- **Standardized I/O**: Robust loaders for GALFORM `galaxies.hdf5` files with support for different output versions.
- **Data Aggregation**: Tools to scan simulation directories and aggregate data across subvolumes using high-performance `polars` dataframes.
- **Mass Functions**: Computation of Stellar Mass Functions (SMF), Halo Mass Functions (HMF), and Halo Occupation Distribution (HOD).
- **Correlation Functions**: Estimators for 2-point, 3-point, and N-point correlation functions (2PCF/NPCF) including subvolume-weighted corrections for convergence analysis.
- **Redshift-Space Distortions**: Estimators for anisotropic clustering ($\xi(s, \mu)$) and multipoles ($\xi_0, \xi_2, \xi_4$).
- **Simulation Management**: Built-in configurations for major N-body simulations including L800, Millennium I/II, COLIBRE, and FLAMINGO.

## Installation

Install the package in your Python environment:

```bash
uv pip install -e .
```

### Dependencies
The library requires `numpy`, `scipy`, `matplotlib`, `polars`, `h5py`, `seaborn`, and `Corrfunc`. These are automatically managed during installation.

## Quick Start

The following example demonstrates how to load a simulation configuration and read galaxy data:

```python
from galform_analysis import SimulationConfig, config
from galform_analysis.readers.loaders import read_snapshot_data

# 1. Access simulation-specific constants (box size, cosmology, etc.)
sim = SimulationConfig('L800')
print(f"Simulation: {sim.name}, Box Size: {sim.box_size} Mpc/h")

# 2. Configure the data location
config.set_base_dir('/path/to/Galform_Out/L800/model_name')

# 3. Load snapshot data for a specific subvolume
data = read_snapshot_data('iz271', ivol=0)
mstar = data['mstar']  # Stellar masses in M_sun/h
```

## Citation & Academic Credit

If you use `galform_analysis` in your research, please cite the software using the DOI provided above or the following BibTeX entry:

```bibtex
@software{galform_analysis,
  author = {Hickman, Oscar},
  title = {galform_analysis: A modular Python framework for analyzing GALFORM simulation outputs},
  version = {0.1.3},
  doi = {10.5281/zenodo.FIXME},
  url = {https://github.com/OscarHickman/galform_analysis}
}
```

## Simulation Metadata

Configurations for supported simulations are stored centrally in `galform_analysis/sim_configs/`. This allows for dynamic access to cosmological parameters and volume metadata:

```python
from galform_analysis import SimulationConfig

flamingo = SimulationConfig('FLAMINGO')
omega_m = flamingo.omega_m
h0 = flamingo.h0
```

## Documentation & Examples

Refer to the `examples/` directory for interactive Jupyter notebooks:
- `examples/readers/load_snapshot.ipynb`: Introduction to data loading.
- `examples/analysis/mass_functions/smf_example.ipynb`: Plotting Stellar Mass Functions.
- `examples/analysis/correlation/correlation_example.ipynb`: Computing clustering statistics.

## Testing & Quality Standards

The project maintains high code quality through automated linting and comprehensive testing:

```bash
# Run the test suite
pytest tests

# Check code style
ruff check galform_analysis
```

## Author

Oscar Hickman
