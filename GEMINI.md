# Project: galform_analysis

Tools for analyzing GALFORM HDF5 outputs on COSMA. This project provides a modular framework for reading simulation data, aggregating it across subvolumes, and performing various astronomical analyses like stellar mass functions, halo mass functions, and correlation functions.

## Project Structure

- `src/`: Main source code.
    - `config.py`: Central configuration for paths, constants, and default binning.
    - `readers/`: Utilities for reading HDF5 outputs (`loaders.py`).
    - `analysis/`: Analysis modules.
        - `aggregation.py`: Tools for scanning and aggregating data across subvolumes using Polars.
        - `mass_functions/`: SMF, HMF, HOD computation.
        - `correlation/`: 2-point correlation functions and galaxy bias.
        - `redshift_space_distortions/`: RSD multipoles and corrections.
    - `utils/`: Helper functions for stats, plotting, etc.
    - `redshift_lists/`: Redshift mappings for different simulations (e.g., `L800.txt`).
- `tests/`: Pytest suite mirroring the `src/` structure.
- `examples/`: Jupyter notebooks demonstrating usage and specific scientific investigations.
- `data/`: Placeholder/small data files (if applicable).
- `scripts/`: Batch processing scripts.

## Core Concepts & Workflow

1.  **Configuration**: Set `BASE_DIR` in `src/config.py` or via `GALFORM_BASE_DIR` environment variable to point to the simulation output.
2.  **Data Ingestion**: Use `src.readers.loaders.read_snapshot_data` to load galaxy properties from a specific snapshot (`iz*`) and subvolume (`ivol*`). Always remember to close the returned HDF5 file using `close_snapshot`.
3.  **Aggregation**: Since simulation data is split into many subvolumes (default `N_SUBVOLUMES = 1024`), use functions in `src.analysis.aggregation` to find completed subvolumes and aggregate data across them.
4.  **Analysis**: Perform scientific analyses using the specialized modules in `src.analysis`. Most modules support both per-subvolume analysis and analysis of aggregated data.

## Development Commands

- **Set up environment**: `pip install -r requirements.txt`
- **Run tests**: `pytest tests`
- **Linting**: `ruff check src`
- **Formatting**: `ruff format src` (if configured)

## COSMA Partition & Mount Guide

Most data for this project lives on `/cosma5`. Only specific partitions can see this file system.

| Partition Group | Mounts `/cosma5`? | Target Nodes | Notes |
|:---|:---:|:---|:---|
| `cosma5` | **Yes** | `m5xxx` | Default for this repo. |
| `cosma7` (standard) | **No** | `m7xxx` | **Avoid.** |
| `cosma8` (standard) | **No** | `m8xxx` | **Avoid.** |
| `cosma7-shm` / `shm2` | **Yes** | `mad01-03` | Use for high-memory/spillover. |
| `cosma8-shm` / `shm2` / `shm3` | **Yes** | `mad04-05`, `ga004-006` | Use for high-memory/spillover. |
| `cosma8-ska` | **Yes** | `mad07` | Valid spillover. |

**Important:** Jobs submitted to standard `cosma7` or `cosma8` partitions will fail immediately when attempting to read simulation data from `/cosma5`.

## Coding Conventions

- **Data Types**: Use `numpy` arrays for numerical data and `polars` for tabular data/aggregation tasks.
- **HDF5 Handling**: Ensure HDF5 files are properly closed after reading.
- **Docstrings**: Follow NumPy/Google-style docstrings.
- **Testing**: Add tests for new analysis functions in `tests/galform_analysis/`.
- **Notebooks**: Use notebooks in `examples/` for exploration and plotting, but keep core logic in `src/`.

## Key Files to Reference

- `src/config.py`: To check or modify default bins, cosmology, or simulation constants.
- `src/readers/loaders.py`: To see how specific HDF5 datasets are being mapped to galaxy properties.
- `src/analysis/aggregation.py`: For logic related to scanning simulation directories and handling completion flags.
