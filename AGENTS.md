# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install in editable mode (use uv for speed)
uv pip install -e ".[dev]"

# Run all tests
pytest tests

# Run a single test file
pytest tests/galform_analysis/analysis/correlation/test_subvol_weighted_correction.py

# Run a single test by name
pytest tests -k "test_name"

# Lint
ruff check galform_analysis

# Format
black galform_analysis
```

## Architecture

`galform_analysis` is a Python library for reading and analysing GALFORM semi-analytic model outputs stored as HDF5 files. The on-disk layout is:

```
<BASE_DIR>/iz<NNN>/ivol<M>/galaxies.hdf5
```

Each `ivol` is an **independent full-box realisation** (not a spatial tile) of the simulation. The full-box size for L800 is 542.16 Mpc/h; `V_ivol` in the HDF5 `Parameters` group is the per-subvolume statistical volume. Never interpret subvolumes as spatially disjoint.

### Layer overview

| Layer | Path | Role |
|---|---|---|
| Config | `galform_analysis/config.py` | `SimulationConfig`, `set_base_dir`, `load_redshift_mapping` |
| Readers | `galform_analysis/readers/loaders.py` | Low-level HDF5 open/read; `read_snapshot_data` returns raw arrays + open file handle |
| Utils | `galform_analysis/utils/read_galaxies.py` | `read_galaxy_arrays`, `read_galaxy_positions`, `read_halo_positions` – filtered, normalised NumPy arrays |
| Analysis | `galform_analysis/analysis/` | SMF, HMF, HOD, 2PCF, NPCF, RSD multipoles |
| Aggregation | `galform_analysis/analysis/aggregation.py` | Scan & stack data across many ivols using polars |

### Key data-flow for correlation functions

1. `read_galaxy_arrays` / `read_galaxy_positions` (utils) → NumPy position arrays
2. `compute_xi_corrfunc` (correlation.py) → calls `Corrfunc.theory.xi` for periodic box xi(r)
3. `subvol_weighted_correction.py` → auto/cross decomposition for subvolume sub-sampling bias correction:
   - `load_subvolume_galaxies` builds a tagged polars DataFrame
   - `compute_weighted_xi_from_catalogue` / `compute_weighted_wp_from_catalogue` apply the alpha/beta weighting scheme
4. `subvol_weighted_multipoles.py` → extends the correction to RSD xi(s, mu) and multipoles xi_0, xi_2, xi_4

### Simulation configs

Sourced from `galform_execution` (PyPI package) at runtime: `galform_execution/config/simulations/<family>.json`. When `galform_execution` is not installed, falls back to the local `galform_analysis/sim_configs/` directory. `SimulationConfig('L800')` loads cosmology, box size, and number of subvolumes. Redshift → iz index mappings live in `galform_analysis/redshift_lists/<sim_name>.txt`.

### HDF5 schema notes

- Stellar mass = `mstars_disk + mstars_bulge` (fields inside `Output001`, `Output002`, … — loader picks the highest-numbered group)
- Halo mass = `mhalo` (subhalo), `mhhalo` (host/FOF)
- Positions = `xgal`, `ygal`, `zgal` in Mpc/h, spanning [0, 542.16)
- Central galaxies: `is_central == 1`
- `CompletionFlag == 1` means the subvolume finished successfully
- All masses in M_sun/h

### Testing

Tests use synthetic HDF5 files built by `tests/conftest.py` (`write_galaxy_hdf5`), matching the real GALFORM schema without requiring access to the simulation outputs. No real data is needed to run the test suite.

### Dependencies

Runtime: `numpy`, `scipy`, `polars`, `h5py`, `matplotlib`, `seaborn`, `Corrfunc`, `sugc`, `galform_execution`
Dev extras: `pytest`, `ruff`, `black`, `build`
Science extras (optional): `astropy`, `hmf`, `halotools`

`N_SUBVOLUMES = 1024` is an internal fallback constant used when the HDF5 `Parameters/n_subvolumes` field is absent. Use `SimulationConfig` for all external access to simulation parameters.
