# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install the package in editable mode
pip install -e .

# Tests
pytest tests -q

# Lint
ruff check galform_analysis
```

SLURM jobs run on COSMA and activate `.venv` (Python 3.12) at repo root. Scripts in `scripts/` should use the installed `galform_analysis` package. Notebooks in `examples/` have logic to dynamically find the repo root and add it to `sys.path` to ensure the local version of `galform_analysis` is used.

When creating or editing notebooks and plotting examples, always import `matplotlib as mpl` and call `mpl.setconfig()` before plotting so figures use the repository's custom formatting. Examples in `examples/` should stay reusable after cloning, so resolve paths dynamically from the current environment or repository root and avoid hardcoded absolute paths.

## Repository layout

| Directory | Git | Purpose |
|-----------|-----|---------|
| `galform_analysis/` | tracked | Reusable GALFORM analysis library |
| `examples/` | tracked | Showcase notebooks for `galform_analysis/` |
| `tests/` | tracked | Unit tests for `galform_analysis/` |
| `science/` | **gitignored** | Personal research notebooks, drafts, paper prose |
| `scripts/` | gitignored | HPC submission scripts |
| `data/` | gitignored | Output CSVs from HPC runs |
| `logs/` | gitignored | SLURM log files |

`science/` is gitignored — it is the active workspace for the research work streams below. It is not intended as a public library; do not add `science/` files to git.

---

## Work stream 1: Reusable analysis library

**Goal:** A clean, tested Python library for reading and analysing GALFORM HDF5 outputs. Showcased in `examples/`, fully unit-tested in `tests/`. This is the public-facing part of the repo.

All modules in `galform_analysis/` use flat absolute imports with the package prefix.

```python
from galform_analysis import config
from galform_analysis.readers.loaders import read_snapshot_data
from galform_analysis.utils.read_galaxies import read_galaxy_arrays
from galform_analysis.analysis.mass_functions import hmf_given_redshift_and_subvolume
```

### Data flow

GALFORM writes output at:
```
/cosma5/data/durham/dc-hick2/Galform_Out/<SIM>/<MODEL>/iz<N>/ivol<K>/galaxies.hdf5
```

`config.BASE_DIR` (overridable via `GALFORM_BASE_DIR` env var or `set_base_dir()`) points here. Scripts accept `--base-dir` / `--sim-name` / `--model` / `--iz` and construct paths themselves.

Snapshot indices (`iz`) map to redshifts via `galform_analysis/redshift_lists/<sim_name>.txt` — use `config.get_snapshot_redshift('iz155', 'L800')` or `config.find_snapshot_at_redshift(0.5, 'L800')`. To add a new simulation, drop a `<sim_name>.txt` file into `galform_analysis/redshift_lists/`.

### Library structure (`galform_analysis/`)

| Module | Role |
|--------|------|
| `config.py` | `BASE_DIR`, `Cosmology` constants, `iz` ↔ redshift lookup |
| `readers/loaders.py` | Low-level HDF5 openers; `read_snapshot_data()` returns a dict with an open `h5py.File` — caller must call `close_snapshot()` |
| `utils/read_galaxies.py` | Higher-level readers → numpy arrays or DataFrames, with optional position/velocity arrays, mass cuts, and centrals-only filtering |
| `analysis/mass_functions/` | HMF, SMF, HOD, theoretical HMF |
| `analysis/aggregation.py` | Cross-subvolume aggregation helpers |
| `analysis/correlation/` | 2PCF modules (see below) |
| `analysis/redshift_space_distortions/` | RSD multipole estimators |

### 2PCF estimators (`analysis/correlation/`)

The core pattern is the **subvolume-weighted correction**: for m selected subvolumes out of k total,
```
DD_corr = alpha * DD_auto + beta * DD_cross
```
where `alpha = m/k`, `beta = m(k-1) / [k(m-1)]`.

| Module | Purpose |
|--------|---------|
| `correlation.py` | Base Corrfunc `DD` wrapper |
| `subvol_weighted_correction.py` | Subvolume-weighted estimator (main workhorse) |
| `halo_sampling_correction.py` | Halo-sampling convergence correction |
| `group_sampling_correlation.py` | Group/galaxy sampling 2PCF |
| `mass_weighted_correlation.py` | Mass-weighted 2PCF |
| `satellite_cross_correlation.py` | Satellite–central cross-correlation |
| `galaxy_bias.py` | `b(r) = sqrt(xi_gal / xi_dm)` |
| `dm_correlation.py` | DM-only 2PCF from merger tree files |

### RSD multipoles (`analysis/redshift_space_distortions/`)

`subvol_weighted_multipoles.py` — direct periodic estimator: `xi(s,mu) = DD(s,mu) / RR_analytic(s,mu) - 1`, projected to monopole (xi0) and quadrupole (xi2). No random catalogue needed for a periodic box.

---

## Work stream 2: SCOPE ξ(r) paper
... (rest of file remains same)
