# Project: galform_analysis

A modular Python framework for reading and analyzing GALFORM HDF5 simulation outputs on the COSMA HPC cluster. It supports large-scale data aggregation across subvolumes and various astronomical probes including Stellar/Halo Mass Functions (SMF/HMF), N-point correlation functions (2PCF/3PCF/NPCF), and Redshift-Space Distortions (RSD).

## Repository Layout

| Directory | Git Status | Purpose |
|-----------|------------|---------|
| `src/` | Tracked | Reusable GALFORM analysis library. |
| `examples/` | Tracked | Showcase notebooks for `src/` usage. |
| `tests/` | Tracked | Comprehensive unit tests mirroring `src/` structure. |
| `science/` | **Ignored** | Research notebooks, drafts, paper prose (active work streams). |
| `scripts/` | Ignored | HPC submission scripts and SLURM wrappers. |
| `data/` | Ignored | Intermediate outputs (CSVs) from HPC runs. |
| `logs/` | Ignored | SLURM stdout/stderr log files. |
| `_plots/` | Ignored | Generated figures and visualizations. |

## Library Structure & Usage (`src/`)

All modules in `src/` use flat absolute imports. To use the library, add the `src/` directory to `sys.path`.

```python
import sys
sys.path.insert(0, '/cosma/apps/durham/dc-hick2/galform_analysis/src')

from config import get_snapshot_redshift
from readers.loaders import read_snapshot_data
from utils.read_galaxies import read_galaxy_arrays
from analysis.mass_functions import smf
```

### Module Roles

| Module | Role |
|--------|------|
| `config.py` | Central config: `BASE_DIR`, Cosmology constants, snapshot-to-redshift mapping. |
| `readers/loaders.py` | Low-level HDF5 IO; `read_snapshot_data` returns a dict with open `h5py.File`. |
| `analysis/aggregation.py` | Tools for scanning simulation dirs and aggregating data via Polars. |
| `analysis/mass_functions/` | SMF, HMF (snapshot/theoretical), and HOD computation. |
| `analysis/correlation/` | 2PCF/NPCF estimators and weighting logic. |
| `analysis/redshift_space_distortions/` | RSD multipole estimators (anisotropic xi0, xi2). |
| `utils/read_galaxies.py` | High-level galaxy loaders returning numpy arrays or DataFrames. |
| `redshift_lists/` | Redshift lookup tables for specific simulations (e.g., `L800.txt`). |

## Core Concepts & Data Flow

### 1. Data Structure
Simulation data is stored at:
`/cosma5/data/durham/dc-hick2/Galform_Out/<SIM>/<MODEL>/iz<N>/ivol<K>/galaxies.hdf5`
- **Base Dir**: Overridable via `GALFORM_BASE_DIR` env var or `config.set_base_dir()`.
- **Aggregation**: Use `analysis.aggregation` to scan `ivol` directories and handle completion flags.

### 2. Subvolume-Weighted Correction (2PCF)
For correlation functions using $m$ selected subvolumes out of $k$ total:
$DD_{corr} = \alpha \cdot DD_{auto} + \beta \cdot DD_{cross}$
- **Standard (Scale-Down)**: $\alpha = m/k$, $\beta = m(k-1) / [k(m-1)]$. (Default in `src/analysis/correlation/`)
- **Legacy (Scale-Up)**: $\alpha = k/m$, $\beta = k(k-1) / [m(m-1)]$. (Used in early SCOPE versions)

## Active Research Work Streams

### 1. SCOPE ξ(r) Convergence
Demonstrating that subvolume-weighted 2PCF converges to the full-box reference.
- **SCOPE Tool**: Rust-backed pair-counter at `/cosma/apps/durham/dc-hick2/SCOPE/`.
- **Reference**: Corrfunc full-box runs (`seed1000`) are the ground truth for residual analysis.

### 2. NPCF Investigation
Generalizing subvolume corrections to N-point correlation functions.
- **Correction Formula**: $w_s = (\frac{m}{k})^N \cdot \frac{(k)_s}{(m)_s}$ where $(x)_s$ is the falling factorial.
- **Operational Regime**: $N=3$ is always feasible; $N=4,5$ requires $m \ge 64$ to control Poisson shot noise.

### 3. RSD Investigation
Investigating the Kaiser boost and multipole convergence in redshift space.
- **Estimator**: $\xi(s, \mu) = DD_{corr}(s, \mu) / RR_{analytic}(s, \mu) - 1$ (analytical randoms for periodic boxes).

### 4. DESI ELG Mocks
Building GALFORM light-cones with DECam photometry for DESI ELG comparison.
- **Photometry**: Requires specific GALFORM build with DECam filters (Band IDs 350, 351, 353).
- **Status**: Mock construction at $z \approx 1.0$ (iz155) underpredicts DESI $w_p$ by $\approx 2.2\times$.

### 5. Dynamical Friction (τ₀) Sensitivity
Quantifying how $\tau_0$ shapes galaxy assembly (SHMR, SMF, HOD).
- **FOF-Central Convention**: `is_central == 1 AND mhalo/mhhalo > 0.5`.
- **Headline Result**: $\tau_0 = \infty$ (no merging) fails to form massive centrals ($M_* > 10^{11.5} M_\odot/h$).

## Simulations & Data

| Simulation | Box [Mpc/h] | N Subvols | Notes |
|------------|-------------|-----------|-------|
| **L800** | 542.16 | 1024 | Main box (Planck 2013). |
| **Mill1/2** | 500.0 / 100.0 | 64 | Millennium I/II (WMAP1). |
| **COLIBRE** | 68.1 | 64 | Statistical realizations (Planck 2018). |
| **FLAMINGO**| 681.0 | 64 | Statistical realizations (Planck 2018). |

## HPC & COSMA Environment

### Partition & Mount Guide (Mounts /cosma5?)
| Partition | Mounts? | Target | Notes |
|:---|:---:|:---|:---|
| `cosma5` | **Yes** | `m5xxx` | Default. |
| `cosma7/8` (Std) | **No** | `m7xxx`/`m8xxx`| **Avoid (FileNotFoundError).** |
| `cosma7/8-shm` | **Yes** | `madxx`/`gaxxx` | High-memory spillover. |
| `cosma8-ska` | **Yes** | `mad07` | Valid spillover (use `--account=durham`). |

### Resource Guidance
- **Environment**: Python 3.12 in `.venv` (module load `python/3.9.19` on some nodes if needed).
- **Compute**: L800 `mstar_none` (all galaxies) is infeasible for $n \ge 64$ due to 72h walltime limit.

## Engineering Standards
- **Testing**: `pytest tests -q`. New features MUST have validation tests.
- **Linting**: `ruff check src` and `ruff format src`.
- **HDF5**: Always use `loaders.close_snapshot()` to prevent resource leaks.
- **Data Flow**: Science notebooks live in `science/` (gitignored) to separate research from library code.
