# galform_analysis

[![PyPI](https://img.shields.io/pypi/v/galform_analysis.svg)](https://pypi.org/project/galform_analysis/)
[![Python versions](https://img.shields.io/pypi/pyversions/galform_analysis.svg)](https://pypi.org/project/galform_analysis/)
[![CI](https://github.com/OscarHickman/galform_analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/OscarHickman/galform_analysis/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular Python framework for reading and analysing outputs of the
[GALFORM](https://ui.adsabs.harvard.edu/abs/2016MNRAS.462.3854L) semi-analytic
model of galaxy formation, stored as `galaxies.hdf5` files. It covers everything
from low-level HDF5 I/O to mass functions, clustering statistics and
redshift-space distortions.

## Features

- **I/O**: robust readers for GALFORM `galaxies.hdf5` files, tolerant of
  different output versions and incomplete subvolumes.
- **Aggregation**: scan a snapshot directory and stack galaxies across
  subvolumes into [polars](https://pola.rs) DataFrames.
- **Mass functions**: stellar mass functions (SMF), halo mass functions (HMF)
  and halo occupation distributions (HOD), per subvolume or averaged over
  subvolumes and snapshots.
- **Theoretical predictions** *(optional)*: halo mass functions from
  [hmf](https://github.com/halofit/hmf) with GALFORM's Mvir mass definition,
  the GPS+ model, and the linear matter correlation function from CAMB.
- **Clustering** *(optional)*: real-space 2-point correlation functions,
  satellite–central cross-correlations and galaxy bias, built on
  [Corrfunc](https://github.com/manodeep/Corrfunc).
- **Subvolume-weighted corrections** *(optional)*: ξ(r) and projected w_p(r_p)
  from an auto/cross pair-count decomposition that removes the sub-sampling
  bias when only some subvolumes are used.
- **Redshift-space distortions** *(optional)*: ξ(s, μ) and the multipoles
  ξ₀, ξ₂, ξ₄.
- **Simulation metadata**: box sizes, cosmologies and snapshot redshifts for
  L800, Millennium I/II, EAGLE, COLIBRE, FLAMINGO, Dove and nIFTy.

## Installation

```bash
pip install galform_analysis
```

The base install covers I/O, aggregation, simulation metadata and the
SMF/HMF/HOD. Optional features are available as extras:

| Extra | Installs | Needed for |
|---|---|---|
| `clustering` | Corrfunc | correlation functions, w_p, bias, RSD multipoles |
| `science` | hmf, CAMB, colossus, SciPy | theoretical HMFs, linear matter ξ(r) |
| `all` | both of the above | everything |

```bash
pip install "galform_analysis[all]"
```

**Note on `clustering`:** Corrfunc is distributed only as source and compiles
during installation. It needs a C compiler with OpenMP and the GSL headers,
for example `sudo apt-get install libgsl-dev` (Debian/Ubuntu),
`brew install gsl` (macOS) or `module load gsl` on an HPC cluster. If an
optional dependency is missing, the functions that need it raise an
`ImportError` naming the extra to install. The rest of the package still works.

galform_analysis supports Python 3.10–3.13 and is tested on Linux.

## Quick start

GALFORM writes one directory per snapshot, each holding one or more
subvolumes:

```
<base_dir>/iz<NNN>/ivol<M>/galaxies.hdf5
```

Each `ivol` is an independent realisation of the full simulation box, not a
spatial tile of it.

### Simulation metadata

```python
from galform_analysis import (
    SimulationConfig,
    find_snapshot_at_redshift,
    get_snapshot_redshift,
)

sim = SimulationConfig("L800")
print(sim.box_size, sim.omega_m, sim.h0, sim.n_subvolumes)

# Snapshot closest to z = 1 for this simulation, e.g. "iz155"
snapshot = find_snapshot_at_redshift(1.0, "L800")
z = get_snapshot_redshift(snapshot, "L800")
```

### Reading a subvolume

```python
from galform_analysis import close_snapshot, read_snapshot_data

data = read_snapshot_data("/path/to/Galform_Out/L800/model/iz271", ivol=0)
mstar = data["mstar"]  # stellar mass (disk + bulge), M_sun/h
mhalo = data["mhalo"]  # halo mass, M_sun/h
print(data["z"], data["V_ivol"])
close_snapshot(data)  # releases the open HDF5 file handle
```

### Stellar and halo mass functions

```python
from galform_analysis import set_base_dir, smf_given_redshift_and_subvolume
from galform_analysis.analysis import avg_smf_given_redshift_and_subvolumes

set_base_dir("/path/to/Galform_Out/L800/model")

smf = smf_given_redshift_and_subvolume("/path/to/Galform_Out/L800/model/iz271", ivol=0)
smf["centers"], smf["phi"]  # log10(M*) bin centres, phi in (Mpc/h)^-3 dex^-1

# Mean and scatter over subvolumes 0-7 of snapshot iz271, under the base dir
avg = avg_smf_given_redshift_and_subvolumes(271, ivols=list(range(8)))
avg["phi"], avg["phi_std"]
```

### Correlation functions (requires `clustering`)

```python
import numpy as np
from galform_analysis import compute_xi_corrfunc

rng = np.random.default_rng(1)
positions = rng.uniform(0, 100.0, size=(20_000, 3))  # (N, 3) array, Mpc/h
rbins = np.logspace(0, 1.5, 11)  # 1-32 Mpc/h
xi = compute_xi_corrfunc(positions, boxsize=100.0, rbins=rbins, nthreads=4)
xi["r"], xi["xi"]  # consistent with zero for a uniform random field
```

Functions that read data take `nthreads` arguments for Corrfunc. On shared
compute nodes, set these (and `OMP_NUM_THREADS`) to your allocation.

## Configuration

- **Data location**: call `galform_analysis.set_base_dir(path)` or set the
  `GALFORM_BASE_DIR` environment variable. Functions that take a snapshot
  number (such as `271` for `iz271`) rather than a path resolve it against
  this directory.
- **Simulation configs**: these come from the
  [galform_execution](https://pypi.org/project/galform_execution/) package, so
  analysis and execution use identical parameters. A copy bundled with this
  package is used as a fallback.
- **Snapshot redshifts**: `load_redshift_mapping(sim_name)` maps `iz` indices
  to redshifts for simulations with a bundled redshift list.

## Examples

Jupyter notebooks covering each module are in the
[`examples/`](https://github.com/OscarHickman/galform_analysis/tree/main/examples)
directory of the repository, including
[reading snapshots](https://github.com/OscarHickman/galform_analysis/blob/main/examples/readers/load_snapshot.ipynb),
[stellar mass functions](https://github.com/OscarHickman/galform_analysis/blob/main/examples/analysis/mass_functions/smf.ipynb),
[correlation functions](https://github.com/OscarHickman/galform_analysis/blob/main/examples/analysis/correlation/correlation.ipynb)
and [RSD multipoles](https://github.com/OscarHickman/galform_analysis/blob/main/examples/analysis/redshift_space_distortions/multipoles.ipynb).

## Development

```bash
git clone https://github.com/OscarHickman/galform_analysis.git
cd galform_analysis
pip install -e ".[all,dev]"

pytest tests --cov       # test suite with coverage (CI requires >= 80%)
ruff check galform_analysis tests
ruff format galform_analysis tests
```

The tests build synthetic `galaxies.hdf5` files that follow the real GALFORM
schema, so no simulation data is needed. Tests that need an optional
dependency are skipped when it is not installed.

## Citation

If you use galform_analysis in your research, please cite it using the
metadata in [`CITATION.cff`](https://github.com/OscarHickman/galform_analysis/blob/main/CITATION.cff):

```bibtex
@software{galform_analysis,
  author  = {Hickman, Oscar},
  title   = {galform_analysis: A modular Python framework for analysing GALFORM simulation outputs},
  version = {0.2.0},
  url     = {https://github.com/OscarHickman/galform_analysis}
}
```

## License

MIT. See [LICENSE](https://github.com/OscarHickman/galform_analysis/blob/main/LICENSE).
