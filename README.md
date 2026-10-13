# galform_analysis

[![PyPI](https://img.shields.io/pypi/v/galform_analysis.svg)](https://pypi.org/project/galform_analysis/)
[![Python versions](https://img.shields.io/pypi/pyversions/galform_analysis.svg)](https://pypi.org/project/galform_analysis/)
[![CI](https://github.com/OscarHickman/galform_analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/OscarHickman/galform_analysis/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Python tools for reading the `galaxies.hdf5` files written by the
[GALFORM](https://ui.adsabs.harvard.edu/abs/2016MNRAS.462.3854L) galaxy
formation model and measuring things from them: mass and luminosity functions,
halo occupation, dust attenuation and clustering.

## Install

```bash
pip install galform_analysis
```

This is enough for reading data, mass functions, luminosity functions and
dust. Two extras add the rest:

| Extra | Adds | For |
|---|---|---|
| `clustering` | Corrfunc | correlation functions, w_p, bias, RSD multipoles |
| `science` | hmf, astropy, CAMB, colossus, SciPy | theoretical halo mass functions, linear matter ξ(r) |
| `all` | both | everything |

```bash
pip install "galform_analysis[all]"
```

Corrfunc is compiled during installation, so `clustering` needs a C compiler
with OpenMP and the GSL headers (`sudo apt-get install libgsl-dev`,
`brew install gsl`, or `module load gsl` on a cluster). If an extra is
missing, only the functions that need it fail, with an error saying which
extra to install.

Python 3.10 to 3.14 is supported.

## Before you start

GALFORM writes one directory per snapshot, with one or more subvolumes in
each:

```
<base_dir>/iz<NNN>/ivol<M>/galaxies.hdf5
```

Each `ivol` is an independent realisation of the whole simulation box, not a
piece of it.

Tell the package where your output is, once per session:

```python
import galform_analysis

galform_analysis.set_base_dir("/path/to/Galform_Out/L800/model")
```

or set the `GALFORM_BASE_DIR` environment variable. Functions that take a
snapshot number, such as `271` for `iz271`, look inside this directory.
Functions that take a path, such as `".../iz271"`, do not need it.

**Units.** Masses are in M☉/h and lengths in Mpc/h, as GALFORM writes
them. Number densities φ are per (Mpc/h)³ per dex (mass functions) or per
magnitude (luminosity functions). Absolute magnitudes are M − 5 log₁₀ h.

## How to

### Read one subvolume

```python
from galform_analysis import close_snapshot, read_snapshot_data

data = read_snapshot_data("/path/to/Galform_Out/L800/model/iz271", ivol=0)
mstar = data["mstar"]   # stellar mass, disk + bulge
mhalo = data["mhalo"]   # halo mass
print(data["z"], data["V_ivol"])
close_snapshot(data)    # close the HDF5 file
```

For a table with positions and other fields, use
`galform_analysis.utils.read_galaxies.read_galaxies_dataframe`.

### Measure a stellar mass function

```python
from galform_analysis import (
    avg_smf_given_redshift_and_subvolumes,
    smf_given_redshift_and_subvolume,
)

# One subvolume
smf = smf_given_redshift_and_subvolume("/path/to/Galform_Out/L800/model/iz271", ivol=0)
smf["centers"], smf["phi"]   # log10(M*) bin centres and phi

# Mean and scatter over subvolumes 0-7 of iz271
avg = avg_smf_given_redshift_and_subvolumes(271, ivols=list(range(8)))
avg["phi"], avg["phi_std"]
```

Halo mass functions work the same way (`hmf_given_redshift_and_subvolume`,
`avg_hmf_given_redshift_and_subvolumes`), as does the halo occupation
distribution (`hod_given_redshift_and_subvolume` in
`galform_analysis.analysis`). Pass `bins=` to change the bins.

### Measure a luminosity function

GALFORM stores a luminosity for every galaxy in every band listed in its run
parameters. To see which bands a run has:

```python
from galform_analysis import list_bands

list_bands("/path/to/Galform_Out/L800/model/iz271")
```

Pick a band by its `label`, which is the first word of the GALFORM band name,
such as `"SDSS-r"`, `"UKIRT-K"` or `"Bj"`. Labels are case-sensitive: `"R"` is
Cousins R and `"SDSS-r"` is SDSS r.

```python
from galform_analysis import (
    avg_lf_given_redshift_and_subvolumes,
    lf_given_redshift_and_subvolume,
)

# r band, one subvolume
lf = lf_given_redshift_and_subvolume(
    "/path/to/Galform_Out/L800/model/iz271", ivol=0, band="SDSS-r"
)
lf["centers"], lf["phi"]   # magnitude bin centres and phi

# K band in Vega magnitudes, averaged over subvolumes 0-7
avg = avg_lf_given_redshift_and_subvolumes(
    271, ivols=list(range(8)), band="UKIRT-K", system="Vega"
)
```

Other options: `frame="observer"` for observer-frame magnitudes, and
`component="disk"` or `"bulge"` for the light of one component only.

### Add dust

By default luminosities are intrinsic, with no dust. Pass `dust="lacey16"` to
apply GALFORM's dust model with the parameters of Lacey et al. (2016).

GALFORM is not public, so this package does not contain its dust model.
Instead it runs the dust code in your own GALFORM checkout. Point
`GALFORM_DIR` at a checkout that you have built with CMake:

```bash
export GALFORM_DIR=/path/to/galform    # build expected in /path/to/galform/build
```

If the build is somewhere else, also set `GALFORM_BUILD_DIR`. The first time
you use dust, a small program is compiled against your GALFORM build, using
the same compiler, and kept in `~/.cache/galform_analysis` for later.

```python
from galform_analysis import (
    avg_lf_given_redshift_and_subvolumes,
    read_band_magnitudes,
)

# bJ luminosity function with dust
avg = avg_lf_given_redshift_and_subvolumes(
    271, ivols=list(range(8)), band="Bj", system="Vega", dust="lacey16"
)

# Magnitudes of every galaxy in one subvolume, with and without dust
iz = "/path/to/Galform_Out/L800/model/iz271"
intrinsic = read_band_magnitudes(iz, 0, "SDSS-r")["mag"]
attenuated = read_band_magnitudes(iz, 0, "SDSS-r", dust="lacey16")["mag"]
```

Each galaxy gets a random orientation, the same in every band, so colours
are consistent. Use `inclination="face-on"`, `"average"` or an angle in
degrees to change this, and `seed=` to change the random draw.

Not included: the extra dimming of very young stars (under 1 Myr old) inside
their birth clouds. GALFORM needs extra tables for this that are not in
`galaxies.hdf5`. It mostly affects the far-UV.

### Measure a correlation function (needs `clustering`)

```python
import numpy as np
from galform_analysis import compute_xi_corrfunc

rng = np.random.default_rng(1)
positions = rng.uniform(0, 100.0, size=(20_000, 3))   # (N, 3) array in Mpc/h
rbins = np.logspace(0, 1.5, 11)                       # 1 to 32 Mpc/h
xi = compute_xi_corrfunc(positions, boxsize=100.0, rbins=rbins, nthreads=4)
xi["r"], xi["xi"]   # a polars DataFrame; xi is close to zero for random points
```

To go straight from a subvolume, use `correlation_given_redshift_and_subvolume`.
On a shared machine, set `nthreads` (and `OMP_NUM_THREADS`) to the number of
cores you have been given.

### Look up a simulation

```python
from galform_analysis import SimulationConfig, find_snapshot_at_redshift

sim = SimulationConfig("L800")
print(sim.box_size, sim.omega_m, sim.h0, sim.n_subvolumes)

find_snapshot_at_redshift(1.0, "L800")   # the snapshot nearest z = 1
```

Simulations included: L800, Millennium I and II, EAGLE, COLIBRE, FLAMINGO,
Dove and nIFTy. Their settings come from the
[galform_execution](https://pypi.org/project/galform_execution/) package, so
analysis and runs use the same values.

## Example notebooks

The [`examples/`](https://github.com/OscarHickman/galform_analysis/tree/main/examples)
folder has one notebook per topic, each run on the L800 simulation:

| Topic | Notebook |
|---|---|
| Reading a snapshot | [load_snapshot](https://github.com/OscarHickman/galform_analysis/blob/main/examples/readers/load_snapshot.ipynb) |
| Galaxy tables and positions | [read_galaxies](https://github.com/OscarHickman/galform_analysis/blob/main/examples/utils/read_galaxies.ipynb) |
| Stacking all subvolumes | [aggregation](https://github.com/OscarHickman/galform_analysis/blob/main/examples/analysis/aggregation.ipynb) |
| Stellar mass function | [smf](https://github.com/OscarHickman/galform_analysis/blob/main/examples/analysis/mass_functions/smf.ipynb) |
| Halo mass function | [hmf](https://github.com/OscarHickman/galform_analysis/blob/main/examples/analysis/mass_functions/hmf.ipynb) |
| Halo occupation | [hod](https://github.com/OscarHickman/galform_analysis/blob/main/examples/analysis/mass_functions/hod.ipynb) |
| Luminosity functions | [lf](https://github.com/OscarHickman/galform_analysis/blob/main/examples/analysis/luminosity_functions/lf.ipynb) |
| Dust attenuation | [dust_attenuation](https://github.com/OscarHickman/galform_analysis/blob/main/examples/dust/dust_attenuation.ipynb) |
| Correlation function | [correlation](https://github.com/OscarHickman/galform_analysis/blob/main/examples/analysis/correlation/correlation.ipynb) |
| Galaxy bias | [bias](https://github.com/OscarHickman/galform_analysis/blob/main/examples/analysis/correlation/bias.ipynb) |
| Satellite–central cross-correlation | [cross_correlation](https://github.com/OscarHickman/galform_analysis/blob/main/examples/analysis/correlation/cross_correlation.ipynb) |
| Using only some subvolumes | [subvol_weighted](https://github.com/OscarHickman/galform_analysis/blob/main/examples/analysis/correlation/subvol_weighted.ipynb) |
| Redshift-space multipoles | [multipoles](https://github.com/OscarHickman/galform_analysis/blob/main/examples/analysis/redshift_space_distortions/multipoles.ipynb) |

The figures use the `k.wang` style from
[oh-mpl-paper](https://pypi.org/project/oh-mpl-paper/), which sizes them for
an MNRAS column.

## Development

```bash
git clone https://github.com/OscarHickman/galform_analysis.git
cd galform_analysis
pip install -e ".[all,dev]"

pytest tests --cov                   # tests and coverage (CI needs 80% or more)
ruff check galform_analysis tests    # lint
ruff format galform_analysis tests   # format
```

The tests make small fake `galaxies.hdf5` files with the same layout as real
ones, so no simulation data is needed. Tests for an extra are skipped if it is
not installed. The dust module is not covered by the tests, because it needs a
private GALFORM checkout.

## Citation

If you use galform_analysis in published work, please cite it
([`CITATION.cff`](https://github.com/OscarHickman/galform_analysis/blob/main/CITATION.cff)):

```bibtex
@software{galform_analysis,
  author  = {Hickman, Oscar},
  title   = {galform_analysis: Python tools for analysing GALFORM galaxy formation model outputs},
  version = {0.4.1},
  url     = {https://github.com/OscarHickman/galform_analysis}
}
```

## License

MIT. See [LICENSE](https://github.com/OscarHickman/galform_analysis/blob/main/LICENSE).
