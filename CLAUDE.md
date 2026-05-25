# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Tests
pytest tests -q

# Lint
ruff check src
```

SLURM jobs run on COSMA and activate `.venv` (Python 3.12) at repo root. All scripts in `scripts/` add `src/` to `sys.path` manually. Notebooks add `src/` via `Path('../../src').resolve()` at the top of their first import cell.

## Repository layout

| Directory | Git | Purpose |
|-----------|-----|---------|
| `src/` | tracked | Reusable GALFORM analysis library |
| `examples/` | tracked | Showcase notebooks for `src/` |
| `tests/` | tracked | Unit tests for `src/` |
| `science/` | **gitignored** | Personal research notebooks, drafts, paper prose |
| `scripts/` | gitignored | HPC submission scripts |
| `data/` | gitignored | Output CSVs from HPC runs |
| `logs/` | gitignored | SLURM log files |

`science/` is gitignored — it is the active workspace for the research work streams below. It is not intended as a public library; do not add `science/` files to git.

---

## Work stream 1: Reusable analysis library

**Goal:** A clean, tested Python library for reading and analysing GALFORM HDF5 outputs. Showcased in `examples/`, fully unit-tested in `tests/`. This is the public-facing part of the repo.

All modules in `src/` use flat absolute imports (no package prefix). To use them, add `src/` to `sys.path`:

```python
import sys
sys.path.insert(0, '/cosma/apps/durham/dc-hick2/galform_analysis/src')

from config import get_base_dir
from readers.loaders import read_snapshot_data
from utils.read_galaxies import read_galaxy_arrays
from analysis.mass_functions import hmf_given_redshift_and_subvolume
```

### Data flow

GALFORM writes output at:
```
/cosma5/data/durham/dc-hick2/Galform_Out/<SIM>/<MODEL>/iz<N>/ivol<K>/galaxies.hdf5
```

`config.BASE_DIR` (overridable via `GALFORM_BASE_DIR` env var or `set_base_dir()`) points here. Scripts accept `--base-dir` / `--sim-name` / `--model` / `--iz` and construct paths themselves.

Snapshot indices (`iz`) map to redshifts via `src/redshift_lists/<sim_name>.txt` — use `config.get_snapshot_redshift('iz155', 'L800')` or `config.find_snapshot_at_redshift(0.5, 'L800')`. To add a new simulation, drop a `<sim_name>.txt` file into `src/redshift_lists/`.

### Library structure (`src/`)

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

**Goal:** Demonstrate that the SCOPE subvolume-weighted 2PCF correction converges to the true full-box ξ(r) as a function of number of subvolumes used, and write a paper on this result.

Science notebooks and paper drafts live in `science/SCOPE/scope_xi/` (gitignored). Output CSVs in `data/2pcf/scope_xi/` (gitignored).

### SCOPE tool

SCOPE is a separate Rust/Python pair-counter at `/cosma/apps/durham/dc-hick2/SCOPE/`.

- **Rust source:** `/cosma/apps/durham/dc-hick2/SCOPE/src/lib.rs` — `count_pairs_1d` and `count_pairs_2d`
- **Python API:** `/cosma/apps/durham/dc-hick2/SCOPE/python/scope/__init__.py`
- **Built .so:** compiled for both Python 3.9 and 3.12 (in `python/scope/`)
- **Add to path:** `export PYTHONPATH="/cosma/apps/durham/dc-hick2/SCOPE/python:$PYTHONPATH"`

Sub-volume IDs label **independent statistical realisations** of the full simulation box — **not spatial cells**. Every realisation spans the full coordinate range [0, boxsize) in all three dimensions; all realisations overlap. Stacking all k realisations recovers the full-density catalogue.

Key functions:

| Function | Purpose |
|----------|---------|
| `compute_xi(coords, subvol_ids, r_bins, box_size, n_subvols, n_subvols_selected)` | **Primary.** Real-space ξ(r); returns dict with `xi`, `dd_auto`, `dd_cross`, `dd_corr`, `rr`, `r_mid` |
| `compute_2pcf(coords, subvol_ids, r_p_bins, pi_bins, box_size, n_subvols, n_subvols_selected, n_total)` | **Legacy.** Projected w_p(r_p) via ξ(r_p,π) |

**α/β conventions — two different normalisation targets:**

- `compute_xi` (primary, Rust backend: `count_pairs_1d`): scales DOWN to selected-catalogue density.  
  `α = m/k`, `β = m(k−1)/[k(m−1)]`, RR uses `N_selected`. Matches `subvol_weighted_correction.py`.

- `compute_2pcf` (legacy, Rust backend: `count_pairs_2d`): scales UP to full-box counts.  
  `α = k/m`, `β = k(k−1)/[m(m−1)]`, `n_total` must be the full-box galaxy count.

**Cell-count bug (fixed 2026-05-06):** `build_cell_list` in `lib.rs` uses `.clamp(3, 128)` (was `.max(1)`). With fewer than 3 cells in a periodic dimension, `rem_euclid` maps two distinct offsets to the same neighbour cell, double-counting pairs. SCOPE rebuilt. Any large-rp CSVs produced before this date are corrupt and were deleted.

### mstar_none: bad physics and infeasible compute

`mstar_none` includes all galaxies down to GALFORM's resolution limit (~10^7 M☉/h). **Bad physics:** galaxies near the resolution limit are numerically poorly resolved — merger trees are incomplete and the physics is unreliable. **Infeasible compute:** with 143M galaxies in L800 full-box, large-n runs hit or exceed the 72h COSMA wall-time limit.

Do not submit `mstar_none` for n≥64 on any simulation. Do not submit corrfunc fullbox `mstar_none` for L800 or Mill1. Use `mstar9.0` as the lowest meaningful stellar mass cut.

Infeasibility details:
- L800 n=128 would need ~192h (cosma8 limit: 8h); n=256 ~768h — always timeout
- L800 cfxi mstar_none: timed out at 48h cosma8-shm limit — do not resubmit
- Mill1 SCOPE n=64 mstar_none: timed out at 72h cosma8-shm max — infeasible

For mstar_none plots use `PLOT_N_MSTAR_NONE = [2, 8, 32]` (excludes n≥128).

### SCOPE ξ(r) convergence campaign (primary)

`scripts/2pcf/compute_scope_xi_convergence.py` — one invocation per (seed, n_subvol); loads exactly n realisations, runs `compute_xi` in corrected and naive mode. Output: `scope_xi_<SIM>_iz<IZ>.csv`.

Data: `data/2pcf/scope_xi/<model>/iz<IZ>/<mstar_tag>/n<N>/seed<S>/scope_xi_<SIM>_iz<IZ>.csv`

`mstar_tag`: `mstar_none`, `mstar9.0`, `mstar10.0`, `mstar11.0`. No halo mass cuts, no centrals filtering.

Submit: `submit_scope_xi_campaign.sh` — one SLURM job per (iz × mstar × n_subvol × seed).  
Seeds per n: 10 (n=1), 7 (n≤8), 5 (n≤64), 3 (n≤256), 2 (n≤512), 1 (n=1024).  
Partitions: cosma5 (n≤32), cosma8 (n≤256), cosma8-shm (n>256).  
r bins: 0.01 → 271 Mpc/h, 30 log-spaced. iz: 155, 207, 271. k_total=1024.

**FLAMINGO n=32/64 mstar_none timing:** n=32 takes 16–60h (node speed variation); always submit with ≥60h on cosma8-shm. n=64 mstar_none: submit with 70h. cosma8-shm max walltime is 72h.

### Corrfunc full-box reference (n=1024)

The reference for all ξ(r) convergence comparisons is a **Corrfunc** full-box run, not SCOPE. It loads all 1024 ivols, applies the same stellar mass cut, and runs Corrfunc for a periodic box.

- **Script:** `scripts/2pcf/compute_corrfunc_xi_fullbox.py`
- **Submit:** `scripts/2pcf/slurm/submit_corrfunc_xi_fullbox.sh`
- **One job per (iz, mstar)** — deterministic, only `seed1000` needed (no seed variation)
- **Output:** `data/2pcf/scope_xi/<model>/iz<IZ>/<mstar_tag>/n1024/seed1000/scope_xi_<SIM>_iz<IZ>.csv`
- Residual columns in the notebook are NaN until the reference for that (iz, mstar) exists — do not use n=512 SCOPE as a proxy fallback

### SCOPE w_p convergence campaign (legacy)

`scripts/2pcf/compute_scope_wp_convergence.py` — one invocation per seed; sweeps all `n_subvol` values. Output: `scope_wp_<SIM>_iz<IZ>.csv`.

Data:
- `data/2pcf/scope_wp/` — small-scale (r_p: 0.1 → 31.6 Mpc/h, 20 log bins)
- `data/2pcf/scope_wp_large_rp/` — large-scale (r_p: 31.6 → 200 Mpc/h, 7 log bins)

Both: `<model>/iz<IZ>/centrals_<0|1>/mstar<val>/<mhalo_val>/seed<S>/n<N>/scope_wp_<SIM>_iz<IZ>.csv`

Submit scripts:

| Script | Purpose |
|--------|---------|
| `submit_scope_wp_split_by_n.sh` | Small-scale campaign; cycles seeds across cosma5/cosma8-shm/cosma8 |
| `submit_scope_wp_large_rp_split_by_n.sh` | Large-scale campaign; all jobs on cosma8-shm; runs iz=155,207,271 |

Default configs: `mhalo_min=1e9`, 5 seeds (base 1000), `pi_max=40`, `n_pi_bins=40`, `k_total=1024`.

---

## Work stream 3: NPCF investigation

**Goal:** Generalise the SCOPE 2PCF subvolume correction to any N-point correlation function. Validate the formula for N=3,4,5, then add callable Python/Rust functions to SCOPE as a new feature.

Science notebooks and drafts in `science/SCOPE/` (gitignored). Implementation in `src/analysis/correlation/`.

### Correction formula

Classify every N-tuple by s = number of distinct parent realisations (s ∈ {1,…,N}). The weight for the scale-down convention is:

$$w_s = \left(\frac{m}{k}\right)^N \cdot \frac{(k)_s}{(m)_s}$$

where (x)_s = x(x−1)…(x−s+1) is the falling factorial. Reduces to α/β at N=2 and to the 3PCF v2 weights at N=3 (verified to machine precision).

### Implementation

| File | Role |
|------|------|
| `src/analysis/correlation/n_point_bruteforce.py` | `compute_npoint_counts` (N-tuple counter) and `scope_weights_npcf(N,m,k)` |
| `src/analysis/correlation/three_point_bruteforce.py` | Dedicated 3PCF counter (SSS/SSD/DDD) |
| `src/analysis/correlation/three_point_reference.py` | Full-box 3PCF reference |
| `src/analysis/correlation/three_point_scope.py` | SCOPE-corrected 3PCF |
| `scripts/2pcf/run_npcf_fullbox.py` | Driver: `--N --iz --m --mstar-min-log10 --output-dir` |
| `scripts/2pcf/slurm/submit_npcf_fullbox.sh` | Submits N=4,5 × 3 iz × 4 m = 24 jobs |

Science notebooks: `science/SCOPE/npcf_scope_validation.ipynb`, `science/SCOPE/3pcf_scope_validation.ipynb`.

### Validation results

| N | iz | m | CV_Poisson | T̂_corr/T̂_ref | Pass? |
|---|---|---|-----------|-------------|-------|
| 3 | all | ≥16 | <10% | 0.85–1.07 | ✓ |
| 4 | 155 | 64 | 43% | 0.90–1.52 | ✓ (noise expected) |
| 4 | 155 | 256 | 5% | 0.89–1.23 | ✓ |
| 4 | 207 | 64 | 6–10% | 0.86–1.66 | ✓ |
| 4 | 207 | 256 | <1% | 0.85–1.28 | ✓ |
| 4 | 271 | 64 | 3–5% | 0.29–1.69 | ✗ sampling var |
| 4 | 271 | 256 | <1% | 0.72–1.18 | marginal |

Naive bias at m=64, N=4: **86×–3720×** off reference. Correction reduces scatter to <2×.

### Two failure modes

**1. Poisson shot noise**  
The s=N term (all N galaxies from distinct realisations) has expected count E[T_{s=N}(m)] = T_{s=N,ref} × (m)_N/(k)_N that falls rapidly with N and m.

Rule of thumb (CV < 20%):
- N=3: m ≥ 1 (always feasible)
- N=4: m ≥ ~20 at iz=155, m ≥ ~5 at iz=271
- N=5: m ≥ ~100 at iz=155 for r < 5 Mpc/h; m ≥ ~20 at larger r

**2. Sampling variance from HOD scatter**  
Even with low Poisson noise, massive clusters dominate N-point functions and their occupancy varies across realisations. At iz=271, m=64: the first 64 realisations have 54% more 1-halo 4-tuples than average, driving T_corr to 29% of the true value at r≈4.6 Mpc/h. Not a formula bug — the correction is unbiased in expectation; multiple seeds are required.

**Operating regime:**
- N=3: always feasible
- N=4: feasible for m ≥ 64 at iz=155/207, m ≥ 256 at iz=271
- N=5: feasible for m ≥ 256 at iz=155; requires multiple seeds at iz=271

For N=4,5 at z=0 (iz=271), run 3–5 seeds and report the mean corrected ratio.

---

## Work stream 4: RSD investigation

**Goal:** Investigate how the SCOPE subvolume correction works in redshift space — the Kaiser boost, galaxy bias b(r), and convergence of the multipoles ξ0(s) and ξ2(s). Build a draft paper on the SCOPE RSD estimator.

Science notebooks and drafts in `science/SCOPE/` (gitignored), primarily `scope_rsd.ipynb` and `rsd_investigation_galform.ipynb`.

The redshift-space estimator applies the α/β correction to anisotropic pair counts in (s, μ) space:
```
xi(s, mu) = DD_corr(s, mu) / RR_analytic(s, mu) - 1
```
then integrates to multipoles. `RR_analytic` is computed analytically for a periodic box — no random catalogue needed.

Key code:
- `src/analysis/redshift_space_distortions/subvol_weighted_multipoles.py` — multipole estimator
- `scripts/redshift_space_distortions/slurm/` — SLURM submission scripts

---

## Work stream 5: DESI ELGs mock

**Goal:** Build a GALFORM light-cone mock with DECam (DES) photometry and compare to real DESI DR1 ELG clustering. Primary comparison: projected correlation function w_p(r_p). Properly assess GALFORM ELG predictions vs observation.

Science notebooks in `science/SCOPE/scope_xi/section4/` (gitignored).

### GALFORM DECam build

A separate GALFORM build computes DECam photometry alongside the standard output.

- **Binary (gfortran):** `/cosma/apps/durham/dc-hick2/galform/build_desi_filters_gcc/galform2` ✓
- **DO NOT USE** the Intel binary at `build_desi_filters/galform2` — Intel ifx `-O3 -ip` causes `FATAL: StarBurst() - tran parameter is zero!` for ~48% of subvolumes due to numerical instability in `t_merge`. Rebuilt 2026-05-25 with GCC 10.2.0 gfortran.

DECam band IDs (in `Data/filters/filters_unique.dat`):

| Band | ID | iselect=0 (rest) | iselect=1 (observer) |
|------|----|-----------------|----------------------|
| DES-g | 350 | yes | yes |
| DES-r | 351 | yes | yes |
| DES-z | 353 | yes | yes |

Bands added at runtime by `GalformSubmitter._generate_bands_block()` — the binary is generic. The standard lc16 outputs at `/cosma5/data/durham/dc-hick2/Galform_Out/L800/lc16/` do **not** contain DECam bands.

**Submission scripts** (in `/cosma/apps/durham/dc-hick2/galform/`):
- `submit_desi_lc16.py` — full campaign (all 1024 ivols × 3 snapshots)
- `submit_desi_lc16_rerun.py` — targeted rerun for missing ivols only

Model: `lc16`, simulation: L800, snapshots: iz=271/207/155. Output: `/cosma5/data/durham/dc-hick2/DESI_LC16_GALFORM/L800/lc16/`

**Status as of 2026-05-25:**
- iz=271: complete (1024/1024 subvolumes)
- iz=207: rerunning ivols 459–1023 (job 11205913, gfortran binary)
- iz=155: rerunning all 1024 ivols (jobs 11205914–11205915, gfortran binary)

### Mock construction and results

**Snapshot:** iz155, Output003 = z=1.007 (NOT Output004 = z=1.496 which `get_output_group()` returns — must read Output003 directly via h5py).

**SFR abundance matching (current):** Target density n = 5×10⁻⁴ h³/Mpc³ (DESI ELG, z=0.8–1.1). Field: `mstardot + mstardot_burst`. OII luminosity in lc16 underpredicts DESI by ~20×, so photometric cuts are impractical without the DECam run.

**Key result:** GALFORM underpredicts DESI ELG w_p by ~2.2× (median ratio 0.45) across 0.3–10 Mpc/h.

**Photometric pipeline (ready, waiting for DECam data):** `scripts/desi/compute_desi_elg_mock_wp_phot.py`. DESI ELG cuts: g < 24.1, −0.2 ≤ g−r ≤ 1.6, 0.3(g−r)+0.9 ≤ g−z ≤ 1.6. Uses iz=155 Output009 (z=0.9505). Run `bash scripts/desi/slurm/submit_desi_elg_mock_wp_phot.sh` when DESI LC GALFORM runs complete.

---

## Work stream 6: Dynamical friction paper

**Goal:** Quantify how the dynamical-friction timescale parameter τ₀ shapes galaxy properties in GALFORM, particularly for massive central galaxies and cluster-scale satellites. Write a paper on τ₀ sensitivity.

Science notebooks in `science/dynamical_friction/` (gitignored). Shared helper: `science/dynamical_friction/tau0_helpers.py`.

### Runs

All under `/cosma5/data/durham/dc-hick2/Tau0_Investigation/`:

| Label | Path | Physics |
|-------|------|---------|
| `Default` | `Galform_Out_Default/L800/lc16.newmg` | Calibrated τ₀ |
| `tau0=0` | `Galform_Out_0tau0/L800/lc16.newmg` | Instant satellite merging |
| `tau0=inf` | `Galform_Out_1e6tau0/L800/lc16.newmg` | No satellite merging |

Only **16 ivols** (out of L800's 1024) per run, at iz271 (z=0) and iz207 (z=0.5). High-mass bins (log Mh ≳ 14.5) are noisy with this sample size.

### Notebooks (`science/dynamical_friction/`)

| File | Probe |
|------|-------|
| `central_shmr.ipynb` | Central SHMR with IQR + bootstrap band; per-halo BCG mass distributions; σ(log M*\|Mh) |
| `stellar_mass_function.ipynb` | Total/cen/sat SMF; ratios vs Default; satellite fraction; cumulative N(>M*) |
| `hod.ipynb` | ⟨N_tot/cen/sat⟩(Mh); ratios; power-law fit α at log Mh ≥ 12.5; M* cut sweep |
| `satellite_cross_correlation.ipynb` | Satellite–central cross-correlation across the three runs |
| `colibre_dynamical_friction.ipynb` | Same analysis on COLIBRE-L100m6 runs |
| `flamingo_dynamical_friction.ipynb` | Same analysis on FLAMINGO-L1000N1800 runs |

**FOF-central / BCG convention:** `is_central == 1 AND mhalo/mhhalo > 0.5`. HOD numerator is galaxies binned by host `mhhalo`; denominator is `Trees/mphalo` (one entry per FOF group).

`tau0_helpers.py` defines `RUNS`, `RUN_LABELS`, `RUN_COLORS` (Default=k, τ₀=0=C3 red, τ₀=∞=C0 blue), `SNAPSHOTS`, `DEFAULT_IVOLS=range(16)`, `BOX_SIZE_MPC_H=542.16`, `FOF_CENTRAL_RATIO_THRESHOLD=0.5`. Provides per-ivol loader `load_galaxy_fields`, summarisers `central_shmr_per_ivol` / `smf_split_per_ivol` / `hod_per_ivol`, aggregators `collect_per_ivol` and `stack_per_ivol` (returns mean / SEM / bootstrap 16-84 ranges with NaN-slice warnings suppressed).

### Headline physics (z=0, 16 ivols)

- BCG mass at log Mh = 13: τ₀=0 → +0.08 dex, τ₀=∞ → **−0.84 dex** (BCG starved)
- Φ_cen at log M* = 11.5: τ₀=0 → 1.46×, τ₀=∞ → **0.00×** (no M* > 10^11.5 centrals form)
- ⟨N_sat⟩ at log Mh = 12: τ₀=0 → 0.83×, τ₀=∞ → **2.85×**

Dynamical friction is essential for assembling massive central galaxies. Default sits much closer to τ₀=0 than to τ₀=∞.

---

## Simulations

| Name  | Box (Mpc/h) | N subvolumes | Notes |
|-------|-------------|--------------|-------|
| L800  | 542.16      | 1024         | Main high-res box; Planck 2013 |
| Mill1 | 500.0       | 64           | Millennium I; WMAP1 cosmology |
| Mill2 | 100.0       | 64           | Millennium II; WMAP1 cosmology |
| COLIBRE-L100m6 | 68.1 | 64        | Planck 2018; statistical realisations; on cosma8 |
| FLAMINGO-L1000N1800 | 681.0 | 64  | Planck 2018; statistical realisations; on cosma8 |

Masses in GALFORM HDF5 files are in M_sun/h. `config.DEFAULT_STELLAR_MASS_BINS` and `config.DEFAULT_HALO_MASS_BINS` are in log10(M_sun/h).

---

## SLURM / COSMA

Scripts in `scripts/2pcf/slurm/` and `scripts/redshift_space_distortions/slurm/` submit jobs via inline heredoc `sbatch`. Key env vars: `SIM_NAME`, `MODEL_NAME`, `IZ`, `OUTPUT_DIR`, `NMAX`, `MHALO_MIN`, `BOXSIZE`. SLURM resource overrides: `PARTITION`, `TIME_LIMIT`, `CPUS_PER_TASK`, `MEMORY`, `ACCOUNT`.

Output CSVs land in `data/`; logs in `logs/`; plots in `_plots/`.

### COSMA Partition & Mount Guide (Mounts /cosma5?)

Most simulation data lives on `/cosma5`. Only specific partitions can see this file system.

| Partition Group | Mounts `/cosma5`? | Target Nodes | Notes |
|:---|:---:|:---|:---|
| `cosma5` | **Yes** | `m5xxx` | Default for this repo. |
| `cosma7` (standard) | **No** | `m7xxx` | **Avoid.** |
| `cosma8` (standard) | **No** | `m8xxx` | **Avoid.** |
| `cosma7-shm` / `shm2` | **Yes** | `mad01-03` | Verified working (2026-05-17). |
| `cosma8-shm` / `shm2` / `shm3` | **Yes** | `mad04-05`, `ga004-006` | Verified working (2026-05-17). |
| `cosma8-ska` | **Yes** | `mad07` | Requires `--account=durham`. |

**Important:** Jobs submitted to standard `cosma7` or `cosma8` partitions will fail with `FileNotFoundError`. Always use one of the specialized `shm`/`ska` partitions if spillover from `cosma5` is needed.

Regular users cannot boost SLURM job priority (`scontrol update` is restricted); only `normal` QOS is available on both dp004 and durham accounts.
