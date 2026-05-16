# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable, required for script imports)
pip install -e .
pip install -r requirements.txt

# Tests
pytest tests -q

# Lint
ruff check src

```

SLURM jobs run on COSMA and activate `.venv` (Python 3.12) at repo root. All scripts in `scripts/` add `src/` to `sys.path` manually, so a dev install is not required for them.

## Architecture

Two top-level concerns share this repo:

**`src/`** — pip-installable library for post-processing GALFORM HDF5 outputs.

### Data flow

GALFORM writes output at:
```
/cosma5/data/durham/dc-hick2/Galform_Out/<SIM>/<MODEL>/iz<N>/ivol<K>/galaxies.hdf5
```

The `config.BASE_DIR` (overridable via `GALFORM_BASE_DIR` env var or `set_base_dir()`) points here. Scripts accept `--base-dir` / `--sim-name` / `--model` / `--iz` arguments and construct paths themselves.

Snapshot indices (`iz`) map to redshifts via `src/redshift_list.txt` — use `config.get_snapshot_redshift('iz155')` or `config.find_snapshot_at_redshift(0.5)`.

### Library structure (`src/`)

| Module | Role |
|--------|------|
| `config.py` | `BASE_DIR`, `Cosmology` constants, `iz` ↔ redshift lookup |
| `io/loaders.py` | Low-level HDF5 openers; `read_snapshot_data()` returns a dict with an open `h5py.File` — caller must call `close_snapshot()` |
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
where `alpha = m/k`, `beta = m(k-1) / [k(m-1)]`. This corrects for the fact that a sub-sample of subvolumes misses cross-pairs.

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

### RSD multipoles

`analysis/redshift_space_distortions/subvol_weighted_multipoles.py` — direct periodic estimator: `xi(s,mu) = DD(s,mu) / RR_analytic(s,mu) - 1`, projected to monopole (xi0) and quadrupole (xi2). No random catalogue needed for a periodic box.

## SCOPE

SCOPE is a separate Rust/Python pair-counter at `/cosma/apps/durham/dc-hick2/SCOPE/`.

- **Rust source:** `/cosma/apps/durham/dc-hick2/SCOPE/src/lib.rs` — `count_pairs_1d` and `count_pairs_2d`
- **Python API:** `/cosma/apps/durham/dc-hick2/SCOPE/python/scope/__init__.py`
- **Built .so:** compiled for both Python 3.9 and 3.12 (in `python/scope/`)
- **Add to path:** `export PYTHONPATH="/cosma/apps/durham/dc-hick2/SCOPE/python:$PYTHONPATH"`

Sub-volume IDs label **independent statistical realisations** of the full simulation box — **not spatial cells**. Every realisation spans the full coordinate range [0, boxsize) in all three dimensions; all realisations overlap. Stacking all k realisations recovers the full-density catalogue. Galaxies from different realisations can sit at the same position; same-subvol pairs are intra-realisation pairs, not spatially co-located pairs.

Key functions:

| Function | Purpose |
|----------|---------|
| `compute_xi(coords, subvol_ids, r_bins, box_size, n_subvols, n_subvols_selected)` | **Primary.** Real-space ξ(r), returns dict with `xi`, `dd_auto`, `dd_cross`, `dd_corr`, `rr`, `r_mid` |
| `compute_2pcf(coords, subvol_ids, r_p_bins, pi_bins, box_size, n_subvols, n_subvols_selected, n_total)` | **Legacy.** Projected w_p(r_p) via ξ(r_p,π), returns dict with `xi`, `wp`, `dd_auto`, `dd_cross`, `dd_corr`, `rr` |

**α/β conventions — two different normalisation targets:**

- `compute_xi` (primary, Rust backend: `count_pairs_1d`): scales DOWN to selected-catalogue density.  
  `α = m/k`, `β = m(k−1)/[k(m−1)]`, RR uses `N_selected`.  
  Matches `subvol_weighted_correction.py`.

- `compute_2pcf` (legacy, Rust backend: `count_pairs_2d`): scales UP to full-box counts.  
  `α = k/m`, `β = k(k−1)/[m(m−1)]`, `n_total` must be the **full-box** galaxy count (pass explicitly or it defaults to `N_sel · k/m`).

**Cell-count bug (fixed 2026-05-06):** `build_cell_list` in `lib.rs` uses `.clamp(3, 128)` (was `.max(1)`). With fewer than 3 cells in a periodic dimension, `rem_euclid` maps two distinct offsets to the same neighbour cell, double-counting pairs. Fix is in `lib.rs`; SCOPE has been rebuilt. Any large-rp CSVs produced before this date are corrupt and were deleted.

### SCOPE ξ(r) convergence campaign (primary)

`scripts/2pcf/compute_scope_xi_convergence.py` — one invocation per (seed, n_subvol); loads exactly n realisations, runs `compute_xi` in corrected and naive mode. Output: `scope_xi_<SIM>_iz<IZ>.csv`.

Data: `data/2pcf/scope_xi/<model>/iz<IZ>/<mstar_tag>/n<N>/seed<S>/scope_xi_<SIM>_iz<IZ>.csv`

`mstar_tag` is one of: `mstar_none`, `mstar9.0`, `mstar10.0`, `mstar11.0`. No halo mass cuts, no centrals filtering.

Submit: `submit_scope_xi_campaign.sh` — one SLURM job per (iz × mstar × n_subvol × seed).  
Seeds per n: 10 (n=1), 7 (n≤8), 5 (n≤64), 3 (n≤256), 2 (n≤512), 1 (n=1024).  
Partitions: cosma5 (n≤32), cosma8 (n≤256), cosma8-shm (n>256).  
r bins: 0.01 → 271 Mpc/h, 30 log-spaced. iz: 155, 207, 271. k_total=1024.  
n=1024 (SCOPE) is formally α=β=1 but is **not used as the reference** — see Corrfunc full-box below.

**mstar_none infeasibility:** The `mstar_none` (all-galaxy) catalogue is very large. n=32 iz271 takes ~12h on cosma5. Consequently:
- **n=128** would need ~192h vs the 8h cosma8 limit → will always timeout, do not submit
- **n=256** would need ~768h vs the 16h cosma8 limit → will always timeout, do not submit
- **n=512** is excluded from the campaign entirely (~1920h estimated)

The notebook `examples/2pcf/scope_xi_convergence.ipynb` defines `PLOT_N_MSTAR_NONE = [2, 8, 32]` (excludes n≥128) and uses it in the section-5 sweep for `mstar_none` only.

### Corrfunc full-box reference (n=1024)

The reference for all ξ(r) convergence comparisons is a **Corrfunc** full-box run, not SCOPE. It loads all 1024 ivols, applies the same stellar mass cut, and runs Corrfunc for a periodic box.

- **Script:** `scripts/2pcf/compute_corrfunc_xi_fullbox.py`
- **Submit:** `scripts/2pcf/slurm/submit_corrfunc_xi_fullbox.sh`
- **One job per (iz, mstar)** — deterministic, only `seed1000` needed (no seed variation)
- **Output:** `data/2pcf/scope_xi/<model>/iz<IZ>/<mstar_tag>/n1024/seed1000/scope_xi_<SIM>_iz<IZ>.csv`
- Residual columns in the notebook are NaN until the reference for that (iz, mstar) exists — do not use n=512 SCOPE as a proxy fallback

### SCOPE w_p convergence campaign (legacy)

`scripts/2pcf/compute_scope_wp_convergence.py` — one invocation per seed; loads up to `max(n_subvol_list)` sub-volumes, sweeps all `n_subvol` values computing both corrected and naive w_p. Output: `scope_wp_<SIM>_iz<IZ>.csv`.

Data:
- `data/2pcf/scope_wp/` — small-scale (r_p: 0.1 → 31.6 Mpc/h, 20 log bins)
- `data/2pcf/scope_wp_large_rp/` — large-scale (r_p: 31.6 → 200 Mpc/h, 7 log bins)

Both: `<model>/iz<IZ>/centrals_<0|1>/mstar<val>/<mhalo_val>/seed<S>/n<N>/scope_wp_<SIM>_iz<IZ>.csv`

Submit scripts (one SLURM job per seed × n_subvol, skips existing CSVs):

| Script | Purpose |
|--------|---------|
| `submit_scope_wp_split_by_n.sh` | Small-scale campaign; cycles seeds across cosma5/cosma8-shm/cosma8 |
| `submit_scope_wp_large_rp_split_by_n.sh` | Large-scale campaign; all jobs on cosma8-shm; runs iz=155,207,271 |

Default configs: `mhalo_min=1e9` and no mass cut; 5 seeds (base 1000); `pi_max=40`, `n_pi_bins=40`; `k_total=1024`.

## Dynamical friction timescale (τ₀) paper

Comparison of three lc16.newmg variants of GALFORM that differ only in the dynamical-friction timescale parameter τ₀. The headline question: how much does τ₀ shape galaxy properties even at the high-mass / cluster regime?

**Runs** (under `/cosma5/data/durham/dc-hick2/Tau0_Investigation/`):

| Label | Path | Physics |
|-------|------|---------|
| `Default` | `Galform_Out_Default/L800/lc16.newmg` | Calibrated τ₀ |
| `tau0=0` | `Galform_Out_0tau0/L800/lc16.newmg` | Instant satellite merging |
| `tau0=inf` | `Galform_Out_1e6tau0/L800/lc16.newmg` | Effectively no satellite merging |

Only **16 ivols** (out of L800's 1024) are written per run, and only **iz271 (z=0)** and **iz207 (z=0.5)**. High-mass bins (log Mh ≳ 14.5) are noisy with this sample size.

**Notebooks** (`examples/dynamical_friction_timescale/`):

| File | Probe |
|------|-------|
| `central_shmr.ipynb` | Central SHMR with IQR + bootstrap band; per-halo BCG mass distributions; σ(log M*\|Mh) |
| `stellar_mass_function.ipynb` | Total/cen/sat SMF; ratios vs Default; satellite fraction; cumulative N(>M*) |
| `hod.ipynb` | ⟨N_tot/cen/sat⟩(Mh); ratios; power-law fit α at log Mh ≥ 12.5; M* cut sweep |
| `satellite_cross_correlation.ipynb` | Pre-existing — satellite–central cross-correlation across the three runs |

**Shared module:** `tau0_helpers.py` (co-located, not packaged) — defines `RUNS`, `RUN_LABELS`, `RUN_COLORS` (Default=k, τ₀=0=C3 red, τ₀=∞=C0 blue), `RUN_MARKERS`, `SNAPSHOTS`, `DEFAULT_IVOLS=range(16)`, `BOX_SIZE_MPC_H=542.16`, `FOF_CENTRAL_RATIO_THRESHOLD=0.5`. Provides per-ivol loader (`load_galaxy_fields`), per-ivol summarisers (`central_shmr_per_ivol`, `smf_split_per_ivol`, `hod_per_ivol`), `collect_per_ivol(run_path, snapshot, ivols, summarise)`, and `stack_per_ivol(summaries, keys, nboot, seed)` which returns mean / SEM / bootstrap 16-84 ranges with NaN-slice warnings suppressed.

**FOF-central / BCG convention:** `is_central == 1 AND mhalo/mhhalo > 0.5`. HOD numerator is galaxies binned by host `mhhalo`; denominator is `Trees/mphalo` (one entry per FOF group).

**Headline physics** (z=0, 16 ivols):

- BCG mass at log Mh = 13: τ₀=0 → +0.08 dex, τ₀=∞ → **−0.84 dex** (BCG starved)
- Φ_cen at log M* = 11.5: τ₀=0 → 1.46×, τ₀=∞ → **0.00×** (no M* > 10^11.5 centrals form)
- ⟨N_sat⟩ at log Mh = 12: τ₀=0 → 0.83×, τ₀=∞ → **2.85×**

The asymmetry — Default sits much closer to τ₀=0 than to τ₀=∞ — is the paper's headline: dynamical friction is essential for assembling massive central galaxies.

## Simulations

| Name  | Box (Mpc/h) | N subvolumes | Notes |
|-------|-------------|--------------|-------|
| L800  | 542.16      | 1024         | Main high-res box |
| Mill1 | 365.0       | 64           | Millennium I |
| Mill2 | 73.0        | 64           | Millennium II |

Masses in GALFORM HDF5 files are in M_sun/h. `config.DEFAULT_STELLAR_MASS_BINS` and `config.DEFAULT_HALO_MASS_BINS` are in log10(M_sun/h).

## SLURM campaigns

Scripts in `scripts/2pcf/slurm/` and `scripts/redshift_space_distortions/slurm/` are shell wrappers that submit jobs directly via inline heredoc `sbatch` (no separate `.slurm` file needed for SCOPE scripts). Key env vars: `SIM_NAME`, `MODEL_NAME`, `IZ`, `OUTPUT_DIR`, `NMAX`, `MHALO_MIN`, `BOXSIZE`. SLURM resource overrides: `PARTITION`, `TIME_LIMIT`, `CPUS_PER_TASK`, `MEMORY`, `ACCOUNT`.

Output CSVs land in `data/`; logs in `logs/`; plots in `_plots/`.

### /cosma5 mount status on compute nodes (as of 2026-05-07)

GALFORM data lives on `/cosma5/data/...`. This filesystem is **not mounted on all compute nodes**. Submitting jobs to unmounted nodes causes immediate `FileNotFoundError`. Always submit to confirmed-working nodes only.

**Confirmed working** (mount verified):

| Node | Partition |
|------|-----------|
| m5005, m5006 | cosma5 |
| ga004, ga006 | cosma8-shm2 |
| mad09 | cosma8-shm3 |
| mad07 | cosma8-ska |

**Confirmed failing** (mount absent): cosma8 regular (m8xxx nodes), cosma7-rp, dine2.

**cosma8-ska requires `--account=durham`** (not `dp004`) — submitting with dp004 gives "Invalid account or account/partition combination".

Regular users cannot boost SLURM job priority (`scontrol update` is restricted); only `normal` QOS is available on both dp004 and durham accounts.
