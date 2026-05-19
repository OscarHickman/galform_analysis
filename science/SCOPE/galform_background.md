# GALFORM and Simulation Background

Reference document for the SCOPE ξ(r) paper.  Covers the galaxy formation model,
simulation suite, sub-volume implementation, and known limitations relevant to
clustering statistics.  Derived from source code in `/cosma/apps/durham/dc-hick2/galform/`
and `/cosma/apps/durham/dc-hick2/SCOPE/`, plus the analysis library in `src/`.

---

## 1  What is GALFORM?

GALFORM is a semi-analytic model (SAM) of galaxy formation developed at Durham
(Cole et al. 2000; Bower et al. 2006; Lacey et al. 2016).  It post-processes the
merger trees of a dark-matter-only N-body simulation to predict the formation and
evolution of galaxies across cosmic time.  The underlying assumption is that
baryonic physics (gas cooling, star formation, feedback) can be encoded as analytic
prescriptions that are applied at each node of the merger tree, cheaply enough to
run over the full halo population of a large simulation.

### Physical processes

| Process | Module(s) | Key physics |
|---------|-----------|-------------|
| **Gas cooling** | `cooling.F90`, `cooling.mass_infalling.F90` | Bower (2006) cooling model (`icool=3`); hot gas cools at rate limited by free-fall time and cooling time; reheated gas re-incorporated |
| **Star formation (quiescent)** | `star_formation.rules.F90`, `star_formation.solve_equations.F90` | Schmidt–Kennicutt law; τ_star = τ₀ (V_circ/200)^α; solved as ODE per timestep |
| **Star formation (bursts)** | `star_formation.bursts.F90` | Triggered by major mergers and disk instabilities; exponential decay timescale |
| **Supernova feedback** | `feedback.expulsion.parameters.F90` | Mass-loaded wind: Ṁ_eject = β_SN × Ṁ_star where β_SN ∝ (V_hot/V_circ)^α_hot; two separate V_hot values for disk and burst SF |
| **AGN feedback** | `supermassive_black_holes.AGN_feedback.F90` | Radio-mode heating: AGN accretion at fraction ε_SMBH of Eddington rate heats ICM; suppresses cooling in massive halos (Bower et al. 2006) |
| **SMBH growth** | `supermassive_black_holes.F90`, `supermassive_black_holes.AGN_physics.F90` | SMBH grows during mergers and disk instabilities; spin evolution tracked but disabled in lc16/gp14 calibration runs |
| **Galaxy mergers** | `merging.find_merger_time.F90`, `merging.nbody_merger_time.F90` | Dynamical friction (Lacey & Cole 1993); satellite placed on orbit when subhalo is last resolved; merger clock ticks until DF timescale exhausted |
| **Dynamical friction timescale** | `merging.dynamical_friction_timescale.F90`, `merging.nbody_merger_time.F90` | t_DF = τ₀ × θ × t_dyn × f(M_sat/M_host); isothermal sphere assumed; τ₀ is a free parameter |
| **Disk instability** | `disk_stability.F90` | Efstathiou et al. (1982) global criterion; unstable disk mass transferred to bulge, triggering a burst |
| **Satellite physics** | `ram_pressure.starvation.F90` | Ram-pressure and tidal stripping of hot halo gas (Benson & Bower 2010); starvation flag controls whether satellites can cool new gas |
| **Reionisation** | `cooling.F90` | Phenomenological: cooling suppressed for V_circ < v_cut AND z < z_cut; mimics photoionisation feedback; default v_cut = 30 km/s, z_cut = 10 |
| **Dust** | `dust.F90`, `dust.attenuate.F90`, `dust.SED.F90` | Two-component model (diffuse ISM + birth clouds); attenuation applied to stellar SEDs |
| **Stellar populations / SEDs** | `stellar_populations.F90`, band output | Bruzual & Charlot or Maraston SSP tables; convolved with SFH to give broadband luminosities |

### How it works (step by step)

1. **Input**: N-body merger trees (HDF5, aquarius format).  Each halo has position,
   velocity, mass, spin, and subhalo structure at every snapshot.
2. **Walk trees**: GALFORM processes each merger tree from high redshift to z=0,
   evolving the gas and stellar content of each halo/galaxy forward in time.
3. **At each timestep**: cool gas → form stars → apply SN feedback → grow SMBH →
   apply AGN heating → check disk stability → handle mergers.
4. **Output**: Galaxy properties (positions, masses, SFRs, luminosities, `is_central`)
   written to HDF5 per ivol per snapshot.

---

## 2  The Two Model Variants: lc16 and gp14

### lc16 — Lacey et al. (2016), modified for L800

**Full reference**: Lacey et al. (2016, MNRAS, 462, 3854), recalibrated for the
P-Millennium (L800) simulation by Baugh et al. (2018, in prep).  Includes the
Simha & Cole "new merging" scheme (`aquarius_nbody_merging_scheme = 1`).

**Calibration targets**: K-band luminosity function; b_J-band LF; sub-millimetre
galaxy number counts; black hole mass–bulge mass relation.  Intentionally
reproduces the high-redshift galaxy population and dust-obscured SF.

**Key parameters** (from `Lacey16_newmg_Nbody_L800.input.ref`):

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `alphahot` | 3.4 | SN feedback mass-loading slope |
| `vhotdisk` | 320 km/s | SN feedback velocity for disk SF |
| `vhotburst` | 320 km/s | SN feedback velocity for burst SF |
| `epsilon_SMBH_Eddington` | 0.01 | AGN accretion rate (fraction of Eddington) |
| `tau0mrg` | 1.0 | Dynamical friction timescale normalisation |
| `stabledisk` | 0.9 | Disk instability threshold (Efstathiou Q) |
| `gasrich_unstabledisk` | `.true.` | Only gas-rich disks trigger instability bursts |
| `nmf` | 2 | Two IMFs: Kennicutt (quiescent) + top-heavy (bursts) |
| `starvation` | `.true.` | Satellites cannot cool new gas (hot gas stripped) |
| `fgasburst` | 0 | Minor-merger bursts not suppressed by gas fraction |

### gp14 — Gonzalez-Perez et al. (2014), Guo & Gonzalez-Perez (2016)

**Full reference**: Guo, Gonzalez-Perez et al. (2016, MNRAS, 461, 3457); based on
Gonzalez-Perez et al. (2014).  Includes the Simha & Cole merging scheme.

**Calibration targets**: b-band and K-band luminosity functions; optical
colour–magnitude relations.  Tuned to reproduce the optical galaxy population
without requiring a top-heavy burst IMF.

**Key parameters** (from `Gonzalez15_newmg_Nbody_L800.input.ref`):

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `alphahot` | 3.2 | SN feedback mass-loading slope |
| `vhotdisk` | 380 km/s | SN feedback velocity for disk SF |
| `vhotburst` | 380 km/s | SN feedback velocity for burst SF |
| `epsilon_SMBH_Eddington` | 0.0398 | AGN accretion rate (~4× stronger than lc16) |
| `tau0mrg` | 1.5 | Dynamical friction timescale normalisation (50% longer) |
| `stabledisk` | 0.8 | Disk instability threshold (easier to trigger) |
| `gasrich_unstabledisk` | `.false.` | All unstable disks trigger bursts |
| `nmf` | 1 | Universal (Kennicutt) IMF everywhere |
| `starvation` | `.false.` | Satellites can continue to cool gas |
| `fgasburst` | 0.1 | Minor-merger bursts suppressed if disk is gas-poor |

### Key physical differences and clustering consequences

| Difference | lc16 | gp14 | Clustering effect |
|-----------|------|------|-------------------|
| AGN feedback | Weak (ε=0.01) | Strong (ε=0.040) | gp14 suppresses BCG mass more aggressively; fewer massive centrals → lower large-scale bias at high M* cuts |
| SN feedback velocity | Lower (320 km/s) | Higher (380 km/s) | gp14 ejects more gas from low-mass halos; fewer low-mass satellites |
| DF timescale τ₀ | 1.0 | 1.5 | gp14 satellites take 50% longer to merge → more satellites per halo at fixed mass → higher satellite fraction → stronger one-halo term |
| Starvation | On | Off | lc16 satellites are gas-starved; lower satellite SFR; does not affect stellar mass (and hence mass-selected clustering) directly |
| IMF | Two-component | Universal | Affects SFR-to-stellar-mass conversion in bursts; relevant for SFR-selected (e.g. emission-line) samples, not mass-selected |

**Why lc16 and gp14 for robustness testing?**  The two models span a meaningful
range of galaxy formation physics while both being calibrated to observational data.
Their key difference for clustering is the HOD: gp14 predicts more satellites per
halo (higher τ₀, no starvation), giving a larger one-halo clustering amplitude at
fixed stellar mass threshold.  If the SCOPE correction factor α = m/k, β = m(k−1)/[k(m−1)]
is truly HOD-independent (as the derivation in §3 requires), the two models must
converge to their respective full-box references at the same rate — which is what
§5.4 demonstrates.

---

## 3  The Simulations

### P-Millennium / L800

The "Planck Millennium" simulation.  Run with the same initial conditions structure
as the original Millennium but with Planck 2013 cosmology.

| Property | Value |
|---------|-------|
| Box side length | 542.16 Mpc/h (800 Mpc physical at z=0) |
| Cosmology | Planck 2013: h=0.6777, Ω_m=0.307, Ω_Λ=0.693, Ω_b=0.0483 |
| Sub-volumes (k) | 1024 |
| Volume per ivol | 155,626 (Mpc/h)³ |
| Galaxy count (lc16, z=0, all) | ~142.7 M total (≈139,400 per ivol) |
| Stellar mass range | ~10³–2×10¹¹ M☉/h |
| Reference in this work | All L800/lc16 convergence, redshift, sample-selection analyses |

### Millennium I (Mill1)

The original Millennium Simulation (Springel et al. 2005).

| Property | Value |
|---------|-------|
| Box side length | 500 Mpc/h |
| Cosmology | WMAP1: h=0.73, Ω_m=0.25, Ω_Λ=0.75 |
| Sub-volumes (k) | 64 |
| Galaxy coordinates | [0, 500] Mpc/h (confirmed from HDF5) |
| Reference in this work | §5.4 model robustness (lc16 vs gp14) |

### Millennium II (Mill2)

Higher-resolution resimulation of a smaller Millennium I sub-volume
(Boylan-Kolchin et al. 2009).

| Property | Value |
|---------|-------|
| Box side length | 100 Mpc/h |
| Cosmology | WMAP1: h=0.73, Ω_m=0.25, Ω_Λ=0.75 |
| Sub-volumes (k) | 64 |
| Galaxy coordinates | [0, 100] Mpc/h (confirmed from HDF5) |
| Reference in this work | §5.4 model robustness (lc16 vs gp14) |

**Note on boxsizes**: Galaxy coordinates in all three simulations are stored in
Mpc/h.  The boxsizes above are the correct values to pass to both SCOPE and
Corrfunc.  An earlier version of the model-robustness submit script incorrectly
used physical boxsizes (365 and 73 Mpc, i.e. Mpc/h × h = 500×0.73 and 100×0.73),
producing corrupt pair counts; this was corrected in May 2026 and all affected
data was recomputed.

**Why these simulations?**  L800 provides k=1024 independent sub-volumes, enabling
robust statistics across N_subvol = 1–512.  Mill1 and Mill2 test that the SCOPE
correction is independent of cosmology (WMAP1 vs Planck 2013) and box size (factor
of 25 in volume).  The two WMAP1 simulations also differ in resolution, allowing a
check that the HOD-independence of α/β holds even when the galaxy occupation shifts
due to better-resolved low-mass halos in Mill2.

---

## 4  Galaxy Definition in GALFORM

### What constitutes a galaxy

In GALFORM's HDF5 output, every row of the `Output{NNN}` group is a galaxy.
A galaxy exists if:

1. It sits in a resolved dark matter (sub)halo at the output snapshot, **or**
2. It has been orphaned (its host subhalo was lost in the N-body tree) and a merger
   clock is still running.

In practice, any object with `mstars_disk + mstars_bulge > 0` is counted.  The
minimum stellar mass is set by the minimum resolved halo mass and the SF efficiency;
for L800/lc16 this is ≈10³ M☉/h.

### Key output arrays (HDF5 `Output{NNN}` group)

| Array | Units | Meaning |
|-------|-------|---------|
| `xgal`, `ygal`, `zgal` | Mpc/h (comoving) | Galaxy position in the full box |
| `mstars_disk` + `mstars_bulge` | M☉/h | Total stellar mass |
| `mhalo` | M☉/h | Subhalo mass (the DM (sub)halo hosting this galaxy) |
| `mhhalo` | M☉/h | Host FOF halo mass (the group containing the subhalo) |
| `mstardot` | M☉/h/Gyr | Total star formation rate |
| `is_central` | 0 or 1 | 1 = central galaxy of its FOF halo |
| `Bands/Band###_Lum_{Disk,Bulge}` | L☉/h² (rest-frame) | Broadband luminosities |

The `Parameters` group stores the per-ivol volume (`volume` in (Mpc/h)³ = V_box/k),
cosmological parameters, and the number of snapshots.

### Stellar mass cuts

The cuts applied in this work are:

| Tag | Cut | L800/lc16 z=0 count (est.) |
|-----|-----|-----------------------------|
| `mstar_none` | All galaxies | ~142.7 M |
| `mstar9.0` | mstars > 10⁹ M☉/h | ~20–30 M |
| `mstar10.0` | mstars > 10¹⁰ M☉/h | ~5–8 M |
| `mstar11.0` | mstars > 10¹¹ M☉/h | ~0.2–0.5 M |

The cut is applied identically to the SCOPE sub-volume catalogue and the Corrfunc
full-box reference.  No centrals-only or satellites-only filtering is applied in
the convergence campaign (unlike the earlier w_p campaign).

---

## 5  Sub-volume Implementation

### How halos are partitioned into k sub-volumes

The P-Millennium merger trees were pre-processed before running GALFORM: each
merger tree (one tree = one FOF halo tracked back through all its progenitors) was
assigned a random integer in [0, k-1] and written to a separate file.  From the
GALFORM parameter file:

> "The MillGas trees have been split into files by drawing a random number in the
> range 0–63 to decide which file to put each tree in.  This means that the volume
> associated with each file is 1/1024 times the simulation volume."

GALFORM is then run once per ivol, reading only the trees assigned to that file.
The `ivolume` parameter (or `--ivol N` on the command line) selects which file to
read.  When `append_ivolume = .true.`, GALFORM appends `.{ivol}` to the tree
filename, opening `trees.{ivol}.hdf5` instead of `trees.hdf5`.

### Spatial structure of sub-volumes

Each ivol is **not a spatial tile** of the box.  Because each merger tree is
assigned randomly (not by position), the halos in any one ivol are scattered
uniformly across the full simulation volume [0, L_box).  Galaxy positions
(`xgal`, `ygal`, `zgal`) therefore span the full box in every ivol.

Consequence: ivol 0 and ivol 1 contain different halos at **overlapping positions**.
Stacking all k ivols recovers the full galaxy catalogue — every halo appears in
exactly one ivol.

### Correlation structure between sub-volumes

Because ivols are random partitions of the same underlying N-body field:

- **Cross-pairs are real**: a close pair in 3D that happens to straddle two ivol
  assignments is a genuine physical pair in the simulation.
- **Sub-volumes are not independent**: they are complementary draws from the same
  density field, not independent Monte Carlo realisations.  Two ivols sample the
  same large-scale modes (the same cosmic web); they differ only in which halos are
  included.
- **Statistical equivalence**: any single ivol has the same two-point statistics
  as any other in expectation, because halo assignment is random.  This is what
  makes the ensemble average over seeds meaningful.

The SCOPE α/β correction (§3) exploits this structure: selecting m of k sub-volumes
gives m/k of the total galaxy density.  The auto-pair counts (same-ivol pairs) and
cross-pair counts (different-ivol pairs) scale differently with m, and α and β are
the weights that restore the correct pair-count normalization.

### Seed vs deterministic sub-volume selection

In the convergence campaign, the `selection_seed` parameter controls which m of the
k available ivols are selected for a given run.  Two runs with the same (m, k) but
different seeds select different subsets of ivols and therefore different galaxy
samples.  The seed-to-seed scatter measures the intrinsic statistical uncertainty
of the SCOPE estimator (§5.5).

---

## 6  Known Assumptions and Limitations

### Relevant to clustering statistics

**No explicit orphan satellites.**  When a satellite's subhalo is lost from the
N-body merger tree (typically because it falls below the resolution limit), GALFORM
starts a dynamical-friction merger clock.  The satellite galaxy's position is
**not tracked** after subhalo loss; it is simply held at the position of the last
known subhalo until the merger clock expires, at which point it merges with the
central.  This means:

- The small-scale clustering of satellites (r ≲ a few × r_vir) is not captured
  after subhalo disruption; the true satellite distribution is narrower than
  the N-body subhalo profile would give.
- The number of satellites at any epoch depends on τ₀ (the merger timescale
  parameter), making ⟨N_sat⟩(M_h) and hence the one-halo clustering amplitude
  sensitive to τ₀ (see §5.4 and the τ₀ investigation notebooks).

**Dynamical friction timescale approximation.**  The Lacey & Cole (1993) formula
assumes an **isothermal sphere** host halo (not NFW).  The implementation
(`merging.nbody_merger_time.F90`, line ~180):

```fortran
ts_db = tau0mrg * theta * tdyn * 0.3722 * m_int / (msat * clog)
```

where `theta` encodes the orbit circularity, `m_int` is the host mass within the
circular orbit radius, and `clog` is the Coulomb logarithm.  Orbital parameters
are drawn from a distribution calibrated to N-body simulations.  The scaling is
approximate at the factor-of-2 level, and the free parameter τ₀ absorbs this
uncertainty.

**Tidal and ram-pressure stripping (simplified).**  Only the **hot gas halo** of
satellites is stripped (ram_pressure.starvation.F90), following Benson & Bower (2010).
Stellar stripping and tidal truncation of the stellar component are not modelled.
For mass-selected clustering at M* > 10⁹–10¹¹ M☉/h, this matters at small
separations (r ≲ 0.5 Mpc/h) where tidally stripped stars form intracluster light
rather than a distinct galaxy.

**Disk instability is global, not local.**  The Efstathiou et al. (1982) criterion
checks global disk stability (Q as a function of disk mass, size, and circular
velocity) rather than a local Toomre instability.  This tends to underestimate
the frequency of clump formation and overestimate the mass transferred to the bulge
in a single event.  For the HOD, this means the bulge-to-total ratio (and hence
the morphological mix at fixed mass) carries a systematic uncertainty.

**Reionisation is phenomenological.**  Cooling is suppressed below V_c = 30 km/s
at z < 10 via a hard threshold, not through a physical reionisation calculation.
This affects the abundance of the faintest galaxies (M* < 10⁷ M☉/h) but is
negligible for the M* > 10⁹ cuts used in this work.

**Periodic boundary conditions in the N-body simulation.**  All pair-count
statistics (both Corrfunc and SCOPE) use periodic boundary conditions matched to
the simulation box.  Galaxy clustering near the fundamental mode (r ~ L_box/2)
is affected by the finite box size; the r_max values in this work (271, 250, 50 Mpc/h
for L800, Mill1, Mill2) are set to half the box side to avoid this regime.

**Box-to-box cosmological variance.**  L800 and the Millennium simulations use
different cosmologies (Planck 2013 vs WMAP1).  ξ(r) amplitudes are not directly
comparable between the two; the model-robustness test (§5.4) compares convergence
*rates* (not absolute values) and is not sensitive to cosmological differences.

---

## 7  Summary Reference Table

| Item | Value / Reference |
|------|------------------|
| GALFORM version | v2.7.0 (lc16/gp14), source at `/cosma/apps/durham/dc-hick2/galform/` |
| lc16 parameter file | `Lacey16_newmg_Nbody_L800.input.ref` |
| gp14 parameter file | `Gonzalez15_newmg_Nbody_L800.input.ref` |
| lc16 reference | Lacey et al. (2016); Baugh et al. (2018, in prep) |
| gp14 reference | Guo, Gonzalez-Perez et al. (2016); Gonzalez-Perez et al. (2014) |
| L800 boxsize | 542.16 Mpc/h; k=1024; Planck 2013 |
| Mill1 boxsize | 500 Mpc/h; k=64; WMAP1; Springel et al. (2005) |
| Mill2 boxsize | 100 Mpc/h; k=64; WMAP1; Boylan-Kolchin et al. (2009) |
| Ivol partitioning | Random tree assignment; each galaxy in exactly one ivol |
| Galaxy position units | Mpc/h (comoving), full box range in every ivol |
| Stellar mass units | M☉/h (as stored in HDF5) |
| Halo mass units | M☉/h (as stored in HDF5) |
| SCOPE source | `/cosma/apps/durham/dc-hick2/SCOPE/src/` |
| SCOPE α convention | α = m/k, β = m(k−1)/[k(m−1)]; scales to selected-catalogue density |
