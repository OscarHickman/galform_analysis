# The Dynamical Friction Timescale in GALFORM

## Physical motivation

The dynamical-friction timescale $\tau_0$ controls how long an infalling satellite spirals inward
before merging with the central galaxy of its host halo. In GALFORM the Lacey–Cole merger time is
$t_\mathrm{merge} = \tau_0 f(\epsilon) t_\mathrm{dyn}$, where $f(\epsilon)$ is a circularity
function and $t_\mathrm{dyn}$ is the halo dynamical time. Two limiting cases bracket the
model space:

| Run | $\tau_0$ | Physics |
|-----|----------|---------|
| `Default` | calibrated | canonical lc16.newmg |
| `tau0=0` | 0 | every infalling satellite merges instantly |
| `tau0=inf` | $10^6$ | satellites never merge (infinite timescale) |

These extremes isolate the contribution of the merger channel to galaxy assembly: $\tau_0=0$ routes
all infalling stellar mass directly into the central, while $\tau_0 \to \infty$ keeps every
satellite alive indefinitely. The bracket between them quantifies how much of the mass budget at
any given halo mass is controlled by dynamical friction rather than in-situ star formation.

**Key question:** does $\tau_0$ matter at the *high-mass* end — groups and clusters — or does
the dynamical time already become so short that the satellites merge quickly regardless?

## Runs

All three variants run `lc16.newmg` on the **L800** N-body box (P-Millennium,
$L_\mathrm{box} = 542.16\,\mathrm{Mpc}/h$, Planck 2013 cosmology).
Currently **16 of 1024 subvolumes** are analysed (ivols 0–15); the remaining 1008 are queued.
Two snapshots: iz271 ($z = 0$) and iz207 ($z = 0.5$).

Output paths:
```
/cosma5/data/durham/dc-hick2/Tau0_Investigation/Galform_Out_{Default,0tau0,1e6tau0}/L800/lc16.newmg/
```

The same three variants are also being run on **COLIBRE-L100m6** and **FLAMINGO-L1000N1800**
(cosma8, 64 subvolumes each) to test cosmology and resolution dependence.

## Probes

Three complementary statistics are used, each sensitive to a different aspect of the merger channel:

### 1. Halo Occupation Distribution

$\langle N_\mathrm{sat} \rangle(M_h)$ is the most direct test. Instant merging ($\tau_0=0$)
destroys satellites immediately, suppressing the satellite HOD. No merging ($\tau_0 \to \infty$)
accumulates every galaxy that ever fell in, inflating $\langle N_\mathrm{sat} \rangle$ at all
halo masses but especially in massive halos with rich merger histories.

**Headline result (z=0, 16 ivols):**

| $\log_{10} M_h$ | $\langle N_\mathrm{sat} \rangle / \langle N_\mathrm{sat} \rangle^\mathrm{default}$ |
|--|--|
| | $\tau_0=0$ | $\tau_0=\infty$ |
| 12 | 0.83 | **2.85** |
| 13 | ~0.7 | **~5–10** (noisy at 16 ivols) |

The asymmetry — Default sits much closer to $\tau_0=0$ than to $\tau_0=\infty$ — means dynamical
friction is the dominant timescale in the Default run. The power-law slope
$\langle N_\mathrm{sat} \rangle \propto M_h^\alpha$ can also shift between runs if dynamical
friction acts differently on low- vs high-mass satellites.

Figures 1–4 in `dynamical_friction.ipynb`.

### 2. Central Stellar-to-Halo Mass Relation

The BCG mass is the most direct stellar-budget consequence. At fixed $M_\mathrm{halo}$:

- $\tau_0=0$: BCG accretes its satellites → higher median $M_\star$.
- $\tau_0 \to \infty$: BCG is starved of accreted mass → lower median $M_\star$.

**Headline result (z=0):**

| $\log_{10} M_h$ | $M_{\star,\mathrm{cen}}^{\tau_0=0} / M_{\star,\mathrm{cen}}^\mathrm{default}$ | $M_{\star,\mathrm{cen}}^{\tau_0=\infty} / M_{\star,\mathrm{cen}}^\mathrm{default}$ |
|--|--|--|
| 12 | +0.08 dex | −0.84 dex |
| 13 | ~+0.1 dex | ≪ −1 dex (BCG starved) |

The $\tau_0 \to \infty$ run almost completely starves the BCG above $\log_{10} M_h \sim 13$:
no massive central galaxies form without the merger channel. The per-halo scatter
$\sigma(\log M_\star | M_h)$ also changes: $\tau_0$ broadens the distribution because
merger histories are stochastic.

Figures 5–7 in `dynamical_friction.ipynb`.

### 3. Stellar Mass Function

$\tau_0$ is a redistribution knob — it does not change total stellar mass production,
only where that mass ends up. Consequently:

- **Total SMF**: approximately invariant across $\tau_0$ (second-order effects from gas
  brought in by merging satellites are small).
- **Central SMF**: suppressed at $\log_{10} M_\star \gtrsim 11$ for $\tau_0 \to \infty$;
  enhanced for $\tau_0=0$.
- **Satellite SMF**: mirror-image response.

**Headline result (z=0):**

| $\log_{10} M_\star$ | $\Phi_\mathrm{cen}^{\tau_0=0} / \Phi_\mathrm{cen}^\mathrm{default}$ | $\Phi_\mathrm{cen}^{\tau_0=\infty} / \Phi_\mathrm{cen}^\mathrm{default}$ |
|--|--|--|
| 11.5 | 1.46× | **0.00×** (no $M_\star > 10^{11.5}$ centrals form) |

Figures 8–10 in `dynamical_friction.ipynb`.

## Paper narrative

The three probes tell a consistent story:

1. **Dynamical friction is essential for assembling massive central galaxies.** The $\tau_0 \to \infty$
   run demonstrates that without satellite merging there are essentially no
   $M_\star \gtrsim 10^{11.5}\,M_\odot/h$ BCGs and the high-mass end of the SHMR collapses by
   $\gtrsim 1\,\mathrm{dex}$ above $\log_{10} M_h \sim 13$.

2. **The Default run sits close to the $\tau_0=0$ limit, not the $\tau_0=\infty$ limit.**
   This is the headline asymmetry: the calibrated GALFORM model already operates in a regime
   where most satellites merge within a Hubble time. The $\tau_0$ knob has a large dynamic range
   in one direction (removing merging) but a modest range in the other (speeding them up).

3. **The satellite HOD and satellite fraction are the most sensitive diagnostics.**
   Even modest changes in $\tau_0$ produce order-of-magnitude shifts in $\langle N_\mathrm{sat} \rangle$
   at group scales ($\log_{10} M_h \sim 13$). The total SMF remains stable, confirming the
   mass-redistribution interpretation.

4. **Multi-simulation comparison (pending):** Running the same three variants on COLIBRE-L100m6
   and FLAMINGO-L1000N1800 will reveal whether these results are robust to N-body resolution and
   cosmology (Planck 2013 vs 2018) or whether the calibrated $\tau_0$ shifts between boxes.

## Notebook

**`dynamical_friction.ipynb`** — all analysis in one notebook:
- Section 1: HOD (Figures 1–4)
- Section 2: Central SHMR (Figures 5–7)
- Section 3: Stellar mass function (Figures 8–10)

**`tau0_helpers.py`** — shared loader, summarisers (`hod_per_ivol`, `central_shmr_per_ivol`,
`smf_split_per_ivol`), and stacking utilities (`collect_per_ivol`, `stack_per_ivol`).

## Status

- [x] Default L800 — 1024 ivols, iz271 + iz207
- [ ] tau0=0 L800 — ivols 0–15 done; 16–1023 queued (SLURM 11116942–11116945)
- [ ] tau0=inf L800 — ivols 0–15 done; 16–1023 queued (SLURM 11116946–11116949)
- [ ] COLIBRE-L100m6 (Default, tau0=0, tau0=inf) — queued (SLURM 11116927–11116932)
- [ ] FLAMINGO-L1000N1800 (Default, tau0=0, tau0=inf) — queued (SLURM 11116933–11116938)
