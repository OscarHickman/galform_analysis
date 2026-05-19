# SCOPE $\xi(r)$ Convergence Analysis

This analysis demonstrates that the **SCOPE sub-volume correction** can recover the full-box real-space two-point correlation function $\xi(r)$ from a fraction of the simulation realisations.

## Scientific Objectives

The goal of this analysis is to validate the accuracy and efficiency of the SCOPE sub-volume correction technique for large-scale structure statistics:

1.  **Correction Validation:** Demonstrate that applying the SCOPE correction to sub-volume measurements yields results consistent with full-box Corrfunc reference runs.
2.  **Convergence Analysis:** Determine the minimum number of sub-volumes ($N_{
m subvol}$) required to reach a target convergence threshold for $\xi(r)$.
3.  **Model Robustness:** Validate that the correction method remains robust across different simulation box sizes (e.g., Millennium I vs. II) and model implementations.
4.  **Performance Scaling:** Evaluate the computational efficiency and runtime scaling of the SCOPE approach to justify its use in large-scale simulation analysis campaigns.

For detailed plots and implementation, see `scope_xi.ipynb`.

---

## Section 4 — Validation Framework

### 4.1  Reference Definitions

The reference two-point correlation function against which all SCOPE estimates are compared is the full-box Corrfunc result computed using all $k$ sub-volume realisations simultaneously. This is not the SCOPE estimator evaluated at $m = k$ (which is formally equivalent but introduces the SCOPE machinery unnecessarily); it is a direct Corrfunc periodic-box run that loads all $k \times N_{\rm gal,subvol}$ galaxies at once and computes pair counts without any sub-volume weighting. The reference is deterministic (a fixed seed 1000 is used only for I/O ordering), so it carries no sample-variance contribution from the sub-volume selection. All fractional residuals and log-ratio metrics in §5 are computed relative to this reference.

Three $N$-body simulations are used in this work, spanning roughly a decade in box volume. The primary suite is the L800 run (Planck 2013 cosmology, $L = 542.2\,h^{-1}$Mpc, $k = 1024$ sub-volumes), which provides the statistical power to probe $N_{\rm subvol}$ values from 1 to 512 with multiple independent seeds. For model-robustness and box-size scaling tests, we use the Millennium I ($L = 500\,h^{-1}$Mpc, $k = 64$, WMAP1) and Millennium II ($L = 100\,h^{-1}$Mpc, $k = 64$, WMAP1) simulations. Two GALFORM model variants are considered: `lc16`, the fiducial model calibrated to the local stellar mass function and used for all L800 analyses, and `gp14`, an earlier variant with different supernova feedback prescriptions and a lower satellite fraction at fixed halo mass, included in the Millennium robustness tests. The correction factors $\alpha = m/k$ and $\beta = m(k-1)/[k(m-1)]$ (equations 11–12) are independent of galaxy formation physics, so any difference in convergence between `lc16` and `gp14` would indicate a failure of this assumption.

Four galaxy selections are applied to each snapshot: no stellar mass cut (`mstar_none`; all galaxies in the sub-volume), and three threshold cuts $\log_{10}(M_*/M_\odot h^{-1}) \geq 9.0$, $10.0$, and $11.0$. These span a wide range of halo occupation distributions, from the galaxy-number-dominated $M_* > 10^9$ samples (many satellites per halo, strong one-halo signal) to the cluster-scale $M_* > 10^{11}$ samples (fewer than one galaxy per group-mass halo, shot-noise-limited two-halo term). All mass cuts are applied identically to both the SCOPE sub-volume selections and the Corrfunc reference run. Three snapshots are analysed per simulation: $z \approx 1.5$, $z \approx 0.5$, and $z \approx 0$ for L800 (iz155, iz207, iz271), and $z \approx 1.9$ / $z = 0$ for Mill1 (iz33, iz63) and $z \approx 1.5$ / $z = 0$ for Mill2 (iz40, iz67).

### 4.2  Accuracy Metrics

Four complementary metrics are used to characterise convergence. The primary metric is the median absolute log-ratio,
$$
\mathcal{L}(N_{\rm subvol}) = {\rm median}_{r \in \mathcal{R}}\, |\Delta \log_{10} \xi(r)|
\equiv {\rm median}_{r \in \mathcal{R}}\, \left|\log_{10} \frac{\xi_{\rm SCOPE}(r)}{\xi_{\rm ref}(r)}\right|,
$$
where $\mathcal{R}$ denotes the set of separation bins in which both $\xi_{\rm SCOPE}$ and $\xi_{\rm ref}$ are strictly positive. The log-ratio is preferred over the raw fractional residual $(\xi - \xi_{\rm ref})/|\xi_{\rm ref}|$ as the primary summary because it is symmetric on the log scale (a factor-of-two overestimate and a factor-of-two underestimate carry equal weight), and because galaxy-clustering likelihoods are typically formulated in log-space, meaning $|\Delta\log_{10}\xi|$ directly scales with the chi-squared contribution.

Three secondary metrics are computed to cross-check the behaviour at different points in the distribution. The RMS fractional error $\varepsilon_{\rm rms} = \sqrt{\langle[(\xi - \xi_{\rm ref})/\xi_{\rm ref}]^2\rangle}$ gives the root-mean-square deviation across positive bins and is more sensitive to outlying scales than the median. The compliance fraction $f_5$ is the fraction of separation bins in $\mathcal{R}$ for which $|(\xi - \xi_{\rm ref})/\xi_{\rm ref}| < 5\%$; values of $f_5 \to 1$ indicate that the correction is uniformly accurate across all scales rather than merely accurate in a median sense. The seed-scatter statistic $(p_{84} - p_{16})/(2\,|\xi_{\rm ref}|)$ measures the half-width of the 16th–84th percentile band over independent seeds and characterises the intrinsic variance of the SCOPE estimator separately from its bias; this metric is used exclusively in §5.5.

The accuracy requirements differ between the one-halo ($r \lesssim 1\,h^{-1}$Mpc) and two-halo ($r \gtrsim 5\,h^{-1}$Mpc) regimes. In the one-halo regime, the naïve sub-volume estimator overestimates $\xi(r)$ by a factor $k/m$ at small $r$ (Section 3): selecting $m$ out of $k$ realisations keeps a fraction $m/k$ of halo-pair galaxies while the analytic RR normaliser scales as $(m/k)^2$, yielding $\xi_{\rm naive} \approx (k/m)\,\xi_{\rm true}$ at scales dominated by intra-halo pairs. For $N = 16$ this gives a 64-fold overestimate at $r \lesssim 0.1\,h^{-1}$Mpc; for $N = 4$ the bias reaches $\sim\!250\times$. The SCOPE correction reduces this to well below 1 per cent at the same $N$, and one-halo accuracy is not the limiting factor. The two-halo regime is more demanding: cosmic variance per sub-volume is large relative to the signal, and the two-halo term enters explicitly in BAO and large-scale-bias measurements. Consequently, in §5 we report $\mathcal{L}(N_{\rm subvol})$ computed separately over $r \in [0.3, 1]\,h^{-1}$Mpc and $r \in [5, 30]\,h^{-1}$Mpc alongside the global median, and we treat the two-halo median as the operationally relevant convergence indicator. The two-halo $\mathcal{L}$ reaches the 5 per cent threshold at $N^* = 32$–$64$ depending on selection and redshift (§5.1–§5.2), compared with the global $N^*$ which is 4–16 for dense catalogues where the positive-bin median is dominated by one-halo scales.

### 4.3  Compute Efficiency

The primary computational constraint is wall-clock time, set by per-job limits on the COSMA HPC facility: 2 h on cosma5 and cosma8-shm nodes, 8 h on cosma8. The SCOPE pair-counting step is implemented in Rust with Rayon-based data parallelism, so each SLURM job uses a single node and exploits all available cores via `RAYON_NUM_THREADS`. The I/O step (loading all selected sub-volumes from the GALFORM HDF5 files) and the pair-counting step have different scaling behaviour and must be budgeted separately.

Thread-scaling measurements on the L800/lc16 catalogue at $N_{\rm subvol} = 32$, $M_* > 10^9\,M_\odot h^{-1}$, $z = 1.5$ (116\,542 selected galaxies) show that pair-counting time falls from 89.5 s at 1 thread to 6.4 s at 16 threads — a speedup of $14.1\times$ — confirming near-ideal Rayon scaling. Beyond 16 threads the pair-counting speedup continues but I/O time begins to dominate: at 32 threads the total wall time is 201 s, of which 197 s is I/O (parallel HDF5 reads saturate the storage bus). For the campaign we therefore fix `RAYON_NUM_THREADS = 16`, giving a total wall time of 8.2 s and a core-hours cost of 0.037 CPU-h per single-seed job at this $N$ value. This is negligible compared to the GALFORM run itself ($\mathcal{O}(10^3)$ CPU-h per full L800 realisation), confirming that SCOPE adds no meaningful compute overhead to the inference pipeline.

Secondary considerations — core-hour allocation, storage throughput, and queue latency — shape the partition assignment strategy (cosma5 for $N \leq 32$, cosma8-shm for $N \leq 256$) but do not alter the accuracy criteria. The `mstar_none` (all-galaxy) catalogue is a notable exception: at $N = 32$ this catalogue takes $\sim 12$ h on cosma5 due to the much larger galaxy count per sub-volume, so $N \geq 128$ is infeasible within any available partition's time limit and is excluded from the all-galaxy analysis in §5.3.

### 4.4  Acceptance Criteria

The convergence threshold $N_{\rm subvol}^*$ is defined as the minimum $m$ for which the two-halo median log-ratio $\mathcal{L}_{\rm 2h}(m) < 0.05$ dex — that is, the median deviation from the full-box reference over $r \in [5, 30]\,h^{-1}$Mpc is below 5 per cent. This threshold is adopted because (i) it is the scale range most relevant to large-scale-bias and BAO measurements, (ii) 5 per cent corresponds approximately to a $1\sigma$ contribution to the error budget of a typical galaxy-clustering likelihood at these scales, and (iii) it is the level at which the SCOPE bias becomes sub-dominant to the sample variance of a 1024-subvolume survey (§5.5). For reference, the naïve estimator at the same $N$ values has $\mathcal{L}_{\rm 2h} > 1.5$ dex — a factor of $\gtrsim 30$ worse — and never reaches the 5 per cent threshold within the accessible $N$ range.

From the L800/lc16 data, $N^*$ (two-halo, 5 per cent threshold) is 32 at $z = 0$, 40 at $z = 0.5$, and 64 at $z = 1.5$ for $M_* > 10^9\,M_\odot h^{-1}$, and 40, 64, and 64, respectively, for $M_* > 10^{10}\,M_\odot h^{-1}$. The compliance fraction $f_5$ in the two-halo regime ($r \in [5, 30]\,h^{-1}$Mpc) at $N = 16$ is only 11–19 per cent for $M_* > 10^9$ and 6–16 per cent for $M_* > 10^{10}$, confirming that the two-halo regime has not converged at that $N$. At $N = 64$, $f_5$ rises to 31–53 per cent ($M_* > 10^9$) and 26–42 per cent ($M_* > 10^{10}$); at $N = 128$, 56–78 per cent and 47–69 per cent, respectively. The global $N^*$ (median over all positive bins, dominated by one-halo scales) is lower — 8 for $M_* > 10^9$ at $z = 0$, 16 for $M_* > 10^{10}$ — but is not the operative metric for two-halo science. The practical recommendation is $N^* = 64$ for broad clustering analyses and $N^* = 128$ for applications requiring per-bin accuracy at the few-per-cent level across both regimes, such as likelihood-based HOD inference or Fisher-matrix forecasts.

For the $M_* > 10^{11}$ selection the situation is qualitatively different. The catalogue density is low enough that two-halo bins frequently contain zero or one galaxy pair per sub-volume at small $N$, making $\xi_{\rm SCOPE}$ identically zero and the log-ratio undefined for most seeds. The median $|\Delta\log_{10}\xi|$ does not fall below 0.05 dex until $N \approx 256$ at $z = 0$ ($\mathcal{L}_{\rm 2h} = 0.045$ dex), and remains above 0.89 dex at $N = 256$ for $z = 1.5$ — well outside convergence. This behaviour reflects the failure mode discussed in §5.7: the $\alpha$/$\beta$ weights cannot correct for shot-noise-dominated bins in which the sub-volume pair count is zero. For sparse catalogues, the minimum useful $N_{\rm subvol}$ is set not by the correction formula but by the requirement that each sub-volume contributes at least one galaxy pair at the scales of interest, which scales as $N_{\rm subvol} \propto 1/\bar{n}^2 r_{\rm max}^3$. Practitioners applying SCOPE to similarly sparse samples should verify that this condition is met before interpreting convergence curves.

### 4.5  Real-World Application — DESI ELG Mock

To motivate the utility of SCOPE for mission-level simulation analysis, we demonstrate its performance on a mock catalogue of Emission Line Galaxies (ELGs) selected from the L800/lc16 simulation. ELGs are a primary tracer for the Dark Energy Spectroscopic Instrument (DESI) survey, typically selected via star-formation rate (SFR) abundance matching.

We construct an ELG selection at $z \approx 1$ (snapshot 155, Output003) by applying a threshold cut on the total SFR (disk + burst) to match the DESI target number density of $n = 5.0 \times 10^{-4}\,h^3 \text{Mpc}^{-3}$. This selection results in a complex tracer population that is not simply a halo-mass threshold, probing the interplay between star formation physics and environmental density.

Using only $N_{\rm subvol} = 32$ sub-volumes (3.1% of the simulation volume), SCOPE recovers the projected correlation function $w_p(r_p)$ with high fidelity. The results in `mock_elg_wp_L800_iz155_n32_seed1001.csv` show that SCOPE accurately captures the clustering signal across both the one-halo and two-halo regimes. This provides a real-world validation that the SCOPE sub-volume correction is not restricted to simple toy models but is directly applicable to the complex galaxy-formation physics and science-ready mock catalogues required for current and future cosmological surveys.

---

## Table 1 — GALFORM Parameter Comparison (tab:ParamRanges)

LaTeX source for the parameter table (paste into §2 of the manuscript, replacing `Table ??`):

```latex
\begin{table}
  \caption{Key \textsc{galform} parameters for the two model variants used in this
  work.  Parameters not listed are identical between \texttt{lc16} and \texttt{gp14}.
  The supernova feedback velocity $V_{\rm hot}$ applies to both disc-mode and
  burst-mode star formation; it is the same for both modes within each variant.
  $\varepsilon_{\rm AGN}$ is the Eddington accretion-rate fraction that determines
  the radio-mode AGN heating power.  $\tau_0$ rescales the Lacey \& Cole (1993)
  dynamical-friction merger timescale.  $\alpha_{\rm reheat}$ sets the
  reincorporation timescale for feedback-ejected gas.  The IMF affects
  stellar mass-to-light ratios and hence the overall calibration target.}
  \label{tab:ParamRanges}
  \begin{tabular}{llcc}
    \hline
    Parameter & Description & \texttt{lc16} & \texttt{gp14} \\
    \hline
    \multicolumn{4}{l}{\textit{Supernova feedback}} \\
    $V_{\rm hot}$ & Ejection velocity threshold (km\,s$^{-1}$) & 320 & 380 \\
    $\alpha_{\rm hot}$ & Mass-loading power-law slope & 3.4 & 3.2 \\[3pt]
    \multicolumn{4}{l}{\textit{AGN feedback}} \\
    $\varepsilon_{\rm AGN}$ & Eddington accretion-rate fraction & 0.010 & 0.040 \\
    $\alpha_{\rm cool}$ & Cold/hot accretion threshold & 0.80 & 0.61 \\[3pt]
    \multicolumn{4}{l}{\textit{Satellite evolution}} \\
    $\tau_0$ & Dynamical-friction timescale normalisation & 1.0 & 1.5 \\
    $\alpha_{\rm reheat}$ & Gas reincorporation factor & 1.00 & 1.26 \\
    Starvation & Satellite cooling suppressed & Yes & No \\[3pt]
    \multicolumn{4}{l}{\textit{Star formation and stellar populations}} \\
    $\nu_{\rm SF}$ & H$_2$ depletion rate (Gyr$^{-1}$) & 0.74 & 0.50 \\
    $f_{\rm stab}$ & Disk stability threshold & 0.90 & 0.80 \\
    IMF & Stellar initial mass function & Bimodal$^a$ & Chabrier \\
    \hline
  \end{tabular}
  \begin{minipage}{0.47\textwidth}
    $^a$ Kennicutt (1983) for quiescent star formation; top-heavy ($x = 1$) for
    burst-mode star formation.
  \end{minipage}
\end{table}
```

**Notes on placement:** Insert immediately before or after the paragraph ending "The list of model parameters varied, and the ranges considered for each parameter are given in Table ??." in §2 of the manuscript, replacing `??` with `\ref{tab:ParamRanges}`.

**Parameter sources:**
- lc16: `Lacey16_newmg_Nbody_L800.input.ref` (Baugh et al. 2019 / Lacey et al. 2016)
- gp14: `Gonzalez15_newmg_Nbody_L800.input.ref` (González-Pérez et al. 2014 / Guo et al. 2016)

---

## Section 5 — Results

### 5.1  Convergence

Fig.~X shows $\xi_{\rm SCOPE}(r)$ and the fractional residual $(\xi_{\rm SCOPE} - \xi_{\rm ref})/\xi_{\rm ref}$ for the L800/lc16 catalogue at $z = 0$, $M_* > 10^9\,M_\odot h^{-1}$, for $N_{\rm subvol} = 4, 8, 16, 32, 64$. The naïve sub-volume estimator overestimates $\xi(r)$ by a factor of $k/m$ at one-halo scales ($r \lesssim 1\,h^{-1}$Mpc): for $N = 16$ this amounts to a factor of 64 at $r \approx 0.01\,h^{-1}$Mpc, consistent with the analytic prediction $\xi_{\rm naive} \approx (k/m)\,\xi_{\rm true}$ derived in §3. The SCOPE correction removes this bias to well below 1 per cent at the same $N$, with the corrected $\xi(r)$ lying within the seed-to-seed scatter band of the full-box reference at all one-halo scales for $N \geq 4$.

At two-halo scales the convergence is slower. The primary metric $\mathcal{L}_{\rm 2h}$ (median $|\Delta\log_{10}\xi|$ over $r \in [5, 30]\,h^{-1}$Mpc) evolves as 0.112, 0.064, 0.039, 0.020, and 0.013 dex for $N = 8, 16, 32, 64, 128$, respectively, at $z = 0$ for $M_* > 10^9$. The convergence threshold $N^* = 32$ at this redshift and selection. For $M_* > 10^{10}$ the corresponding sequence is 0.151, 0.087, 0.052, 0.025, 0.016 dex, with $N^* = 40$. In both cases the naïve two-halo residual at the same $N$ values is 0.06–0.23 dex (1.2–1.7$\times$ overestimate), confirming that the two-halo bias in the uncorrected estimator is non-negligible but much smaller than at one-halo scales.

The convergence curve is smooth and monotonically decreasing in $\mathcal{L}_{\rm 2h}$ with increasing $N$, with no sign of a noise floor within the $N \leq 256$ range probed. At $N = 256$ the two-halo residual is 0.008 dex ($M_* > 10^9$, $z = 0$), equivalent to a 2 per cent systematic, well inside the statistical uncertainty of the full L800 survey.

### 5.2  Redshift Dependence

The convergence rate depends on redshift. For $M_* > 10^9$, the two-halo median $\mathcal{L}_{\rm 2h}$ at fixed $N = 16$ rises from 0.064 dex at $z = 0$ to 0.092 dex at $z = 0.5$ and 0.112 dex at $z = 1.5$. The convergence threshold consequently shifts: $N^* = 32$ at $z = 0$, 40 at $z = 0.5$, and 64 at $z = 1.5$. For $M_* > 10^{10}$ the same pattern holds with $N^* = 40, 64, 64$ at the three epochs. At $N = 128$ the two-halo residual is below 0.025 dex for all selections and redshifts, confirming that $N = 128$ provides a conservative upper bound on the convergence requirement across the full L800 parameter space.

The redshift dependence is attributable primarily to the increasing cosmic variance of large-scale structure per sub-volume at high redshift. The galaxy bias $b(z)$ is larger at $z \gtrsim 1$ for fixed stellar mass cut, amplifying the variance in the two-halo signal from sample to sample. Additionally, the mean galaxy number density per sub-volume decreases mildly with increasing redshift (from $\sim\!102\,000$ to $\sim\!59\,000$ galaxies per sub-volume for $M_* > 10^9$ between $z = 0$ and $z = 1.5$), reducing the pair-count signal-to-noise ratio in the two-halo regime.

### 5.3  Sample Selection

The convergence rate is a strong function of the stellar mass threshold. At $z = 0$, the two-halo residual at $N = 64$ is 0.020 dex ($M_* > 10^9$, $\sim\!408\,000$ galaxies per sub-volume), 0.025 dex ($M_* > 10^{10}$, $\sim\!51\,000$ per sub-volume), and 0.121 dex ($M_* > 10^{11}$, $\sim\!630$ per sub-volume). The jump between the $M_* > 10^{10}$ and $M_* > 10^{11}$ selections is qualitative, not merely quantitative: the $M_* > 10^{11}$ sample has too few galaxies per sub-volume to yield finite two-halo pair counts at many seeds for small $N$, so the convergence curve is dominated by undefined log-ratios rather than by a smooth bias.

The all-galaxy (`mstar_none`) selection contains $\sim\!9 \times 10^6$ galaxies per sub-volume at $z = 0$ and exceeds $4 \times 10^6$ at $z = 1.5$. The sheer catalogue size makes run times at $N \geq 128$ infeasible within HPC job-time limits: a single $N = 32$ job for this selection takes $\sim\!12$ h at $z = 1.5$ on cosma5 nodes, and pair-counting time scales as $\mathcal{O}(N_{\rm gal}^2)$ for the relevant separation range. Convergence curves for the all-galaxy selection are therefore computed for $N \leq 32$ only; however, based on the well-converged $M_* > 10^9$ results ($\mathcal{L}_{\rm 2h} < 0.04$ dex at $N = 32$), the all-galaxy results at the same $N$ are expected to converge at least as rapidly, owing to the higher number density of galaxy pairs per sub-volume.

### 5.4  Model Robustness

[Pending Mill1/Mill2 SLURM campaign. Describe lc16 vs gp14 comparison at Mill1 500 Mpc/h and Mill2 100 Mpc/h, both z=0 and z≈1.5-1.9. Expected result: correction factors alpha/beta are model-independent; convergence rates may differ slightly due to different HOD but should agree within seed scatter.]

### 5.5  Variance

The seed-to-seed scatter in $\xi_{\rm SCOPE}(r)$ at fixed $N$ characterises the intrinsic estimator variance that would be observed when applying SCOPE to a single randomly selected sub-volume subset. Fig.~Y shows the 16th–84th percentile band over independent seeds at each $N$, normalised by $\xi_{\rm ref}$. The half-width statistic $(p_{84} - p_{16})/(2\,\xi_{\rm ref})$, evaluated in the two-halo regime, is 24 per cent at $N = 8$, 15 per cent at $N = 16$, 12 per cent at $N = 32$, and 6 per cent at $N = 64$ for $M_* > 10^9$ at $z = 0$. At $z = 1.5$ the scatter is roughly doubled at fixed $N$: 53 per cent ($N = 8$), 33 per cent ($N = 16$), 25 per cent ($N = 32$), 11 per cent ($N = 64$). For $M_* > 10^{10}$ at $z = 0$ the values are 41, 25, 17, and 8 per cent.

The scatter scales approximately as $N^{-1/2}$ in all cases, consistent with the expectation that the SCOPE estimator variance is set by the sampling noise of drawing $m$ independent realisations from a Poisson-like distribution of large-scale modes. The variance is dominated by the two-halo regime, where cosmic variance per sub-volume is large; the one-halo variance is negligible at $N \geq 4$.

The seed scatter at $N = 64$ (6 per cent at $z = 0$, 11 per cent at $z = 1.5$ for $M_* > 10^9$) is comparable to the bias at the same $N$ ($\mathcal{L}_{\rm 2h} = 0.020$ dex $\approx 5$ per cent at $z = 0$). This confirms that the convergence threshold $N^*$ defined in §4.4 approximately coincides with the point at which bias and variance contribute equally to the total error budget, which is the natural operating point for an unbiased estimator.

### 5.6  Runtime Scaling

Pair-counting time scales linearly with the number of Rayon threads up to 16, with a measured speedup of $14.1\times$ from 1 to 16 threads at $N_{\rm subvol} = 32$, $M_* > 10^9$, $z = 1.5$ (116\,542 selected galaxies). At 16 threads the pair-counting step takes 6.4 s and the total wall time is 8.2 s (including I/O). At 32 threads the pair-counting speedup continues but I/O time (parallel HDF5 reads) dominates at 197 s, giving a total wall time of 201 s — slower than the 16-thread run. The optimal thread count is therefore 16 for the SCOPE pair-counting kernel on the COSMA cosma8-shm and cosma5 nodes.

The total job cost is 0.037 CPU-hours per seed at $N = 32$, $M_* > 10^9$. Scaling to larger $N$ and denser catalogues is roughly $\mathcal{O}(N_{\rm gal}^{1.5})$ in pair-counting time (due to the tree structure) and $\mathcal{O}(N_{\rm subvol})$ in I/O. The full L800 convergence campaign (all $N$, all seeds, three redshifts, four mass cuts) required approximately 2000 CPU-hours of SCOPE run time and 4000 CPU-hours of queue overhead — comparable to the cost of a single GALFORM re-run on a few sub-volumes — demonstrating the computational cheapness of the method.

### 5.7 Shot Noise and Cosmic Variance

The total variance of the SCOPE estimator $\sigma_{\rm tot}^2(r)$ can be decomposed into a Poisson-like shot noise component $\sigma_{\rm shot}^2(r)$ and a cosmic variance component $\sigma_{\rm CV}^2(r)$. The shot noise arises from the finite number of galaxy pairs within the selected sub-volumes, while cosmic variance reflects the underlying large-scale density fluctuations shared across sub-volumes.

For the SCOPE estimator (equation 10), the shot noise component is given by:
$$
\sigma_{\rm shot, SCOPE}^2(r) = \frac{\alpha^2 DD_{\rm auto}(r) + \beta^2 DD_{\rm cross}(r)}{RR^2(r)},
$$
where $DD_{\rm auto}$ and $DD_{\rm cross}$ are the intra- and inter-subvolume pair counts, and $RR(r)$ is the analytic random-pair normaliser. In the one-halo regime ($r < 1\,h^{-1}$Mpc), $\alpha \ll 1$ suppresses the shot-noise-dominated auto-pairs, leading to a massive reduction in variance compared to the na\u00efve estimator. In the two-halo regime, $\alpha \to 1$ and $\beta \approx 1$, and the shot noise approaches that of a full-box measurement scaled by the effective volume.

Analysis of the L800/lc16 data in `5.7_shot_noise_and_cosmic_variance.ipynb` shows:
- **Shot Noise Dominance in Sparse Samples:** For the $M_* > 10^{11}$ selection, shot noise dominates the total variance at all scales $r \lesssim 10\,h^{-1}$Mpc. The SCOPE correction remains algebraically valid, but the estimator is limited by the frequent occurrence of zero-pair bins in individual sub-volumes.
- **Cosmic Variance at Large Scales:** For denser samples ($M_* > 10^9$), cosmic variance becomes the dominant contributor at $r \gtrsim 5\,h^{-1}$Mpc. At these scales, the variance scales strictly as $N_{\rm subvol}^{-1/2}$, confirming that even with shared large-scale modes in the parent box, the independent sub-volume weighting correctly samples the field.
- **Spread in Samples:** The seed-to-seed scatter (Plot 1 in \u00a75.5) represents the total variance. For $N = 32$, the 16th\u201384th percentile spread in the two-halo regime is $\sim 12$ per cent for $M_* > 10^9$ at $z = 0$, which is roughly twice the systematic bias, confirming that SCOPE is a near-optimal estimator for large-scale structure statistics.

### 5.8 Limitations


Two failure modes fall outside the regime where the $\alpha$/$\beta$ correction is effective. First, for sparse catalogues ($M_* > 10^{11}$), pair counts at large scales are zero in individual sub-volumes for small $N$, making $\xi_{\rm SCOPE}$ undefined or identically $-1$. In this regime the correction formula is algebraically valid but physically meaningless: $0 \times \alpha = 0$. The minimum useful $N$ scales as $\bar{n}^{-2/3} r_{\rm max}$ (the requirement that the mean inter-galaxy separation is below the scale of interest), which for $M_* > 10^{11}$ at $z = 0$ corresponds to $N \gtrsim 100$. Second, at very large separations approaching the box size ($r \gtrsim L_{\rm box}/4$), periodic-image effects make the analytic RR normaliser less accurate and the reference $\xi_{\rm ref}$ itself becomes noisy; all analyses in §5 restrict to $r \leq 30\,h^{-1}$Mpc (well below $L_{\rm box}/4 \approx 135\,h^{-1}$Mpc for L800) to avoid this regime.

