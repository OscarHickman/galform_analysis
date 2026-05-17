# SCOPE RSD Multipoles and Kaiser Model Validation

This notebook demonstrates the use of the `compute_xi_smu` function within the SCOPE framework to measure the redshift-space correlation function $\xi(s, \mu)$.

## Objectives
1.  **Measurement:** Compute the redshift-space correlation multipoles—the monopole $\xi_0(s)$ and quadrupole $\xi_2(s)$—from simulation data.
2.  **Validation:** Compare the measured multipoles against the linear-theory Kaiser prediction (assuming plane-parallel, distant-observer approximation) to validate the effectiveness of the sub-volume correction in capturing the underlying clustering signal while mitigating artificial anisotropic smearing.
3.  **Analysis:** Investigate the impact of different sub-volume configurations on the recovery of the RSD signal.
