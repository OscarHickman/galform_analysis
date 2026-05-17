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
