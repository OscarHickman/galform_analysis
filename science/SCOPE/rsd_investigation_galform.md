# RSD Investigation: Galform and Sub-volume Corrections

This notebook explores Redshift-Space Distortion (RSD) effects in Galform simulations and the application of sub-volume weighted corrections.

## Objectives
1.  **Validation of Corrections:** Validate the `compute_weighted_rsd_multipoles` algorithm implemented in `subvol_weighted_multipoles.py`.
2.  **Mitigation of Bias:** Demonstrate how sub-volume weighting successfully isolates the expected Monopole ($\xi_0$) and Quadrupole ($\xi_2$), bypassing the artificial anisotropic smearing (the "Fingers-of-God" effect) typically introduced by standard sub-volume subsampling in RSD measurements.
3.  **Investigation:** Explore alternative ways of handling RSDs within the Galform simulation context and analyze the impact of weighted corrections on final clustering results.
