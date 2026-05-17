# Science Investigation: Probing the Dynamical Friction Timescale

This investigation uses two key statistics to probe the impact of the dynamical friction timescale ($	au_0$) in GALFORM simulations:

1.  **The Halo Occupation Distribution (HOD):** $\langle N_{gal} | M_h 
angle$
2.  **The Central Stellar-to-Halo Mass Relation (SHMR):** $M_*$ vs. $M_h$ for central galaxies.

## Scientific Objectives

By comparing simulation runs with varying $	au_0$ values (e.g., $	au_0=0$, $	au_0=\infty$, and a default), we aim to:

1.  **Quantify the impact of merging:** Determine how the instantaneous merging of satellites ($	au_0=0$) versus the complete absence of merging ($	au_0=\infty$) affects the satellite HOD and the stellar mass buildup of central galaxies.
2.  **Identify the mass regime of interest:** Isolate the halo mass regime where dynamical friction plays a significant role in galaxy assembly.
3.  **Validate the model:** Use these statistics to validate the implementation of dynamical friction within GALFORM and understand its consequences for observable galaxy properties.

## Notebooks

*   **`dynamical_friction.ipynb`**: Contains the combined analysis of both the HOD and the central SHMR.
