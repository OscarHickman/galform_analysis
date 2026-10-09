# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/).

## [0.2.0] - unreleased

Several fixes below change numerical results. **Recompute** any subvolume-
weighted xi(r)/w_p(r_p), stacked HMF, theoretical HMF or correlation-function
results made with earlier versions.

### Fixed — results change

- **Subvolume-weighted xi(r) and w_p(r_p)** (`compute_weighted_xi_*`,
  `compute_weighted_wp_*`) were offset by about +1 and +2·pimax. Corrfunc's
  auto-correlation counts each pair twice, but the counts were normalised by
  the number of unique pairs. A uniform random field now gives xi ≈ 0.
- **Correlation functions used the wrong box size.** The periodic box was
  inferred from the extent of the galaxy positions, which is smaller than the
  true box for sparse samples and biased xi(r) low. It is now
  `(V_ivol · n_subvolumes)^(1/3)` from the HDF5 file, or a new `boxsize=`
  argument. This also affects `satellite_central_cross_correlation`.
- **Theoretical HMFs** (`create_theoretical_hmf`, `compute_theoretical_hmfs`):
  - They used hmf's default Planck18 cosmology rather than the L800 one
    (σ8 = 0.8288, Ωm = 0.307, h = 0.6777) assumed everywhere else in the
    module. Abundances at 10^15 M_sun/h change by about 9%.
  - hmf returns each fit in its native mass definition (M200m for Tinker08 and
    PS, Mvir for SMT), but the module treated every fit as M200c. It now
    converts from the native definition.
  - `get_mvir_to_m200c_ratio` returned 0.72–0.91 at z = 0, which is
    impossible because Δ_vir < 200. It now uses an NFW conversion (about 1.2).
  - GPS+ was silently all-NaN on NumPy ≥ 2 (`np.trapz` was removed). Its
    m200b → Mvir conversion is now an NFW conversion, and it uses the L800
    cosmology.
- `avg_hmf_given_redshift_and_subvolumes` normalised phi by one subvolume's
  volume instead of the total volume of the subvolumes used.
- `aggregate_snapshot` now uses only completed subvolumes
  (`CompletionFlag == 1`) with a positive volume. A subvolume whose `mstar`
  and `mhalo` arrays differ in length is skipped with a warning; previously
  the stacked arrays were misaligned.
- The redshift parsed from `zsnap.dat` matched the `z=` inside `iz=` and
  returned the snapshot number.
- A missing derived field (e.g. `mstardot`) made `read_galaxy_arrays` and
  `read_galaxy_positions` silently return zero galaxies.
- Luminosity band lookup preferred the generic `_r` suffix over `sdss_r`.
- `nvol_range` values such as `"0-63"` gave 63 subvolumes instead of 64.
- The analytic random pairs for RSD xi(s, mu) ignored `mu_max`, so
  `compute_direct_rsd_multipoles` and `compute_weighted_direct_rsd_multipoles`
  were biased for `mu_max < 1` (by 2x at `mu_max = 0.5`). The default
  `mu_max = 1` was unaffected.

### Fixed — errors and edge cases

- An empty galaxy selection crashed the correlation functions. Single-
  subvolume calls now return NaN xi and stacked calls skip the empty
  subvolume.
- `compute_xi_corrfunc` accepted a bin edge exactly at L/2, which Corrfunc
  rejects. `attrs["rbins"]` now holds the bins actually used.
- `completed_galaxies` and `incomplete_subvolumes` now read `CompletionFlag`
  (the docstrings said they did).
- Directories like `ivol_old/` no longer crash subvolume scans, and
  subvolumes are listed in numeric order (0, 2, 10, not 0, 10, 2).
- HDF5 files are closed when a read fails part-way, and after each
  correlation-function call.
- A missing optional dependency now raises `ImportError` with the install hint
  everywhere. Previously `create_theoretical_hmf` turned it into a
  `ValueError`, `compute_theoretical_hmfs` into NaN models and
  `matter_xi_at_snapshot` into `None`.
- The HOD works on files without `mhalo` (centrals/satellites split is then
  `None`).
- Non-finite halo IDs no longer produce random `halo_id_hash` labels.
- `compute_galaxy_bias` raises a clear `ValueError` for mismatched bins, and
  `compute_matter_xi` accepts a list of bins.

### Changed

- **No default data directory.** The base directory used to default to a
  path on the Durham COSMA cluster. Call `set_base_dir(path)`, set
  `GALFORM_BASE_DIR`, or pass `base_dir=` explicitly; otherwise
  `get_base_dir()` raises a `RuntimeError` explaining this. Function defaults
  are resolved at call time, so `set_base_dir` after import now takes effect.
- `read_galaxy_arrays` leaves out absent optional fields instead of returning
  empty arrays, and raises `ValueError` for arrays of inconsistent length.
- `completed_galaxies` reports a file with no `CompletionFlag` as incomplete
  (it used to count any readable file as complete).
- `create_theoretical_hmf` returns `mass_ratio` and `native_mass_definition`
  in place of `ratio_mvir_to_m200c`. `get_concentration` is now the Duffy+08
  c200c(M200c) relation, so its values change.
- The correlation helpers raise `ValueError` if galaxy positions do not fit in
  the box derived from the file (for example when `n_subvolumes` is missing
  and the 1024 fallback is wrong); pass `boxsize=` in that case.

### Added

- RSD hexadecapole ξ₄ (`xi4`, `xi4_standard`, `xi4_corrected`, `xi4_naive`)
  from all multipole functions.
- `boxsize=` keyword on the correlation-function helpers, and `base_dir=` on
  the dark-matter correlation helpers.
- `n_subvolumes` in the dictionary returned by `read_snapshot_data`, and
  `ivols` in the one returned by `aggregate_snapshot`.
- Loader helpers `list_subvolume_dirs`, `read_completion_flag` and
  `read_volumes`. `read_galaxies_dataframe` takes `mstar_min`.
- `py.typed` marker, and Python 3.14 support.

### Fixed — packaging

- **Installation from PyPI.** Versions 0.1.6–0.1.9 depended on `sugc`, which
  is not published on PyPI, so they could not be installed and
  `pip install galform_analysis` fell back to 0.1.5. The dependency is gone.
- `import galform_analysis` failed unless the optional `hmf` package was
  installed, because `theoretical_hmf` imported it at module level.
- `SimulationConfig("COLIBRE-L200m6")` raised `FileNotFoundError` with
  `galform_execution` < 0.2.4 installed. The package now requires
  `galform_execution>=0.2.4`.
- The subvolume-weighted and RSD multipole functions were silently left out
  of `galform_analysis.analysis.correlation` if any of their imports failed.
  They are now always exported.

### Changed — packaging and dependencies

- **Corrfunc is now optional.** Install it with
  `pip install "galform_analysis[clustering]"`. Corrfunc compiles from source
  (C compiler, OpenMP and GSL), so the base install now works from wheels
  alone.
- The `science` extra now contains exactly what the code uses: `hmf`,
  `astropy`, `camb`, `colossus` and `scipy`. A new `all` extra installs `clustering` and
  `science`.
- Calling a function whose optional dependency is missing raises an
  `ImportError` that names the extra to install.
- Python 3.10 and 3.11 are supported again (`requires-python >=3.10`, was
  `>=3.12`).
- `scipy` moved from the core dependencies to the `science` extra.

### Removed

- 3-point and N-point correlation functions (`compute_3pcf_counts_with_sugc`,
  `compute_triplet_counts`, `sugc_weights`, `compute_npoint_counts`,
  `sugc_weights_npcf`). They depend on SaUCE/`sugc` and will return once that
  package is public.
- Unused dependencies: `seaborn` (core), and `halotools`, `packaging` and
  `deprecation` (`science` extra).

### Added — testing and release tooling

- Test suite expanded to ~460 tests and 98% coverage, with independent
  reference checks: brute-force pair counts, analytic random pairs, Legendre
  projections and hand-built catalogues with known mass functions.
- CI builds the wheel and runs the tests against the *installed* package, both
  without extras and with `[all]`, so packaging errors cannot hide behind the
  source tree. CI also enforces 80% test coverage and runs `twine check`.
- The release workflow refuses to publish when the git tag and
  `__version__` disagree.
- PyPI metadata: classifiers, keywords, SPDX license expression and a
  changelog link. The source distribution now includes the tests.

## [0.1.5] and earlier

Initial releases. 0.1.6–0.1.9 cannot be installed from PyPI (see above).

[0.2.0]: https://github.com/OscarHickman/galform_analysis/compare/v0.1.9...v0.2.0
[0.1.5]: https://github.com/OscarHickman/galform_analysis/releases/tag/v0.1.5
