# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/).

## [0.2.0] - 2026-10-04

### Fixed

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

### Changed

- **Corrfunc is now optional.** Install it with
  `pip install "galform_analysis[clustering]"`. Corrfunc compiles from source
  (C compiler, OpenMP and GSL), so the base install now works from wheels
  alone.
- The `science` extra now contains exactly what the code uses: `hmf`, `camb`,
  `colossus` and `scipy`. A new `all` extra installs `clustering` and
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
- Unused dependencies: `seaborn` (core), and `astropy`, `halotools`,
  `packaging` and `deprecation` (`science` extra).

### Added

- Expanded test suite with independent reference checks (brute-force pair
  counts, analytic random pairs, hand-built catalogues with known mass
  functions).
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
