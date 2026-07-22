# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2026-07-22

### Removed

- The deprecated typo aliases introduced in 0.2.0: the
  `banditry.optimisation_subroutines.contextal_problem` shim module (import
  `contextual_problem` instead) and `DesignSpace.register_parmeter_type`
  (use `register_parameter_type`).

## [0.3.0] - 2026-07-22

### Added

- **Documentation site** at <https://vahanarsenian.github.io/banditry/>:
  getting-started, guides (design spaces, contextual bandits, choosing an
  agent, configuring samplers), benchmark CLI reference, and a full
  mkdocstrings API reference; deployed automatically from `main`.
- **Docstrings across the public API**, including field-by-field documentation
  of `OFUGPConfig`/`TSConfig` and every key of `DEFAULT_LANGEVIN_CONFIG` and
  `DEFAULT_NUTS_CONFIG`.
- **`py.typed` marker** and type annotations on the core loop API
  (`suggest`/`observe`/`quasi_sample`/`get_best_id`), so type checkers and
  IDEs pick up the library's inline types.
- **`examples/` directory** with four seeded, CPU-only scripts: quickstart
  (Branin), mixed design space, contextual bandit, and a TS vs Feel-Good TS
  comparison that generates the README plot.
- `docs` extra (`pip install "banditry[docs]"`) and `Documentation` project
  URL.

## [0.2.0] - 2026-07-22

### Changed

- **Relicensed from CC BY-NC-SA 4.0 to MIT.** HEBO-derived portions remain
  MIT-licensed upstream; attribution moved to the new `NOTICE` file.
- The benchmark runner moved from the repository-root `main.py` into the
  package (`banditry.benchmark`) and is installed as the `banditry-bench`
  console script.
- Renamed `banditry.optimisation_subroutines.contextal_problem` to
  `contextual_problem`. The old module still imports with a
  `DeprecationWarning`.
- Renamed `DesignSpace.register_parmeter_type` to `register_parameter_type`.
  The old name still works with a `DeprecationWarning`.

### Added

- Public API re-exports: `from banditry import OFUGPAgent, TSAgent,
  AbstractAgent` now works, and every subpackage re-exports its main classes
  (e.g. `banditry.sampling_oracles.LangevinSampler`).
- Test suite (pytest): design-space/parameter unit tests plus seeded
  suggest/observe smoke tests for all four agent variants.
- CI workflow (lint + tests on Python 3.10 and 3.13) and ruff
  lint/format configuration; `dev` extra with pytest and ruff.
- `CITATION.cff`, `CONTRIBUTING.md`, and this changelog.

## [0.1.0] - 2026-07-22

### Added

- Initial release: `OFUGPAgent` (GP-UCB with exact/sparse-variational GP
  surrogates), `TSAgent` (neural Thompson sampling with Langevin or NUTS
  posterior sampling, optional Feel-Good reweighting), mixed design spaces
  (numeric/int/bool/categorical), evolutionary acquisition optimisation
  (pymoo), and contextual bandit support via `fix_input`.

[0.3.1]: https://github.com/VahanArsenian/banditry/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/VahanArsenian/banditry/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/VahanArsenian/banditry/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/VahanArsenian/banditry/releases/tag/v0.1.0
