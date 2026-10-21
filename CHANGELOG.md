# Changelog

All notable changes to MechAudit are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) during the 0.x alpha series.

## [Unreleased]

## [0.1.0] - 2026-10-22

Initial public release.

### Added

- Finite environments with scalar or structured type spaces, explicit social outcomes,
  independent or correlated joint priors, and private or interdependent quasi-linear values.
- Fixed-mechanism audits for DSIC, ex-post IC, BIC, and ex-post, interim, and ex-ante individual
  rationality, together with payment, budget, capacity, symmetry, and custom linear requirements.
- Randomized mechanism design as sparse linear programs solved with HiGHS (`highspy` 1.11 or
  newer); every returned optimum is checked against the unscaled model and audited again in
  economic units.
- Revenue, welfare, residual-surplus, weighted-utility, and custom linear objectives.
- Optimal-face activity classification, slack and dual ranges, redundancy, ablation, and
  essentiality diagnostics, with optional process-based parallelism.
- One stable identifier per constraint across audit reports, compiled LP rows,
  selected-solution activity, and optimal-face reports.
- Numerical-health reports and conditional-belief (Crémer–McLean) diagnostics.
- Mechanism import from mappings, callbacks, NumPy, pandas, and optional xarray; deterministic
  JSON reports with a published schema, labeled pandas exports, and domain, mechanism, and model
  fingerprints.
- Documentation site, tested examples, an executable Jupyter workflow notebook, and a
  reproducible smoke benchmark.

[Unreleased]: https://github.com/studio-strata/mechaudit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/studio-strata/mechaudit/releases/tag/v0.1.0
