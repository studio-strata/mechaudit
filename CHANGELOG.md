# Changelog

All notable changes to MechAudit are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) during the 0.x alpha series.

## [Unreleased]

### Added

- User, contributor, maintainer, security, and release documentation.
- Artifact-install, minimum-dependency, documentation, and trusted-publishing workflows.
- Issue and pull-request templates, dependency-update configuration, and a reproducible smoke
  benchmark.
- Initial finite-environment, mechanism-audit, and randomized-design APIs.
- DSIC, ex-post IC, BIC, and ex-post/interim/ex-ante IR checks.
- Revenue, welfare, residual-surplus, weighted-utility, and custom linear objectives with a
  direct HiGHS backend.
- Optimal-face activity, redundancy, ablation, and conditioning diagnostics.
- Deterministic report serialization, labeled pandas exports, and tested examples.
- Spawned-process optimal-face task execution with workload-aware automatic parallelism.
- Public outside-option convenience classes and adversarial contract tests.
- A clean, executable Jupyter notebook covering audit, design, optimal-face analysis, candidate
  assessment, and report exports.

### Changed

- Made API scope, limitations, and error messages release-neutral.
- Made the Ruff pre-commit hooks self-contained and version-pinned.
- Clarified release, security, conduct, and project documentation for maintainers and
  contributors.
- Hardened repository and source-distribution exclusions for generated and local-only artifacts.
- Expanded installation guidance for package-index, release-wheel, source, and contributor
  workflows.
- Hardened CI with immutable Action pins, bounded runtimes, concurrency control, and an explicit
  85% coverage floor.
- `ExpectedResidualSurplus` is now unweighted; use `WeightedAgentUtilities` for weighted agent
  utilities.
- `FiniteEnvironment` accepts valuations only through `values=`.
- The minimum supported `highspy` is now 1.11, the first release that reports the solution-quality
  statistics used to certify optimal-face solves; older versions are rejected with a clear error.
- `DesignProblem.compile()` no longer takes a `probability_tolerance` argument, which had no
  effect on the compiled model.
- Model compilation is several times faster, and fixed-mechanism audits about four times faster,
  on domains with thousands of profiles.

### Fixed

- Audit checks, compiled LP rows, selected-solution activity, and optimal-face records now use
  one identifier per constraint, so identifiers from an audit report can be passed to the
  redundancy, cost, and essentiality diagnostics.
- Audits under independent or one-agent priors no longer warn about conditional-belief rank.
- Candidate assessment audits with the tolerances of the optimum it is compared against, and no
  longer reports violated constraints as active.
- `JointPrior` lookup accepts mappings, ordered tuples, and one-agent type labels, and `align`
  reports a missing type space as a `PriorValidationError`.
