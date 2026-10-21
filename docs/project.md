# Project and support

## Status and compatibility

MechAudit is alpha research software. The project follows Semantic Versioning, but 0.x minor
releases may contain documented breaking changes. Patch releases should preserve documented
Python and report-schema contracts except where a security or correctness fix requires an
explicitly announced change.

Supported Python versions and runtime dependency bounds are declared in `pyproject.toml` and
exercised by the CI matrix. Numerical equivalence always depends on the declared finite model,
dependency versions, tolerances, and solver configuration.

## Get help or report a bug

Use the [GitHub issue tracker](https://github.com/studio-strata/mechaudit/issues) for reproducible
bugs, documentation problems, and feature proposals. Include:

- MechAudit, Python, operating-system, and dependency versions;
- the smallest complete environment and mechanism that reproduces the behavior;
- the exact exception and its `path` or `context`, when present;
- relevant fingerprints, tolerances, and solver options;
- whether the result changes across machines or dependency versions.

Do not put vulnerabilities or sensitive research data in a public issue. Follow the
[security policy](https://github.com/studio-strata/mechaudit/blob/main/SECURITY.md) instead.

## Governance and conduct

MechAudit uses a maintainer-led governance model. Alessio Staffini of Studio Strata,
Tokyo, Japan is the maintainer. Responsibilities and decision practices are documented in
[MAINTAINERS.md](https://github.com/studio-strata/mechaudit/blob/main/MAINTAINERS.md).

Participation is governed by the
[Code of Conduct](https://github.com/studio-strata/mechaudit/blob/main/CODE_OF_CONDUCT.md), and
contributions follow
[CONTRIBUTING.md](https://github.com/studio-strata/mechaudit/blob/main/CONTRIBUTING.md).

## Citation and reproducibility

Use the repository's `CITATION.cff` metadata when citing the software. A computational result
should additionally report:

- the MechAudit and HiGHS versions;
- the domain, mechanism, and model fingerprints that apply;
- the prior, objective, and economic requirements;
- the solver options and tolerance policy;
- the requested and effective near-optimal-face delta;
- any numerical-health warnings or inconclusive auxiliary tasks.

The canonical software citation is maintained in `CITATION.cff`.

## Releases and documentation

Release preparation and host-side trusted-publisher requirements are documented in
[RELEASING.md](https://github.com/studio-strata/mechaudit/blob/main/RELEASING.md). Published
documentation is available at <https://studio-strata.github.io/mechaudit/>. The source repository and
a release artifact remain the authority when versioned documentation and the development branch
differ.
