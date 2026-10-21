# Contributing to MechAudit

Thank you for improving MechAudit. Economic semantics, numerical verification, stable labels,
and exported evidence are all part of the public contract. A change is not complete when only
the solver output is correct.

## Before opening a change

- Use the repository issue tracker to discuss substantial features, API changes, or new
  economic semantics before investing in an implementation.
- Search existing issues and pull requests to avoid duplicate work.
- Do not disclose suspected vulnerabilities in a public issue; follow [SECURITY.md](SECURITY.md).
- Follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) in all project spaces.

## Development setup

MechAudit supports Python 3.10 and newer. Create an isolated environment, then install the
development and optional xarray dependencies:

```console
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev,xarray]"
```

Activate the environment if desired, or invoke its Python executable directly. On POSIX the
executable is `.venv/bin/python`; on Windows it is `.venv\Scripts\python.exe`.

## Make a focused change

1. Branch from the current default branch.
2. Keep unrelated formatting or refactoring out of the change.
3. Add tests at the semantic boundary affected by the change.
4. Update the API reference, methodology, examples, and changelog when behavior visible to users
   changes.
5. Run the focused tests while iterating, followed by the full local checks below.

Externally visible inequalities use canonical `lhs - rhs >= 0` form. Preserve stable semantic
identifiers through solver transformations, retain original economic units in reports, and keep
fixed-mechanism audits separate from objective or optimal-face claims.

## Testing expectations

Tests should cover both the mathematical result and the labels or metadata shown to researchers.
In particular:

- use exact small analytical cases when an expected economic value is known;
- test both successful and malformed labeled inputs;
- check original-unit residuals independently of backend status;
- add a regression test for every bug fix;
- keep property tests deterministic in the ordinary suite;
- make standalone examples explanatory and fast enough for CI;
- do not weaken a tolerance merely to make a numerical test pass.

Optional xarray tests must continue to skip cleanly when xarray is absent. Core functionality
must not acquire an undeclared optional dependency.

## Required local checks

```console
ruff format --check .
ruff check .
mypy -p mechaudit
pytest --cov=mechaudit --cov-report=term-missing --cov-fail-under=85
mkdocs build --strict
python -m build
```

The pre-commit configuration runs the fast formatting and lint checks. Install it with
`python -m pip install pre-commit` if it is not already available, then run:

```console
pre-commit run --all-files
```

The CI matrix remains authoritative for supported Python versions and operating systems.

## Documentation and compatibility

Public objects are selectively exported from `mechaudit`. Avoid adding a new top-level symbol
without documenting its intended stability. During the 0.x alpha series, breaking changes are
possible, but they must be explicit in `CHANGELOG.md`, migration guidance, and the pull-request
description. Report JSON schema changes require separate consideration from Python API changes.

Code snippets must be runnable or clearly marked as pseudocode. Prefer linking to a tested
example over duplicating a long program in several pages.

## Pull-request checklist

- [ ] The change is focused and its motivation is explained.
- [ ] New behavior and failure modes have tests.
- [ ] Economic labels, signs, timing, and units were checked.
- [ ] Documentation and changelog entries are updated where needed.
- [ ] Formatting, lint, typing, tests, strict docs, and package build pass locally.
- [ ] No generated build, coverage, environment, or documentation-site files are included.

Maintainers may ask for a smaller change or additional analytical evidence before review. See
[MAINTAINERS.md](MAINTAINERS.md) for the current governance model.
