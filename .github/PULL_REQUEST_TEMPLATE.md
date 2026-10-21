## Summary

Describe the problem, the chosen approach, and any alternatives considered.

## Economic and numerical contract

Explain affected information assumptions, timing, signs, units, tolerances, semantic row IDs,
and report fields. State explicitly when none are affected.

## Verification

List focused analytical tests and full checks that were run.

## Checklist

- [ ] The change is focused and linked to an issue or prior discussion when substantial.
- [ ] New behavior and failure modes have tests.
- [ ] Solver results are independently checked in original economic units.
- [ ] Public API, methodology, examples, and changelog are updated where needed.
- [ ] JSON/report-schema compatibility was considered.
- [ ] `ruff format --check .`, `ruff check .`, `mypy -p mechaudit`, full tests with coverage,
      strict MkDocs, and the package build pass.
- [ ] No generated environments, sites, coverage files, or build artifacts are included.
