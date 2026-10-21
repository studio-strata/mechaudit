# Release process

This document records MechAudit's public release procedure for current and future maintainers.
Administrative credentials and repository-host configuration are managed outside the source
repository.

## Release infrastructure

Tagged releases are built by `.github/workflows/release.yml` and published to PyPI through
trusted publishing. The workflow contains no password or long-lived package-index token. The
`pypi` GitHub environment and the corresponding PyPI trusted-publisher record must identify the
release workflow before publication can succeed.

Documentation is built by `.github/workflows/docs.yml` and deployed through GitHub Pages. The
package metadata, source repository, issue tracker, changelog, and documentation site must use
the canonical project URLs.

## Prepare a release

1. Select a Semantic Versioning-compatible version. During alpha development, breaking changes
   may use a new minor version.
2. Update `src/mechaudit/_version.py` and the `version` field in `CITATION.cff`. Keep
   `project.version` dynamic in `pyproject.toml`; Hatch reads the version from `_version.py`.
3. Add `date-released` to `CITATION.cff` using the intended publication date.
4. Move the accumulated `CHANGELOG.md` entries from `Unreleased` into a dated version section.
   Add comparison links once the corresponding repository tag exists.
5. Document public API changes, report-schema changes, dependency-bound changes, deprecations,
   and migration guidance.
6. Run the complete local validation suite:

   ```console
   ruff format --check .
   ruff check .
   mypy -p mechaudit
   pytest --cov=mechaudit --cov-report=term-missing --cov-fail-under=85
   mkdocs build --strict
   python -m build
   python -m twine check dist/*
   ```

7. Run `python .github/scripts/check_release_version.py vX.Y.Z`.
8. Inspect the wheel and source distribution for unexpected files, credentials, and missing
   license, schema, or typing metadata.
9. Install the wheel in a clean environment and run at least one standalone example against the
   installed artifact.
10. Submit the release changes for review and merge them only after CI passes.

## Publish and verify

1. Create and push the exact tag `vX.Y.Z` from the reviewed release commit.
2. Confirm that the `Release` workflow validates version synchronization, builds and checks both
   distributions, records provenance, and publishes through the trusted publisher.
3. Verify the published package metadata and install the PyPI wheel in a clean environment.
4. Publish release notes from the matching changelog section and link the immutable artifacts
   and provenance.
5. Confirm that the documentation site represents the released version.

## Failure policy

Published package artifacts are immutable. A failed publication must be diagnosed before any
retry. If an artifact may have reached the package index, the version number is incremented and
the incident is recorded in the release notes; a released version number is never reused.
