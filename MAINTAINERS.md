# Maintainers and governance

## Current maintainer

- **Alessio Staffini** — Studio Strata, Tokyo, Japan

The canonical source repository is <https://github.com/studio-strata/mechaudit>. Project issues and
pull requests are the ordinary public coordination channels. Use the private repository reporting
flow described in [SECURITY.md](SECURITY.md) for vulnerabilities and the private flow in
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for conduct concerns.

## Governance model

MechAudit uses a maintainer-led model. The maintainer is responsible for:

- public API, economic semantics, and numerical-method decisions;
- reviewing and merging changes;
- release and security decisions;
- documentation and community-policy stewardship;
- appointing additional maintainers when sustained contributions justify it.

Material design changes should be discussed in a public issue or proposal before implementation.
Decisions should record the alternatives considered, mathematical or numerical evidence, API
compatibility impact, and any unresolved risks. During the 0.x alpha period, the maintainer may
make breaking changes, but must document them and provide migration guidance.

## Conflicts and succession

A maintainer with a personal conflict should seek an independent reviewer before deciding a
conduct, security-credit, or authorship dispute. If maintainership changes, this file, package
metadata, citation metadata, trusted-publisher settings, and security contacts must be updated in
the same release process.
