# Security policy

MechAudit is a local numerical research library. Ordinary use makes no network requests, but
valuation and mechanism callbacks execute Python supplied by the caller in the current process.
Do not run untrusted callbacks or load untrusted pickle files. JSON reports are numerical exports;
they are not executable and cannot round-trip callbacks.

## Supported versions

| Version | Security support |
| --- | --- |
| `0.1.x` | Supported |
| Earlier development snapshots | Not supported |

Only the latest patch release in a supported minor series receives security fixes during the
alpha period.

## Report a vulnerability privately

Use this repository's
[private vulnerability reporting form](https://github.com/studio-strata/mechaudit/security/advisories/new).
Include:

- the affected MechAudit and Python versions;
- operating system and relevant dependency versions, especially `highspy`;
- a minimal reproduction or proof of concept;
- expected impact and any known workaround;
- whether the issue is already public.

Do not open a public issue containing exploit details or sensitive research data.

If the private form is unavailable, do not disclose the vulnerability publicly. A public issue
may request a private contact channel, but it must contain no exploit details, sensitive data, or
identifying information.

## Response and disclosure

The maintainer will acknowledge a usable private report, assess affected versions, coordinate a
fix and release where appropriate, and credit the reporter unless anonymity is requested. Exact
response times are not promised for this alpha research project. Please allow coordinated
disclosure until a fix or mitigation is available.

Dependency vulnerabilities in NumPy, pandas, SciPy, HiGHS, or optional xarray should also be
reported to the affected upstream project when the issue is not specific to MechAudit.
