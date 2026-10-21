# MechAudit

MechAudit is a Python research workbench for finite direct mechanisms. It preserves the
economic identity of every constraint while it:

- audits supplied allocation and payment rules;
- designs randomized mechanisms with sparse linear programming;
- separates activity at one returned solution from activity across a declared optimal or
  near-optimal face;
- measures the effect of removing a constraint or an economic family;
- reports numerical conditioning in economically interpretable units.

!!! warning "Research software"

    MechAudit is alpha research software. All solver and optimal-face conclusions are
    floating-point and tolerance-relative, not formal certificates.

## Three workflow boundaries

### Fixed-mechanism audit

`FiniteEnvironment.audit(...)` returns a `MechanismAuditReport`. It checks allocation
validity, declared IC/IR/budget conditions, epsilon-IC, rents, and deviation witnesses. A fixed
audit has no objective and therefore no optimal-face methods.

### Design and diagnose

`FiniteEnvironment.design(...).solve()` returns a certified `DesignSolution`: a mechanism,
objective value, solver evidence, and a fresh independent economic audit. Calling
`solution.analyze_optima(...)` returns an `OptimalFaceReport` with universal/nonuniversal and,
on demand, sometimes/never classifications.

### Candidate assessment

`DesignProblem.assess(...)` returns a `CandidateAssessment`. It aligns an imported mechanism
to the reference domain, audits feasibility, computes maximize-convention objective gaps, and
keeps candidate-local activity separate from invariant activity.

## Supported model

MechAudit supports finite agents and outcomes, scalar or structured finite types, independent or
correlated joint priors, private or interdependent values, quasi-linear utility, and
report-contingent payments. Built-ins include:

- DSIC, ex-post IC, and BIC;
- ex-post, interim, and ex-ante IR;
- payment bounds, no subsidies, and ex-post/ex-ante budget balance;
- capacity, explicit symmetry actions, and custom labeled linear rows;
- expected revenue, welfare, residual surplus, weighted utility, and custom linear objectives;
- optimal-face slack ranges, multiplier ranges, redundancy, essentiality, and family ablation;
- conditional-belief and Crémer–McLean rank/conditioning diagnostics.

Deterministic-mechanism optimization, continuous spaces, indirect or dynamic mechanisms, and
exact rational certificates are outside the supported scope. Fixed deterministic mechanisms
remain fully auditable.

## Start here

Install MechAudit using [Installation](installation.md), then follow
[Getting started](getting-started.md) for a runnable two-type buyer. Continue with:

- [Audit a mechanism](auditing.md) for information and timing semantics;
- [Design a mechanism](designing.md) for compilation and solver behavior;
- [Optimal-face analysis](optimal-face.md) before interpreting activity labels;
- [Numerical diagnostics](numerical-health.md) for scaling and conditional-belief checks;
- [Benchmarks](benchmarks.md) for reproducible smoke measurements;
- [API reference](api.md) for the supported public surface.
- [Project and support](project.md) for compatibility, citation, governance, and reporting.

The package-wide payment convention is fixed: a positive payment is made **from an agent to the
designer**.
