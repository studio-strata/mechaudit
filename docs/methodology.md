# Methodology and limits

MechAudit is a numerical workbench for **finite, direct, quasi-linear mechanisms**. It combines
three activities that must not be conflated:

1. exhaustive evaluation of a supplied mechanism;
2. linear-program design of a randomized mechanism;
3. auxiliary optimization over a declared near-optimal face.

The library implements established finite mechanism-design inequalities and records their
economic identities. Its outputs are tolerance-relative numerical evidence, not symbolic or
formal proofs.

## Model and sign conventions

Let `N` be the finite agent set, `T_i` agent `i`'s finite type set, `T` their Cartesian
product, and `O` the finite set of explicit social outcomes. A direct mechanism consists of:

```text
x_o(r)   probability of outcome o after report profile r
p_i(r)   payment from agent i to the designer after report profile r
```

Every allocation row must satisfy:

```text
x_o(r) >= 0  for every o
sum_o x_o(r) = 1
```

Outcomes are complete social alternatives. `x_o(r)` is not an independent per-agent
allocation probability unless the user explicitly encoded such alternatives as outcomes.

The sign convention is fixed package-wide:

```text
p_i(r) > 0   means agent i pays the designer
p_i(r) < 0   means the designer subsidizes agent i
```

For a true profile `t` and report profile `r`, define quasi-linear utility:

```text
U_i(t; r) = sum_o x_o(r) v_i(t, o) - p_i(r)
```

Under `PrivateValues`, `v_i(t, o)` is restricted by construction to `v_i(t_i, o)`. Under
`InterdependentValues`, it may depend on the complete true profile. Value callbacks are
eagerly tabulated, so subsequent callback mutation cannot change an existing environment.

## Incentive compatibility

All implemented IC rows compare truthful utility with one unilateral report deviation. Rows
for `r_i' = t_i` are omitted because they are identities.

### Private-value DSIC

`DSIC` requires an environment explicitly declared with `PrivateValues`. For every selected
agent `i`, true type `t_i`, alternative report `r_i'`, and arbitrary other-agent report profile
`r_-i`, MechAudit imposes:

```text
U_i(t_i; (t_i, r_-i)) - U_i(t_i; (r_i', r_-i)) >= 0
```

Expanded using the package sign convention:

```text
sum_o [x_o(t_i, r_-i) - x_o(r_i', r_-i)] v_i(t_i, o)
    - p_i(t_i, r_-i) + p_i(r_i', r_-i) >= 0
```

The other agents' coordinates range over possible reports; they are not interpreted as a
truthful realized profile. Because an interdependent-value deviation needs a complete true
profile, `DSIC` is rejected there rather than silently assigned an ambiguous meaning.

### Ex-post IC

`ExPostIC` supports both information models. For every complete true profile `t` and
alternative report `r_i'`, other agents report truthfully and the same complete true profile is
used to evaluate both utilities:

```text
U_i(t; t) - U_i(t; (r_i', t_-i)) >= 0
```

For private values this is closely related to the usual dominant-strategy inequalities, but
the semantic quantification is different and the constraint IDs retain that distinction.

### Bayesian IC

Let `mu(t)` be the declared joint prior and

```text
m_i(t_i) = sum_(t_-i) mu(t_i, t_-i)
```

be the own-type marginal. For positive `m_i(t_i)`, `BIC` imposes:

```text
sum_(t_-i) mu(t_-i | t_i)
    [U_i((t_i, t_-i); (t_i, t_-i))
     - U_i((t_i, t_-i); (r_i', t_-i))] >= 0
```

The deviating report does not change the conditional belief. The compiler stores the
joint-mass-weighted numerator to avoid dividing every coefficient by a small marginal:

```text
joint_slack = sum_(t_-i) mu(t_i, t_-i) [truthful utility - deviation utility]
reported conditional slack = joint_slack / m_i(t_i)
```

Audit records retain both `joint_weighted_slack` and `conditioning_mass`. Thus epsilon-IC is
reported in conditional utility units rather than being artificially small because a type is
rare.

### Zero-probability conditional rows

The BIC and interim-IR branches apply the declared `ZeroProbabilityPolicy` only when
`m_i(t_i) = 0`. A small positive mass remains a well-defined conditional row and receives a
numerical-health warning when it is at or below `probability_tolerance`:

| Policy | Implemented behavior |
| --- | --- |
| `"error"` | Raise `ZeroProbabilityTypeError`. |
| `"vacuous"` | Do not add an LP row; record a labeled vacuous audit check. |
| `"enforce_ex_post"` | Replace the undefined conditional statement with one profile-wise row for each `t_-i`. |

This is a semantic choice, not an automatic limiting argument. Omitted prior profiles are an
independent issue and are errors unless the prior was explicitly constructed with
`missing="zero"`.

## Participation

Let `b_i(t)` be the outside option supplied by a scalar, own-type table, profile-aware provider,
or callback. Truthful utility uses report profile `t`.

### Ex-post IR

For every selected complete profile:

```text
U_i(t; t) - b_i(t) >= 0
```

### Interim IR

For every selected own type with positive marginal:

```text
sum_(t_-i) mu(t_-i | t_i) [U_i(t; t) - b_i(t)] >= 0
```

As with BIC, the compiled row is joint weighted and the reported slack is divided by
`m_i(t_i)`. The zero-probability policy is identical to BIC's policy.

### Ex-ante IR

For each selected agent:

```text
sum_t mu(t) [U_i(t; t) - b_i(t)] >= 0
```

Interim and ex-ante IR require a joint prior. Ex-post IR does not.

The audit stores truthful utility, outside option, and their difference as information rent.
For interim and ex-ante rows these quantities are conditional or unconditional expectations,
respectively.

## Payments, budget, and other linear restrictions

With `R(r) = sum_i p_i(r)` denoting designer revenue:

| Requirement | Canonical condition |
| --- | --- |
| `NoSubsidies` | `p_i(r) >= 0` for selected coordinates. |
| `PaymentBounds(lower=L)` | `p_i(r) - L >= 0`. |
| `PaymentBounds(upper=H)` | `H - p_i(r) >= 0`. |
| Ex-post weak budget balance | `R(r) - floor >= 0` for every report. |
| Ex-ante weak budget balance | `sum_t mu(t) R(t) - floor >= 0`. |
| Ex-post strong budget balance | `R(r) - target = 0` for every report. |
| Ex-ante strong budget balance | `sum_t mu(t) R(t) - target = 0`. |

`Capacity` assigns a numeric usage to each explicit outcome and bounds its expected use under
each selected report:

```text
lower <= sum_o usage(o) x_o(r) <= upper
```

`Symmetry` never guesses how identity-bearing labels transform. Each `SymmetryOperation`
explicitly maps agents, their types, and outcomes; the compiled equalities compare the mapped
allocation and payment coordinates.

Custom affine rows use typed `AllocationCoordinate` and `PaymentCoordinate` keys. A user-facing
`<=` row is multiplied by `-1`; an equality remains an equality. Backend column numbers are not
part of the custom-row interface.

## Canonical rows, slack, and epsilon

The compiler uses one solver-independent maximization form:

```text
maximize      c @ z + offset
subject to    A_j @ z >= b_j    for inequality rows
              A_j @ z == b_j    for equality rows
              lower <= z <= upper
```

Every economic inequality has canonical slack:

```text
s_j(z) = A_j @ z - b_j
```

Larger slack always means more satisfaction. An equality instead has signed residual
`A_j @ z - b_j`; reports summarize its absolute magnitude. Stable semantic row IDs carry the
family and labeled qualifiers independently of backend row positions or presolve.

For a fixed audit at feasibility tolerance `tau`:

```text
inequality passes  iff  slack >= -tau
equality passes    iff  abs(residual) <= tau
local activity     iff  slack <= tau
```

`active_constraints()` additionally retains only passing inequalities. Activity at one
mechanism is not an invariant optimal-face statement.

For the audited IC section:

```text
epsilon_IC = max(0, -min_j reported_IC_slack_j)
```

This is the smallest common additive relaxation of the materialized IC rows, in their reported
units. Vacuous rows have no numerical slack and do not affect the minimum.

## Scaling and original units

The canonical LP stores original economic coefficients. Each backend row may receive a
strictly positive power-of-two equilibration factor `alpha_j`; report normalization uses a
separate positive factor `beta_j`:

```text
solver_slack_j   = alpha_j * original_slack_j
reported_slack_j = beta_j  * original_slack_j
reported_dual_j  = (alpha_j / beta_j) * normalized_solver_dual_j
```

Usually `beta_j = 1`. For positive-mass BIC and interim-IR rows,
`beta_j = 1 / m_i(t_i)`, converting the joint-weighted numerator to conditional utility.
Positive scaling preserves the feasible set and inequality direction.

The HiGHS adapter uses its own canonical sign conversion internally and normalizes returned
statuses and multipliers. MechAudit independently recomputes primal row and bound residuals
from the unscaled model instead of relying only on backend-reported feasibility.

## Fixed audit, design, and assessment boundaries

### Fixed-mechanism audit

`audit_mechanism` or `FiniteEnvironment.audit`:

- binds and aligns the mechanism;
- always evaluates allocation nonnegativity and normalization;
- exhaustively evaluates the declared economic rows;
- produces witnesses, rents, fingerprints, and an audit-level health report.

It has no objective, compiled LP, solver status, optimal value, or optimal face. Consequently,
a `MechanismAuditReport` exposes no optimal-face method.

### Randomized-mechanism design

`DesignProblem.solve` performs the following implemented sequence:

1. compile allocation/payment variables, baseline feasibility, requirements, and objective;
2. positively scale rows for the HiGHS backend;
3. require normalized status `optimal` and a usable primal vector;
4. recompute original-unit LP residuals and enforce the feasibility tolerance;
5. decode a bound `DirectMechanism`;
6. run the independent economic audit over all declared requirements;
7. return `DesignSolution` only if that audit passes.

Infeasible, unbounded, nonoptimal, or uncertifiable solves raise typed exceptions rather than
returning partially valid design solutions. No IR condition or payment bound is inserted
automatically to repair an unbounded formulation.

### Candidate assessment

`DesignProblem.assess` binds a candidate to the same domain, audits the full requirement set,
evaluates it under the compiled objective, and compares it with a reference or cached optimum.
The maximization gap is:

```text
gap = optimal_value - candidate_value
```

For a feasible candidate, the reported absolute gap is truncated below at zero to absorb a
small numerical overshoot. `is_optimal` additionally requires feasibility and a gap no larger
than `optimality_tolerance`. Candidate-local activity and universally active rows are stored as
different fields.

## Optimal-face analysis

Let `V*` be the selected certified optimum and let `delta >= 0`. The analyzed primal
near-optimal face is:

```text
F_delta = {z feasible : c @ z + offset >= V* - delta}
```

If `objective_delta` is omitted, MechAudit uses:

```text
delta = max(
    objective_absolute,
    objective_relative * max(1, abs(V*))
)
```

The report records both requested and effective delta, the model fingerprint, tolerances,
auxiliary solve counts, retained task failures, and numerical witnesses.

For each classifiable inequality `j`, define:

```text
s_j^max = max_(z in F_delta) s_j(z)
s_j^min = min_(z in F_delta) s_j(z)
```

The implemented classifications are:

| Classification | Numerical meaning |
| --- | --- |
| Universally active | The authoritative maximum-slack solve returns `s_j^max <= activity tolerance`. |
| Nonuniversal | A positive-slack witness exists, but a minimum-slack solve has not classified the boundary behavior. This is a normal fast-level result. |
| Sometimes active | `s_j^max` is positive and the full-level minimum satisfies `s_j^min <= activity tolerance`. |
| Never active | The full-level minimum is reliably positive. |
| Inconclusive | An auxiliary solve failed, hit a limit, or returned a value in the configured tolerance gray zone. |

“Reliably positive” uses a separation threshold of ten times the relevant activity/partition
tolerance (with a tiny numerical floor). Values between the activity threshold and this
positive threshold remain inconclusive rather than being forced into a binary label.

An IPM solve with presolve and crossover disabled is used as a numerical maximum-support
candidate. Positive slack or dual coordinates can classify clear cases at zero delta; ambiguous
fast-level coordinates receive targeted maximum-slack LPs. For a positive-width face, a
candidate dual coordinate alone is not accepted as proof of universal activity. At `"full"`
and `"essentiality"` levels, every selected row receives a targeted maximum-slack LP, and every
row then known to be nonuniversal receives a targeted minimum-slack LP.

A targeted optimum is authoritative only after checking optimal status, a usable primal-dual
solution, objective value/bound consistency, independent primal feasibility, dual
infeasibility, primal-dual objective error, and membership in the declared near-optimal face.
The corresponding `ActivityRecord` retains these quantities in an auxiliary certificate.

### Dual relevance, redundancy, and removal

Dual-range analysis constructs an explicit dual near-optimal face and reports multipliers in
the same normalized report units as the corresponding slack. Its labels distinguish never,
possibly, and universally priced rows. Dual relevance is not the same question as primal
activity.

The removal diagnostics also answer different questions:

| Diagnostic | Question |
| --- | --- |
| Redundancy | After removing the inequality, do the remaining feasibility rows still imply it? |
| Objective cost | How much can the maximized objective increase after removing the row/family? |
| Full-mechanism implementation essentiality | If objective cost is zero, does removal add a near-optimal mechanism that violates the removed inequality? |
| Allocation/payment projection | Can a relaxed witness be separated from the original face by a coordinate range in that projection? |

The full-mechanism violation test is exact up to tolerances for the stated LP. Projection tests
are one-sided: failure to find a coordinate separator is reported as
inconclusive, because coordinate-wise ranges are only an outer approximation to a projected
polytope. The redundancy and implementation-essentiality helpers do not support equalities.

## Correlation and Crémer–McLean conditioning

For each agent `i`, the diagnostic constructs a matrix whose rows are own types and whose
columns are other-agent type profiles:

```text
B_i[t_i, t_-i] = mu(t_i, t_-i) / m_i(t_i)    when m_i(t_i) > 0
B_i[t_i, t_-i] = 0                           when m_i(t_i) = 0
```

It computes singular values with NumPy and declares numerical rank as the number strictly above
`rank_tolerance`. Full row rank means:

```text
rank(B_i) = number of declared types of agent i
```

The reported condition number is infinite when the smallest returned singular value is at or
below `rank_tolerance`; otherwise it is the largest singular value divided by the smallest.
Pairs of rows within `duplicate_tolerance` in Euclidean norm are retained as near duplicates.
Zero-marginal types are explicit warnings because their conditional beliefs are undefined.

When a mechanism is supplied, the report also compares its largest absolute payment with the
largest absolute tabulated value. A large ratio and a small positive singular value are
fragility evidence. Full numerical rank alone is neither a robustness guarantee nor proof that
a particular Crémer–McLean extraction mechanism has been constructed.

## Reproducibility and serialization boundaries

Declaration order is semantic: it determines canonical profile, outcome, and matrix axes.
Mappings used as structured labels are key-sorted before hashing, while axes remain explicitly
ordered. Direct-mechanism imports with coordinates are reordered into those axes.

Three fingerprints serve different scopes:

- the domain fingerprint covers ordered agent/type/outcome axes only;
- the mechanism fingerprint adds canonical allocation/payment bytes and the payment sign;
- the model fingerprint covers the compiled LP's canonical numerical data and semantic row and
  column identities.

None hashes Python callback source. Report JSON stores evaluated numerical evidence and display
labels, not callback code or an executable environment. Human-readable summaries are for
inspection; JSON envelopes and labeled DataFrames are the research artifact interfaces.

## Scope and nonclaims

MechAudit supports:

- finite agents, types, reports, and explicit outcomes;
- scalar and structured stable labels;
- arbitrary finite joint priors, including correlation and explicitly declared zero mass;
- private or interdependent quasi-linear values;
- imported deterministic or randomized direct mechanisms;
- linear randomized-mechanism design with HiGHS;
- linear IC, IR, budget, capacity, symmetry, and custom restrictions;
- tolerance-relative optimal-face, dual-range, redundancy, and ablation diagnostics.

MechAudit does not:

- optimize deterministic mechanisms (`randomized=False` would require a mixed-integer model);
- discretize continuous type, report, or outcome spaces;
- model indirect, dynamic, repeated, or cryptographic mechanisms;
- infer an information model, prior, outside option, or missing economic restriction;
- serialize callbacks into executable model files;
- provide exact rational arithmetic, theorem-prover export, or formal certificates;
- turn full conditional-belief rank into a theorem about robustness or implemented extraction.

Every conclusion should therefore be read with its declared finite model, sign convention,
prior, objective, fingerprint, effective delta, and tolerance policy attached.
