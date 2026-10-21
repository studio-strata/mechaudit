# API reference

MechAudit exposes a curated research API from `mechaudit`. The package works with finite
direct mechanisms, explicit social outcomes, quasi-linear utility, and linear randomized-
mechanism design problems. Positive payments always mean transfers **from an agent to the
designer**.

This page documents the implemented public signatures. Objects in `mechaudit.design` and
`mechaudit.backends` that are not re-exported at the package root are lower-level compilation
interfaces; most users should enter through `FiniteEnvironment`, `DirectMechanism`, and
`DesignProblem`.

## Top-level surface

The following names are importable directly from `mechaudit`.

| Area | Public names |
| --- | --- |
| Version | `__version__` |
| Domains and values | `FiniteEnvironment`, `JointPrior`, `MissingProfilePolicy`, `PrivateValues`, `InterdependentValues`, `ValuationKind`, `StructuredLabel`, `TypeProfile`, `ReportProfile` |
| Outcomes | `Outcome`, `OutcomeSpace`, `single_item_outcomes`, `multiple_item_outcomes`, `public_project_outcomes`, `bilateral_trade_outcomes`, `procurement_outcomes` |
| Mechanisms | `DirectMechanism`, `MechanismTables` |
| Audit | `audit_mechanism`, `AuditOptions`, `CheckStatus`, `ConstraintCheck`, `AuditSection`, `MechanismAuditReport` |
| Incentives and participation | `DSIC`, `ExPostIC`, `BIC`, `ExPostIR`, `InterimIR`, `ExAnteIR`, `ZeroProbabilityPolicy` |
| Payments and budget | `PaymentBounds`, `PaymentFloor`, `PaymentCap`, `NoSubsidies`, `NoPositiveTransfers`, `WeakBudgetBalance`, `StrongBudgetBalance`, `BudgetTiming` |
| Other requirements | `Capacity`, `Symmetry`, `Anonymity`, `SymmetryOperation`, `CustomLinearConstraint`, `CustomLinearRequirement`, `AllocationCoordinate`, `PaymentCoordinate` |
| Objectives | `LinearObjective`, `ExpectedRevenue`, `ExpectedSocialWelfare`, `ExpectedWelfare`, `ExpectedResidualSurplus`, `WeightedAgentUtilities`, `LinearCombination`, `CustomLinearObjective` |
| Design and assessment | `DesignProblem`, `DesignSolution`, `CandidateAssessment`, `OptimalFaceReport`, `SolveOptions`, `SolveMethod`, `TolerancePolicy`, `OptimalFaceOptions`, `OptimalFaceLevel` |
| Diagnostics | `NumericalHealthReport`, `ConditionalBeliefDiagnostic`, `CremerMcLeanReport`, `correlation_diagnostics`, `cremer_mclean_diagnostic` |
| Scenarios | `ScenarioResult`, `ScenarioComparison` |
| Exceptions | `MechAuditError`, `ValidationError`, `LabelValidationError`, `PriorValidationError`, `EnvironmentValidationError`, `RequirementValidationError`, `DomainMismatchError`, `ZeroProbabilityTypeError`, `SolverError`, `InfeasibleDesignError`, `UnboundedDesignError`, `NumericalCertificationError` |

Aliases are exact aliases, not subclasses or additional semantic families:

- `ExpectedWelfare is ExpectedSocialWelfare`;
- `NoPositiveTransfers is NoSubsidies` under the package payment convention;
- `Anonymity is Symmetry`.

## Environments, labels, and priors

### `FiniteEnvironment`

```python
FiniteEnvironment(
    *,
    agents,
    outcomes,
    prior=None,
    values=None,
)
```

`agents` is an ordered mapping from agent labels to ordered, nonempty finite type axes.
`outcomes` is an ordered, nonempty iterable. `values` is required. Prefer an explicit valuation
wrapper:

```python
PrivateValues(function)
InterdependentValues(function)
PrivateValues.from_mapping(values)
InterdependentValues.from_mapping(values)
```

A private-value callback has signature `(agent, own_type, outcome) -> real`. An
interdependent callback has signature `(agent, true_profile, outcome) -> real`. A bare callable
passed to `FiniteEnvironment` is treated as `InterdependentValues`; the library does not infer
private values from callback behavior. Values are evaluated over the complete finite domain,
checked for finiteness, and tabulated during construction.

Important attributes and methods are:

| Member | Contract |
| --- | --- |
| `agents`, `outcomes` | Canonical ordered axes. |
| `types` | Read-only mapping from each canonical agent to its type tuple. |
| `profiles`, `report_profiles` | Cartesian-product true and report profiles in canonical order. |
| `values` | Tabulated values indexed `[agent][true_profile][outcome]`. |
| `prior` | Aligned `JointPrior` or `None`. |
| `n_agents`, `n_profiles`, `n_outcomes` | Axis sizes. |
| `valuation_kind`, `is_private_values` | Declared information model. |
| `coerce_profile(profile)` | Return a validated `TypeProfile`. |
| `coerce_report_profile(report)` | Return a validated `ReportProfile`. |
| `agent_index(agent)`, `type_index(agent, type_value)`, `profile_index(profile)`, `outcome_index(outcome)` | Convert labeled coordinates to canonical integer positions. |
| `evaluate_value(agent, profile, outcome)` | Read one tabulated value by label. |
| `with_prior(prior)` | Rebuild the same domain and valuation model with another prior. |
| `audit(mechanism, requirements, *, options=None)` | Run the fixed-mechanism audit workflow. |
| `design(*, objective, requirements=(), randomized=True)` | Create a `DesignProblem`. `randomized=False` is not supported. |

`domain_fingerprint` is a SHA-256 digest of the ordered agent labels, each ordered type axis,
and the ordered outcomes. It does **not** hash the prior or the tabulated values.
Use report/model fingerprints and stored numerical evidence when those objects must also be
distinguished.

### Labels and profiles

Scalar hashable values may be used directly as labels. JSON-like mappings become immutable
`StructuredLabel` objects, and nested lists/tuples become tuples. NaN, sets, empty axes, and
duplicate or equality-colliding axis labels are rejected.

```python
StructuredLabel(values)
Outcome(attributes=None, **fields)
Outcome.from_mapping(attributes)
OutcomeSpace(outcomes)
```

`TypeProfile` and `ReportProfile` are immutable agent-indexed mappings. Environment coercion is
the safest way to create one because it also validates domain membership. Their common profile
operations include `from_mapping`, `from_values`, `values_tuple`, `agents_tuple`,
`with_report`, `without`, and `to_dict`.

The outcome factories return explicit `OutcomeSpace` instances:

| Factory | Signature |
| --- | --- |
| `single_item_outcomes` | `(agents, *, include_unallocated=True)` |
| `multiple_item_outcomes` | `(agents, items, *, allow_unallocated=True)` |
| `public_project_outcomes` | `()` |
| `bilateral_trade_outcomes` | `()` |
| `procurement_outcomes` | `(suppliers, *, include_no_procurement=True)` |

`multiple_item_outcomes` enumerates the complete assignment product and can therefore grow
exponentially.

### `JointPrior`

Use a named constructor rather than manually instantiating `JointPrior`:

| Constructor | Implemented signature |
| --- | --- |
| `JointPrior.from_independent` | `(marginals, *, normalization_tolerance=1e-12)` |
| `JointPrior.from_mapping` | `(probabilities, *, agent_order=None, type_spaces=None, missing=MissingProfilePolicy.ERROR, normalization_tolerance=1e-12)` |
| `JointPrior.from_records` | `(records, *, agent_order, probability_field="probability", type_spaces=None, missing=MissingProfilePolicy.ERROR, normalization_tolerance=1e-12)` |
| `JointPrior.from_xarray` | `(data, *, normalization_tolerance=1e-12)` |

`from_mapping` accepts scalar profile keys as a one-agent shorthand. Multi-agent tuple keys
follow `agent_order`. Unless all keys are already `TypeProfile` objects, `agent_order` is
required when constructing a prior independently of an environment. Omitted Cartesian-product
profiles raise by default; `missing="zero"` is the explicit zero-fill policy. Probabilities
must be finite, nonnegative, and sum to one within `normalization_tolerance`.

Public accessors include `probabilities`, `marginal(agent, type_value)`,
`zero_marginal_types(*, tolerance=0.0)`, `is_independent(*, tolerance=1e-12)`, and
`align(*, agents, type_spaces)`. `FiniteEnvironment` calls `align` when given an existing
`JointPrior`. Indexing accepts a `TypeProfile`, an agent-to-type mapping, or an ordered type
tuple; a one-agent prior also accepts the bare type label, as in `prior["low"]`.

## Direct mechanisms

### Construction and binding

```python
DirectMechanism(
    *,
    allocation,
    payments,
    environment=None,
    domain_fingerprint=None,
)
```

Without `environment`, construction records an immutable input source and returns an unbound
mechanism. `bind(environment)` evaluates callbacks, validates labels, and creates canonical
read-only `float64` arrays. Binding an already-bound mechanism to the same domain is
idempotent; binding it to another domain raises `DomainMismatchError`.

The explicit constructors are:

| Constructor | Implemented signature and input contract |
| --- | --- |
| `DirectMechanism.from_mapping` | `(allocation, payments=None, *, environment=None, domain_fingerprint=None)`. Normally accepts separate nested profile mappings. If `payments` is omitted, `allocation` must be the combined mapping returned by `to_dict()`. |
| `DirectMechanism.from_functions` | `(allocation, payments, *, environment=None, domain_fingerprint=None)`. Each callback may be row-style or cell-style. |
| `DirectMechanism.from_arrays` | `(allocation, payments, *, environment=None, report_profiles=None, outcomes=None, agents=None, payment_report_profiles=None, domain_fingerprint=None)`. Unlabeled arrays assert canonical order; supplied labels are reordered at binding. |
| `DirectMechanism.from_tables` | `(allocation, payments, *, environment=None, domain_fingerprint=None)`. Accepts two wide pandas-like tables. |
| `DirectMechanism.from_xarray` | `(allocation, payments=None, *, environment=None, report_dims=None, outcome_dim="outcome", payment_agent_dim="agent", domain_fingerprint=None)`. Accepts two DataArrays or one Dataset with `allocation` and `payments` variables. |

Allocation callbacks accept either `(report) -> complete outcome mapping` or
`(report, outcome) -> number`. Payment callbacks analogously accept `(report) -> complete agent
mapping` or `(report, agent) -> number`. One-agent payment mappings and arrays may use scalar
rows.

Binding rejects missing or duplicate rows/columns, unknown labels, wrong shapes, and NaN or
infinite values. It retains negative allocations and rows that do not sum to one so the audit can
report them as economic feasibility failures.

### Bound representation

For `R` report profiles, `K` outcomes, and `n` agents:

| Member | Contract |
| --- | --- |
| `allocation` | Read-only `float64` array with shape `(R, K)`. |
| `payments` | Read-only `float64` array with shape `(R, n)`. |
| `payment_sign` | The literal string `"agent_to_designer"`. |
| `agents`, `outcomes`, `report_profiles` | Canonical labels for the three axes. |
| `fingerprint` | SHA-256 over the domain fingerprint, sign convention, shapes, and canonical numeric bytes. Available only when bound. |
| `allocation_row(report, environment)` | One labeled allocation row. |
| `payment_row(report, environment)` | One labeled payment row. |
| `allocation_probability(report, outcome, environment)` | One allocation cell. |
| `payment(report, agent, environment)` | One payment cell. |
| `to_dict()` | Round-trippable nested **Python** mapping containing allocations, payments, sign, and fingerprints. Profile and label objects remain mapping keys. |
| `to_pandas()` | `MechanismTables(allocation, payments)` containing two labeled DataFrames. |

The DataFrames use a one-level index for a one-agent environment and a `MultiIndex` for
multiple agents. Both frames carry `domain_fingerprint`, `mechanism_fingerprint`, and
`payment_sign` in `.attrs`.

## Auditing

The function and environment method are equivalent entry points:

```python
audit_mechanism(*, environment, mechanism, requirements=(), options=None)
environment.audit(mechanism, requirements, *, options=None)
```

The audit always checks allocation nonnegativity and normalization. It then exhaustively
evaluates the declared requirements in original economic units. It does not compile or solve a
design problem.

### Requirement constructors

Selector arguments use labels, not integer positions. `None` means all labels; an empty
selector means none.

| Requirement | Implemented signature | Meaning |
| --- | --- | --- |
| `DSIC` | `(*, agents=None, true_types=None, deviations=None)` | Private-value dominant-strategy IC. Rejected for interdependent values. |
| `ExPostIC` | `(*, agents=None, true_types=None, deviations=None)` | Profile-wise IC with other agents truthful; supports either value model. |
| `BIC` | `(*, agents=None, true_types=None, deviations=None, zero_probability_types=ZeroProbabilityPolicy.ERROR)` | Conditional expected IC under the joint prior. |
| `ExPostIR` | `(outside_option=0.0, *, agents=None, types=None)` | Profile-wise participation. |
| `InterimIR` | `(outside_option=0.0, *, agents=None, types=None, zero_probability_types=ZeroProbabilityPolicy.ERROR)` | Type-conditional expected participation. |
| `ExAnteIR` | `(outside_option=0.0, *, agents=None)` | Agent-level expected participation before types are realized. |
| `PaymentBounds` | `(*, lower=None, upper=None, agents=None, reports=None)` | Report/agent-specific finite bounds; at least one side is required. |
| `PaymentFloor` | `(floor, *, agents=None, reports=None)` | Convenience constructor for `PaymentBounds(lower=floor, ...)`. |
| `PaymentCap` | `(cap, *, agents=None, reports=None)` | Convenience constructor for `PaymentBounds(upper=cap, ...)`. |
| `NoSubsidies` | `(*, agents=None, reports=None)` | Require selected payments to be nonnegative. |
| `WeakBudgetBalance` | `(*, timing=BudgetTiming.EX_POST, floor=0.0)` | Designer revenue is at least the floor. |
| `StrongBudgetBalance` | `(*, timing=BudgetTiming.EX_POST, target=0.0)` | Designer revenue equals the target. |
| `Capacity` | `(*, resource, usage, upper, lower=None, profiles=None)` | Bound expected resource use under each selected report. |
| `SymmetryOperation` | `(*, agents, types, outcomes, name="symmetry")` | Explicit relabeling of all affected axes. |
| `Symmetry` | `(*operations)` | Require allocations and payments to be invariant under each operation. |

Outside options may be a finite number, an object implementing
`value(agent, true_profile)`, a callable `(agent, true_profile) -> number`, or a nested
agent/own-type mapping. Profile-dependent tables are available through
`ProfileOutsideOptions` and require canonical `TypeProfile` keys. `OutsideOption`,
`ConstantOutsideOption`, `CallableOutsideOption`, `OwnTypeOutsideOptions`, and
`ProfileOutsideOptions` are available from both `mechaudit` and `mechaudit.requirements`.

`BudgetTiming` has values `"ex_post"` and `"ex_ante"`. Ex-ante budget constraints, BIC,
interim IR, and ex-ante IR require a joint prior. `ZeroProbabilityPolicy` has values
`"error"`, `"vacuous"`, and `"enforce_ex_post"`.

### Custom linear requirements

```python
AllocationCoordinate(report, outcome)
PaymentCoordinate(report, agent)

CustomLinearConstraint(
    name,
    coefficients,
    *,
    sense=">=",
    rhs=0.0,
    constant=0.0,
    family="custom",
    economic_unit="custom",
    metadata=None,
)

CustomLinearRequirement(name, constraints)
```

Coordinates require canonical `ReportProfile` instances. `sense` accepts `">="`, `"<="`,
or `"=="`; less-than rows are sign-flipped into the package's canonical representation.
Names and family labels become stable semantic row identifiers.

### Audit options and report

```python
AuditOptions(
    tolerances=TolerancePolicy(),
    max_witnesses=10,
    include_all_rows=True,
)
```

`MechanismAuditReport` exposes:

| Member | Result |
| --- | --- |
| `passed`, `is_feasible` | Whether every baseline and declared check passed or was vacuous. |
| `checks` | Complete tuple of `ConstraintCheck` records. |
| `feasibility`, `ic`, `ir`, `budget` | `AuditSection` views by semantic family. |
| `epsilon_ic` | `max(0, -minimum IC slack)` in the reported IC units. |
| `rents` | Long-form DataFrame of truthful utility, outside option, and information rent. |
| `active_constraints()` | Passing inequalities active at this one mechanism within their row tolerance. |
| `numerical_health` | Audit-level `NumericalHealthReport`. |
| `summary()` | Human-readable text; not a machine-stable format. |

Every `ConstraintCheck` contains `constraint_id`, `family`, `relation`, `status`, `tolerance`,
`slack` or `residual`, `unit`, and labeled `metadata`. A constraint has the same
`constraint_id` in audit reports, compiled LP rows, selected-solution activity, and optimal-face
reports, so any of these identifiers can be passed to the redundancy, cost, and essentiality
diagnostics. `CheckStatus` defines `"pass"`,
`"fail"`, `"vacuous"`, and `"ambiguous"`; the fixed-mechanism audit engine emits pass/fail and,
for an explicit zero-probability policy, vacuous rows.

`AuditSection` is a sequence and provides `passed`, `minimum_slack`, `maximum_residual`,
`epsilon`, `worst_deviations(k=10)`, `by_type()`, and `to_pandas()`.

## Design

### Objectives

All objectives use a maximization convention.

| Objective | Implemented signature | Quantity |
| --- | --- | --- |
| `ExpectedRevenue` | `(agent_weights=None)` | Expected weighted payments to the designer. |
| `ExpectedSocialWelfare` | `(agent_weights=None)` | Expected weighted total value. |
| `ExpectedResidualSurplus` | `()` | Expected total agent utility, welfare minus payments. Use `WeightedAgentUtilities` for weights. |
| `WeightedAgentUtilities` | `(weights)` | Expected weighted truthful utilities; weights are required. |
| `LinearCombination` | `(terms, *, constant=0.0)` | Finite weighted combination of `LinearObjective` objects. |
| `CustomLinearObjective` | `(name, coefficients, *, constant=0.0)` | Affine objective over typed mechanism coordinates. |

Expected objectives require a joint prior. `LinearObjective.weighted(coefficient)`, scalar
multiplication, and addition create `LinearCombination` objects.

### `DesignProblem`

```python
DesignProblem(
    *,
    environment,
    objective,
    requirements=(),
    randomized=True,
)
```

| Method | Implemented signature | Behavior |
| --- | --- | --- |
| `require` | `(*requirements)` | Mutate this builder by appending requirements; clear cached compilation/solution. |
| `with_requirements` | `(*requirements)` | Return a separate problem with additional requirements. |
| `compile` | `()` | Return the cached solver-independent `CompiledDesign`. |
| `solve` | `(*, backend="highs", options=None)` | Solve, independently check, decode, and audit; return `DesignSolution`. |
| `assess` | `(mechanism, *, optimality_tolerance=1e-9, solution=None, analyze_optimal_face=True)` | Compare an imported mechanism with this complete design problem. |
| `compare_scenarios` | `(scenarios)` | Solve a labeled batch of additions, removals, replacements, prior changes, or objective changes. |

The built-in backend is selected with `backend="highs"`; callers may instead pass an
`LPBackend` implementation. `randomized=False` raises `NotImplementedError`; deterministic
mechanisms may still be imported and audited.

### Solve and tolerance options

```python
SolveOptions(
    tolerances=TolerancePolicy(),
    method=SolveMethod.AUTO,
    crossover=None,
    presolve=True,
    time_limit=None,
    threads=None,
    random_seed=None,
    backend_options=(),
)
```

`SolveMethod` values are `"auto"`, `"simplex"`, and `"ipm"`. A `backend_options` mapping or
tuple of `(name, value)` pairs is the explicit backend escape hatch.

```python
TolerancePolicy(
    probability=1e-12,
    feasibility=1e-8,
    activity=1e-8,
    objective_absolute=1e-9,
    objective_relative=1e-9,
    partition_primal=1e-9,
    partition_dual=1e-9,
    complementarity=1e-8,
)
```

`TolerancePolicy.objective_delta(objective_value)` returns the larger of the absolute
tolerance and the relative tolerance times `max(1, abs(objective_value))`.

### `DesignSolution` and assessment

A successful solve returns one solver-selected optimum with these primary members:

| Member | Contract |
| --- | --- |
| `mechanism` | Bound `DirectMechanism` decoded from the primal vector. |
| `objective_value` | Recomputed value in the declared maximization convention. |
| `audit` | Independent `MechanismAuditReport` over the decoded mechanism. |
| `backend_result` | Backend status, primal/dual data, iterations, timing, and residual metadata. |
| `selected_activity` | Local activity records in report units. |
| `active_at_selected_solution` | Locally active subset; not an optimal-face claim. |
| `numerical_health` | Audit and solver-independent residual evidence. |
| `model_fingerprint`, `solver_name`, `solver_version` | Reproducibility metadata. |

`DesignSolution.analyze_optima` has the implemented convenience signature:

```python
solution.analyze_optima(
    level="fast",
    *,
    options=None,
    dual_ranges=False,
    constraint_families=None,
    n_jobs=1,
    objective_delta=None,
)
```

Pass either one `OptimalFaceOptions` object or the individual arguments, not both.

`CandidateAssessment` exposes feasibility, candidate and optimal objective values,
absolute/relative gaps, local and universal activity labels, `to_dict()`, `to_json()`,
`write_json()`, `to_pandas()`, and `summary()`.

## Optimal-face module

Detailed optimal-face result types and auxiliary diagnostics live in `mechaudit.optimality`.
The main entry point is normally `solution.analyze_optima(...)`; the equivalent function is:

```python
analyze_optimal_face(solution, options=None)
```

```python
OptimalFaceOptions(
    level=OptimalFaceLevel.FAST,
    tolerances=TolerancePolicy(),
    objective_delta=None,
    dual_ranges=False,
    constraint_families=None,
    n_jobs=1,
)
```

`OptimalFaceLevel` values are `"fast"`, `"full"`, and `"essentiality"`. The returned
`OptimalFaceReport` provides `constraints`, `universally_active`, `nonuniversal`,
`sometimes_active`, `never_active`, `inconclusive`, `activity_for`, and report exports. Its
on-demand methods `constraint_cost`, `redundancy`, `implementation_essential`,
`ablate_family`, and `dual_range` require the live `DesignSolution`; they are unavailable on a
detached/deserialized report.

Each `ActivityRecord` distinguishes candidate evidence from authoritative maximum/minimum
slack results. Certified auxiliary optima retain an `AuxiliaryCertificate` containing the
objective value and bound, independent primal violation, maximum dual infeasibility, and
primal-dual objective error.

The module also exports these functional forms:

| Function | Implemented signature |
| --- | --- |
| `constraint_cost` | `(solution, constraint, *, delta=None, projection="full")` |
| `test_redundancy` / `redundancy` | `(solution, constraint)` |
| `implementation_essentiality` / `implementation_essential` | `(solution, constraint, *, delta=None, projection="full")` |
| `ablate_family` | `(solution, family, *, agent=None, delta=None, projection="full")` |
| `compute_dual_ranges` | `(solution, row_indices, *, delta, n_jobs=1, tolerances=None)` |

`projection` accepts `"full"`, `"allocation"`, or `"payment"`. Projected essentiality is
one-sided: absence of a coordinate-range separating witness can produce an inconclusive
result rather than a false proof of containment.

## Correlation diagnostics

```python
correlation_diagnostics(
    environment,
    *,
    rank_tolerance=1e-10,
    duplicate_tolerance=1e-7,
)

cremer_mclean_diagnostic(
    environment,
    mechanism=None,
    *,
    rank_tolerance=1e-10,
    duplicate_tolerance=1e-7,
)
```

Both require a prior. `ConditionalBeliefDiagnostic` stores the labeled conditional-belief
matrix, singular values, numerical rank, full-row-rank flag, smallest singular value,
condition number, zero-marginal types, near-duplicate row pairs, and warnings. It provides
`to_dict()` and `to_pandas()`.

`CremerMcLeanReport` provides `full_rank`, `worst_condition_number`, `for_agent(agent)`, and
`to_dict()`. If a mechanism is supplied, it also reports maximum absolute payment and its ratio
to the largest absolute tabulated value. This is a conditioning diagnostic, not a claim that a
particular extraction construction was implemented.

## Scenario comparisons

`DesignProblem.compare_scenarios(scenarios)` accepts a mapping from scenario name to a change
mapping. Implemented change keys are `"add"`, `"remove"`, `"replace"`, `"prior"`,
`"objective"`, and `"solve_options"`.

A scenario failure caused by `MechAuditError` or `ValueError` is retained as a failed
`ScenarioResult`; it does not abort the remaining scenarios. `ScenarioComparison` provides
`reports`, `to_pandas()`, `to_dict()`, `to_json()`, `write_json()`, and `summary()`.

## Result and export contracts

### JSON envelopes

`to_json(indent=2)` and `write_json(path, indent=2)` on report objects produce deterministic,
key-sorted JSON with this common envelope:

```text
schema_version = "1.0"
library_version = mechaudit.__version__
report_kind = workflow-specific string
... workflow payload ...
```

Implemented report kinds are `"mechanism_audit"`, `"design_solution"`,
`"candidate_assessment"`, `"optimal_face_report"`, and `"scenario_comparison"`.
Nonfinite floats are encoded as `null` for NaN and the strings `"Infinity"` or
`"-Infinity"`. Arbitrary unsupported label objects fall back to a tagged `repr`; callback code
is never serialized. A report is numerical evidence, not an executable environment snapshot.

### DataFrame collections

| Object | Export | Keys or shape |
| --- | --- | --- |
| `DirectMechanism` | `to_pandas()` | `MechanismTables(allocation, payments)`; both frames are wide and labeled. |
| `MechanismAuditReport` | `to_dataframes()` | `checks`, `ic`, `ir`, `budget`, `feasibility`. |
| `DesignSolution` | `to_dataframes()` | Audit keys plus `selected_activity`, `allocation`, `payments`. |
| `CandidateAssessment` | `to_pandas()` | One-row summary frame. |
| `OptimalFaceReport` | `to_dataframes()` | `optimal_face_activity`, `dual_ranges`, `task_failures`, `optimal_face_metadata`. |
| `ConditionalBeliefDiagnostic` | `to_pandas()` | Types by other-agent type profiles. |
| `ScenarioComparison` | `to_pandas()` | One row per named scenario, indexed by name. |

DataFrame schemas are long-form evidence contracts where stated; human-readable `summary()`
strings are not intended for machine parsing.

### Fingerprints

| Fingerprint | Covers | Does not cover |
| --- | --- | --- |
| `FiniteEnvironment.domain_fingerprint` | Ordered agent/type/outcome axes and their label representations. | Prior probabilities and valuation table. |
| `DirectMechanism.fingerprint` | Domain digest, payment sign, numeric shapes, canonical allocation/payment bytes. | Prior probabilities and valuation table. |
| `DesignSolution.model_fingerprint` | Canonical sparse matrix, right-hand side, objective, bounds, row/column semantic IDs, senses, and scale metadata. | Solver options and a selected solver basis. |

Fingerprint equality is a reproducibility check, not proof that two Python callbacks have the
same source code.

## Exceptions

All library-defined exceptions inherit from `MechAuditError`.

| Exception | Typical cause |
| --- | --- |
| `ValidationError` | Base class for invalid user-supplied domain objects; also a `ValueError`. Carries `message`, optional `path`, and `context`. |
| `LabelValidationError` | Unstable, unordered, empty, duplicate, or colliding labels/axes. |
| `PriorValidationError` | Invalid support, probability, normalization, or missing-profile policy. |
| `EnvironmentValidationError` | Missing/incompatible prior or valuation data, callback failure, or invalid environment construction. |
| `RequirementValidationError` | Malformed requirement, incompatible information model, or duplicate semantic row. |
| `DomainMismatchError` | Mechanism, profile, prior, or labeled axes do not align. |
| `ZeroProbabilityTypeError` | BIC or interim IR encounters an exactly zero-marginal type under the `"error"` policy. |
| `SolverError` | Base class for backend and certification failures. |
| `InfeasibleDesignError` | The declared LP is infeasible. |
| `UnboundedDesignError` | The declared maximization problem is unbounded. |
| `NumericalCertificationError` | A nominal solver result lacks a usable primal, violates the independent LP tolerance, or fails the independent economic audit. |

`mechaudit.exceptions.UnsupportedRequirementError` is a public submodule exception used when a
workflow cannot compile or audit a requirement; it is not re-exported from the package root.
Programming-contract errors may also use standard `TypeError`, `ValueError`, `KeyError`,
`IndexError`, or `NotImplementedError`, as documented by the corresponding method.
