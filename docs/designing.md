# Design a mechanism

A `DesignProblem` is a finite sparse linear program for randomized direct mechanisms. All
designer objectives use a maximization convention.

## Variables and canonical rows

For every report profile `r`, outcome `o`, and agent `i`, the compiler creates:

- `x[r, o]`, the probability of the explicit social outcome `o`;
- `p[r, i]`, the payment from agent `i` to the designer.

It always imposes allocation normalization and nonnegativity. Payments remain free unless IC,
IR, budget, or explicit payment-bound rows restrict them. Every economic inequality is stored
as `lhs - rhs >= 0`, so reported slack always has the same sign and meaning.

Rows have stable semantic identities independent of HiGHS presolve indices. The compiled-model
fingerprint covers canonical numerical data and economic row/objective semantics, but not solver
options.

## Objectives

Built-in linear objectives are:

- `ExpectedRevenue(agent_weights=None)`;
- `ExpectedSocialWelfare(agent_weights=None)` (also exported as `ExpectedWelfare`);
- `ExpectedResidualSurplus()`, defined as welfare minus revenue, hence the sum of agents'
  quasi-linear utilities;
- `WeightedAgentUtilities(weights)`;
- `LinearCombination([(weight, objective), ...])`;
- `CustomLinearObjective(...)` over typed allocation/payment coordinates.

Expected objectives require a joint prior. The compiler never invents one for a belief-free
environment.

## Build and solve

```python
from mechaudit import DSIC, ExPostIR, ExpectedRevenue, NoSubsidies, SolveOptions

problem = env.design(
    objective=ExpectedRevenue(),
    requirements=(DSIC(), ExPostIR(), NoSubsidies()),
)

solution = problem.solve(
    backend="highs",
    options=SolveOptions(method="simplex", presolve=True),
)
```

`problem.with_requirements(extra)` creates a separate branch. For exploratory notebooks,
`problem.require(extra)` appends in place and invalidates compiled/solution caches.

Only a certified optimal solve returns a `DesignSolution`. Infeasible and unbounded problems
raise `InfeasibleDesignError` and `UnboundedDesignError`; they do not return half-populated
solutions. An unbounded revenue problem often lacks IR or an upper payment bound, but MechAudit
only diagnoses that possibility—it never adds restrictions automatically.

Optimization of deterministic rules is not supported because it requires a mixed-integer model.
A fixed deterministic mechanism can still be imported and audited.

## Custom linear restrictions

Custom constraints use typed coordinates rather than backend column numbers:

```python
from mechaudit import (
    CustomLinearConstraint,
    CustomLinearRequirement,
    PaymentCoordinate,
)

high_report = env.coerce_report_profile({"buyer": "high"})
row = CustomLinearConstraint(
    "payment_cap_at_high",
    {PaymentCoordinate(report=high_report, agent="buyer"): 1.0},
    sense="<=",
    rhs=2.0,
)

problem = problem.with_requirements(CustomLinearRequirement("payment_caps", (row,)))
```

Coordinates are aligned and validated against the environment at compilation. Custom rows are
still converted to the package's canonical inequality/equality representation and retain their
name and metadata in reports.

## Result and numerical evidence

`DesignSolution` contains:

- the bound `mechanism`;
- the original-unit `objective_value`;
- an independent `audit`;
- local `selected_activity` and transformed dual multipliers;
- a `NumericalHealthReport`;
- backend status, iterations, version, and residuals;
- a deterministic `model_fingerprint`.

Use `solution.analyze_optima(...)` for invariant claims. The selected active set by itself is
solver-dependent.

## Assess an imported mechanism

```python
assessment = problem.assess(candidate, solution=solution)
print(assessment.summary())
```

Assessment aligns the candidate to the same domain, audits all problem requirements, evaluates
the maximize-convention objective, and reports absolute/relative optimality gaps. It labels
candidate-local activity separately from rows known to be universally active on the reference
optimal face.

## Scenario comparisons

The thin scenario runner supports requirement and prior perturbations without becoming a full
comparative-statics framework:

```python
comparison = problem.compare_scenarios(
    {
        "baseline": {},
        "extra_budget_rule": {"add": [NoSubsidies()]},
        "perturbed_prior": {"prior": another_prior},
    }
)

print(comparison.to_pandas())
```

Each failed scenario is retained as labeled data rather than aborting unrelated solves.
