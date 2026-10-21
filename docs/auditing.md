# Audit a mechanism

`FiniteEnvironment.audit` exhaustively evaluates a supplied direct mechanism on its finite
domain. It does not compile or solve an optimization problem, and it never makes claims about
an optimal face.

## Information models and incentive notions

Choose the valuation wrapper explicitly:

| Valuation model | Incentive requirement | Meaning |
| --- | --- | --- |
| `PrivateValues` | `DSIC` | Other agents' reports vary; the deviator's value depends only on its own true type. |
| Either model | `ExPostIC` | Other agents report their true types, and utility uses the complete true profile. |
| Either model with a prior | `BIC` | Expected utility is conditional on the agent's true type under the declared joint prior. |

For correlated types, BIC deviation weights are conditioned on the true own type. A
misreport does not change the agent's belief. Internally, BIC rows use joint masses for
numerical stability; reports divide a positive-mass row by the true-type marginal and express
epsilon in conditional utility units.

## Participation and budget timing

Available participation requirements are `ExPostIR`, `InterimIR`, and `ExAnteIR`. Outside
options may be a finite scalar, a labeled mapping, or a callback. Profile-dependent outside
options are evaluated ex post and averaged for interim or ex-ante checks.

Budget requirements always use the package sign convention `p_i > 0`: agent `i` pays the
designer. Use an explicit `BudgetTiming.EX_POST` or `BudgetTiming.EX_ANTE` when interpretation
could be ambiguous:

```python
from mechaudit import BudgetTiming, StrongBudgetBalance, WeakBudgetBalance

requirements = (
    WeakBudgetBalance(timing=BudgetTiming.EX_POST, floor=0),
    StrongBudgetBalance(timing=BudgetTiming.EX_ANTE, target=0),
)
```

`NoSubsidies` means `p_i >= 0`. `NoPositiveTransfers` is a documented alias for exactly the
same restriction, not a second semantic family.

## Zero-probability types

`BIC` and `InterimIR` require an explicit policy when a selected true type has zero marginal
probability:

- `ZeroProbabilityPolicy.ERROR` raises a semantic validation error;
- `VACUOUS` records a labeled vacuous check;
- `ENFORCE_EX_POST` expands the condition into profile-wise checks.

Exact zero mass is different from small positive mass. Positive masses at or below the
probability tolerance receive numerical-health warnings but are still conditioned normally;
only exact zero invokes the selected policy. Omitted prior profiles are never silently treated
as zero. Use `JointPrior.from_mapping(..., missing="zero")` to opt in to zero filling.

## Mechanism inputs

The constructor accepts nested mappings, row/cell callbacks, or numeric arrays. Explicit
constructors make the intended input style clear:

```python
DirectMechanism.from_mapping(allocation, payments, environment=env)
DirectMechanism.from_functions(allocation_fn, payment_fn, environment=env)
DirectMechanism.from_arrays(
    allocation_array,
    payment_array,
    environment=env,
    report_profiles=report_labels,
    outcomes=outcome_labels,
    agents=agent_labels,
)
DirectMechanism.from_tables(allocation_frame, payment_frame, environment=env)
DirectMechanism.from_xarray(dataset, environment=env)
```

Binding checks coverage, coordinates, dimensions, duplicates, and finite numeric values. It
retains negative or non-normalized allocation entries so the audit can report them as economic
feasibility failures rather than unreadable input.

## Report structure

`MechanismAuditReport` is immutable and provides typed views:

- `report.feasibility`: baseline allocation checks;
- `report.ic`: DSIC, ex-post IC, or BIC checks;
- `report.ir`: participation checks;
- `report.budget`: payment bounds and budget checks;
- `report.rents`: truthful utility, outside option, and information-rent records;
- `report.numerical_health`: independent scaling and residual diagnostics.

Each `ConstraintCheck` has a stable semantic identifier, family, relation, status, original-unit
slack or residual, tolerance, and labeled witness metadata. Status is not collapsed to a bare
Boolean: a row can pass, fail, be vacuous, or remain numerically ambiguous.

```python
report = env.audit(mechanism, requirements)

if not report.ic.passed:
    print(report.ic.worst_deviations(k=10))

print("smallest additive IC epsilon:", report.epsilon_ic)
print(report.ir.by_type())
print(report.rents)
```

Equalities have residuals, not slacks. “Active” is meaningful only for inequalities at this
particular mechanism; it is never evidence of universal activity across optima.

## Serialization

`to_dict()`, `to_json()`, `write_json(path)`, and `to_dataframes()` retain long-form semantic
coordinates rather than opaque solver row numbers. Labels must be serializable by the report
codec. Callback code itself is not serialized.
