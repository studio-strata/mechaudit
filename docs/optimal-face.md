# Optimal-face analysis

A linear-programming solver returns one optimum. Economic conclusions often concern every
optimum. MechAudit keeps those questions separate.

For canonical inequality `j`, let

```text
s_j(z) = a_j @ z - b_j >= 0
```

and define the reported near-optimal face

```text
F_delta = {z feasible : c @ z + offset >= optimal_value - delta}.
```

## Classification vocabulary

- **Active at the selected solution**: one returned mechanism has
  `s_j <= activity_tolerance`. This is local and solver-dependent.
- **Universally active**: the maximum of `s_j` over `F_delta` is zero within tolerance.
- **Nonuniversal**: a feasible mechanism in `F_delta` has positive slack.
- **Sometimes active**: the minimum slack is zero while the maximum is positive.
- **Never active**: the minimum slack is strictly positive within the declared tolerance band.
- **Inconclusive**: available objective/residual bounds do not justify one of the labels.

Equalities are accounting or structural identities. They have residual, rank, dependence, and
redundancy diagnostics; “active equality” is not a useful classification.

## The maximum-support correction

A strictly complementary maximum-support primal–dual pair separates universally active from
nonuniversally active inequalities. It does **not** distinguish a row slack at every optimum
from a row active only at boundary points. Both appear slack at a relative-interior primal
point and have zero multiplier.

Full analysis therefore minimizes every nonuniversal row slack. This targeted second pass is
what separates “sometimes active” from “never active.”

## Numerical algorithm

MechAudit:

1. starts from the independently audited baseline optimum;
2. constructs the explicit objective-band row for `F_delta`;
3. asks HiGHS for an IPM, no-crossover candidate with presolve disabled;
4. independently checks primal/dual residuals, objective error, and complementarity;
5. uses positive primal slack and positive dual coordinates only as provisional evidence;
6. in `fast` mode, runs targeted maximum-slack LPs where the provisional split or a positive
   delta requires them;
7. in `full` and `essentiality` modes, maximizes every selected row slack, then minimizes every
   nonuniversal row slack;
8. retains witnesses, verified optimized values, residuals, and failures for every
   authoritative task.

When the near-optimal face is widened, exact-face dual support is not enough. Provisional
universal rows are rechecked against the widened `F_delta`.

## Computational cost

The baseline mechanism-design LP and the post-optimal analysis have different scaling behavior.
If `r` inequalities are selected, full mode solves `r` maximum-slack LPs and as many as `r`
additional minimum-slack LPs, besides the baseline and partition solves. This row-wise work is
inherent in certifying each reported slack range; a fast baseline solve does not imply a fast
full-face report.

Use `fast` first, select only the economic families needed for the research question, and retain
the solve counts recorded in every report. Full analysis is convenient for models with hundreds
of rows, can become minute-scale around a thousand rows, and may be impractical for many
thousands of rows. These are hardware- and geometry-dependent observations rather than hard
limits; the [benchmark guide](benchmarks.md) includes a reproducible size sweep.

## Analysis levels

```python
fast = solution.analyze_optima(level="fast")
full = solution.analyze_optima(level="full", objective_delta=0.0)
advanced = solution.analyze_optima(
    level="essentiality",
    dual_ranges=True,
    constraint_families=("dsic", "ex_post_ir"),
    n_jobs=-1,
)
```

- `fast` emphasizes the universal/nonuniversal partition and uses targeted fallback solves.
- `full` computes both endpoints of nonuniversal slack ranges and reports universal,
  sometimes, never, or inconclusive.
- `essentiality` includes full activity information and exposes the same on-demand removal,
  ablation, and multiplier analyses.

`n_jobs=-1` uses up to four spawned worker processes for sufficiently large task batches and
stays serial for tiny batches where process startup would dominate. A positive integer requests
that many processes. Each HiGHS instance is restricted to one solver thread, model fingerprints
are checked after serialization, and result order is deterministic. In interactive entry points
that cannot safely be imported by spawned workers, execution falls back to serial. Failures and
limits produce retained inconclusive task records rather than silently disappearing.

## Delta and tolerances

The report serializes both the requested objective delta and the effective delta used to build
the face. If no explicit delta is given, the effective value is derived from the absolute and
relative objective tolerances.

A positive delta can change the economic classification. A row universal on the exact face may
become slack at a slightly suboptimal mechanism. Read every label as:

> across this delta-optimal face, at these reported feasibility and activity tolerances.

## Witnesses and slack ranges

Each activity record includes the selected slack, maximum/minimum slack when solved, transformed
candidate dual, solver status, method, and witness mechanisms where available.

```python
row = full.activity_for(full.sometimes_active[0])
print(row.to_dict())

tables = full.to_dataframes()
tables["optimal_face_activity"].to_csv("activity.csv", index=False)
```

Never clamp small negative numerical slacks when inspecting evidence. Classification uses
tolerances, while the raw optimized value is retained.

## Multiplier ranges

One solver multiplier is not a unique economic price on a degenerate dual face. Request all
ranges during analysis or compute one on demand:

```python
with_duals = solution.analyze_optima(level="full", dual_ranges=True)
price_range = with_duals.dual_range(with_duals.universally_active[0])
```

The explicit dual LP includes finite lower/upper variable-bound multipliers. Returned row
multipliers are transformed to the row's declared report units.

## Redundancy, cost, and ablation

```python
row = full.universally_active[0]

redundancy = full.redundancy(row)
cost = full.constraint_cost(row)
implementation = full.implementation_essential(row)
family = full.ablate_family("dsic")
```

- **Structural redundancy** removes a row and minimizes its slack over the remaining complete
  feasible system.
- **Objective essentiality** compares the relaxed objective interval with the baseline interval.
- **Implementation essentiality** asks whether removal enlarges the set of full near-optimal
  mechanisms even when it does not improve the objective.
- **Family ablation** removes every selected semantic row together and repeats the analysis.

`projection="full"` gives the exact full-mechanism row-violation test. Allocation- or payment-
only projections use one-sided coordinate-range witnesses: a separated coordinate proves
expansion, while failure to find one is explicitly not reported as proof of equal projections.

An individually redundant duplicate row can coexist with a nonredundant family. Objective
essentiality, structural redundancy, implementation essentiality, local activity, and universal
activity are independent axes of interpretation.
