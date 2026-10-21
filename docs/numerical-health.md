# Numerical health

Mechanism-design LPs can be economically correct and numerically fragile. Small type masses,
nearly duplicate conditional beliefs, large contingent transfers, redundant IC rows, and broad
payment ranges all deserve explicit diagnostics.

## Original units and row scaling

MechAudit compiles the economic model in original units, then creates a positively row-scaled
backend view. For a solver row scale `alpha > 0`,

```text
solver slack = alpha * original slack
```

The library divides by `alpha` before reporting a slack. BIC and interim-IR rows also record the
conditioning marginal, so conditional-utility slack is not confused with a small joint-weighted
number. If `beta > 0` converts an original slack to its reported unit, then, after sign
normalization, `reported dual = (alpha / beta) * solver dual`.

The HiGHS adapter rejects coefficients that would fall at or below the configured solver drop
threshold after scaling instead of allowing the backend to silently erase them.

## Health report

`NumericalHealthReport` records the diagnostics available for its workflow, including:

- minimum positive joint and marginal probability;
- allocation nonnegativity and normalization residuals;
- coefficient and row-scale ranges;
- maximum payments and payment-to-value scale ratio;
- independent primal, dual, and complementarity residuals when a solve exists;
- warnings for near-zero probability and conditioning concerns.

Conditional-belief conditioning is diagnosed only when the prior is correlated. Under an
independent prior, including any one-agent prior, every type shares the same conditional
beliefs; that rank deficiency is what independence means, so it produces no warning.
`details["independent_prior"]` records which case applied. The standalone
`cremer_mclean_diagnostic` still reports the rank of every prior.

Audit reports contain audit-level evidence. Design solutions augment it with solver-independent
LP residuals and backend KKT information. Missing diagnostics are represented as unavailable,
not silently replaced with zero.

## Tolerances

`TolerancePolicy` separates probability, feasibility, activity, objective absolute/relative,
optimal-partition primal/dual, and complementarity thresholds. This separation is intentional:
an economically meaningful epsilon-IC statement is not the same object as a solver feasibility
tolerance.

```python
from mechaudit import SolveOptions, TolerancePolicy

tolerances = TolerancePolicy(
    probability=1e-12,
    feasibility=1e-8,
    activity=1e-8,
    objective_absolute=1e-9,
    objective_relative=1e-9,
)
solution = problem.solve(options=SolveOptions(tolerances=tolerances))
```

Every optimal-face report persists its effective objective delta. Conclusions should be read as
“across this delta-optimal face at these tolerances,” never as exact symbolic identities.

## Conditional-belief diagnostics

`cremer_mclean_diagnostic(env)` constructs each agent's conditional-belief matrix and reports:

- numerical rank, full-row-rank status, and the labeled type count;
- singular values and smallest singular value;
- condition number;
- pairs of nearly duplicate conditional-belief rows;
- warnings tied to the agent and type labels.

```python
from mechaudit import cremer_mclean_diagnostic

report = cremer_mclean_diagnostic(env)
print(report.to_dict())
print(report.for_agent("buyer").to_pandas())
```

Full numerical rank is not a robustness guarantee. A very small positive singular value can
permit formal contingent-payment extraction while making payments extremely sensitive to tiny
prior perturbations. MechAudit reports this as conditioning evidence, not as a new economic
theorem or a formal certificate.
