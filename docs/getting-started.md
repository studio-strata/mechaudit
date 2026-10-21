# Getting started

MechAudit separates three workflows:

1. audit a supplied direct mechanism;
2. design a randomized direct mechanism;
3. assess a supplied mechanism against a solved design problem.

A fixed audit does not have an objective or an optimal face. Optimal-face conclusions only
belong to a successfully solved `DesignSolution`.

## Installation

Create a virtual environment and install MechAudit:

```console
python -m venv .venv
python -m pip install mechaudit
```

The required optimization backend is HiGHS, installed through the `highspy` dependency. See
[Installation](installation.md) for release-wheel, source-checkout, optional xarray, and
contributor setups.

## A two-type buyer

This model has one buyer, values 1 and 2, and an explicit social outcome: either no sale or a
sale. Labels stay attached to every array row, audit check, and exported table.

```python
from mechaudit import (
    DSIC,
    ExPostIR,
    ExpectedRevenue,
    FiniteEnvironment,
    NoSubsidies,
    PrivateValues,
)

env = FiniteEnvironment(
    agents={"buyer": ["low", "high"]},
    outcomes=["no_sale", "sale"],
    prior={"low": 0.5, "high": 0.5},
    values=PrivateValues(
        lambda agent, own_type, outcome: (
            {"low": 1.0, "high": 2.0}[own_type] if agent == "buyer" and outcome == "sale" else 0.0
        )
    ),
)
```

`PrivateValues` is an explicit semantic declaration. MechAudit never guesses private values by
inspecting a callback. Use `InterdependentValues` when value can depend on the full true type
profile; private-value `DSIC` is then rejected, while `ExPostIC` remains available.

The environment validates and tabulates values eagerly. Its `domain_fingerprint`
covers only the ordered agents, type axes, and outcomes, so the same mechanism rule can be
rebound under a different prior or valuation table. Numerical model and report fingerprints
separately capture the economic data used in a solve or audit.

## Design and independently certify a mechanism

```python
problem = env.design(
    objective=ExpectedRevenue(),
    requirements=(DSIC(), ExPostIR(), NoSubsidies()),
)
solution = problem.solve()

print(solution.objective_value)
print(solution.mechanism.to_pandas().allocation)
print(solution.mechanism.to_pandas().payments)
print(solution.audit.summary())
```

Every allocation variable is the probability of one explicit social outcome. These are not
independent per-agent allocation probabilities. Payments are report-contingent, and a positive
payment always means money paid by the agent to the designer.

The solver result is not trusted on its own. MechAudit reconstructs the mechanism, recomputes
the sparse-model residuals in original units, and runs the independent economic audit before
returning a `DesignSolution`.

## Study every near-optimal mechanism

The example has several revenue-optimal mechanisms. A solver returns only one of them, so a
zero slack at that selected solution is not an invariant conclusion.

```python
face = solution.analyze_optima(level="full", objective_delta=0.0)

print(face.summary())
print(face.universally_active)
print(face.sometimes_active)
print(face.never_active)
```

Full analysis maximizes every selected inequality slack to decide universal versus non-universal
activity. It then minimizes the slack of each non-universal row to distinguish “sometimes
active” from “never active.” The second pass is essential: one relative-interior point cannot
make that distinction.

The report always records the requested and effective objective tolerance. Treat the labels as
floating-point, tolerance-relative classifications rather than formal certificates.

## Audit a supplied mechanism

Mechanisms can be supplied as mappings, callbacks, NumPy arrays, pandas tables, or optional
xarray objects. The following fixed-price menu is not DSIC:

```python
from mechaudit import DirectMechanism

candidate = DirectMechanism(
    allocation={
        "low": {"no_sale": 0.0, "sale": 1.0},
        "high": {"no_sale": 0.0, "sale": 1.0},
    },
    payments={"low": 1.0, "high": 2.0},
)

audit = env.audit(candidate, (DSIC(), ExPostIR()))
print(audit.summary())
print(audit.ic.worst_deviations(k=5))
```

The high type can report `low` and gain 1. The report therefore has `epsilon_ic == 1.0` and a
fully labeled deviation witness. Core allocation nonnegativity and normalization are audited
even when the requirements tuple does not request them.

## Export numerical artifacts

The audit, design-solution, and optimal-face reports provide dictionary, JSON, and long-form
DataFrame exports. Candidate and scenario results provide corresponding pandas/JSON helpers:

```python
audit.write_json("audit.json")
tables = solution.to_dataframes()
tables["selected_activity"].to_csv("selected_activity.csv", index=False)
```

JSON exports preserve numerical evidence, fingerprints, semantic row identifiers, solver
metadata, and tolerances. They are not executable environment snapshots when inputs contain
Python callbacks.

Continue with [Audit a mechanism](auditing.md), [Design a mechanism](designing.md), and
[Optimal-face terminology](optimal-face.md).

For an executable tour of all three workflows, open the
[guided Jupyter notebook](https://github.com/studio-strata/mechaudit/blob/main/examples/mechaudit_workflow.ipynb).
