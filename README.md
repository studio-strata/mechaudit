# MechAudit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23540591.svg)](https://doi.org/10.5281/zenodo.23540591)

MechAudit is a Python research workbench for finite direct mechanisms. It audits supplied
mechanisms, designs randomized mechanisms as sparse linear programs, and distinguishes
constraints that are active at one returned solution from constraints that are active across
the declared optimal or near-optimal face.

> **Project status:** MechAudit is research software. Numerical conclusions are floating-point,
> tolerance-relative diagnostics rather than formal certificates.

## Why MechAudit?

Mechanism-design code often loses the economic meaning of a constraint after it becomes a
solver row. MechAudit keeps agents, types, outcomes, deviations, timing, units, and stable row
identifiers attached to the numerical model and its reports.

It can answer:

- Is a finite mechanism DSIC, ex-post IC, or BIC under the declared information model?
- Does it satisfy ex-post, interim, or ex-ante participation and the requested budget rules?
- What is the most profitable deviation and the smallest additive IC epsilon?
- Which inequalities are active only at the selected optimum, and which are active at every
  mechanism on the declared near-optimal face?
- Is a correlated-type or large-transfer conclusion numerically fragile?
- What changes after removing a constraint or an economic family?

## Installation

MechAudit requires Python 3.10 or newer. HiGHS is the required optimization backend and is
installed through the `highspy` dependency.

Install the library and its HiGHS backend with:

```console
python -m pip install mechaudit
```

Install optional labeled xarray input support with:

```console
python -m pip install "mechaudit[xarray]"
```

Contributors and source users can install a checked-out tree in editable mode:

```console
python -m pip install -e ".[dev,xarray]"
```

See the [installation guide](https://studio-strata.github.io/mechaudit/installation/) for isolated
environments, development setup, and backend troubleshooting.

## Quick start

The following model has one buyer with values 1 and 2 and two explicit social outcomes.
Positive payments always mean money paid **from an agent to the designer**.

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

problem = env.design(
    objective=ExpectedRevenue(),
    requirements=(DSIC(), ExPostIR(), NoSubsidies()),
)
solution = problem.solve()

print(solution.objective_value)
print(solution.mechanism.to_pandas().payments)
print(solution.audit.summary())
```

The solver result is independently checked against the unscaled sparse model and audited again
in economic units before `solve()` returns. Optimal-face analysis remains a separate step:

```python
face = solution.analyze_optima(level="full", objective_delta=0.0)
print(face.summary())
```

Optimal-face analysis can be much more expensive than the baseline design solve. Full mode
solves at least one auxiliary LP per selected economic inequality and a second LP for every
nonuniversal row. Start with `level="fast"`, restrict `constraint_families` on larger models,
and consult the [scaling benchmark](https://studio-strata.github.io/mechaudit/benchmarks/) before
launching a full analysis with hundreds or thousands of rows. `n_jobs=-1` uses bounded
process-based workers for sufficiently large task batches.

Read the [optimal-face terminology](https://studio-strata.github.io/mechaudit/optimal-face/) before
treating any activity label as an invariant claim.

## What is included

- finite scalar or structured type spaces and explicit finite outcomes;
- independent or correlated finite joint priors;
- private or interdependent quasi-linear values;
- mapping, callback, NumPy, pandas, and optional xarray mechanism inputs;
- DSIC, ex-post IC, BIC, ex-post/interim/ex-ante IR, payment, budget, capacity,
  symmetry, and custom linear requirements;
- revenue, welfare, residual-surplus, weighted-utility, and custom linear objectives;
- deterministic JSON, labeled DataFrame exports, and domain/mechanism/model fingerprints;
- optimal-face activity, dual ranges, redundancy, essentiality, family ablation, and
  conditional-belief diagnostics.

The current scope does not optimize deterministic rules, discretize continuous spaces, model
indirect or dynamic mechanisms, or provide exact theorem-prover certificates. Fixed
deterministic mechanisms remain auditable.

## Documentation and examples

Start with the repository's
[documentation source](https://github.com/studio-strata/mechaudit/tree/main/docs) and
[getting-started guide](https://github.com/studio-strata/mechaudit/blob/main/docs/getting-started.md).
The canonical documentation site is <https://studio-strata.github.io/mechaudit/>.
The repository also contains standalone, tested examples:

- [guided Jupyter workflow](https://github.com/studio-strata/mechaudit/blob/main/examples/mechaudit_workflow.ipynb);
- [discrete Myerson design](https://github.com/studio-strata/mechaudit/blob/main/examples/discrete_myerson.py);
- [profitable-deviation audit](https://github.com/studio-strata/mechaudit/blob/main/examples/profitable_deviation.py);
- [bilateral trade](https://github.com/studio-strata/mechaudit/blob/main/examples/bilateral_trade.py);
- [correlated interdependent values](https://github.com/studio-strata/mechaudit/blob/main/examples/correlated_interdependent.py);
- [Crémer–McLean diagnostics](https://github.com/studio-strata/mechaudit/blob/main/examples/cremer_mclean.py);
- [multidimensional buyer](https://github.com/studio-strata/mechaudit/blob/main/examples/multidimensional_buyer.py);
- [revenue-equivalent optima](https://github.com/studio-strata/mechaudit/blob/main/examples/revenue_equivalent_optima.py).

## Development and project policy

MechAudit is maintained by Alessio Staffini of Studio Strata, Tokyo, Japan.

See [CONTRIBUTING.md](https://github.com/studio-strata/mechaudit/blob/main/CONTRIBUTING.md) for the
development workflow and required checks. Project governance and attribution are recorded in
[MAINTAINERS.md](https://github.com/studio-strata/mechaudit/blob/main/MAINTAINERS.md), release steps in
[RELEASING.md](https://github.com/studio-strata/mechaudit/blob/main/RELEASING.md), and notable changes
in [CHANGELOG.md](https://github.com/studio-strata/mechaudit/blob/main/CHANGELOG.md).

Use the [repository issue tracker](https://github.com/studio-strata/mechaudit/issues) for reproducible
bugs and feature proposals. Security concerns
must follow [SECURITY.md](https://github.com/studio-strata/mechaudit/blob/main/SECURITY.md); conduct
concerns follow
[CODE_OF_CONDUCT.md](https://github.com/studio-strata/mechaudit/blob/main/CODE_OF_CONDUCT.md).

## Citation

If MechAudit contributes to research, cite the software using
[CITATION.cff](https://github.com/studio-strata/mechaudit/blob/main/CITATION.cff). Every release is
archived on Zenodo under the DOI
[10.5281/zenodo.23540591](https://doi.org/10.5281/zenodo.23540591), which always resolves to the
latest version; the Zenodo record also lists a DOI for each individual release. Record the exact
library version, HiGHS version, fingerprints, effective objective delta, and tolerance policy
used in the reported computation.

## License

MechAudit is copyright Alessio Staffini and released under the
[MIT License](https://github.com/studio-strata/mechaudit/blob/main/LICENSE).
