"""Finite bilateral trade under DSIC, ex-post IR, and strong budget balance."""

from mechaudit import (
    DSIC,
    ExpectedWelfare,
    ExPostIR,
    FiniteEnvironment,
    PrivateValues,
    StrongBudgetBalance,
)


def value(agent, own_type, outcome):
    if outcome == "no_trade":
        return 0.0
    return own_type if agent == "buyer" else -own_type


def main() -> int:
    environment = FiniteEnvironment(
        agents={"buyer": [0.0, 2.0], "seller": [0.0, 1.0]},
        outcomes=["no_trade", "trade"],
        prior={
            (0.0, 0.0): 0.25,
            (0.0, 1.0): 0.25,
            (2.0, 0.0): 0.25,
            (2.0, 1.0): 0.25,
        },
        values=PrivateValues(value),
    )
    problem = environment.design(
        objective=ExpectedWelfare(),
        requirements=(DSIC(), ExPostIR(), StrongBudgetBalance()),
    )
    solution = problem.solve()

    print(solution.summary())
    print(solution.mechanism.to_pandas().allocation)
    print(solution.mechanism.to_pandas().payments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
