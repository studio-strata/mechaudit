"""A finite one-buyer monopoly problem with a unique posted-price optimum."""

from mechaudit import DSIC, ExpectedRevenue, ExPostIR, FiniteEnvironment, NoSubsidies, PrivateValues


def main() -> int:
    environment = FiniteEnvironment(
        agents={"buyer": [1.0, 3.0]},
        outcomes=["no_sale", "sale"],
        prior={1.0: 0.5, 3.0: 0.5},
        values=PrivateValues(
            lambda agent, own_type, outcome: own_type if outcome == "sale" else 0.0
        ),
    )
    problem = environment.design(
        objective=ExpectedRevenue(),
        requirements=(DSIC(), ExPostIR(), NoSubsidies()),
    )
    solution = problem.solve()

    print(solution.summary())
    print(solution.mechanism.to_pandas().allocation)
    print(solution.mechanism.to_pandas().payments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
