"""The elevator-pitch example: selected activity is not invariant activity."""

from mechaudit import DSIC, ExpectedRevenue, ExPostIR, FiniteEnvironment, NoSubsidies, PrivateValues


def main() -> int:
    environment = FiniteEnvironment(
        agents={"buyer": ["low", "high"]},
        outcomes=["no_sale", "sale"],
        prior={"low": 0.5, "high": 0.5},
        values=PrivateValues(
            lambda agent, own_type, outcome: (
                {"low": 1.0, "high": 2.0}[own_type] if outcome == "sale" else 0.0
            )
        ),
    )
    problem = environment.design(
        objective=ExpectedRevenue(),
        requirements=(DSIC(), ExPostIR(), NoSubsidies()),
    )
    solution = problem.solve()

    print(solution.summary())
    print("\nSelected mechanism")
    print(solution.mechanism.to_pandas().allocation)
    print(solution.mechanism.to_pandas().payments)
    face = solution.analyze_optima(level="full")
    print("\n" + face.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
