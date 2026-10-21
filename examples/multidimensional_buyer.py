"""Revenue design for a buyer with two-dimensional values for two items."""

from mechaudit import (
    DSIC,
    ExpectedRevenue,
    ExPostIR,
    FiniteEnvironment,
    JointPrior,
    NoSubsidies,
    PrivateValues,
)

TYPES = [
    {"item_1": 1.0, "item_2": 1.0},
    {"item_1": 1.0, "item_2": 3.0},
    {"item_1": 3.0, "item_2": 1.0},
    {"item_1": 3.0, "item_2": 3.0},
]
OUTCOMES = ["none", "item_1", "item_2", "bundle"]


def value(agent, own_type, outcome):
    if outcome == "item_1":
        return own_type["item_1"]
    if outcome == "item_2":
        return own_type["item_2"]
    if outcome == "bundle":
        return own_type["item_1"] + own_type["item_2"]
    return 0.0


def main() -> int:
    prior = JointPrior.from_records(
        ({"buyer": type_value, "probability": 0.25} for type_value in TYPES),
        agent_order=("buyer",),
        type_spaces={"buyer": TYPES},
    )
    environment = FiniteEnvironment(
        agents={"buyer": TYPES},
        outcomes=OUTCOMES,
        prior=prior,
        values=PrivateValues(value),
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
