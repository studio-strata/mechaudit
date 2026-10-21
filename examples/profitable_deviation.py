"""Audit a direct price menu with a profitable high-to-low deviation."""

from mechaudit import DSIC, DirectMechanism, ExPostIR, FiniteEnvironment, PrivateValues


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
    mechanism = DirectMechanism(
        allocation={
            "low": {"no_sale": 0.0, "sale": 1.0},
            "high": {"no_sale": 0.0, "sale": 1.0},
        },
        payments={"low": 1.0, "high": 2.0},
    )
    report = environment.audit(mechanism, (DSIC(), ExPostIR()))

    print(report.summary())
    print("\nWorst deviations")
    print(report.ic.worst_deviations())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
