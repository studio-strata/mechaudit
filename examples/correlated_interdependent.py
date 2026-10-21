"""BIC and ex-post IC are distinct in a correlated interdependent environment."""

from mechaudit import BIC, DirectMechanism, ExPostIC, FiniteEnvironment, InterdependentValues


def main() -> int:
    environment = FiniteEnvironment(
        agents={"buyer": ["low", "high"], "signal": ["a", "b"]},
        outcomes=["none", "allocate"],
        prior={
            ("low", "a"): 0.4,
            ("low", "b"): 0.1,
            ("high", "a"): 0.1,
            ("high", "b"): 0.4,
        },
        values=InterdependentValues(
            lambda agent, types, outcome: (
                (1.0 if types["signal"] == "a" else 2.0)
                if agent == "buyer" and outcome == "allocate"
                else 0.0
            )
        ),
    )
    mechanism = DirectMechanism(
        allocation=lambda report, outcome: 1.0 if outcome == "allocate" else 0.0,
        payments=lambda report, agent: (
            {
                ("low", "a"): 0.0,
                ("low", "b"): 2.0,
                ("high", "a"): 2.0,
                ("high", "b"): 0.0,
            }[(report["buyer"], report["signal"])]
            if agent == "buyer"
            else 0.0
        ),
    )
    bic = environment.audit(mechanism, (BIC(agents=("buyer",)),))
    ex_post = environment.audit(mechanism, (ExPostIC(agents=("buyer",)),))

    print("BIC")
    print(bic.summary())
    print("\nEx-post IC")
    print(ex_post.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
