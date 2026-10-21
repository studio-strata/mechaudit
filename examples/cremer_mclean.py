"""Track conditional-belief conditioning on a path toward independence."""

from mechaudit import FiniteEnvironment, InterdependentValues, cremer_mclean_diagnostic


def environment_at(epsilon: float) -> FiniteEnvironment:
    # Uniform own-type marginals with conditional matrix
    # [[1/2+eps, 1/2-eps], [1/2-eps, 1/2+eps]].
    prior = {
        ("low", "a"): 0.5 * (0.5 + epsilon),
        ("low", "b"): 0.5 * (0.5 - epsilon),
        ("high", "a"): 0.5 * (0.5 - epsilon),
        ("high", "b"): 0.5 * (0.5 + epsilon),
    }
    return FiniteEnvironment(
        agents={"buyer": ["low", "high"], "signal": ["a", "b"]},
        outcomes=["outside"],
        prior=prior,
        values=InterdependentValues(lambda agent, types, outcome: 0.0),
    )


def main() -> int:
    for epsilon in (0.1, 1e-3, 1e-6, 0.0):
        report = cremer_mclean_diagnostic(environment_at(epsilon))
        buyer = report.for_agent("buyer")
        print(
            f"epsilon={epsilon:<8g} rank={buyer.rank} "
            f"sigma_min={buyer.smallest_singular_value:.6g} "
            f"condition={buyer.condition_number:.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
