"""Reproducible smoke benchmark for the public MechAudit workflow."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
import tracemalloc
from functools import partial
from pathlib import Path
from typing import Any

import highspy

import mechaudit
from mechaudit import (
    DSIC,
    ExpectedRevenue,
    ExPostIR,
    FiniteEnvironment,
    JointPrior,
    NoSubsidies,
    PrivateValues,
)

EXPECTED_OBJECTIVE = 1.5


def _problem() -> Any:
    environment = FiniteEnvironment(
        agents={"buyer": ("low", "high")},
        outcomes=("no_sale", "sale"),
        prior={"low": 0.5, "high": 0.5},
        values=PrivateValues(
            lambda agent, own_type, outcome: (
                {"low": 1.0, "high": 3.0}[own_type]
                if agent == "buyer" and outcome == "sale"
                else 0.0
            )
        ),
    )
    return environment.design(
        objective=ExpectedRevenue(),
        requirements=(DSIC(), ExPostIR()),
    )


def _scaling_problem(n_agents: int, n_types: int) -> Any:
    """Build the symmetric public-outcome family used for the size sweep."""

    if n_agents <= 0 or n_types <= 0:
        raise ValueError("scaling dimensions must be positive")
    agents = tuple(f"agent_{index}" for index in range(n_agents))
    types = tuple(range(1, n_types + 1))
    type_spaces = {agent: types for agent in agents}
    prior = JointPrior.from_independent(
        {agent: {type_value: 1.0 / n_types for type_value in types} for agent in agents}
    )
    environment = FiniteEnvironment(
        agents=type_spaces,
        outcomes=("off", "on"),
        prior=prior,
        values=PrivateValues(
            lambda _agent, own_type, outcome: float(own_type) if outcome == "on" else 0.0
        ),
    )
    return environment.design(
        objective=ExpectedRevenue(),
        requirements=(DSIC(), ExPostIR(), NoSubsidies()),
    )


def _timed(callable_: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    result = callable_()
    return result, time.perf_counter() - started


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "minimum": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "maximum": max(values),
    }


def run_benchmark(repeat: int, *, n_jobs: int = 1) -> dict[str, Any]:
    """Run the analytical smoke model and return JSON-compatible evidence."""

    if repeat <= 0:
        raise ValueError("repeat must be a positive integer")

    measurements: list[dict[str, Any]] = []
    tracemalloc.start()
    try:
        for index in range(repeat):
            problem = _problem()
            compiled, compile_seconds = _timed(problem.compile)
            solution, solve_seconds = _timed(problem.solve)
            fast, fast_seconds = _timed(
                partial(
                    solution.analyze_optima,
                    level="fast",
                    objective_delta=0.0,
                    n_jobs=n_jobs,
                )
            )
            full, full_seconds = _timed(
                partial(
                    solution.analyze_optima,
                    level="full",
                    objective_delta=0.0,
                    n_jobs=n_jobs,
                )
            )

            if not solution.audit.passed:
                raise RuntimeError("benchmark solution failed its independent audit")
            if abs(solution.objective_value - EXPECTED_OBJECTIVE) > 1e-8:
                raise RuntimeError(
                    "benchmark objective changed: "
                    f"expected {EXPECTED_OBJECTIVE}, received {solution.objective_value}"
                )
            if fast.failures or full.failures:
                raise RuntimeError("benchmark optimal-face analysis retained task failures")

            measurements.append(
                {
                    "repeat": index + 1,
                    "compile_seconds": compile_seconds,
                    "solve_seconds": solve_seconds,
                    "fast_seconds": fast_seconds,
                    "full_seconds": full_seconds,
                    "objective_value": solution.objective_value,
                    "model_fingerprint": compiled.fingerprint,
                    "mechanism_fingerprint": solution.mechanism.fingerprint,
                    "fast_solve_counts": fast.solve_counts.to_dict(),
                    "full_solve_counts": full.solve_counts.to_dict(),
                }
            )
        _current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    fingerprints = {item["model_fingerprint"] for item in measurements}
    if len(fingerprints) != 1:
        raise RuntimeError("model fingerprint changed across benchmark repetitions")

    timing_keys = (
        "compile_seconds",
        "solve_seconds",
        "fast_seconds",
        "full_seconds",
    )
    return {
        "benchmark": "two_type_revenue_design_v1",
        "repeat": repeat,
        "n_jobs": n_jobs,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "mechaudit": mechaudit.__version__,
            "highs": highspy.Highs().version(),
        },
        "analytical_expected_objective": EXPECTED_OBJECTIVE,
        "model_fingerprint": measurements[0]["model_fingerprint"],
        "peak_traced_python_bytes": peak_bytes,
        "timing_seconds": {
            key.removesuffix("_seconds"): _summary([float(item[key]) for item in measurements])
            for key in timing_keys
        },
        "measurements": measurements,
    }


def run_scaling_sweep(
    cases: tuple[tuple[int, int], ...], *, n_jobs: int = 1
) -> list[dict[str, Any]]:
    """Measure how row-wise auxiliary analysis grows across finite domains."""

    measurements: list[dict[str, Any]] = []
    for n_agents, n_types in cases:
        problem = _scaling_problem(n_agents, n_types)
        compiled, compile_seconds = _timed(problem.compile)
        solution, solve_seconds = _timed(problem.solve)
        full, full_seconds = _timed(
            partial(
                solution.analyze_optima,
                level="full",
                objective_delta=0.0,
                n_jobs=n_jobs,
            )
        )
        if full.failures:
            raise RuntimeError(
                f"scaling case {n_agents}x{n_types} retained auxiliary task failures"
            )
        model = compiled.model
        measurements.append(
            {
                "agents": n_agents,
                "types_per_agent": n_types,
                "profiles": n_types**n_agents,
                "rows": model.n_rows,
                "classifiable_rows": sum(row.classifiable for row in model.rows),
                "columns": model.n_columns,
                "compile_seconds": compile_seconds,
                "solve_seconds": solve_seconds,
                "full_seconds": full_seconds,
                "full_solve_counts": full.solve_counts.to_dict(),
                "model_fingerprint": compiled.fingerprint,
            }
        )
    return measurements


def _parse_sweep_case(value: str) -> tuple[int, int]:
    try:
        raw_agents, raw_types = value.lower().split("x", maxsplit=1)
        result = (int(raw_agents), int(raw_types))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("sweep cases must look like AGENTSxTYPES") from exc
    if result[0] <= 0 or result[1] <= 0:
        raise argparse.ArgumentTypeError("sweep dimensions must be positive")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=3, help="positive repetition count")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="auxiliary worker count; -1 selects bounded automatic process workers",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="also run the finite-domain scaling sweep (can take several minutes)",
    )
    parser.add_argument(
        "--sweep-case",
        action="append",
        type=_parse_sweep_case,
        metavar="AGENTSxTYPES",
        help="scaling case; repeat the option to override the default sweep",
    )
    parser.add_argument("--json", type=Path, help="optional path for the JSON report")
    return parser


def main() -> int:
    """Run the command-line benchmark."""

    arguments = _parser().parse_args()
    report = run_benchmark(arguments.repeat, n_jobs=arguments.jobs)
    if arguments.sweep:
        cases = tuple(arguments.sweep_case or ((2, 4), (3, 3), (3, 4)))
        report["scaling_sweep"] = run_scaling_sweep(cases, n_jobs=arguments.jobs)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if arguments.json is not None:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
