"""Thin, reproducible batches of economic formulation changes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from mechaudit._serialization import dumps_report, write_report
from mechaudit.design.solution import DesignSolution
from mechaudit.exceptions import MechAuditError


@dataclass(frozen=True)
class ScenarioResult:
    """One scenario's successful solution or labeled failure."""

    name: str
    status: str
    solution: DesignSolution | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        if self.solution is None:
            return {"name": self.name, "status": self.status, "error": self.error}
        return {
            "name": self.name,
            "status": self.status,
            "objective_name": self.solution.compiled.objective.name,
            "objective_value": self.solution.objective_value,
            "model_fingerprint": self.solution.model_fingerprint,
            "mechanism_fingerprint": self.solution.mechanism.fingerprint,
            "audit_passed": self.solution.audit.passed,
            "solver_runtime_seconds": self.solution.backend_result.run_time_seconds,
        }


@dataclass(frozen=True)
class ScenarioComparison:
    """Side-by-side results for a finite set of formulation changes."""

    results: tuple[ScenarioResult, ...]

    @property
    def reports(self) -> dict[str, DesignSolution]:
        return {
            result.name: result.solution for result in self.results if result.solution is not None
        }

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame.from_records([result.to_dict() for result in self.results]).set_index(
            "name"
        )

    def to_dict(self) -> dict[str, Any]:
        return {"scenarios": [result.to_dict() for result in self.results]}

    def to_json(self, *, indent: int = 2) -> str:
        return dumps_report("scenario_comparison", self.to_dict(), indent=indent)

    def write_json(self, path: str | Path, *, indent: int = 2) -> None:
        write_report(path, "scenario_comparison", self.to_dict(), indent=indent)

    def summary(self) -> str:
        lines = ["Scenario comparison", "-------------------"]
        for result in self.results:
            if result.solution is None:
                lines.append("{:<24} {}".format(result.name + ":", result.status))
            else:
                lines.append(
                    "{:<24} {:<10} objective={:.10g}".format(
                        result.name + ":", result.status, result.solution.objective_value
                    )
                )
        return "\n".join(lines)


def _match_requirement(existing: Any, target: Any) -> bool:
    if existing == target:
        return True
    if isinstance(target, str):
        return getattr(existing, "family", None) == target
    return type(existing) is type(target)


def _replace_requirements(requirements: list[Any], replacement: Any) -> None:
    if isinstance(replacement, Mapping):
        pairs = tuple(replacement.items())
    else:
        values = tuple(replacement)
        if len(values) == 2 and not isinstance(values[0], (tuple, list)):
            pairs = ((values[0], values[1]),)
        else:
            pairs = tuple(tuple(pair) for pair in values)
    for old, new in pairs:
        indices = [
            index
            for index, existing in enumerate(requirements)
            if _match_requirement(existing, old)
        ]
        if not indices:
            raise ValueError(f"scenario replacement did not match {old!r}")
        first = indices[0]
        requirements[:] = [
            existing for index, existing in enumerate(requirements) if index not in set(indices)
        ]
        requirements.insert(first, new)


def compare_scenarios(
    base_problem: Any, scenarios: Mapping[str, Mapping[str, Any]]
) -> ScenarioComparison:
    """Solve named additions, replacements, removals, and prior perturbations."""

    from mechaudit.design.problem import DesignProblem

    if not scenarios:
        raise ValueError("at least one scenario is required")
    results: list[ScenarioResult] = []
    for name, changes in scenarios.items():
        if not isinstance(name, str) or not name:
            raise ValueError("scenario names must be nonempty strings")
        allowed = {"add", "remove", "replace", "prior", "objective", "solve_options"}
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"unknown scenario keys: {sorted(unknown)}")
        requirements = list(base_problem.requirements)
        for target in tuple(changes.get("remove", ())):
            requirements = [
                existing for existing in requirements if not _match_requirement(existing, target)
            ]
        if "replace" in changes:
            _replace_requirements(requirements, changes["replace"])
        requirements.extend(tuple(changes.get("add", ())))
        environment = (
            base_problem.environment.with_prior(changes["prior"])
            if "prior" in changes
            else base_problem.environment
        )
        problem = DesignProblem(
            environment=environment,
            objective=changes.get("objective", base_problem.objective),
            requirements=requirements,
            randomized=True,
        )
        try:
            solution = problem.solve(options=changes.get("solve_options"))
        except (MechAuditError, ValueError) as exc:
            results.append(
                ScenarioResult(
                    name=name,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
        else:
            results.append(ScenarioResult(name=name, status="optimal", solution=solution))
    return ScenarioComparison(tuple(results))
