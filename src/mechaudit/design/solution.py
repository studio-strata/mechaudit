"""Selected optimum and candidate-assessment result objects."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from mechaudit._serialization import dumps_report, write_report
from mechaudit.audit.report import MechanismAuditReport
from mechaudit.backends.base import SolveResult
from mechaudit.design.compiler import CompiledDesign
from mechaudit.diagnostics.health import NumericalHealthReport
from mechaudit.mechanisms.direct import DirectMechanism
from mechaudit.options import OptimalFaceLevel, OptimalFaceOptions, TolerancePolicy

if TYPE_CHECKING:
    from mechaudit.design.problem import DesignProblem
    from mechaudit.optimality.report import OptimalFaceReport


@dataclass(frozen=True)
class SelectedActivity:
    """Original/report-unit slack for a row at the returned optimum."""

    row_index: int
    constraint_id: str
    family: str
    slack: float
    active: bool
    tolerance: float
    dual_multiplier: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "constraint_id": self.constraint_id,
            "family": self.family,
            "slack": self.slack,
            "active_at_selected_solution": self.active,
            "activity_tolerance": self.tolerance,
            "dual_multiplier": self.dual_multiplier,
        }


@dataclass(frozen=True)
class DesignSolution:
    """One solver-selected optimal mechanism plus an independent audit."""

    problem: DesignProblem
    compiled: CompiledDesign
    mechanism: DirectMechanism
    objective_value: float
    audit: MechanismAuditReport
    backend_result: SolveResult
    tolerances: TolerancePolicy
    numerical_health: NumericalHealthReport
    selected_activity: tuple[SelectedActivity, ...]

    @property
    def model_fingerprint(self) -> str:
        return self.compiled.fingerprint

    @property
    def solver_name(self) -> str:
        return self.backend_result.solver_name

    @property
    def solver_version(self) -> str:
        return self.backend_result.solver_version

    @property
    def active_at_selected_solution(self) -> tuple[SelectedActivity, ...]:
        return tuple(item for item in self.selected_activity if item.active)

    def analyze_optima(
        self,
        level: str = "fast",
        *,
        options: OptimalFaceOptions | None = None,
        dual_ranges: bool = False,
        constraint_families: list[str] | tuple[str, ...] | None = None,
        n_jobs: int = 1,
        objective_delta: float | None = None,
    ) -> OptimalFaceReport:
        """Analyze invariant and noninvariant activity across optima."""

        from mechaudit.optimality.analysis import analyze_optimal_face

        if options is not None and (
            level != "fast"
            or dual_ranges
            or constraint_families is not None
            or n_jobs != 1
            or objective_delta is not None
        ):
            raise ValueError("pass either options or individual optimal-face arguments, not both")
        selected_options = options or OptimalFaceOptions(
            level=cast(OptimalFaceLevel, level),
            tolerances=self.tolerances,
            objective_delta=objective_delta,
            dual_ranges=dual_ranges,
            constraint_families=(
                None if constraint_families is None else tuple(constraint_families)
            ),
            n_jobs=n_jobs,
        )
        return analyze_optimal_face(self, selected_options)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_fingerprint": self.model_fingerprint,
            "mechanism_fingerprint": self.mechanism.fingerprint,
            "objective_name": self.compiled.objective.name,
            "objective_value": self.objective_value,
            "solver": {
                "name": self.solver_name,
                "version": self.solver_version,
                "status": self.backend_result.status.value,
                "message": self.backend_result.message,
                "runtime_seconds": self.backend_result.run_time_seconds,
                "simplex_iterations": self.backend_result.simplex_iterations,
                "ipm_iterations": self.backend_result.ipm_iterations,
                "crossover_iterations": self.backend_result.crossover_iterations,
            },
            "tolerances": self.tolerances.__dict__,
            "selected_activity": [item.to_dict() for item in self.selected_activity],
            "audit": self.audit.to_dict(),
            "numerical_health": self.numerical_health.to_dict(),
            "mechanism": self.mechanism.to_dict(),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return dumps_report("design_solution", self.to_dict(), indent=indent)

    def write_json(self, path: str | Path, *, indent: int = 2) -> None:
        write_report(path, "design_solution", self.to_dict(), indent=indent)

    def to_dataframes(self) -> dict[str, pd.DataFrame]:
        tables = self.audit.to_dataframes()
        mechanism_tables = self.mechanism.to_pandas()
        return {
            **tables,
            "selected_activity": pd.DataFrame.from_records(
                [item.to_dict() for item in self.selected_activity]
            ),
            "allocation": mechanism_tables.allocation,
            "payments": mechanism_tables.payments,
        }

    def summary(self) -> str:
        return "\n".join(
            (
                "Design solution",
                "---------------",
                f"Objective:                    {self.compiled.objective.name}",
                f"Optimal value:                {self.objective_value:.10g}",
                f"Solver:                       {self.solver_name} {self.solver_version}",
                "Independent audit:            {}".format(
                    "PASSED" if self.audit.passed else "FAILED"
                ),
                f"Active at selected solution:  {len(self.active_at_selected_solution)}",
                f"Model fingerprint:            {self.model_fingerprint}",
            )
        )


def build_selected_activity(
    compiled: CompiledDesign,
    result: SolveResult,
    tolerances: TolerancePolicy,
) -> tuple[SelectedActivity, ...]:
    """Recompute local activity from the original unscaled matrix."""

    if result.primal is None:
        return ()
    residuals = compiled.model.evaluate_primal(result.primal)
    records: list[SelectedActivity] = []
    for index in compiled.economic_row_indices:
        row = compiled.model.rows[index]
        reported_slack = row.scale.original_slack_to_reported(float(residuals.row_slack[index]))
        dual = None
        if result.row_dual is not None:
            dual = float(result.row_dual[index])
        records.append(
            SelectedActivity(
                row_index=index,
                constraint_id=row.key,
                family=row.family,
                slack=reported_slack,
                active=reported_slack <= tolerances.activity,
                tolerance=tolerances.activity,
                dual_multiplier=dual,
            )
        )
    return tuple(records)


def with_solver_health(
    audit_health: NumericalHealthReport, result: SolveResult
) -> NumericalHealthReport:
    """Attach backend residual information to the independent audit report."""

    info = dict(result.solver_info)
    primal = result.residuals.max_violation if result.residuals is not None else None
    dual = info.get("max_dual_infeasibility")
    complementarity = info.get("max_complementarity_violation")
    return replace(
        audit_health,
        primal_residual=primal,
        dual_residual=float(dual) if isinstance(dual, (int, float)) else None,
        complementarity_residual=(
            float(complementarity) if isinstance(complementarity, (int, float)) else None
        ),
    )
