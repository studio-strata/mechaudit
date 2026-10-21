"""User-facing finite randomized direct-mechanism design workflow."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from mechaudit.backends import (
    HighsBackend,
    LPBackend,
    ModelStatus,
    SolverConfig,
)
from mechaudit.core.environment import FiniteEnvironment
from mechaudit.design.assessment import CandidateAssessment
from mechaudit.design.compiler import CompiledDesign, compile_design_problem
from mechaudit.design.solution import (
    DesignSolution,
    build_selected_activity,
    with_solver_health,
)
from mechaudit.exceptions import (
    InfeasibleDesignError,
    NumericalCertificationError,
    SolverError,
    UnboundedDesignError,
)
from mechaudit.mechanisms.direct import DirectMechanism
from mechaudit.objectives import ExpectedRevenue, LinearObjective
from mechaudit.options import AuditOptions, SolveMethod, SolveOptions
from mechaudit.requirements.budget import PaymentBounds
from mechaudit.requirements.participation import ExAnteIR, ExPostIR, InterimIR

if TYPE_CHECKING:
    from mechaudit.experiments.scenarios import ScenarioComparison
    from mechaudit.requirements.base import Requirement


def _solver_config(options: SolveOptions) -> SolverConfig:
    solver = {
        SolveMethod.AUTO: "simplex",
        SolveMethod.SIMPLEX: "simplex",
        SolveMethod.IPM: "ipm",
    }[options.method]
    crossover = {None: "choose", True: "on", False: "off"}[options.crossover]
    tolerances = options.tolerances
    solver_tolerance = max(1e-10, min(tolerances.partition_primal, 1e-7))
    dual_tolerance = max(1e-10, min(tolerances.partition_dual, 1e-7))
    return SolverConfig(
        solver=solver,
        presolve="on" if options.presolve else "off",
        run_crossover=crossover,
        threads=options.threads or 1,
        primal_feasibility_tolerance=solver_tolerance,
        dual_feasibility_tolerance=dual_tolerance,
        ipm_optimality_tolerance=max(solver_tolerance, dual_tolerance),
        time_limit=options.time_limit,
        random_seed=options.random_seed or 0,
        extra_options=options.backend_options,
    )


def _backend(value: str | LPBackend) -> LPBackend:
    if isinstance(value, LPBackend):
        return value
    if value == "highs":
        return HighsBackend()
    raise ValueError("unsupported backend; expected 'highs' or an LPBackend instance")


class DesignProblem:
    """A finite randomized direct-mechanism LP before solver selection.

    ``require`` is a small mutable builder convenience for the documented
    research workflow. Any existing compiled cache is discarded
    when a requirement is appended. ``with_requirements`` is available for code
    that prefers immutable-style branching.
    """

    def __init__(
        self,
        *,
        environment: FiniteEnvironment,
        objective: LinearObjective,
        requirements: Iterable[Requirement] = (),
        randomized: bool = True,
    ) -> None:
        if not isinstance(environment, FiniteEnvironment):
            raise TypeError("environment must be a FiniteEnvironment")
        if not isinstance(objective, LinearObjective):
            raise TypeError("objective must be a LinearObjective")
        if not randomized:
            raise NotImplementedError(
                "randomized=False is not supported because deterministic-mechanism "
                "optimization requires a MILP; "
                "fixed deterministic mechanisms can still be audited"
            )
        self.environment = environment
        self.objective = objective
        self.randomized = True
        self._requirements = list(requirements)
        self._compiled_cache: CompiledDesign | None = None
        self._solution_cache: DesignSolution | None = None

    @property
    def requirements(self) -> tuple[Requirement, ...]:
        return tuple(self._requirements)

    def require(self, *requirements: Requirement) -> DesignProblem:
        """Append economic restrictions and return this builder."""

        self._requirements.extend(requirements)
        self._compiled_cache = None
        self._solution_cache = None
        return self

    def with_requirements(self, *requirements: Requirement) -> DesignProblem:
        """Return a separate problem with additional restrictions."""

        return type(self)(
            environment=self.environment,
            objective=self.objective,
            requirements=(*self.requirements, *requirements),
            randomized=True,
        )

    def compile(self) -> CompiledDesign:
        """Compile and cache the immutable sparse numerical model."""

        if self._compiled_cache is None:
            self._compiled_cache = compile_design_problem(
                self.environment,
                self.objective,
                self.requirements,
            )
        return self._compiled_cache

    def solve(
        self,
        *,
        backend: str | LPBackend = "highs",
        options: SolveOptions | None = None,
    ) -> DesignSolution:
        """Solve, decode, and independently audit an optimal mechanism."""

        solve_options = options or SolveOptions()
        if not isinstance(solve_options, SolveOptions):
            raise TypeError("options must be a SolveOptions object")
        compiled = self.compile()
        adapter = _backend(backend)
        result = adapter.solve(compiled.model, _solver_config(solve_options))
        if result.status is ModelStatus.INFEASIBLE:
            raise InfeasibleDesignError(
                "The declared mechanism-design problem is infeasible. Check whether IC, "
                "participation, budget, and custom restrictions are mutually consistent."
            )
        if result.status is ModelStatus.UNBOUNDED:
            details = ["The declared design objective is unbounded."]
            has_upper_payment_control = any(
                isinstance(item, (ExPostIR, InterimIR, ExAnteIR))
                or (isinstance(item, PaymentBounds) and item.upper is not None)
                for item in self.requirements
            )
            if isinstance(self.objective, ExpectedRevenue) and not has_upper_payment_control:
                details.append(
                    "Payments can increase without a declared IR condition or payment cap."
                )
            details.append("MechAudit has not added any restriction automatically.")
            raise UnboundedDesignError(" ".join(details))
        if result.status is ModelStatus.INFEASIBLE_OR_UNBOUNDED:
            raise SolverError(
                "HiGHS could not distinguish infeasibility from unboundedness; rerun with "
                "presolve disabled for diagnosis."
            )
        if result.status is not ModelStatus.OPTIMAL:
            raise SolverError(
                f"The solver did not produce an optimal design (status={result.status.value}, message={result.message!r})."
            )
        if result.primal is None or result.objective_value is None:
            raise NumericalCertificationError(
                "The solver reported optimality without a usable primal solution."
            )
        if (
            result.residuals is None
            or result.residuals.max_violation > solve_options.tolerances.feasibility
        ):
            residual = None if result.residuals is None else result.residuals.max_violation
            raise NumericalCertificationError(
                "The selected solution failed the independent LP feasibility check "
                f"(maximum residual={residual!r})."
            )

        mechanism = compiled.mechanism_from_vector(result.primal)
        audit = self.environment.audit(
            mechanism,
            self.requirements,
            options=AuditOptions(tolerances=solve_options.tolerances),
        )
        if not audit.passed:
            failed = [check.constraint_id for check in audit.checks if not check.passed][:5]
            raise NumericalCertificationError(
                "The selected optimum failed its independent economic audit: {}".format(
                    ", ".join(failed)
                )
            )
        health = with_solver_health(audit.numerical_health, result)
        row_scales = [row.scale.factor for row in compiled.model.rows]
        health = replace(
            health,
            row_scale_min=min(row_scales, default=None),
            row_scale_max=max(row_scales, default=None),
        )
        solution = DesignSolution(
            problem=self,
            compiled=compiled,
            mechanism=mechanism,
            objective_value=float(result.objective_value),
            audit=audit,
            backend_result=result,
            tolerances=solve_options.tolerances,
            numerical_health=health,
            selected_activity=build_selected_activity(compiled, result, solve_options.tolerances),
        )
        self._solution_cache = solution
        return solution

    def assess(
        self,
        mechanism: DirectMechanism,
        *,
        optimality_tolerance: float = 1e-9,
        solution: DesignSolution | None = None,
        analyze_optimal_face: bool = True,
    ) -> CandidateAssessment:
        """Assess an imported mechanism against this complete design problem."""

        if optimality_tolerance < 0.0 or not math.isfinite(optimality_tolerance):
            raise ValueError("optimality_tolerance must be finite and nonnegative")
        optimum = solution or self._solution_cache or self.solve()
        bound = mechanism.bind(self.environment)
        # Judge the candidate with the same tolerances that certified the optimum.
        audit = self.environment.audit(
            bound,
            self.requirements,
            options=AuditOptions(tolerances=optimum.tolerances),
        )
        vector = optimum.compiled.vector_from_mechanism(bound)
        candidate_value = optimum.compiled.model.objective_value(vector)
        gap = optimum.objective_value - candidate_value
        absolute_gap = max(0.0, gap) if audit.passed else abs(gap)
        relative_gap = absolute_gap / max(1.0, abs(optimum.objective_value))
        is_optimal = audit.passed and absolute_gap <= optimality_tolerance

        residuals = optimum.compiled.model.evaluate_primal(vector)
        active = []
        for row_index in optimum.compiled.economic_row_indices:
            row = optimum.compiled.model.rows[row_index]
            slack = row.scale.original_slack_to_reported(float(residuals.row_slack[row_index]))
            # A violated row is infeasible at the candidate, not active there.
            if -optimum.tolerances.feasibility <= slack <= optimum.tolerances.activity:
                active.append(row.key)

        universal: tuple[str, ...] = ()
        if is_optimal and analyze_optimal_face:
            face = optimum.analyze_optima(level="fast")
            universal = tuple(item.constraint_id for item in face.universally_active)
        active_set = tuple(active)
        universal_set = set(universal)
        return CandidateAssessment(
            mechanism=bound,
            audit=audit,
            objective_value=candidate_value,
            optimal_value=optimum.objective_value,
            absolute_optimality_gap=absolute_gap,
            relative_optimality_gap=relative_gap,
            optimality_tolerance=optimality_tolerance,
            is_optimal=is_optimal,
            active_constraints=active_set,
            universally_active_constraints=universal,
            active_but_nonuniversal_constraints=tuple(
                item for item in active_set if item not in universal_set
            ),
            model_fingerprint=optimum.model_fingerprint,
        )

    def compare_scenarios(self, scenarios: Mapping[str, Mapping[str, Any]]) -> ScenarioComparison:
        """Solve a thin batch of requirement/prior perturbations."""

        from mechaudit.experiments.scenarios import compare_scenarios

        return compare_scenarios(self, scenarios)

    def __repr__(self) -> str:
        return f"DesignProblem(objective={self.objective.name!r}, requirements={len(self.requirements)}, profiles={self.environment.n_profiles}, outcomes={self.environment.n_outcomes})"
