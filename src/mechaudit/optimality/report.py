"""Public result objects for optimal-face and essentiality analysis.

The classes in this module distinguish observations at one
solver-selected solution from statements about the complete
``delta``-optimal face.  Numerical witnesses and failed auxiliary solves are
kept in the report so a partial analysis never silently becomes a categorical
claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mechaudit._serialization import dumps_report, write_report

if TYPE_CHECKING:
    from mechaudit.design.solution import DesignSolution


class ActivityClassification(str, Enum):
    """Invariant activity classes on the explicitly reported near-optimal face."""

    UNIVERSALLY_ACTIVE = "universally_active"
    NONUNIVERSAL = "nonuniversal"
    SOMETIMES_ACTIVE = "sometimes_active"
    NEVER_ACTIVE = "never_active"
    INCONCLUSIVE = "inconclusive"


class AuxiliaryStatus(str, Enum):
    """Normalized status of an auxiliary optimality calculation."""

    OPTIMAL = "optimal"
    UNBOUNDED = "unbounded"
    INFEASIBLE = "infeasible"
    LIMIT = "limit"
    FAILED = "failed"
    NOT_RUN = "not_run"


class DualRelevance(str, Enum):
    """Classification based on a range over the dual near-optimal face."""

    NEVER_PRICED = "never_priced"
    POSSIBLY_PRICED = "possibly_priced"
    UNIVERSALLY_PRICED = "universally_priced"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class MechanismWitness:
    """A solver-independent primal witness for an optimal-face conclusion."""

    primal: tuple[float, ...]
    design_objective_value: float
    target_value: float | None = None
    mechanism_fingerprint: str | None = None
    mechanism: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "primal": list(self.primal),
            "design_objective_value": self.design_objective_value,
            "target_value": self.target_value,
            "mechanism_fingerprint": self.mechanism_fingerprint,
            "mechanism": self.mechanism,
        }


@dataclass(frozen=True)
class TaskFailure:
    """A retained failure or numerically inconclusive auxiliary task."""

    stage: str
    status: AuxiliaryStatus
    message: str
    row_index: int | None = None
    constraint_id: str | None = None
    direction: str | None = None
    witness: MechanismWitness | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status.value,
            "message": self.message,
            "row_index": self.row_index,
            "constraint_id": self.constraint_id,
            "direction": self.direction,
            "witness": None if self.witness is None else self.witness.to_dict(),
        }


@dataclass(frozen=True)
class SolveCounts:
    """Solve accounting for a reproducible analysis."""

    ipm_candidates: int = 0
    maximum_slack: int = 0
    minimum_slack: int = 0
    dual_range: int = 0
    other: int = 0

    @property
    def total(self) -> int:
        return (
            self.ipm_candidates
            + self.maximum_slack
            + self.minimum_slack
            + self.dual_range
            + self.other
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "ipm_candidates": self.ipm_candidates,
            "maximum_slack": self.maximum_slack,
            "minimum_slack": self.minimum_slack,
            "dual_range": self.dual_range,
            "other": self.other,
            "total": self.total,
        }


@dataclass(frozen=True)
class AuxiliaryCertificate:
    """Numerical evidence retained for an authoritative auxiliary optimum."""

    objective_value: float | None
    objective_bound: float | None
    max_primal_violation: float | None
    max_dual_infeasibility: float | None
    primal_dual_objective_error: float | None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "objective_value": self.objective_value,
            "objective_bound": self.objective_bound,
            "max_primal_violation": self.max_primal_violation,
            "max_dual_infeasibility": self.max_dual_infeasibility,
            "primal_dual_objective_error": self.primal_dual_objective_error,
        }


@dataclass(frozen=True)
class ActivityRecord:
    """One economic inequality's activity over the near-optimal face."""

    row_index: int
    constraint_id: str
    family: str
    classification: ActivityClassification
    selected_slack: float | None
    activity_tolerance: float
    candidate_slack: float | None = None
    candidate_dual: float | None = None
    candidate_slack_original: float | None = None
    candidate_dual_original: float | None = None
    minimum_slack: float | None = None
    maximum_slack: float | None = None
    minimum_is_exact: bool = False
    maximum_is_exact: bool = False
    classified_by: str = "inconclusive"
    positive_slack_witness: MechanismWitness | None = None
    active_witness: MechanismWitness | None = None
    max_status: AuxiliaryStatus = AuxiliaryStatus.NOT_RUN
    min_status: AuxiliaryStatus = AuxiliaryStatus.NOT_RUN
    maximum_certificate: AuxiliaryCertificate | None = None
    minimum_certificate: AuxiliaryCertificate | None = None
    note: str | None = None

    @property
    def universally_active(self) -> bool:
        return self.classification is ActivityClassification.UNIVERSALLY_ACTIVE

    @property
    def nonuniversal(self) -> bool:
        return self.classification in {
            ActivityClassification.NONUNIVERSAL,
            ActivityClassification.SOMETIMES_ACTIVE,
            ActivityClassification.NEVER_ACTIVE,
        }

    @property
    def active_at_selected_solution(self) -> bool | None:
        if self.selected_slack is None:
            return None
        return self.selected_slack <= self.activity_tolerance

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "constraint_id": self.constraint_id,
            "family": self.family,
            "classification": self.classification.value,
            "selected_slack": self.selected_slack,
            "active_at_selected_solution": self.active_at_selected_solution,
            "activity_tolerance": self.activity_tolerance,
            "candidate_slack": self.candidate_slack,
            "candidate_dual": self.candidate_dual,
            "candidate_slack_original": self.candidate_slack_original,
            "candidate_dual_original": self.candidate_dual_original,
            "minimum_slack": self.minimum_slack,
            "maximum_slack": self.maximum_slack,
            "minimum_is_exact": self.minimum_is_exact,
            "maximum_is_exact": self.maximum_is_exact,
            "classified_by": self.classified_by,
            "positive_slack_witness": (
                None
                if self.positive_slack_witness is None
                else self.positive_slack_witness.to_dict()
            ),
            "active_witness": (
                None if self.active_witness is None else self.active_witness.to_dict()
            ),
            "max_status": self.max_status.value,
            "min_status": self.min_status.value,
            "maximum_certificate": (
                None if self.maximum_certificate is None else self.maximum_certificate.to_dict()
            ),
            "minimum_certificate": (
                None if self.minimum_certificate is None else self.minimum_certificate.to_dict()
            ),
            "note": self.note,
        }


@dataclass(frozen=True)
class DualRangeResult:
    """Range of one normalized multiplier on the dual near-optimal face."""

    row_index: int
    constraint_id: str
    family: str
    lower: float | None
    upper: float | None
    relevance: DualRelevance
    lower_status: AuxiliaryStatus
    upper_status: AuxiliaryStatus
    tolerance: float
    normalization: str = "reported-slack units"
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "constraint_id": self.constraint_id,
            "family": self.family,
            "lower": self.lower,
            "upper": self.upper,
            "relevance": self.relevance.value,
            "lower_status": self.lower_status.value,
            "upper_status": self.upper_status.value,
            "tolerance": self.tolerance,
            "normalization": self.normalization,
            "message": self.message,
        }


@dataclass(frozen=True)
class RedundancyResult:
    """Independent implication test after removing one or more rows."""

    row_indices: tuple[int, ...]
    constraint_ids: tuple[str, ...]
    redundant: bool | None
    minimum_reported_slack: float | None
    status: AuxiliaryStatus
    tolerance: float
    witness: MechanismWitness | None
    solve_count: int
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_indices": list(self.row_indices),
            "constraint_ids": list(self.constraint_ids),
            "redundant": self.redundant,
            "minimum_reported_slack": self.minimum_reported_slack,
            "status": self.status.value,
            "tolerance": self.tolerance,
            "witness": None if self.witness is None else self.witness.to_dict(),
            "solve_count": self.solve_count,
            "message": self.message,
        }


@dataclass(frozen=True)
class ImplementationEssentialityResult:
    """Change to the relaxed optimal mechanism set, with no projection overclaim."""

    row_indices: tuple[int, ...]
    constraint_ids: tuple[str, ...]
    essential: bool | None
    projection: str
    method: str
    maximum_violation: float | None
    status: AuxiliaryStatus
    witness: MechanismWitness | None
    solve_count: int
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_indices": list(self.row_indices),
            "constraint_ids": list(self.constraint_ids),
            "essential": self.essential,
            "projection": self.projection,
            "method": self.method,
            "maximum_violation": self.maximum_violation,
            "status": self.status.value,
            "witness": None if self.witness is None else self.witness.to_dict(),
            "solve_count": self.solve_count,
            "message": self.message,
        }


@dataclass(frozen=True)
class ConstraintCostResult:
    """Objective and implementation consequences of removing constraints."""

    row_indices: tuple[int, ...]
    constraint_ids: tuple[str, ...]
    baseline_objective: float
    relaxed_objective: float | None
    objective_gain: float | None
    objective_essential: bool | None
    status: AuxiliaryStatus
    objective_tolerance: float
    witness: MechanismWitness | None
    solve_count: int
    implementation: ImplementationEssentialityResult | None = None
    message: str = ""

    @property
    def objective_cost(self) -> float | None:
        """Alias for the maximization gain available after removal."""

        return self.objective_gain

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_indices": list(self.row_indices),
            "constraint_ids": list(self.constraint_ids),
            "baseline_objective": self.baseline_objective,
            "relaxed_objective": self.relaxed_objective,
            "objective_gain": self.objective_gain,
            "objective_cost": self.objective_cost,
            "objective_essential": self.objective_essential,
            "status": self.status.value,
            "objective_tolerance": self.objective_tolerance,
            "witness": None if self.witness is None else self.witness.to_dict(),
            "solve_count": self.solve_count,
            "implementation": (
                None if self.implementation is None else self.implementation.to_dict()
            ),
            "message": self.message,
        }


@dataclass(frozen=True)
class FamilyAblationResult:
    """Objective/implementation result for removing an economic row family."""

    family: str
    row_indices: tuple[int, ...]
    constraint_ids: tuple[str, ...]
    cost: ConstraintCostResult

    @property
    def objective_gain(self) -> float | None:
        return self.cost.objective_gain

    @property
    def objective_essential(self) -> bool | None:
        return self.cost.objective_essential

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "row_indices": list(self.row_indices),
            "constraint_ids": list(self.constraint_ids),
            "cost": self.cost.to_dict(),
        }


@dataclass(frozen=True)
class OptimalFaceReport:
    """Invariant analysis of the explicitly stated ``delta``-optimal face."""

    model_fingerprint: str
    objective_name: str
    optimal_value: float
    delta: float
    requested_delta: float | None
    level: str
    tolerances: dict[str, float]
    records: tuple[ActivityRecord, ...]
    solve_counts: SolveCounts
    failures: tuple[TaskFailure, ...] = ()
    dual_ranges: tuple[DualRangeResult, ...] = ()
    analyzed_families: tuple[str, ...] | None = None
    skipped_constraint_count: int = 0
    candidate_status: AuxiliaryStatus = AuxiliaryStatus.NOT_RUN
    candidate_message: str = ""
    candidate_objective_value: float | None = None
    candidate_max_primal_violation: float | None = None
    candidate_max_complementarity: float | None = None
    methodology: str = (
        "IPM without crossover is a numerical maximum-support candidate; "
        "targeted auxiliary LPs are authoritative where run."
    )
    _solution: DesignSolution | None = field(default=None, repr=False, compare=False)

    @property
    def constraints(self) -> tuple[ActivityRecord, ...]:
        return self.records

    @property
    def universally_active(self) -> tuple[ActivityRecord, ...]:
        return tuple(item for item in self.records if item.universally_active)

    @property
    def nonuniversally_active(self) -> tuple[ActivityRecord, ...]:
        return tuple(item for item in self.records if item.nonuniversal)

    @property
    def nonuniversal(self) -> tuple[ActivityRecord, ...]:
        return self.nonuniversally_active

    @property
    def sometimes_active(self) -> tuple[ActivityRecord, ...]:
        return tuple(
            item
            for item in self.records
            if item.classification is ActivityClassification.SOMETIMES_ACTIVE
        )

    @property
    def never_active(self) -> tuple[ActivityRecord, ...]:
        return tuple(
            item
            for item in self.records
            if item.classification is ActivityClassification.NEVER_ACTIVE
        )

    @property
    def inconclusive(self) -> tuple[ActivityRecord, ...]:
        return tuple(
            item
            for item in self.records
            if item.classification is ActivityClassification.INCONCLUSIVE
        )

    def activity_for(self, constraint: Any) -> ActivityRecord:
        """Resolve an analyzed row by index, semantic key, or row-like object."""

        if isinstance(constraint, int) and not isinstance(constraint, bool):
            for item in self.records:
                if item.row_index == constraint:
                    return item
        key = (
            constraint
            if isinstance(constraint, str)
            else getattr(constraint, "constraint_id", None)
        )
        if key is not None:
            for item in self.records:
                if item.constraint_id == key:
                    return item
        raise KeyError(f"constraint is not present in this analysis: {constraint!r}")

    def _require_solution(self) -> DesignSolution:
        if self._solution is None:
            raise RuntimeError(
                "on-demand analysis is unavailable on a detached/deserialized report"
            )
        return self._solution

    def constraint_cost(self, constraint: Any, *, projection: str = "full") -> ConstraintCostResult:
        """Remove one row and measure objective and optimal-set consequences."""

        from mechaudit.optimality.essentiality import constraint_cost

        return constraint_cost(
            self._require_solution(), constraint, delta=self.delta, projection=projection
        )

    def redundancy(self, constraint: Any) -> RedundancyResult:
        """Test whether the remaining complete feasible system implies a row."""

        from mechaudit.optimality.essentiality import test_redundancy

        return test_redundancy(self._require_solution(), constraint)

    def implementation_essential(
        self, constraint: Any, *, projection: str = "full"
    ) -> ImplementationEssentialityResult:
        """Test expansion of the relaxed optimal mechanism set."""

        from mechaudit.optimality.essentiality import implementation_essentiality

        return implementation_essentiality(
            self._require_solution(), constraint, delta=self.delta, projection=projection
        )

    def ablate_family(
        self, family: str, *, agent: Any = None, projection: str = "full"
    ) -> FamilyAblationResult:
        """Remove a labeled row family, optionally filtered by metadata agent."""

        from mechaudit.optimality.essentiality import ablate_family

        return ablate_family(
            self._require_solution(),
            family,
            agent=agent,
            delta=self.delta,
            projection=projection,
        )

    def dual_range(self, constraint: Any) -> DualRangeResult:
        """Compute one reported-unit multiplier range on demand."""

        from mechaudit.optimality.dual import compute_dual_ranges

        item = self.activity_for(constraint)
        results, _ = compute_dual_ranges(
            self._require_solution(), (item.row_index,), delta=self.delta, n_jobs=1
        )
        return results[0]

    def to_dict(self) -> dict[str, Any]:
        counts = {
            classification.value: sum(
                item.classification is classification for item in self.records
            )
            for classification in ActivityClassification
        }
        return {
            "model_fingerprint": self.model_fingerprint,
            "objective_name": self.objective_name,
            "optimal_value": self.optimal_value,
            "requested_delta": self.requested_delta,
            "effective_delta": self.delta,
            "delta": self.delta,
            "near_optimal_face": "c @ z + offset >= optimal_value - delta",
            "level": self.level,
            "tolerances": dict(self.tolerances),
            "classification_counts": counts,
            "constraints": [item.to_dict() for item in self.records],
            "solve_counts": self.solve_counts.to_dict(),
            "failures": [item.to_dict() for item in self.failures],
            "dual_ranges": [item.to_dict() for item in self.dual_ranges],
            "analyzed_families": self.analyzed_families,
            "skipped_constraint_count": self.skipped_constraint_count,
            "candidate": {
                "status": self.candidate_status.value,
                "message": self.candidate_message,
                "objective_value": self.candidate_objective_value,
                "max_primal_violation": self.candidate_max_primal_violation,
                "max_complementarity": self.candidate_max_complementarity,
            },
            "methodology": self.methodology,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return dumps_report("optimal_face_report", self.to_dict(), indent=indent)

    def write_json(self, path: str | Path, *, indent: int = 2) -> None:
        write_report(path, "optimal_face_report", self.to_dict(), indent=indent)

    def to_dataframes(self) -> dict[str, Any]:
        import pandas as pd

        metadata = {
            "model_fingerprint": self.model_fingerprint,
            "objective_name": self.objective_name,
            "optimal_value": self.optimal_value,
            "delta": self.delta,
            "requested_delta": self.requested_delta,
            "level": self.level,
            **{f"tolerance_{key}": value for key, value in self.tolerances.items()},
            **{f"solves_{key}": value for key, value in self.solve_counts.to_dict().items()},
        }
        return {
            "optimal_face_activity": pd.DataFrame.from_records(
                [item.to_dict() for item in self.records]
            ),
            "dual_ranges": pd.DataFrame.from_records([item.to_dict() for item in self.dual_ranges]),
            "task_failures": pd.DataFrame.from_records([item.to_dict() for item in self.failures]),
            "optimal_face_metadata": pd.DataFrame.from_records([metadata]),
        }

    def summary(self) -> str:
        return "\n".join(
            (
                "Optimal-face analysis",
                "---------------------",
                f"Objective:              {self.objective_name}",
                f"Optimal value:          {self.optimal_value:.10g}",
                f"Near-optimal delta:     {self.delta:.6g}",
                f"Universally active:     {len(self.universally_active)}",
                f"Sometimes active:       {len(self.sometimes_active)}",
                f"Never active:           {len(self.never_active)}",
                f"Nonuniversal (fast):    {sum(item.classification is ActivityClassification.NONUNIVERSAL for item in self.records)}",
                f"Inconclusive:           {len(self.inconclusive)}",
                f"Auxiliary solves:       {self.solve_counts.total}",
                f"Retained task failures: {len(self.failures)}",
                f"IPM status:             {self.candidate_status.value}",
            )
        )
