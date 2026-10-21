"""Imported-mechanism assessment against a complete design problem."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from mechaudit._serialization import dumps_report, write_report
from mechaudit.audit.report import MechanismAuditReport
from mechaudit.mechanisms.direct import DirectMechanism


@dataclass(frozen=True)
class CandidateAssessment:
    """Feasibility and optimality of an imported mechanism."""

    mechanism: DirectMechanism
    audit: MechanismAuditReport
    objective_value: float
    optimal_value: float
    absolute_optimality_gap: float
    relative_optimality_gap: float
    optimality_tolerance: float
    is_optimal: bool
    active_constraints: tuple[str, ...]
    universally_active_constraints: tuple[str, ...]
    active_but_nonuniversal_constraints: tuple[str, ...]
    model_fingerprint: str

    @property
    def is_feasible(self) -> bool:
        return self.audit.passed

    @property
    def optimality_gap(self) -> float:
        return self.absolute_optimality_gap

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_fingerprint": self.model_fingerprint,
            "mechanism_fingerprint": self.mechanism.fingerprint,
            "is_feasible": self.is_feasible,
            "objective_value": self.objective_value,
            "optimal_value": self.optimal_value,
            "absolute_optimality_gap": self.absolute_optimality_gap,
            "relative_optimality_gap": self.relative_optimality_gap,
            "optimality_tolerance": self.optimality_tolerance,
            "is_optimal": self.is_optimal,
            "active_constraints": list(self.active_constraints),
            "universally_active_constraints": list(self.universally_active_constraints),
            "active_but_nonuniversal_constraints": list(self.active_but_nonuniversal_constraints),
            "audit": self.audit.to_dict(),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return dumps_report("candidate_assessment", self.to_dict(), indent=indent)

    def write_json(self, path: str | Path, *, indent: int = 2) -> None:
        write_report(path, "candidate_assessment", self.to_dict(), indent=indent)

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            [
                {
                    "is_feasible": self.is_feasible,
                    "objective_value": self.objective_value,
                    "optimal_value": self.optimal_value,
                    "absolute_optimality_gap": self.absolute_optimality_gap,
                    "relative_optimality_gap": self.relative_optimality_gap,
                    "is_optimal": self.is_optimal,
                }
            ]
        )

    def summary(self) -> str:
        return "\n".join(
            (
                "Candidate mechanism",
                "-------------------",
                "Feasible:                         {}".format("yes" if self.is_feasible else "no"),
                f"Candidate objective:              {self.objective_value:.10g}",
                f"Optimal objective:                {self.optimal_value:.10g}",
                f"Absolute optimality gap:          {self.absolute_optimality_gap:.10g}",
                "Candidate is optimal:             {}".format("yes" if self.is_optimal else "no"),
                f"Active at candidate:              {len(self.active_constraints)} constraints",
                f"Universally active:               {len(self.universally_active_constraints)} constraints",
                f"Candidate-specific active:        {len(self.active_but_nonuniversal_constraints)} constraints",
            )
        )
