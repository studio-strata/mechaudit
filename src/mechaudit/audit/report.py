"""Structured, workflow-safe fixed-mechanism audit reports."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, overload

import pandas as pd

from mechaudit._serialization import dumps_report, json_safe, write_report
from mechaudit.diagnostics.health import NumericalHealthReport


class CheckStatus(str, Enum):
    """Numerical disposition of one semantic audit check."""

    PASS = "pass"
    FAIL = "fail"
    VACUOUS = "vacuous"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class ConstraintCheck:
    """The original-unit evaluation of one economic condition."""

    constraint_id: str
    family: str
    relation: str
    status: CheckStatus
    tolerance: float
    slack: float | None = None
    residual: float | None = None
    unit: str = "utility"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Whether this check is satisfied (including a declared vacuous row)."""

        return self.status in (CheckStatus.PASS, CheckStatus.VACUOUS)

    @property
    def active(self) -> bool:
        """Whether this inequality is locally active within its tolerance."""

        return self.slack is not None and self.slack <= self.tolerance

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible long-form record."""

        record = {
            "constraint_id": self.constraint_id,
            "family": self.family,
            "relation": self.relation,
            "status": self.status.value,
            "passed": self.passed,
            "tolerance": self.tolerance,
            "slack": self.slack,
            "residual": self.residual,
            "unit": self.unit,
        }
        record.update({str(key): json_safe(value) for key, value in self.metadata.items()})
        return record


@dataclass(frozen=True)
class AuditSection(Sequence[ConstraintCheck]):
    """A typed view over one or more families of audit checks."""

    name: str
    checks: tuple[ConstraintCheck, ...]

    @overload
    def __getitem__(self, index: int) -> ConstraintCheck: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[ConstraintCheck, ...]: ...

    def __getitem__(self, index: int | slice) -> ConstraintCheck | tuple[ConstraintCheck, ...]:
        return self.checks[index]

    def __len__(self) -> int:
        return len(self.checks)

    @property
    def passed(self) -> bool:
        """Whether every row in the section passed or was vacuous."""

        return all(check.passed for check in self.checks)

    @property
    def minimum_slack(self) -> float | None:
        """Smallest inequality slack, excluding equality and vacuous rows."""

        values = [check.slack for check in self.checks if check.slack is not None]
        return min(values) if values else None

    @property
    def maximum_residual(self) -> float | None:
        """Largest absolute equality residual."""

        values = [abs(check.residual) for check in self.checks if check.residual is not None]
        return max(values) if values else None

    @property
    def epsilon(self) -> float:
        """Smallest additive epsilon satisfying all IC rows in this section."""

        minimum = self.minimum_slack
        return max(0.0, -minimum) if minimum is not None else 0.0

    def worst_deviations(self, k: int = 10) -> pd.DataFrame:
        """Return the *k* most profitable deviations in economic labels."""

        if k < 0:
            raise ValueError("k must be nonnegative")
        records = [
            check.to_dict()
            for check in self.checks
            if check.family in {"dsic", "ex_post_ic", "bic"}
        ]
        records.sort(key=lambda row: float(row.get("slack") or 0.0))
        return pd.DataFrame.from_records(records[:k])

    def by_type(self) -> pd.DataFrame:
        """Return long-form rows retaining agent and true-type labels."""

        return pd.DataFrame.from_records([check.to_dict() for check in self.checks])

    def to_pandas(self) -> pd.DataFrame:
        """Return this section as a long-form data frame."""

        return pd.DataFrame.from_records([check.to_dict() for check in self.checks])


@dataclass(frozen=True)
class MechanismAuditReport:
    """Exhaustive checks for one supplied mechanism.

    A fixed-mechanism report exposes no optimal-face methods because it has no
    declared objective or associated set of optimal solutions.
    """

    checks: tuple[ConstraintCheck, ...]
    environment_fingerprint: str
    mechanism_fingerprint: str
    feasibility_tolerance: float
    probability_tolerance: float
    numerical_health: NumericalHealthReport
    requirements: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        """Whether all declared and baseline checks passed."""

        return all(check.passed for check in self.checks)

    @property
    def is_feasible(self) -> bool:
        """Alias emphasizing use from candidate-assessment workflows."""

        return self.passed

    def section(self, name: str, families: Iterable[str]) -> AuditSection:
        """Create a view containing the requested semantic families."""

        accepted = set(families)
        return AuditSection(name, tuple(check for check in self.checks if check.family in accepted))

    @property
    def feasibility(self) -> AuditSection:
        return self.section("feasibility", {"allocation_nonnegative", "allocation_normalization"})

    @property
    def ic(self) -> AuditSection:
        return self.section("ic", {"dsic", "ex_post_ic", "bic"})

    @property
    def ir(self) -> AuditSection:
        return self.section("ir", {"ex_post_ir", "interim_ir", "ex_ante_ir"})

    @property
    def budget(self) -> AuditSection:
        return self.section(
            "budget",
            {"weak_budget_balance", "strong_budget_balance", "no_subsidies", "payment_bounds"},
        )

    @property
    def epsilon_ic(self) -> float:
        """Smallest additive epsilon satisfying the audited IC notion(s)."""

        return self.ic.epsilon

    @property
    def rents(self) -> pd.DataFrame:
        """Return truthful utilities, outside options, and information rents."""

        records = []
        for check in self.ir:
            if "information_rent" not in check.metadata:
                continue
            records.append(
                {
                    "constraint_id": check.constraint_id,
                    "family": check.family,
                    "agent": json_safe(check.metadata.get("agent")),
                    "true_type": json_safe(check.metadata.get("true_type")),
                    "true_profile": json_safe(check.metadata.get("true_profile")),
                    "truthful_utility": check.metadata.get("truthful_utility"),
                    "outside_option": check.metadata.get("outside_option"),
                    "information_rent": check.metadata.get("information_rent"),
                }
            )
        return pd.DataFrame.from_records(records)

    def active_constraints(self) -> tuple[ConstraintCheck, ...]:
        """Return inequalities active only at this supplied mechanism."""

        return tuple(check for check in self.checks if check.active and check.passed)

    def to_dict(self) -> dict[str, Any]:
        """Return a schema-ready numerical payload."""

        return {
            "environment_fingerprint": self.environment_fingerprint,
            "mechanism_fingerprint": self.mechanism_fingerprint,
            "payment_sign_convention": "positive_to_designer",
            "passed": self.passed,
            "epsilon_ic": self.epsilon_ic,
            "feasibility_tolerance": self.feasibility_tolerance,
            "probability_tolerance": self.probability_tolerance,
            "requirements": list(self.requirements),
            "checks": [check.to_dict() for check in self.checks],
            "numerical_health": self.numerical_health.to_dict(),
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize this report as deterministic JSON."""

        return dumps_report("mechanism_audit", self.to_dict(), indent=indent)

    def write_json(self, path: str | Path, *, indent: int = 2) -> None:
        """Write this report to a UTF-8 JSON file."""

        write_report(path, "mechanism_audit", self.to_dict(), indent=indent)

    def to_dataframes(self) -> dict[str, pd.DataFrame]:
        """Return long-form tables grouped for common research workflows."""

        return {
            "checks": pd.DataFrame.from_records([check.to_dict() for check in self.checks]),
            "ic": self.ic.to_pandas(),
            "ir": self.ir.to_pandas(),
            "budget": self.budget.to_pandas(),
            "feasibility": self.feasibility.to_pandas(),
        }

    def summary(self) -> str:
        """Render a concise human-readable audit summary."""

        lines = ["Mechanism audit", "---------------"]
        for label, section in (
            ("Feasibility", self.feasibility),
            ("IC", self.ic),
            ("IR", self.ir),
            ("Budget", self.budget),
        ):
            if not section:
                continue
            state = "PASSED" if section.passed else "FAILED"
            lines.append("{:<28} {}".format(label + ":", state))
            if section.minimum_slack is not None:
                lines.append(
                    "{:<28} {:.10g}".format(f"Minimum {label} slack:", section.minimum_slack)
                )
            if section.maximum_residual is not None:
                lines.append(
                    "{:<28} {:.10g}".format(f"Maximum {label} residual:", section.maximum_residual)
                )
        if self.ic:
            lines.append("{:<28} {:.10g}".format("Smallest valid epsilon:", self.epsilon_ic))
        return "\n".join(lines)


def checks_from_iterable(checks: Iterable[ConstraintCheck]) -> tuple[ConstraintCheck, ...]:
    """Materialize and type-check a stream of audit checks."""

    result = tuple(checks)
    if not all(isinstance(check, ConstraintCheck) for check in result):
        raise TypeError("audit check streams must contain ConstraintCheck objects")
    return result
