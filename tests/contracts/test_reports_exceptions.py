"""Stable export and domain-exception contracts for public result objects."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

from mechaudit import (
    CandidateAssessment,
    DesignProblem,
    DomainMismatchError,
    EnvironmentValidationError,
    ExpectedRevenue,
    ExPostIR,
    FiniteEnvironment,
    LabelValidationError,
    MechAuditError,
    OptimalFaceReport,
    Outcome,
    PriorValidationError,
    PrivateValues,
    RequirementValidationError,
    ScenarioComparison,
    ScenarioResult,
    ValidationError,
    ZeroProbabilityTypeError,
)
from mechaudit.core.prior import JointPrior
from mechaudit.exceptions import SolverError
from mechaudit.optimality import SolveCounts
from mechaudit.requirements.budget import PaymentBounds

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPORT_SCHEMA_PATH = REPOSITORY_ROOT / "schemas" / "mechaudit-report-v1.schema.json"
REPORT_SCHEMA = json.loads(REPORT_SCHEMA_PATH.read_text(encoding="utf-8"))
REPORT_VALIDATOR = Draft202012Validator(REPORT_SCHEMA)


@pytest.fixture(scope="module")
def solved_problem() -> tuple[DesignProblem, object]:
    environment = FiniteEnvironment(
        agents={"buyer": ("only",)},
        outcomes=("off", "on"),
        prior={"only": 1.0},
        values=PrivateValues(lambda _agent, _type, outcome: 1.0 if outcome == "on" else 0.0),
    )
    problem = DesignProblem(
        environment=environment,
        objective=ExpectedRevenue(),
        requirements=(ExPostIR(),),
    )
    return problem, problem.solve()


def _assert_written_envelope(report: object, path: Path, *, report_kind: str) -> dict[str, object]:
    report.write_json(path, indent=2)  # type: ignore[attr-defined]
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    payload = json.loads(text)
    assert payload["schema_version"] == "1.0"
    assert payload["report_kind"] == report_kind
    assert payload == json.loads(report.to_json(indent=2))  # type: ignore[attr-defined]
    REPORT_VALIDATOR.validate(payload)
    return payload


def test_report_json_schema_is_valid_and_rejects_invalid_envelopes() -> None:
    Draft202012Validator.check_schema(REPORT_SCHEMA)

    invalid_envelopes = (
        {"schema_version": "2.0", "library_version": "0.1.0", "report_kind": "mechanism_audit"},
        {"schema_version": "1.0", "library_version": "0.1.0", "report_kind": "unknown"},
        {"schema_version": "1.0", "library_version": "0.1.0", "report_kind": "mechanism_audit"},
    )
    for envelope in invalid_envelopes:
        with pytest.raises(JsonSchemaValidationError):
            REPORT_VALIDATOR.validate(envelope)


def test_audit_solution_and_assessment_export_contracts(
    solved_problem: tuple[DesignProblem, object], tmp_path: Path
) -> None:
    problem, solution = solved_problem
    assessment = problem.assess(
        solution.mechanism,  # type: ignore[attr-defined]
        solution=solution,  # type: ignore[arg-type]
        analyze_optimal_face=False,
    )

    audit_payload = _assert_written_envelope(
        solution.audit,  # type: ignore[attr-defined]
        tmp_path / "audit.json",
        report_kind="mechanism_audit",
    )
    solution_payload = _assert_written_envelope(
        solution,
        tmp_path / "solution.json",
        report_kind="design_solution",
    )
    assessment_payload = _assert_written_envelope(
        assessment,
        tmp_path / "assessment.json",
        report_kind="candidate_assessment",
    )

    assert audit_payload["payment_sign_convention"] == "positive_to_designer"
    assert solution_payload["mechanism_fingerprint"] == assessment_payload["mechanism_fingerprint"]
    audit_tables = solution.audit.to_dataframes()  # type: ignore[attr-defined]
    assert set(audit_tables) == {"checks", "ic", "ir", "budget", "feasibility"}
    assert all(isinstance(table, pd.DataFrame) for table in audit_tables.values())
    solution_tables = solution.to_dataframes()  # type: ignore[attr-defined]
    assert set(solution_tables) == {
        "checks",
        "ic",
        "ir",
        "budget",
        "feasibility",
        "selected_activity",
        "allocation",
        "payments",
    }
    assert not solution_tables["allocation"].empty
    assert not solution_tables["payments"].empty
    assert assessment.to_pandas().to_dict(orient="records") == [
        {
            "is_feasible": True,
            "objective_value": pytest.approx(1.0),
            "optimal_value": pytest.approx(1.0),
            "absolute_optimality_gap": pytest.approx(0.0),
            "relative_optimality_gap": pytest.approx(0.0),
            "is_optimal": True,
        }
    ]
    assert isinstance(assessment, CandidateAssessment)


def test_optimal_face_and_scenario_export_contracts(
    solved_problem: tuple[DesignProblem, object], tmp_path: Path
) -> None:
    _problem, solution = solved_problem
    face = OptimalFaceReport(
        model_fingerprint="model-123",
        objective_name="expected_revenue",
        optimal_value=1.0,
        delta=1e-9,
        requested_delta=None,
        level="fast",
        tolerances={"activity": 1e-8},
        records=(),
        solve_counts=SolveCounts(ipm_candidates=1),
    )
    comparison = ScenarioComparison(
        (
            ScenarioResult("baseline", "optimal", solution=solution),  # type: ignore[arg-type]
            ScenarioResult("infeasible", "failed", error="declared infeasible"),
        )
    )

    face_payload = _assert_written_envelope(
        face,
        tmp_path / "face.json",
        report_kind="optimal_face_report",
    )
    scenario_payload = _assert_written_envelope(
        comparison,
        tmp_path / "scenarios.json",
        report_kind="scenario_comparison",
    )

    assert face_payload["requested_delta"] is None
    assert face_payload["effective_delta"] == pytest.approx(1e-9)
    assert face_payload["solve_counts"]["total"] == 1  # type: ignore[index]
    face_tables = face.to_dataframes()
    assert set(face_tables) == {
        "optimal_face_activity",
        "dual_ranges",
        "task_failures",
        "optimal_face_metadata",
    }
    metadata = face_tables["optimal_face_metadata"].iloc[0]
    assert metadata["solves_total"] == 1
    assert metadata["tolerance_activity"] == pytest.approx(1e-8)
    scenarios = comparison.to_pandas()
    assert scenarios.index.tolist() == ["baseline", "infeasible"]
    assert comparison.reports == {"baseline": solution}
    assert scenario_payload["scenarios"][1] == {  # type: ignore[index]
        "error": "declared infeasible",
        "name": "infeasible",
        "status": "failed",
    }


def test_validation_exceptions_retain_path_context_and_standard_hierarchies() -> None:
    error = DomainMismatchError(
        "unknown outcome",
        path="mechanism.allocation",
        context={"outcome": "ghost"},
    )

    assert str(error) == "mechanism.allocation: unknown outcome"
    assert error.message == "unknown outcome"
    assert error.path == "mechanism.allocation"
    assert error.context == {"outcome": "ghost"}
    assert error.args == ("mechanism.allocation: unknown outcome",)
    assert isinstance(error, (MechAuditError, ValidationError, ValueError))
    assert issubclass(LabelValidationError, ValidationError)
    assert issubclass(PriorValidationError, ValidationError)
    assert issubclass(EnvironmentValidationError, ValidationError)
    assert issubclass(RequirementValidationError, ValidationError)
    assert issubclass(ZeroProbabilityTypeError, RequirementValidationError)
    assert issubclass(SolverError, MechAuditError)


def test_domain_operations_raise_specific_errors_with_actionable_context() -> None:
    environment = FiniteEnvironment(
        agents={"buyer": ("low", "high")},
        outcomes=("off", "on"),
        prior={"low": 0.5, "high": 0.5},
        values=PrivateValues(lambda _agent, _type, _outcome: 0.0),
    )

    with pytest.raises(DomainMismatchError, match="unknown outcome") as outcome_error:
        environment.outcome_index("ghost")
    assert outcome_error.value.path is None
    assert outcome_error.value.context == {"outcome": "ghost"}

    with pytest.raises(LabelValidationError, match="supplied more than once") as label_error:
        Outcome({"kind": "old"}, kind="new")
    assert label_error.value.context == {"attributes": ["kind"]}

    with pytest.raises(
        PriorValidationError, match=r"prior\[.*probability cannot be negative"
    ) as prior_error:
        JointPrior.from_mapping(
            {"low": -0.1, "high": 1.1},
            agent_order=("buyer",),
        )
    assert prior_error.value.path is not None
    assert prior_error.value.path.startswith("prior[")
    assert prior_error.value.context == {}

    with pytest.raises(EnvironmentValidationError, match="valuation provider is required"):
        FiniteEnvironment(agents={"buyer": ("only",)}, outcomes=("off",))

    with pytest.raises(RequirementValidationError, match="at least one finite bound"):
        PaymentBounds()
