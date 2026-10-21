"""Packaging and typed-public-surface regression contracts."""

from __future__ import annotations

import inspect
import runpy
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the Python 3.10 CI job
    import tomli as tomllib

import mechaudit
import mechaudit.design as design
from mechaudit.core.environment import FiniteEnvironment
from mechaudit.design.assessment import CandidateAssessment
from mechaudit.design.compiler import (
    CompiledDesign,
    MechanismVariableLayout,
    VacuousConstraint,
    compile_design_problem,
)
from mechaudit.design.problem import DesignProblem
from mechaudit.design.solution import DesignSolution, SelectedActivity

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _project_configuration() -> dict[str, Any]:
    return tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_runtime_and_distribution_versions_share_the_hatch_source() -> None:
    configuration = _project_configuration()
    project = configuration["project"]
    version_path = configuration["tool"]["hatch"]["version"]["path"]

    assert project["dynamic"] == ["version"]
    assert "version" not in project
    assert version_path == "src/mechaudit/_version.py"
    version_namespace = runpy.run_path(str(REPOSITORY_ROOT / version_path))
    assert version_namespace["__version__"] == mechaudit.__version__
    assert distribution_version("mechaudit") == mechaudit.__version__


def test_published_metadata_identifies_the_maintainer_and_project() -> None:
    configuration = _project_configuration()
    project = configuration["project"]

    assert project["authors"] == [{"name": "Alessio Staffini"}]
    assert project["maintainers"] == [{"name": "Alessio Staffini"}]
    assert project["license"] == "MIT"
    assert project["license-files"] == ["LICENSE"]
    assert "Typing :: Typed" in project["classifiers"]
    assert project["urls"] == {
        "Homepage": "https://github.com/studio-strata/mechaudit",
        "Documentation": "https://studio-strata.github.io/mechaudit/",
        "Repository": "https://github.com/studio-strata/mechaudit",
        "Issues": "https://github.com/studio-strata/mechaudit/issues",
        "Changelog": "https://github.com/studio-strata/mechaudit/blob/main/CHANGELOG.md",
    }
    build_targets = configuration["tool"]["hatch"]["build"]["targets"]
    assert build_targets["wheel"]["force-include"] == {
        "schemas/mechaudit-report-v1.schema.json": (
            "mechaudit/schemas/mechaudit-report-v1.schema.json"
        )
    }
    assert build_targets["sdist"]["exclude"] == ["/output", "/paper", "/tmp"]


def test_environment_workflow_signatures_retain_public_domain_types() -> None:
    audit = inspect.signature(FiniteEnvironment.audit)
    design_problem = inspect.signature(FiniteEnvironment.design)

    assert audit.parameters["mechanism"].annotation == "DirectMechanism"
    assert audit.parameters["requirements"].annotation == "Iterable[Requirement]"
    assert audit.parameters["options"].annotation == "AuditOptions | None"
    assert audit.return_annotation == "MechanismAuditReport"
    assert design_problem.parameters["objective"].annotation == "LinearObjective"
    assert design_problem.parameters["requirements"].annotation == "Iterable[Requirement]"
    assert design_problem.return_annotation == "DesignProblem"


def test_design_problem_signatures_retain_requirement_and_result_types() -> None:
    constructor = inspect.signature(DesignProblem)
    requirements = inspect.signature(DesignProblem.require)
    branch = inspect.signature(DesignProblem.with_requirements)
    scenarios = inspect.signature(DesignProblem.compare_scenarios)

    assert constructor.parameters["requirements"].annotation == "Iterable[Requirement]"
    assert requirements.parameters["requirements"].annotation == "Requirement"
    assert requirements.return_annotation == "DesignProblem"
    assert branch.parameters["requirements"].annotation == "Requirement"
    assert branch.return_annotation == "DesignProblem"
    assert scenarios.return_annotation == "ScenarioComparison"


def test_design_namespace_exposes_stable_workflow_and_compilation_types() -> None:
    expected = {
        "CandidateAssessment": CandidateAssessment,
        "CompiledDesign": CompiledDesign,
        "DesignProblem": DesignProblem,
        "DesignSolution": DesignSolution,
        "MechanismVariableLayout": MechanismVariableLayout,
        "SelectedActivity": SelectedActivity,
        "VacuousConstraint": VacuousConstraint,
        "compile_design_problem": compile_design_problem,
    }

    assert len(design.__all__) == len(set(design.__all__))
    for name, value in expected.items():
        assert name in design.__all__
        assert getattr(design, name) is value
    assert design.DesignProblem is mechaudit.DesignProblem
    assert design.DesignSolution is mechaudit.DesignSolution
    assert design.CandidateAssessment is mechaudit.CandidateAssessment
