"""Optimal-partition candidate and targeted slack analysis."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from mechaudit.backends.base import ModelStatus
from mechaudit.backends.highs import HighsBackend
from mechaudit.design.solution import DesignSolution
from mechaudit.optimality.report import (
    ActivityClassification,
    ActivityRecord,
    AuxiliaryCertificate,
    AuxiliaryStatus,
    DualRangeResult,
    MechanismWitness,
    OptimalFaceReport,
    SolveCounts,
    TaskFailure,
)
from mechaudit.optimality.tasks import (
    SlackTask,
    SlackTaskResult,
    certified_optimal_result,
    make_face_model,
    run_slack_tasks,
    solver_config,
)
from mechaudit.options import OptimalFaceLevel, OptimalFaceOptions


def _status(status: ModelStatus) -> AuxiliaryStatus:
    if status is ModelStatus.OPTIMAL:
        return AuxiliaryStatus.OPTIMAL
    if status is ModelStatus.UNBOUNDED:
        return AuxiliaryStatus.UNBOUNDED
    if status in {ModelStatus.INFEASIBLE, ModelStatus.INFEASIBLE_OR_UNBOUNDED}:
        return AuxiliaryStatus.INFEASIBLE
    if status.is_limit:
        return AuxiliaryStatus.LIMIT
    return AuxiliaryStatus.FAILED


def _witness(
    solution: DesignSolution,
    task_result: SlackTaskResult,
) -> MechanismWitness | None:
    if task_result.primal is None or task_result.design_objective_value is None:
        return None
    mechanism_dict = None
    mechanism_fingerprint = None
    try:
        mechanism = solution.compiled.mechanism_from_vector(
            np.asarray(task_result.primal, dtype=float)
        )
        mechanism_dict = mechanism.to_dict()
        mechanism_fingerprint = mechanism.fingerprint
    except Exception:
        # The raw primal is still a complete solver-independent witness.
        pass
    return MechanismWitness(
        primal=task_result.primal,
        design_objective_value=task_result.design_objective_value,
        target_value=task_result.slack,
        mechanism_fingerprint=mechanism_fingerprint,
        mechanism=mechanism_dict,
    )


def _failure(
    solution: DesignSolution,
    result: SlackTaskResult,
    *,
    stage: str,
) -> TaskFailure:
    row = solution.compiled.model.rows[result.task.row_index]
    return TaskFailure(
        stage=stage,
        status=result.status,
        message=result.message,
        row_index=result.task.row_index,
        constraint_id=row.key,
        direction=result.task.direction,
        witness=_witness(solution, result),
    )


def _certificate(result: SlackTaskResult) -> AuxiliaryCertificate:
    return AuxiliaryCertificate(
        objective_value=result.auxiliary_objective_value,
        objective_bound=result.objective_bound,
        max_primal_violation=result.max_primal_violation,
        max_dual_infeasibility=result.max_dual_infeasibility,
        primal_dual_objective_error=result.primal_dual_objective_error,
    )


def analyze_optimal_face(
    solution: DesignSolution,
    options: OptimalFaceOptions | None = None,
) -> OptimalFaceReport:
    """Classify economic inequalities over a declared near-optimal face.

    The IPM/no-crossover solve is used only as a numerical maximum-support
    candidate.  Every ambiguous universal/nonuniversal coordinate receives a
    targeted maximum-slack LP.  At ``full`` and ``essentiality`` levels every
    nonuniversal coordinate additionally receives a minimum-slack LP; this is
    the step required to distinguish sometimes-active from never-active rows.
    """

    selected_options = options or OptimalFaceOptions(tolerances=solution.tolerances)
    if not isinstance(selected_options, OptimalFaceOptions):
        raise TypeError("options must be an OptimalFaceOptions instance")
    model = solution.compiled.model
    tolerances = selected_options.tolerances
    optimal_value = float(solution.objective_value)
    delta = (
        float(selected_options.objective_delta)
        if selected_options.objective_delta is not None
        else max(
            tolerances.objective_absolute,
            tolerances.objective_relative * max(1.0, abs(optimal_value)),
        )
    )
    families = selected_options.constraint_families
    family_set = None if families is None else set(families)
    row_indices = tuple(
        index
        for index in solution.compiled.economic_row_indices
        if family_set is None or model.rows[index].family in family_set
    )
    skipped = len(solution.compiled.economic_row_indices) - len(row_indices)
    selected_slacks: dict[int, float] = {}
    if solution.backend_result.primal is not None:
        selected_residuals = model.evaluate_primal(solution.backend_result.primal)
        for index in row_indices:
            selected_slacks[index] = model.rows[index].scale.original_slack_to_reported(
                float(selected_residuals.row_slack[index])
            )

    if not row_indices:
        return OptimalFaceReport(
            model_fingerprint=solution.model_fingerprint,
            objective_name=solution.compiled.objective.name,
            optimal_value=optimal_value,
            delta=delta,
            requested_delta=selected_options.objective_delta,
            level=selected_options.level.value,
            tolerances=dict(tolerances.__dict__),
            records=(),
            solve_counts=SolveCounts(),
            analyzed_families=families,
            skipped_constraint_count=skipped,
            _solution=solution,
        )

    face_model = make_face_model(model, optimal_value=optimal_value, delta=delta)
    failures: list[TaskFailure] = []
    candidate_status = AuxiliaryStatus.FAILED
    candidate_message = ""
    candidate_value: float | None = None
    candidate_violation: float | None = None
    candidate_complementarity: float | None = None
    candidate_slack: dict[int, float] = {}
    candidate_slack_original: dict[int, float] = {}
    candidate_dual: dict[int, float] = {}
    candidate_dual_original: dict[int, float] = {}
    candidate_primal: tuple[float, ...] | None = None
    candidate_valid = False

    try:
        result = HighsBackend().solve(
            model,
            solver_config(tolerances, solver="ipm", presolve="off", crossover="off"),
        )
        candidate_status = _status(result.status)
        candidate_message = result.message
        candidate_value = result.objective_value
        if result.residuals is not None:
            candidate_violation = float(result.residuals.max_violation)
        certified_candidate, certificate_message = certified_optimal_result(result, tolerances)
        candidate_valid = (
            certified_candidate
            and result.primal is not None
            and result.solver_row_dual is not None
            and candidate_violation is not None
            and candidate_violation <= tolerances.feasibility
            and candidate_value is not None
            and candidate_value
            >= optimal_value - delta - max(tolerances.feasibility, tolerances.objective_absolute)
        )
        if result.primal is not None:
            candidate_primal = tuple(float(value) for value in result.primal)
            residuals = model.evaluate_primal(result.primal)
            for index in row_indices:
                row = model.rows[index]
                original = float(residuals.row_slack[index])
                candidate_slack_original[index] = original
                candidate_slack[index] = row.scale.original_slack_to_reported(original)
        if result.solver_row_dual is not None:
            for index in row_indices:
                row = model.rows[index]
                original_dual = row.scale.factor * (-float(result.solver_row_dual[index]))
                candidate_dual_original[index] = original_dual
                candidate_dual[index] = original_dual / row.scale.report_factor
        products = [
            abs(candidate_slack[index] * candidate_dual[index])
            for index in row_indices
            if index in candidate_slack and index in candidate_dual
        ]
        candidate_complementarity = max(products, default=0.0)
        if candidate_complementarity > tolerances.complementarity:
            candidate_valid = False
            candidate_message = (
                f"{candidate_message}; recomputed complementarity "
                f"{candidate_complementarity:.6g} exceeds tolerance"
            )
        if not candidate_valid:
            failures.append(
                TaskFailure(
                    stage="maximum_support_candidate",
                    status=candidate_status,
                    message=(
                        candidate_message
                        or certificate_message
                        or "IPM candidate failed independent feasibility/face checks"
                    ),
                )
            )
    except Exception as exc:
        candidate_message = f"{type(exc).__name__}: {exc}"
        failures.append(
            TaskFailure(
                stage="maximum_support_candidate",
                status=AuxiliaryStatus.FAILED,
                message=candidate_message,
            )
        )

    primal_positive = 10.0 * max(tolerances.activity, tolerances.partition_primal, 1e-15)
    dual_positive = 10.0 * max(tolerances.partition_dual, 1e-15)
    records: dict[int, ActivityRecord] = {}
    ambiguous: list[int] = []
    for index in row_indices:
        row = model.rows[index]
        slack = candidate_slack.get(index)
        dual = candidate_dual.get(index)
        classification = ActivityClassification.INCONCLUSIVE
        classified_by = "ambiguous_ipm_candidate"
        note = "targeted maximum-slack solve required"
        witness = None
        if candidate_valid and slack is not None and dual is not None:
            if slack > primal_positive and abs(dual) <= dual_positive:
                classification = ActivityClassification.NONUNIVERSAL
                classified_by = "ipm_maximum_support_candidate"
                note = (
                    "numerical positive-slack witness; IPM output is not an exact "
                    "strict-complementarity certificate"
                )
                if candidate_primal is not None and candidate_value is not None:
                    mechanism_dict = None
                    mechanism_fingerprint = None
                    try:
                        mechanism = solution.compiled.mechanism_from_vector(
                            np.asarray(candidate_primal, dtype=float)
                        )
                        mechanism_dict = mechanism.to_dict()
                        mechanism_fingerprint = mechanism.fingerprint
                    except Exception:
                        pass
                    witness = MechanismWitness(
                        primal=candidate_primal,
                        design_objective_value=candidate_value,
                        target_value=slack,
                        mechanism_fingerprint=mechanism_fingerprint,
                        mechanism=mechanism_dict,
                    )
            elif dual > dual_positive and slack <= tolerances.activity:
                classification = ActivityClassification.UNIVERSALLY_ACTIVE
                classified_by = "ipm_maximum_support_candidate"
                note = (
                    "numerical positive-dual coordinate; IPM output is not an exact "
                    "strict-complementarity certificate"
                )
        # A dual coordinate from the exact optimum cannot establish activity
        # throughout a positive-width near-optimal face.  Make that claim
        # authoritative with a targeted maximum-slack solve.
        if classification is ActivityClassification.UNIVERSALLY_ACTIVE and delta > 0.0:
            classification = ActivityClassification.INCONCLUSIVE
            classified_by = "ipm_candidate_requires_delta_face_check"
            note = "positive-width near-optimal face requires targeted maximum slack"
        if classification is ActivityClassification.INCONCLUSIVE:
            ambiguous.append(index)
        records[index] = ActivityRecord(
            row_index=index,
            constraint_id=row.key,
            family=row.family,
            classification=classification,
            selected_slack=selected_slacks.get(index),
            activity_tolerance=tolerances.activity,
            candidate_slack=slack,
            candidate_dual=dual,
            candidate_slack_original=candidate_slack_original.get(index),
            candidate_dual_original=candidate_dual_original.get(index),
            maximum_slack=(
                slack if classification is ActivityClassification.NONUNIVERSAL else None
            ),
            maximum_is_exact=False,
            classified_by=classified_by,
            positive_slack_witness=witness,
            note=note,
        )

    maximum_target_indices = list(ambiguous)
    if selected_options.level in {
        OptimalFaceLevel.FULL,
        OptimalFaceLevel.ESSENTIALITY,
    }:
        maximum_target_indices.extend(
            index for index in row_indices if index not in maximum_target_indices
        )
    maximum_tasks = tuple(SlackTask(index, "max") for index in maximum_target_indices)
    maximum_results = run_slack_tasks(
        model,
        face_model,
        maximum_tasks,
        optimal_value=optimal_value,
        delta=delta,
        tolerances=tolerances,
        n_jobs=selected_options.n_jobs,
    )
    for task_result in maximum_results:
        index = task_result.task.row_index
        current = records[index]
        if task_result.status is AuxiliaryStatus.OPTIMAL and task_result.slack is not None:
            slack = task_result.slack
            if slack <= tolerances.activity:
                records[index] = replace(
                    current,
                    classification=ActivityClassification.UNIVERSALLY_ACTIVE,
                    maximum_slack=slack,
                    maximum_is_exact=True,
                    classified_by="targeted_maximum_slack",
                    max_status=task_result.status,
                    maximum_certificate=_certificate(task_result),
                    note=None,
                )
            elif slack > primal_positive:
                records[index] = replace(
                    current,
                    classification=ActivityClassification.NONUNIVERSAL,
                    maximum_slack=slack,
                    maximum_is_exact=True,
                    classified_by="targeted_maximum_slack",
                    positive_slack_witness=_witness(solution, task_result),
                    max_status=task_result.status,
                    maximum_certificate=_certificate(task_result),
                    note=None,
                )
            else:
                records[index] = replace(
                    current,
                    maximum_slack=slack,
                    maximum_is_exact=True,
                    max_status=task_result.status,
                    maximum_certificate=_certificate(task_result),
                    note="optimized maximum lies in the activity tolerance gray zone",
                )
                failures.append(_failure(solution, task_result, stage="maximum_slack_gray_zone"))
        elif task_result.status is AuxiliaryStatus.UNBOUNDED:
            records[index] = replace(
                current,
                classification=ActivityClassification.NONUNIVERSAL,
                maximum_slack=float("inf"),
                maximum_is_exact=True,
                classified_by="targeted_maximum_slack_unbounded",
                max_status=task_result.status,
                note="reported slack is unbounded above on the near-optimal face",
            )
        else:
            records[index] = replace(
                current, max_status=task_result.status, note=task_result.message
            )
            failures.append(_failure(solution, task_result, stage="maximum_slack"))

    minimum_count = 0
    if selected_options.level in {
        OptimalFaceLevel.FULL,
        OptimalFaceLevel.ESSENTIALITY,
    }:
        nonuniversal = tuple(
            index
            for index in row_indices
            if records[index].classification is ActivityClassification.NONUNIVERSAL
        )
        minimum_tasks = tuple(SlackTask(index, "min") for index in nonuniversal)
        minimum_results = run_slack_tasks(
            model,
            face_model,
            minimum_tasks,
            optimal_value=optimal_value,
            delta=delta,
            tolerances=tolerances,
            n_jobs=selected_options.n_jobs,
        )
        minimum_count = len(minimum_tasks)
        for task_result in minimum_results:
            index = task_result.task.row_index
            current = records[index]
            if task_result.status is AuxiliaryStatus.OPTIMAL and task_result.slack is not None:
                slack = task_result.slack
                if slack <= tolerances.activity:
                    records[index] = replace(
                        current,
                        classification=ActivityClassification.SOMETIMES_ACTIVE,
                        minimum_slack=slack,
                        minimum_is_exact=True,
                        classified_by=current.classified_by + "+targeted_minimum_slack",
                        active_witness=_witness(solution, task_result),
                        min_status=task_result.status,
                        minimum_certificate=_certificate(task_result),
                        note=None,
                    )
                elif slack > primal_positive:
                    records[index] = replace(
                        current,
                        classification=ActivityClassification.NEVER_ACTIVE,
                        minimum_slack=slack,
                        minimum_is_exact=True,
                        classified_by=current.classified_by + "+targeted_minimum_slack",
                        min_status=task_result.status,
                        minimum_certificate=_certificate(task_result),
                        note=None,
                    )
                else:
                    records[index] = replace(
                        current,
                        classification=ActivityClassification.INCONCLUSIVE,
                        minimum_slack=slack,
                        minimum_is_exact=True,
                        min_status=task_result.status,
                        minimum_certificate=_certificate(task_result),
                        note="optimized minimum lies in the activity tolerance gray zone",
                    )
                    failures.append(
                        _failure(solution, task_result, stage="minimum_slack_gray_zone")
                    )
            else:
                records[index] = replace(
                    current,
                    classification=ActivityClassification.INCONCLUSIVE,
                    min_status=task_result.status,
                    note=task_result.message,
                )
                failures.append(_failure(solution, task_result, stage="minimum_slack"))

        # A universally active enforced inequality has the structural lower
        # bound zero, so a certified zero maximum also completes its full
        # slack range without another optimization.
        for index in row_indices:
            current = records[index]
            if (
                current.classification is ActivityClassification.UNIVERSALLY_ACTIVE
                and current.maximum_is_exact
            ):
                records[index] = replace(current, minimum_slack=0.0, minimum_is_exact=True)

    dual_results: tuple[DualRangeResult, ...] = ()
    dual_solve_count = 0
    if selected_options.dual_ranges:
        from mechaudit.optimality.dual import compute_dual_ranges

        dual_results, dual_failures = compute_dual_ranges(
            solution,
            row_indices,
            delta=delta,
            n_jobs=selected_options.n_jobs,
            tolerances=tolerances,
        )
        dual_solve_count = 2 * len(row_indices)
        failures.extend(dual_failures)

    return OptimalFaceReport(
        model_fingerprint=solution.model_fingerprint,
        objective_name=solution.compiled.objective.name,
        optimal_value=optimal_value,
        delta=delta,
        requested_delta=selected_options.objective_delta,
        level=selected_options.level.value,
        tolerances=dict(tolerances.__dict__),
        records=tuple(records[index] for index in row_indices),
        solve_counts=SolveCounts(
            ipm_candidates=1,
            maximum_slack=len(maximum_tasks),
            minimum_slack=minimum_count,
            dual_range=dual_solve_count,
        ),
        failures=tuple(failures),
        dual_ranges=tuple(dual_results),
        analyzed_families=families,
        skipped_constraint_count=skipped,
        candidate_status=candidate_status,
        candidate_message=candidate_message,
        candidate_objective_value=candidate_value,
        candidate_max_primal_violation=candidate_violation,
        candidate_max_complementarity=candidate_complementarity,
        _solution=solution,
    )
