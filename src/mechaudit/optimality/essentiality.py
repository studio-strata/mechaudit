"""Redundancy, ablation, and implementation-essentiality diagnostics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import numpy as np

from mechaudit.backends.base import ModelStatus
from mechaudit.backends.highs import HighsBackend
from mechaudit.design.solution import DesignSolution
from mechaudit.design.sparse_model import CompiledLP, RowSense
from mechaudit.optimality.report import (
    AuxiliaryStatus,
    ConstraintCostResult,
    FamilyAblationResult,
    ImplementationEssentialityResult,
    MechanismWitness,
    RedundancyResult,
)
from mechaudit.optimality.tasks import (
    SlackTask,
    SlackTaskResult,
    certified_optimal_result,
    make_face_model,
    run_slack_tasks,
    solver_config,
)


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


def _effective_delta(solution: DesignSolution, delta: float | None) -> float:
    if delta is None:
        return max(
            solution.tolerances.objective_absolute,
            solution.tolerances.objective_relative * max(1.0, abs(solution.objective_value)),
        )
    value = float(delta)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("delta must be finite and nonnegative")
    return value


def _row_indices(solution: DesignSolution, constraint: Any) -> tuple[int, ...]:
    model = solution.compiled.model
    if isinstance(constraint, int) and not isinstance(constraint, bool):
        index = constraint
    elif isinstance(constraint, str):
        try:
            index = model.row_index(constraint)
        except KeyError as exc:
            raise KeyError(f"unknown constraint id: {constraint!r}") from exc
    else:
        candidate = getattr(constraint, "row_index", None)
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            index = candidate
        else:
            key = getattr(constraint, "constraint_id", None)
            if not isinstance(key, str):
                raise TypeError("constraint must be a row index, semantic id, or row-like result")
            try:
                index = model.row_index(key)
            except KeyError as exc:
                raise KeyError(f"unknown constraint id: {key!r}") from exc
    if index < 0 or index >= model.n_rows:
        raise IndexError(f"constraint row index out of range: {index}")
    return (index,)


def _validated_rows(solution: DesignSolution, rows: Iterable[int]) -> tuple[int, ...]:
    model = solution.compiled.model
    result = tuple(dict.fromkeys(int(index) for index in rows))
    if not result:
        raise ValueError("at least one constraint row is required")
    for index in result:
        if index < 0 or index >= model.n_rows:
            raise IndexError(f"constraint row index out of range: {index}")
    return result


def _witness(
    solution: DesignSolution,
    primal: np.ndarray | tuple[float, ...],
    *,
    target_value: float | None = None,
) -> MechanismWitness:
    vector = np.asarray(primal, dtype=float)
    mechanism_dict = None
    mechanism_fingerprint = None
    try:
        mechanism = solution.compiled.mechanism_from_vector(vector)
        mechanism_dict = mechanism.to_dict()
        mechanism_fingerprint = mechanism.fingerprint
    except Exception:
        pass
    return MechanismWitness(
        primal=tuple(float(value) for value in vector),
        design_objective_value=solution.compiled.model.objective_value(vector),
        target_value=target_value,
        mechanism_fingerprint=mechanism_fingerprint,
        mechanism=mechanism_dict,
    )


def _objective_ablation(solution: DesignSolution, rows: tuple[int, ...]) -> ConstraintCostResult:
    model = solution.compiled.model
    rows = _validated_rows(solution, rows)
    ids = tuple(model.rows[index].key for index in rows)
    tolerance = max(
        solution.tolerances.objective_absolute,
        solution.tolerances.objective_relative * max(1.0, abs(solution.objective_value)),
    )
    try:
        with HighsBackend().session(
            model, solver_config(solution.tolerances, solver="simplex")
        ) as session:
            session.remove_rows(rows)
            result = session.solve()
    except Exception as exc:
        return ConstraintCostResult(
            row_indices=rows,
            constraint_ids=ids,
            baseline_objective=solution.objective_value,
            relaxed_objective=None,
            objective_gain=None,
            objective_essential=None,
            status=AuxiliaryStatus.FAILED,
            objective_tolerance=tolerance,
            witness=None,
            solve_count=1,
            message=f"{type(exc).__name__}: {exc}",
        )
    status = _status(result.status)
    if status is AuxiliaryStatus.UNBOUNDED:
        return ConstraintCostResult(
            row_indices=rows,
            constraint_ids=ids,
            baseline_objective=solution.objective_value,
            relaxed_objective=float("inf"),
            objective_gain=float("inf"),
            objective_essential=True,
            status=status,
            objective_tolerance=tolerance,
            witness=None,
            solve_count=1,
            message="the relaxed design problem is unbounded",
        )
    certified, certificate_message = certified_optimal_result(result, solution.tolerances)
    if not certified or result.primal is None:
        return ConstraintCostResult(
            row_indices=rows,
            constraint_ids=ids,
            baseline_objective=solution.objective_value,
            relaxed_objective=result.objective_value,
            objective_gain=None,
            objective_essential=None,
            status=status if status is not AuxiliaryStatus.OPTIMAL else AuxiliaryStatus.FAILED,
            objective_tolerance=tolerance,
            witness=None,
            solve_count=1,
            message=certificate_message,
        )
    relaxed = model.objective_value(result.primal)
    gain = relaxed - solution.objective_value
    if abs(gain) <= tolerance:
        gain = 0.0
    if gain < -tolerance:
        return ConstraintCostResult(
            row_indices=rows,
            constraint_ids=ids,
            baseline_objective=solution.objective_value,
            relaxed_objective=relaxed,
            objective_gain=gain,
            objective_essential=None,
            status=AuxiliaryStatus.FAILED,
            objective_tolerance=tolerance,
            witness=_witness(solution, result.primal, target_value=gain),
            solve_count=1,
            message="removing constraints numerically lowered a maximization optimum",
        )
    return ConstraintCostResult(
        row_indices=rows,
        constraint_ids=ids,
        baseline_objective=solution.objective_value,
        relaxed_objective=relaxed,
        objective_gain=gain,
        objective_essential=gain > tolerance,
        status=AuxiliaryStatus.OPTIMAL,
        objective_tolerance=tolerance,
        witness=_witness(solution, result.primal, target_value=gain),
        solve_count=1,
        message=result.message,
    )


def test_redundancy(solution: DesignSolution, constraint: Any) -> RedundancyResult:
    """Remove a row and maximize its violation under the remaining system."""

    rows = _row_indices(solution, constraint)
    model = solution.compiled.model
    row = model.rows[rows[0]]
    if row.sense is not RowSense.GREATER_EQUAL:
        raise NotImplementedError("redundancy testing requires inequality rows")
    tasks = (SlackTask(rows[0], "min", removed_rows=rows, label="redundancy"),)
    results = run_slack_tasks(
        model,
        model,
        tasks,
        optimal_value=solution.objective_value,
        delta=0.0,
        tolerances=solution.tolerances,
        n_jobs=1,
        require_face=False,
    )
    result = results[0]
    ids = (row.key,)
    positive = 10.0 * max(solution.tolerances.activity, 1e-15)
    if result.status is AuxiliaryStatus.UNBOUNDED:
        return RedundancyResult(
            rows,
            ids,
            False,
            float("-inf"),
            result.status,
            solution.tolerances.activity,
            None,
            1,
            "the removed row can be violated without bound",
        )
    if result.status is not AuxiliaryStatus.OPTIMAL or result.slack is None:
        return RedundancyResult(
            rows,
            ids,
            None,
            result.slack,
            result.status,
            solution.tolerances.activity,
            None,
            1,
            result.message,
        )
    if result.slack >= -solution.tolerances.activity:
        redundant: bool | None = True
        message = "the remaining feasible system implies the removed row"
    elif result.slack < -positive:
        redundant = False
        message = "a feasible violation witness exists after row removal"
    else:
        redundant = None
        message = "maximum violation lies in the redundancy tolerance gray zone"
    witness = (
        None
        if result.primal is None
        else _witness(solution, result.primal, target_value=result.slack)
    )
    return RedundancyResult(
        rows,
        ids,
        redundant,
        result.slack,
        result.status,
        solution.tolerances.activity,
        witness,
        1,
        message,
    )


def _coordinate_separation(
    solution: DesignSolution,
    face_model: CompiledLP,
    witness: MechanismWitness,
    projection: str,
) -> tuple[bool | None, int, str]:
    """Search for a coordinate outside every original-face coordinate range.

    Failure to find such a coordinate is only non-detection: coordinate-wise
    ranges are an outer approximation to a projected polytope.
    """

    model = solution.compiled.model
    columns = tuple(
        index for index, record in enumerate(model.columns) if record.block == projection
    )
    if not columns:
        return None, 0, f"the compiled model has no {projection!r} coordinates"
    vector = np.asarray(witness.primal, dtype=float)
    tolerance = solution.tolerances.activity
    solves = 0
    try:
        with HighsBackend().session(
            face_model, solver_config(solution.tolerances, solver="simplex")
        ) as session:
            basis = None
            for column in columns:
                unit = np.zeros(model.n_columns, dtype=float)
                unit[column] = 1.0
                bounds: list[float] = []
                for sign in (-1.0, 1.0):
                    session.set_objective(sign * unit)
                    result = session.solve(basis=basis)
                    solves += 1
                    if result.basis is not None:
                        basis = result.basis
                    certified, _ = certified_optimal_result(result, solution.tolerances)
                    if not certified or result.primal is None:
                        return (
                            None,
                            solves,
                            "a coordinate-range auxiliary solve was inconclusive",
                        )
                    value = float(result.primal[column])
                    bounds.append(value)
                minimum, maximum = bounds[0], bounds[1]
                if vector[column] < minimum - tolerance or vector[column] > maximum + tolerance:
                    return (
                        True,
                        solves,
                        "a relaxed witness lies outside an original-face coordinate range",
                    )
    except Exception as exc:
        return None, solves, f"coordinate-range solve failed: {type(exc).__name__}: {exc}"
    return (
        None,
        solves,
        "no coordinate-range separation was found; this does not prove projection containment",
    )


def _implementation_after_unchanged(
    solution: DesignSolution,
    rows: tuple[int, ...],
    *,
    delta: float,
    projection: str,
) -> ImplementationEssentialityResult:
    model = solution.compiled.model
    ids = tuple(model.rows[index].key for index in rows)
    for index in rows:
        if model.rows[index].sense is not RowSense.GREATER_EQUAL:
            raise NotImplementedError("implementation essentiality requires inequality rows")
    face_model = make_face_model(model, optimal_value=solution.objective_value, delta=delta)
    tasks = tuple(
        SlackTask(index, "min", removed_rows=rows, label="implementation_essentiality")
        for index in rows
    )
    results = run_slack_tasks(
        model,
        face_model,
        tasks,
        optimal_value=solution.objective_value,
        delta=delta,
        tolerances=solution.tolerances,
        n_jobs=1,
    )
    positive = 10.0 * max(solution.tolerances.activity, 1e-15)
    violating: list[tuple[SlackTaskResult, MechanismWitness]] = []
    gray = False
    for result in results:
        if result.status is AuxiliaryStatus.UNBOUNDED:
            return ImplementationEssentialityResult(
                rows,
                ids,
                True if projection == "full" else None,
                projection,
                "exact_full_mechanism_violation"
                if projection == "full"
                else "coordinate_range_witness",
                float("inf"),
                result.status,
                None,
                len(tasks),
                "a removed inequality can be violated without bound on the relaxed face",
            )
        if result.status is not AuxiliaryStatus.OPTIMAL or result.slack is None:
            return ImplementationEssentialityResult(
                rows,
                ids,
                None,
                projection,
                "exact_full_mechanism_violation"
                if projection == "full"
                else "coordinate_range_witness",
                None,
                result.status,
                None,
                len(tasks),
                result.message,
            )
        if result.slack < -positive and result.primal is not None:
            violating.append(
                (
                    result,
                    _witness(solution, result.primal, target_value=-result.slack),
                )
            )
        elif result.slack < -solution.tolerances.activity:
            gray = True
    if not violating:
        if gray:
            return ImplementationEssentialityResult(
                rows,
                ids,
                None,
                projection,
                "exact_full_mechanism_violation",
                None,
                AuxiliaryStatus.OPTIMAL,
                None,
                len(tasks),
                "maximum violation lies in the implementation tolerance gray zone",
            )
        return ImplementationEssentialityResult(
            rows,
            ids,
            False,
            projection,
            "exact_full_mechanism_containment",
            0.0,
            AuxiliaryStatus.OPTIMAL,
            None,
            len(tasks),
            "the relaxed near-optimal face satisfies every removed inequality",
        )

    worst_result, worst_witness = max(violating, key=lambda item: -float(item[0].slack or 0.0))
    violation = -float(worst_result.slack or 0.0)
    if projection == "full":
        return ImplementationEssentialityResult(
            rows,
            ids,
            True,
            projection,
            "exact_full_mechanism_violation",
            violation,
            AuxiliaryStatus.OPTIMAL,
            worst_witness,
            len(tasks),
            "a relaxed near-optimal mechanism violates a removed inequality",
        )

    coordinate_solves = 0
    messages: list[str] = []
    for _, witness in violating:
        separated, used, message = _coordinate_separation(solution, face_model, witness, projection)
        coordinate_solves += used
        messages.append(message)
        if separated:
            return ImplementationEssentialityResult(
                rows,
                ids,
                True,
                projection,
                "coordinate_range_witness",
                violation,
                AuxiliaryStatus.OPTIMAL,
                witness,
                len(tasks) + coordinate_solves,
                message,
            )
    return ImplementationEssentialityResult(
        rows,
        ids,
        None,
        projection,
        "coordinate_range_witness",
        violation,
        AuxiliaryStatus.OPTIMAL,
        worst_witness,
        len(tasks) + coordinate_solves,
        "; ".join(messages),
    )


def _equality_implementation_result(
    solution: DesignSolution,
    rows: tuple[int, ...],
    *,
    projection: str,
    solve_count: int,
) -> ImplementationEssentialityResult:
    model = solution.compiled.model
    return ImplementationEssentialityResult(
        row_indices=rows,
        constraint_ids=tuple(model.rows[index].key for index in rows),
        essential=None,
        projection=projection,
        method="not_implemented_equality_removal",
        maximum_violation=None,
        status=AuxiliaryStatus.NOT_RUN,
        witness=None,
        solve_count=solve_count,
        message=(
            "full implementation essentiality for equality removal requires "
            "two-sided violation analysis and is not supported"
        ),
    )


def implementation_essentiality(
    solution: DesignSolution,
    constraint: Any,
    *,
    delta: float | None = None,
    projection: str = "full",
) -> ImplementationEssentialityResult:
    """Test whether removal expands the optimal mechanism set.

    ``projection='full'`` is an exact row-violation test.  Allocation/payment
    projections use a one-sided coordinate-range witness and return ``None``
    when no separating coordinate is found.
    """

    if projection not in {"full", "allocation", "payment"}:
        raise ValueError("projection must be 'full', 'allocation', or 'payment'")
    rows = _row_indices(solution, constraint)
    effective_delta = _effective_delta(solution, delta)
    cost = _objective_ablation(solution, rows)
    if cost.objective_essential is not False:
        return ImplementationEssentialityResult(
            rows,
            cost.constraint_ids,
            None,
            projection,
            "not_applicable_objective_changed",
            None,
            cost.status,
            cost.witness,
            cost.solve_count,
            (
                "implementation essentiality is evaluated separately only when "
                "removal leaves the objective unchanged"
            ),
        )
    if any(solution.compiled.model.rows[index].sense is RowSense.EQUAL for index in rows):
        return _equality_implementation_result(
            solution, rows, projection=projection, solve_count=cost.solve_count
        )
    result = _implementation_after_unchanged(
        solution, rows, delta=effective_delta, projection=projection
    )
    return replace(result, solve_count=result.solve_count + cost.solve_count)


def constraint_cost(
    solution: DesignSolution,
    constraint: Any,
    *,
    delta: float | None = None,
    projection: str = "full",
) -> ConstraintCostResult:
    """Measure objective cost and, when applicable, implementation essentiality."""

    if projection not in {"full", "allocation", "payment"}:
        raise ValueError("projection must be 'full', 'allocation', or 'payment'")
    rows = _row_indices(solution, constraint)
    cost = _objective_ablation(solution, rows)
    if cost.objective_essential is not False:
        return cost
    if any(solution.compiled.model.rows[index].sense is RowSense.EQUAL for index in rows):
        implementation = _equality_implementation_result(
            solution, rows, projection=projection, solve_count=0
        )
        return replace(cost, implementation=implementation)
    effective_delta = _effective_delta(solution, delta)
    implementation = _implementation_after_unchanged(
        solution, rows, delta=effective_delta, projection=projection
    )
    return replace(
        cost,
        implementation=implementation,
        solve_count=cost.solve_count + implementation.solve_count,
    )


def ablate_family(
    solution: DesignSolution,
    family: str,
    *,
    agent: Any = None,
    delta: float | None = None,
    projection: str = "full",
) -> FamilyAblationResult:
    """Remove all rows in a semantic family, with an optional agent filter."""

    if not isinstance(family, str) or not family:
        raise ValueError("family must be a nonempty string")
    model = solution.compiled.model
    rows = []
    for index, record in enumerate(model.rows):
        if record.family != family:
            continue
        if agent is not None:
            recorded = record.metadata_dict.get("agent")
            if recorded != agent and recorded != repr(agent):
                continue
        rows.append(index)
    selected = _validated_rows(solution, rows)
    if projection not in {"full", "allocation", "payment"}:
        raise ValueError("projection must be 'full', 'allocation', or 'payment'")
    cost = _objective_ablation(solution, selected)
    if cost.objective_essential is False:
        if any(model.rows[index].sense is RowSense.EQUAL for index in selected):
            implementation = _equality_implementation_result(
                solution, selected, projection=projection, solve_count=0
            )
            cost = replace(cost, implementation=implementation)
            return FamilyAblationResult(
                family=family,
                row_indices=selected,
                constraint_ids=tuple(model.rows[index].key for index in selected),
                cost=cost,
            )
        effective_delta = _effective_delta(solution, delta)
        implementation = _implementation_after_unchanged(
            solution, selected, delta=effective_delta, projection=projection
        )
        cost = replace(
            cost,
            implementation=implementation,
            solve_count=cost.solve_count + implementation.solve_count,
        )
    return FamilyAblationResult(
        family=family,
        row_indices=selected,
        constraint_ids=tuple(model.rows[index].key for index in selected),
        cost=cost,
    )


# Short public aliases used in notebooks and by older design sketches.
redundancy = test_redundancy
implementation_essential = implementation_essentiality
