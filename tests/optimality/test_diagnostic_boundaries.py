"""Input, projection, dual-construction, and auxiliary-task boundaries."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from mechaudit import (
    AllocationCoordinate,
    CustomLinearConstraint,
    CustomLinearObjective,
    FiniteEnvironment,
    PaymentCoordinate,
    PrivateValues,
)
from mechaudit.backends.base import ModelStatus
from mechaudit.design.sparse_model import ColumnRecord, CompiledLP, RowSense
from mechaudit.optimality import (
    AuxiliaryStatus,
    ablate_family,
    build_explicit_dual,
    compute_dual_ranges,
    constraint_cost,
    implementation_essentiality,
)
from mechaudit.optimality import (
    test_redundancy as check_redundancy,
)
from mechaudit.optimality.dual import _status as _dual_status
from mechaudit.optimality.essentiality import _status as _essentiality_status
from mechaudit.optimality.tasks import (
    SlackTask,
    make_face_model,
    run_slack_tasks,
    solve_slack_task,
    solver_config,
)


def _projection_solution():
    environment = FiniteEnvironment(
        agents={"agent": ("only",)},
        outcomes=("off", "on"),
        prior={"only": 1.0},
        values=PrivateValues(lambda _agent, _type, _outcome: 0.0),
    )
    report = environment.report_profiles[0]
    payment = PaymentCoordinate(report, "agent")
    on = AllocationCoordinate(report, "on")
    problem = environment.design(
        objective=CustomLinearObjective("on_probability", {on: 1.0}),
        requirements=(
            CustomLinearConstraint(
                "loose_floor", {payment: 1.0}, sense=">=", rhs=-1.0, family="loose"
            ),
            CustomLinearConstraint(
                "binding_floor", {payment: 1.0}, sense=">=", rhs=0.0, family="binding"
            ),
            CustomLinearConstraint("cap", {payment: 1.0}, sense="<=", rhs=1.0, family="cap"),
        ),
    )
    return problem.solve()


def _row(solution, family: str) -> int:
    return next(
        index
        for index, record in enumerate(solution.compiled.model.rows)
        if record.family == family
    )


def test_projected_implementation_essentiality_is_one_sided_and_witnessed() -> None:
    solution = _projection_solution()
    binding = _row(solution, "binding")

    full = implementation_essentiality(solution, binding, delta=0.0, projection="full")
    payment = implementation_essentiality(solution, binding, delta=0.0, projection="payment")
    allocation = implementation_essentiality(solution, binding, delta=0.0, projection="allocation")

    assert full.essential is True
    assert full.method == "exact_full_mechanism_violation"
    assert full.maximum_violation == pytest.approx(1.0)
    assert payment.essential is True
    assert payment.method == "coordinate_range_witness"
    assert payment.solve_count > full.solve_count
    assert allocation.essential is None
    assert allocation.method == "coordinate_range_witness"
    assert "does not prove projection containment" in allocation.message


def test_essentiality_constraint_selectors_and_validation_errors() -> None:
    solution = _projection_solution()
    binding = _row(solution, "binding")
    binding_id = solution.compiled.model.rows[binding].key

    assert constraint_cost(solution, binding_id).row_indices == (binding,)
    assert constraint_cost(solution, SimpleNamespace(row_index=binding)).row_indices == (binding,)
    assert constraint_cost(solution, SimpleNamespace(constraint_id=binding_id)).row_indices == (
        binding,
    )

    with pytest.raises(TypeError, match="row index, semantic id"):
        constraint_cost(solution, object())
    with pytest.raises(TypeError, match="row index, semantic id"):
        constraint_cost(solution, True)
    with pytest.raises(KeyError, match="unknown constraint id"):
        constraint_cost(solution, "missing")
    with pytest.raises(KeyError, match="unknown constraint id"):
        constraint_cost(solution, SimpleNamespace(constraint_id="missing"))
    with pytest.raises(IndexError, match="out of range"):
        constraint_cost(solution, 99)
    with pytest.raises(ValueError, match="projection"):
        constraint_cost(solution, binding, projection="utility")
    with pytest.raises(ValueError, match="projection"):
        implementation_essentiality(solution, binding, projection="utility")
    with pytest.raises(ValueError, match="delta"):
        implementation_essentiality(solution, binding, delta=-1.0)
    with pytest.raises(ValueError, match="delta"):
        constraint_cost(solution, binding, delta=float("nan"))


def test_family_ablation_and_equality_redundancy_boundaries() -> None:
    solution = _projection_solution()
    equality = next(
        index
        for index, record in enumerate(solution.compiled.model.rows)
        if record.sense is RowSense.EQUAL
    )

    with pytest.raises(NotImplementedError, match="inequality rows"):
        check_redundancy(solution, equality)
    with pytest.raises(ValueError, match="nonempty string"):
        ablate_family(solution, "")
    with pytest.raises(ValueError, match="at least one constraint row"):
        ablate_family(solution, "missing")
    with pytest.raises(ValueError, match="at least one constraint row"):
        ablate_family(solution, "binding", agent="other")
    with pytest.raises(ValueError, match="projection"):
        ablate_family(solution, "binding", projection="utility")


@pytest.mark.parametrize(
    ("status", "expected"),
    (
        (ModelStatus.OPTIMAL, AuxiliaryStatus.OPTIMAL),
        (ModelStatus.UNBOUNDED, AuxiliaryStatus.UNBOUNDED),
        (ModelStatus.INFEASIBLE, AuxiliaryStatus.INFEASIBLE),
        (ModelStatus.INFEASIBLE_OR_UNBOUNDED, AuxiliaryStatus.INFEASIBLE),
        (ModelStatus.TIME_LIMIT, AuxiliaryStatus.LIMIT),
        (ModelStatus.ERROR, AuxiliaryStatus.FAILED),
    ),
)
def test_internal_status_normalization_is_consistent(
    status: ModelStatus, expected: AuxiliaryStatus
) -> None:
    assert _essentiality_status(status) is expected
    assert _dual_status(status) is expected


def test_explicit_dual_dummy_column_and_input_validation() -> None:
    primal = CompiledLP(
        sparse.csr_matrix((0, 1), dtype=float),
        np.array([], dtype=float),
        np.array([0.0]),
        columns=(ColumnRecord("free", "auxiliary"),),
    )

    explicit = build_explicit_dual(primal, optimal_value=7.0, delta=0.5)

    assert explicit.inequality_columns == {}
    assert explicit.model.n_columns == 1
    assert explicit.model.columns[0].key == "dual.dummy"
    assert explicit.model.lower_bounds[0] == 0.0
    assert explicit.model.upper_bounds[0] == 0.0
    assert explicit.model.rows[-1].key == "dual.near_optimal_face"
    for invalid in (-1.0, float("nan")):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            build_explicit_dual(primal, optimal_value=0.0, delta=invalid)


def test_dual_range_and_slack_task_input_boundaries() -> None:
    solution = _projection_solution()
    model = solution.compiled.model
    equality = next(
        index for index, record in enumerate(model.rows) if record.sense is RowSense.EQUAL
    )
    inequality = _row(solution, "binding")
    face_model = make_face_model(model, optimal_value=solution.objective_value, delta=0.0)
    nested_face = make_face_model(face_model, optimal_value=solution.objective_value, delta=0.0)

    assert compute_dual_ranges(solution, (), delta=0.0) == ((), ())
    with pytest.raises(IndexError, match="out of range"):
        compute_dual_ranges(solution, (99,), delta=0.0)
    with pytest.raises(ValueError, match="inequality rows"):
        compute_dual_ranges(solution, (equality,), delta=0.0)
    with pytest.raises(ValueError, match="nonnegative"):
        make_face_model(model, optimal_value=0.0, delta=-1.0)
    assert nested_face.rows[-1].key.endswith("[1]")

    with pytest.raises(ValueError, match="direction"):
        SlackTask(inequality, "sideways")
    with pytest.raises(IndexError, match="outside the base model"):
        solve_slack_task(
            model,
            face_model,
            SlackTask(999, "max"),
            optimal_value=solution.objective_value,
            delta=0.0,
            tolerances=solution.tolerances,
        )
    assert (
        run_slack_tasks(
            model,
            face_model,
            (),
            optimal_value=solution.objective_value,
            delta=0.0,
            tolerances=solution.tolerances,
        )
        == ()
    )
    with pytest.raises(ValueError, match="n_jobs"):
        run_slack_tasks(
            model,
            face_model,
            (SlackTask(inequality, "max"),),
            optimal_value=solution.objective_value,
            delta=0.0,
            tolerances=solution.tolerances,
            n_jobs=0,
        )
    config = solver_config(solution.tolerances, solver="ipm", presolve="off", crossover="off")
    assert config.solver == "ipm"
    assert config.presolve == "off"
    assert config.run_crossover == "off"
    assert config.threads == 1
