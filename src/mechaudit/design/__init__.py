"""Public design workflow and solver-independent compilation primitives."""

from mechaudit.design.assessment import CandidateAssessment
from mechaudit.design.compiler import (
    CompiledDesign,
    MechanismVariableLayout,
    VacuousConstraint,
    compile_design_problem,
)
from mechaudit.design.indexing import VariableIndex, VariableIndexBuilder, VariableKey
from mechaudit.design.linear import LinearExpr, LinearExpression, LinearExpressionBuilder
from mechaudit.design.problem import DesignProblem
from mechaudit.design.solution import DesignSolution, SelectedActivity
from mechaudit.design.sparse_model import (
    ColumnRecord,
    CompiledLP,
    PrimalResiduals,
    RowRecord,
    RowScale,
    RowSense,
    SolverLPData,
    SparseRow,
)

__all__ = [
    "CandidateAssessment",
    "ColumnRecord",
    "CompiledDesign",
    "CompiledLP",
    "DesignProblem",
    "DesignSolution",
    "LinearExpr",
    "LinearExpression",
    "LinearExpressionBuilder",
    "MechanismVariableLayout",
    "PrimalResiduals",
    "RowRecord",
    "RowScale",
    "RowSense",
    "SelectedActivity",
    "SolverLPData",
    "SparseRow",
    "VacuousConstraint",
    "VariableIndex",
    "VariableIndexBuilder",
    "VariableKey",
    "compile_design_problem",
]
