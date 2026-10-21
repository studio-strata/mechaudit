"""Invariant activity, dual-range, redundancy, and ablation analysis."""

from mechaudit.optimality.analysis import analyze_optimal_face
from mechaudit.optimality.dual import build_explicit_dual, compute_dual_ranges
from mechaudit.optimality.essentiality import (
    ablate_family,
    constraint_cost,
    implementation_essential,
    implementation_essentiality,
    redundancy,
    test_redundancy,
)
from mechaudit.optimality.report import (
    ActivityClassification,
    ActivityRecord,
    AuxiliaryCertificate,
    AuxiliaryStatus,
    ConstraintCostResult,
    DualRangeResult,
    DualRelevance,
    FamilyAblationResult,
    ImplementationEssentialityResult,
    MechanismWitness,
    OptimalFaceReport,
    RedundancyResult,
    SolveCounts,
    TaskFailure,
)

__all__ = [
    "ActivityClassification",
    "ActivityRecord",
    "AuxiliaryCertificate",
    "AuxiliaryStatus",
    "ConstraintCostResult",
    "DualRangeResult",
    "DualRelevance",
    "FamilyAblationResult",
    "ImplementationEssentialityResult",
    "MechanismWitness",
    "OptimalFaceReport",
    "RedundancyResult",
    "SolveCounts",
    "TaskFailure",
    "ablate_family",
    "analyze_optimal_face",
    "build_explicit_dual",
    "compute_dual_ranges",
    "constraint_cost",
    "implementation_essential",
    "implementation_essentiality",
    "redundancy",
    "test_redundancy",
]
