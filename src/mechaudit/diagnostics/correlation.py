"""Conditional-belief and Crémer-McLean conditioning diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from mechaudit.core.environment import FiniteEnvironment
from mechaudit.core.labels import AgentId, TypeValue, thaw_label
from mechaudit.exceptions import EnvironmentValidationError


@dataclass(frozen=True)
class ConditionalBeliefDiagnostic:
    """Singular-value report for one agent's conditional-belief rows."""

    agent: AgentId
    types: tuple[TypeValue, ...]
    other_type_profiles: tuple[tuple[tuple[AgentId, TypeValue], ...], ...]
    matrix: np.ndarray
    singular_values: tuple[float, ...]
    rank: int
    full_row_rank: bool
    smallest_singular_value: float
    condition_number: float
    rank_tolerance: float
    zero_marginal_types: tuple[TypeValue, ...]
    nearly_duplicate_pairs: tuple[tuple[TypeValue, TypeValue, float], ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a numerical, labeled representation."""

        return {
            "agent": thaw_label(self.agent),
            "types": [thaw_label(value) for value in self.types],
            "other_type_profiles": [
                {str(thaw_label(agent)): thaw_label(value) for agent, value in profile}
                for profile in self.other_type_profiles
            ],
            "conditional_belief_matrix": self.matrix.tolist(),
            "singular_values": list(self.singular_values),
            "rank": self.rank,
            "full_row_rank": self.full_row_rank,
            "smallest_singular_value": self.smallest_singular_value,
            "condition_number": self.condition_number,
            "rank_tolerance": self.rank_tolerance,
            "zero_marginal_types": [thaw_label(value) for value in self.zero_marginal_types],
            "nearly_duplicate_pairs": [
                {
                    "first_type": thaw_label(first),
                    "second_type": thaw_label(second),
                    "distance": distance,
                }
                for first, second, distance in self.nearly_duplicate_pairs
            ],
            "warnings": list(self.warnings),
        }

    def to_pandas(self) -> pd.DataFrame:
        """Return the conditional-belief matrix with labeled axes."""

        columns = [repr(dict(profile)) for profile in self.other_type_profiles]
        return pd.DataFrame(
            self.matrix, index=[repr(value) for value in self.types], columns=columns
        )


@dataclass(frozen=True)
class CremerMcLeanReport:
    """Rank and conditioning diagnostics for all declared agents."""

    diagnostics: tuple[ConditionalBeliefDiagnostic, ...]
    maximum_absolute_payment: float | None = None
    payment_to_value_ratio: float | None = None

    @property
    def full_rank(self) -> bool:
        """Whether every conditional-belief matrix has full row rank."""

        return all(item.full_row_rank for item in self.diagnostics)

    @property
    def worst_condition_number(self) -> float:
        """Largest reported two-norm condition number."""

        return max((item.condition_number for item in self.diagnostics), default=math.nan)

    def for_agent(self, agent: AgentId) -> ConditionalBeliefDiagnostic:
        """Return one agent's diagnostic."""

        for diagnostic in self.diagnostics:
            if diagnostic.agent == agent:
                return diagnostic
        raise KeyError(agent)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible numerical report."""

        return {
            "full_rank": self.full_rank,
            "worst_condition_number": self.worst_condition_number,
            "maximum_absolute_payment": self.maximum_absolute_payment,
            "payment_to_value_ratio": self.payment_to_value_ratio,
            "agents": [item.to_dict() for item in self.diagnostics],
        }


def _diagnose_agent(
    environment: FiniteEnvironment,
    agent: AgentId,
    *,
    rank_tolerance: float,
    duplicate_tolerance: float,
) -> ConditionalBeliefDiagnostic:
    prior = environment.prior
    if prior is None:
        raise EnvironmentValidationError("conditional-belief diagnostics require a joint prior")

    types = environment.types[agent]
    stride = environment._strides[environment.agent_index(agent)]
    others_entries = environment._other_profiles[agent]
    others = [labels for labels, _ in others_entries]
    masses = environment._profile_masses
    if masses is None:
        raise EnvironmentValidationError("conditional-belief diagnostics require a joint prior")

    matrix = np.zeros((len(types), len(others)), dtype=float)
    zero_types: list[TypeValue] = []
    for row, type_value in enumerate(types):
        marginal = prior.marginal(agent, type_value)
        if marginal <= 0.0:
            zero_types.append(type_value)
            continue
        for column, (_, base) in enumerate(others_entries):
            matrix[row, column] = masses[base + row * stride] / marginal

    singular = np.linalg.svd(matrix, compute_uv=False)
    singular_values = tuple(float(value) for value in singular)
    rank = int(np.sum(singular > rank_tolerance))
    smallest = float(singular[-1]) if len(singular) else 0.0
    condition = math.inf if smallest <= rank_tolerance else float(singular[0] / smallest)
    full_row_rank = rank == len(types)

    duplicates: list[tuple[TypeValue, TypeValue, float]] = []
    for first in range(len(types)):
        for second in range(first + 1, len(types)):
            distance = float(np.linalg.norm(matrix[first] - matrix[second]))
            if distance <= duplicate_tolerance:
                duplicates.append((types[first], types[second], distance))

    warnings: list[str] = []
    if zero_types:
        warnings.append(
            f"Agent {agent!r} has zero-marginal types; their conditional beliefs are undefined."
        )
    if not full_row_rank:
        warnings.append(
            f"Agent {agent!r}'s conditional-belief matrix is not full row rank at tolerance {rank_tolerance:.3g}."
        )
    elif not math.isfinite(condition) or condition >= 1.0 / max(rank_tolerance, 1e-16):
        warnings.append(
            f"Agent {agent!r}'s conditional-belief system is numerically fragile (condition {condition:.3g})."
        )
    if duplicates:
        warnings.append(f"Agent {agent!r} has nearly identical conditional-belief rows.")

    matrix.setflags(write=False)
    return ConditionalBeliefDiagnostic(
        agent=agent,
        types=types,
        other_type_profiles=tuple(others),
        matrix=matrix,
        singular_values=singular_values,
        rank=rank,
        full_row_rank=full_row_rank,
        smallest_singular_value=smallest,
        condition_number=condition,
        rank_tolerance=rank_tolerance,
        zero_marginal_types=tuple(zero_types),
        nearly_duplicate_pairs=tuple(duplicates),
        warnings=tuple(warnings),
    )


def cremer_mclean_diagnostic(
    environment: FiniteEnvironment,
    mechanism: Any | None = None,
    *,
    rank_tolerance: float = 1e-10,
    duplicate_tolerance: float = 1e-7,
) -> CremerMcLeanReport:
    """Evaluate rank and distance-to-singularity of conditional beliefs.

    If a mechanism is supplied, its maximum absolute transfer and transfer-to-
    value scale ratio are included. The report is a conditioning diagnostic,
    not a theorem that a particular extraction mechanism has been implemented.
    """

    if rank_tolerance <= 0.0 or duplicate_tolerance < 0.0:
        raise ValueError("rank tolerance must be positive and duplicate tolerance nonnegative")
    diagnostics = tuple(
        _diagnose_agent(
            environment,
            agent,
            rank_tolerance=rank_tolerance,
            duplicate_tolerance=duplicate_tolerance,
        )
        for agent in environment.agents
    )
    maximum_payment: float | None = None
    ratio: float | None = None
    if mechanism is not None:
        bound = mechanism.bind(environment)
        maximum_payment = float(np.max(np.abs(bound.payments))) if bound.payments.size else 0.0
        values = np.asarray(environment.values, dtype=float)
        value_scale = float(np.max(np.abs(values))) if values.size else 0.0
        ratio = (
            math.inf
            if value_scale == 0.0 and maximum_payment > 0
            else (maximum_payment / value_scale if value_scale > 0.0 else 0.0)
        )
    return CremerMcLeanReport(diagnostics, maximum_payment, ratio)


def correlation_diagnostics(
    environment: FiniteEnvironment,
    *,
    rank_tolerance: float = 1e-10,
    duplicate_tolerance: float = 1e-7,
) -> tuple[ConditionalBeliefDiagnostic, ...]:
    """Convenience wrapper returning one conditional-belief report per agent."""

    return cremer_mclean_diagnostic(
        environment,
        rank_tolerance=rank_tolerance,
        duplicate_tolerance=duplicate_tolerance,
    ).diagnostics
