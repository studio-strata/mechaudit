"""Immutable direct mechanisms on explicit finite report domains."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, NamedTuple, cast

import numpy as np
from numpy.typing import NDArray

from mechaudit.core.labels import AgentId, OutcomeValue, ReportProfile
from mechaudit.exceptions import DomainMismatchError, ValidationError
from mechaudit.mechanisms.importers import (
    EnvironmentDomain,
    MechanismSource,
    canonicalize_source,
    mechanism_source,
    source_from_tables,
    source_from_xarray,
)

FloatArray = NDArray[np.float64]


class MechanismTables(NamedTuple):
    """Pandas allocation and payment tables returned by :meth:`to_pandas`.

    The named tuple supports both attribute access and ordinary unpacking::

        tables = mechanism.to_pandas()
        allocation_df = tables.allocation
        allocation_df, payments_df = tables
    """

    allocation: Any
    payments: Any


def _immutable_array(values: FloatArray, *, shape: tuple[int, int]) -> FloatArray:
    """Return a C-contiguous array backed by immutable bytes."""

    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape:
        raise ValidationError(
            "canonical mechanism matrix has an unexpected shape",
            context={"shape": array.shape, "expected": shape},
        )
    if not np.all(np.isfinite(array)):
        raise ValidationError("canonical mechanism matrices must be finite")
    # A normal read-only owning ndarray can have writeability re-enabled by a
    # caller.  A bytes-backed buffer makes the public immutability guarantee
    # effective even when users manipulate NumPy flags directly.
    frozen = np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=np.float64)
    return frozen.reshape(shape)


def _mechanism_fingerprint(
    *,
    domain_fingerprint: str,
    allocation: FloatArray,
    payments: FloatArray,
    payment_sign: str,
) -> str:
    """Hash canonical numeric mechanism content with unambiguous framing."""

    digest = hashlib.sha256()
    digest.update(b"mechaudit.direct-mechanism.v1\0")

    def add(payload: bytes) -> None:
        digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
        digest.update(payload)

    add(domain_fingerprint.encode("utf-8"))
    add(payment_sign.encode("ascii"))
    add(np.asarray(allocation.shape, dtype="<i8").tobytes(order="C"))
    add(np.asarray(payments.shape, dtype="<i8").tobytes(order="C"))
    add(np.asarray(allocation, dtype="<f8", order="C").tobytes(order="C"))
    add(np.asarray(payments, dtype="<f8", order="C").tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, init=False, eq=False)
class DirectMechanism:
    """A finite randomized direct mechanism.

    A bound mechanism stores allocation probabilities in a read-only
    ``float64`` array of shape ``(report_profiles, outcomes)`` and payments in
    a read-only ``float64`` array of shape ``(report_profiles, agents)``.
    Positive payments are transfers from an agent to the designer.

    Mappings, callables, or unlabeled arrays may be supplied before an
    environment exists.  :meth:`bind` evaluates and aligns those sources and
    returns a new immutable bound mechanism.  Binding validates numeric
    finiteness and complete labeled coverage. Negative or non-normalized
    allocation entries remain available to the economic audit rather than being
    rejected as import errors.

    Parameters
    ----------
    allocation:
        A profile-to-outcome mapping, callback, or numeric array.  Allocation
        callbacks may accept ``(report)`` and return a complete outcome mapping,
        or accept ``(report, outcome)`` and return one numeric cell.
    payments:
        A profile-to-agent mapping, callback, or numeric array.  In a one-agent
        environment, profile-to-number mappings and one-dimensional arrays are
        accepted.  Payment callbacks follow the analogous ``(report)`` or
        ``(report, agent)`` conventions.
    environment:
        If supplied, bind immediately.  Otherwise call :meth:`bind` later.
    domain_fingerprint:
        Optional expected environment-domain fingerprint.  It is checked before
        any callbacks are evaluated.
    """

    PAYMENT_SIGN: ClassVar[str] = "agent_to_designer"

    _allocation: FloatArray | None
    _payments: FloatArray | None
    _source: MechanismSource | None
    _agents: tuple[AgentId, ...]
    _outcomes: tuple[OutcomeValue, ...]
    _report_profiles: tuple[ReportProfile, ...]
    _domain_fingerprint: str | None
    _fingerprint: str | None

    def __init__(
        self,
        *,
        allocation: Any,
        payments: Any,
        environment: EnvironmentDomain | None = None,
        domain_fingerprint: str | None = None,
    ) -> None:
        source = mechanism_source(
            allocation,
            payments,
            domain_fingerprint=domain_fingerprint,
        )
        if environment is None:
            self._install_unbound(source)
        else:
            allocation_array, payment_array = canonicalize_source(source, environment)
            self._install_bound(
                allocation_array,
                payment_array,
                environment=environment,
            )

    def _install_unbound(self, source: MechanismSource) -> None:
        object.__setattr__(self, "_allocation", None)
        object.__setattr__(self, "_payments", None)
        object.__setattr__(self, "_source", source)
        object.__setattr__(self, "_agents", ())
        object.__setattr__(self, "_outcomes", ())
        object.__setattr__(self, "_report_profiles", ())
        object.__setattr__(self, "_domain_fingerprint", source.domain_fingerprint)
        object.__setattr__(self, "_fingerprint", None)

    def _install_bound(
        self,
        allocation: FloatArray,
        payments: FloatArray,
        *,
        environment: EnvironmentDomain,
    ) -> None:
        frozen_allocation = _immutable_array(
            allocation,
            shape=(environment.n_profiles, environment.n_outcomes),
        )
        frozen_payments = _immutable_array(
            payments,
            shape=(environment.n_profiles, environment.n_agents),
        )
        object.__setattr__(
            self,
            "_allocation",
            frozen_allocation,
        )
        object.__setattr__(
            self,
            "_payments",
            frozen_payments,
        )
        object.__setattr__(self, "_source", None)
        object.__setattr__(self, "_agents", tuple(environment.agents))
        object.__setattr__(self, "_outcomes", tuple(environment.outcomes))
        object.__setattr__(self, "_report_profiles", tuple(environment.report_profiles))
        object.__setattr__(self, "_domain_fingerprint", environment.domain_fingerprint)
        object.__setattr__(
            self,
            "_fingerprint",
            _mechanism_fingerprint(
                domain_fingerprint=environment.domain_fingerprint,
                allocation=frozen_allocation,
                payments=frozen_payments,
                payment_sign=self.PAYMENT_SIGN,
            ),
        )

    @classmethod
    def _from_source(
        cls,
        source: MechanismSource,
        *,
        environment: EnvironmentDomain | None,
    ) -> DirectMechanism:
        instance = object.__new__(cls)
        if environment is None:
            instance._install_unbound(source)
        else:
            allocation, payments = canonicalize_source(source, environment)
            instance._install_bound(allocation, payments, environment=environment)
        return instance

    @classmethod
    def from_mapping(
        cls,
        allocation: Mapping[Any, Any],
        payments: Mapping[Any, Any] | None = None,
        *,
        environment: EnvironmentDomain | None = None,
        domain_fingerprint: str | None = None,
    ) -> DirectMechanism:
        """Construct from canonical nested profile mappings.

        ``allocation`` normally maps each report profile to a complete
        outcome-to-probability mapping, while ``payments`` maps each report
        profile to a complete agent-to-payment mapping.  A combined mapping
        produced by :meth:`to_dict` is also accepted when ``payments`` is
        omitted.
        """

        raw_allocation: Any = allocation
        raw_payments: Any = payments
        fingerprint = domain_fingerprint
        if payments is None:
            if "allocation" not in allocation or "payments" not in allocation:
                raise ValidationError(
                    "payments are required unless a combined mechanism mapping is supplied"
                )
            raw_allocation = allocation["allocation"]
            raw_payments = allocation["payments"]
            embedded_fingerprint = allocation.get("domain_fingerprint")
            if (
                fingerprint is not None
                and embedded_fingerprint is not None
                and (fingerprint != embedded_fingerprint)
            ):
                raise DomainMismatchError("explicit and embedded mechanism fingerprints disagree")
            if fingerprint is None:
                fingerprint = embedded_fingerprint
            embedded_sign = allocation.get("payment_sign", cls.PAYMENT_SIGN)
            if embedded_sign != cls.PAYMENT_SIGN:
                raise ValidationError(
                    "unsupported payment sign convention",
                    context={"payment_sign": embedded_sign},
                )
        if not isinstance(raw_allocation, Mapping) or not isinstance(raw_payments, Mapping):
            raise ValidationError("from_mapping requires nested allocation/payment mappings")
        source = mechanism_source(
            raw_allocation,
            raw_payments,
            domain_fingerprint=fingerprint,
        )
        return cls._from_source(source, environment=environment)

    @classmethod
    def from_functions(
        cls,
        allocation: Callable[..., Any],
        payments: Callable[..., Any],
        *,
        environment: EnvironmentDomain | None = None,
        domain_fingerprint: str | None = None,
    ) -> DirectMechanism:
        """Construct from row or cell callbacks, evaluated during binding."""

        if not callable(allocation) or not callable(payments):
            raise ValidationError("from_functions requires two callable objects")
        source = mechanism_source(
            allocation,
            payments,
            domain_fingerprint=domain_fingerprint,
        )
        return cls._from_source(source, environment=environment)

    @classmethod
    def from_arrays(
        cls,
        allocation: Any,
        payments: Any,
        *,
        environment: EnvironmentDomain | None = None,
        report_profiles: Sequence[Any] | None = None,
        outcomes: Sequence[Any] | None = None,
        agents: Sequence[Any] | None = None,
        payment_report_profiles: Sequence[Any] | None = None,
        domain_fingerprint: str | None = None,
    ) -> DirectMechanism:
        """Construct from numeric arrays with optional explicit coordinates.

        Without coordinate arguments the arrays are declared to already use
        the bound environment's canonical axis order.  Explicit coordinates
        may appear in any order and are reordered during binding.
        ``payment_report_profiles`` defaults to ``report_profiles``.
        """

        # Use the shared validator rather than duplicating its detailed errors.
        from mechaudit.mechanisms.importers import array_provider

        allocation_provider = array_provider(
            allocation,
            path="allocation",
            row_labels=report_profiles,
            column_labels=outcomes,
        )
        payment_provider = array_provider(
            payments,
            path="payments",
            row_labels=(
                payment_report_profiles if payment_report_profiles is not None else report_profiles
            ),
            column_labels=agents,
            allow_one_dimensional=True,
        )
        source = cls._source_with_fingerprint(
            MechanismSource(allocation_provider, payment_provider, None),
            domain_fingerprint,
        )
        return cls._from_source(source, environment=environment)

    @classmethod
    def from_tables(
        cls,
        allocation: Any,
        payments: Any,
        *,
        environment: EnvironmentDomain | None = None,
        domain_fingerprint: str | None = None,
    ) -> DirectMechanism:
        """Construct from two wide pandas-like labeled tables."""

        source = source_from_tables(allocation, payments)
        source = cls._source_with_fingerprint(source, domain_fingerprint)
        return cls._from_source(source, environment=environment)

    @classmethod
    def from_xarray(
        cls,
        allocation: Any,
        payments: Any | None = None,
        *,
        environment: EnvironmentDomain | None = None,
        report_dims: Sequence[Any] | None = None,
        outcome_dim: Any = "outcome",
        payment_agent_dim: Any = "agent",
        domain_fingerprint: str | None = None,
    ) -> DirectMechanism:
        """Construct from two labeled xarray DataArrays or one Dataset.

        If ``payments`` is omitted, ``allocation`` must be Dataset-like and
        expose variables named ``"allocation"`` and ``"payments"``.
        Every non-column dimension is interpreted as a report axis and flattened
        in ``report_dims`` order.
        """

        allocation_data = allocation
        payment_data = payments
        if payments is None:
            try:
                allocation_data = allocation["allocation"]
                payment_data = allocation["payments"]
            except Exception as exc:
                raise ValidationError(
                    "a single xarray object must contain allocation and payments variables"
                ) from exc
        source = source_from_xarray(
            allocation_data,
            payment_data,
            report_dims=report_dims,
            outcome_dim=outcome_dim,
            payment_agent_dim=payment_agent_dim,
        )
        source = cls._source_with_fingerprint(source, domain_fingerprint)
        return cls._from_source(source, environment=environment)

    @staticmethod
    def _source_with_fingerprint(
        source: MechanismSource,
        explicit: str | None,
    ) -> MechanismSource:
        if explicit is not None and not isinstance(explicit, str):
            raise ValidationError("domain_fingerprint must be a string or None")
        if (
            explicit is not None
            and source.domain_fingerprint is not None
            and (explicit != source.domain_fingerprint)
        ):
            raise DomainMismatchError("explicit and imported domain fingerprints disagree")
        return MechanismSource(
            source.allocation,
            source.payments,
            explicit if explicit is not None else source.domain_fingerprint,
        )

    @property
    def is_bound(self) -> bool:
        """Whether numeric matrices and canonical axes are available."""

        return self._allocation is not None

    @property
    def payment_sign(self) -> str:
        """The global payment convention: positive is agent to designer."""

        return self.PAYMENT_SIGN

    @property
    def domain_fingerprint(self) -> str | None:
        """Bound or expected environment-domain fingerprint."""

        return self._domain_fingerprint

    @property
    def fingerprint(self) -> str:
        """Stable SHA-256 fingerprint of this bound numeric mechanism.

        The digest covers the domain fingerprint, payment convention, matrix
        shapes, and canonical little-endian ``float64`` matrix bytes.  It is
        therefore distinct for different mechanisms on the same environment.
        """

        self._require_bound()
        assert self._fingerprint is not None
        return self._fingerprint

    def _require_bound(self) -> None:
        if not self.is_bound:
            raise ValidationError(
                "mechanism is not bound to an environment; call bind(environment) first"
            )

    @property
    def allocation(self) -> FloatArray:
        """Read-only allocation matrix with shape ``(R, K)``."""

        self._require_bound()
        assert self._allocation is not None
        return self._allocation

    @property
    def payments(self) -> FloatArray:
        """Read-only payment matrix with shape ``(R, n)``."""

        self._require_bound()
        assert self._payments is not None
        return self._payments

    @property
    def agents(self) -> tuple[AgentId, ...]:
        """Canonical payment-axis labels."""

        self._require_bound()
        return self._agents

    @property
    def outcomes(self) -> tuple[OutcomeValue, ...]:
        """Canonical allocation-axis labels."""

        self._require_bound()
        return self._outcomes

    @property
    def report_profiles(self) -> tuple[ReportProfile, ...]:
        """Canonical allocation/payment row labels."""

        self._require_bound()
        return self._report_profiles

    @property
    def n_agents(self) -> int:
        """Number of payment columns."""

        return len(self.agents)

    @property
    def n_profiles(self) -> int:
        """Number of report rows."""

        return len(self.report_profiles)

    @property
    def n_outcomes(self) -> int:
        """Number of allocation columns."""

        return len(self.outcomes)

    def bind(self, environment: EnvironmentDomain) -> DirectMechanism:
        """Return this mechanism aligned to *environment*.

        Binding is idempotent for the same domain.  A bound mechanism cannot be
        silently reinterpreted on a different domain, even if matrix shapes
        happen to agree.
        """

        if self.is_bound:
            if self._domain_fingerprint != environment.domain_fingerprint:
                raise DomainMismatchError(
                    "bound mechanism cannot be used with a different environment domain",
                    context={
                        "mechanism": self._domain_fingerprint,
                        "environment": environment.domain_fingerprint,
                    },
                )
            if (
                self._agents != tuple(environment.agents)
                or self._outcomes != tuple(environment.outcomes)
                or self._report_profiles != tuple(environment.report_profiles)
            ):
                raise DomainMismatchError(
                    "environment axes disagree despite an equal domain fingerprint"
                )
            return self
        assert self._source is not None
        return type(self)._from_source(self._source, environment=environment)

    def allocation_row(self, report: Any, environment: EnvironmentDomain) -> FloatArray:
        """Return one read-only allocation row after verifying domain alignment."""

        mechanism = self.bind(environment)
        return cast(FloatArray, mechanism.allocation[environment.profile_index(report)])

    def payment_row(self, report: Any, environment: EnvironmentDomain) -> FloatArray:
        """Return one read-only payment row after verifying domain alignment."""

        mechanism = self.bind(environment)
        return cast(FloatArray, mechanism.payments[environment.profile_index(report)])

    def allocation_probability(
        self,
        report: Any,
        outcome: Any,
        environment: EnvironmentDomain,
    ) -> float:
        """Return ``x(outcome | report)`` using labeled coordinates."""

        mechanism = self.bind(environment)
        return float(
            mechanism.allocation[
                environment.profile_index(report), environment.outcome_index(outcome)
            ]
        )

    def payment(
        self,
        report: Any,
        agent: Any,
        environment: EnvironmentDomain,
    ) -> float:
        """Return the agent-to-designer payment for one report profile."""

        mechanism = self.bind(environment)
        candidate = environment.coerce_report_profile(report)
        row_index = environment.profile_index(candidate)
        for agent_index, expected in enumerate(environment.agents):
            if expected == agent:
                return float(mechanism.payments[row_index, agent_index])
        # Structured and sequence labels may need environment-style freezing.
        from mechaudit.core.labels import freeze_label

        frozen = freeze_label(agent)
        for agent_index, expected in enumerate(environment.agents):
            if expected == frozen:
                return float(mechanism.payments[row_index, agent_index])
        raise DomainMismatchError("unknown payment agent", context={"agent": frozen})

    def to_dict(self) -> dict[str, Any]:
        """Return a round-trippable canonical nested Python mapping."""

        self._require_bound()
        allocation = {
            profile: {
                outcome: float(self.allocation[row_index, outcome_index])
                for outcome_index, outcome in enumerate(self.outcomes)
            }
            for row_index, profile in enumerate(self.report_profiles)
        }
        payments = {
            profile: {
                agent: float(self.payments[row_index, agent_index])
                for agent_index, agent in enumerate(self.agents)
            }
            for row_index, profile in enumerate(self.report_profiles)
        }
        return {
            "allocation": allocation,
            "payments": payments,
            "payment_sign": self.PAYMENT_SIGN,
            "domain_fingerprint": self.domain_fingerprint,
            "fingerprint": self.fingerprint,
        }

    def to_pandas(self) -> MechanismTables:
        """Return labeled allocation and payment ``pandas.DataFrame`` objects."""

        self._require_bound()
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - pandas is a runtime dependency
            raise ValidationError("pandas is required for to_pandas()") from exc

        def object_index(values: Sequence[Any], *, name: Any) -> Any:
            raw = np.empty(len(values), dtype=object)
            raw[:] = list(values)
            return pd.Index(raw, name=name)

        if self.n_agents == 1:
            agent = self.agents[0]
            index = object_index(
                [profile[agent] for profile in self.report_profiles],
                name=agent,
            )
        else:
            index = pd.MultiIndex.from_tuples(
                [profile.values_tuple for profile in self.report_profiles],
                names=list(self.agents),
            )

        allocation = pd.DataFrame(
            np.array(self.allocation, copy=True),
            index=index.copy(),
            columns=object_index(self.outcomes, name="outcome"),
        )
        payments = pd.DataFrame(
            np.array(self.payments, copy=True),
            index=index.copy(),
            columns=object_index(self.agents, name="agent"),
        )
        metadata = {
            "domain_fingerprint": self.domain_fingerprint,
            "mechanism_fingerprint": self.fingerprint,
            "payment_sign": self.PAYMENT_SIGN,
        }
        allocation.attrs.update(metadata)
        payments.attrs.update(metadata)
        return MechanismTables(allocation=allocation, payments=payments)

    def __repr__(self) -> str:
        if not self.is_bound:
            expected = (
                ""
                if self.domain_fingerprint is None
                else f", fingerprint={self.domain_fingerprint!r}"
            )
            return f"DirectMechanism(unbound{expected})"
        return f"DirectMechanism(n_profiles={self.n_profiles}, n_outcomes={self.n_outcomes}, n_agents={self.n_agents})"


__all__ = ["DirectMechanism", "MechanismTables"]
