"""Public outside-option table contracts."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from mechaudit import OwnTypeOutsideOptions, ProfileOutsideOptions, TypeProfile
from mechaudit.exceptions import RequirementValidationError


class _RepeatedKeys(Mapping[Any, Any]):
    """Adversarial mapping that exposes equality-colliding input records."""

    def __init__(self, keys: tuple[Any, ...], value: Any) -> None:
        self._keys = keys
        self._value = value

    def __getitem__(self, key: Any) -> Any:
        return self._value

    def __iter__(self) -> Iterator[Any]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)


def _profile(kind: str = "low") -> TypeProfile:
    return TypeProfile((("buyer", kind),))


def test_own_type_table_lookup_and_missing_entry_context() -> None:
    options = OwnTypeOutsideOptions({"buyer": {"low": 0.25, "high": 1.5}})

    assert options.value("buyer", _profile("low")) == pytest.approx(0.25)
    assert options.value("buyer", _profile("high")) == pytest.approx(1.5)

    with pytest.raises(RequirementValidationError, match="missing an agent/type") as missing_type:
        options.value("buyer", _profile("other"))
    assert missing_type.value.context == {"agent": "buyer", "type": "other"}

    with pytest.raises(RequirementValidationError, match="missing an agent/type") as missing_agent:
        options.value("seller", TypeProfile((("seller", "low"),)))
    assert missing_agent.value.context == {"agent": "seller", "type": "low"}


def test_own_type_table_rejects_duplicate_and_nonfinite_records() -> None:
    duplicate_agents = _RepeatedKeys(("buyer", "buyer"), {"low": 0.0})
    duplicate_types = _RepeatedKeys(("low", "low"), 0.0)

    with pytest.raises(RequirementValidationError, match="duplicate agents"):
        OwnTypeOutsideOptions(duplicate_agents)
    with pytest.raises(RequirementValidationError, match="duplicate types"):
        OwnTypeOutsideOptions({"buyer": duplicate_types})
    with pytest.raises(RequirementValidationError, match="finite"):
        OwnTypeOutsideOptions({"buyer": {"low": float("inf")}})
    with pytest.raises(RequirementValidationError, match="real numbers"):
        OwnTypeOutsideOptions({"buyer": {"low": "free"}})  # type: ignore[dict-item]


def test_profile_table_lookup_and_key_validation() -> None:
    low = _profile("low")
    high = _profile("high")
    options = ProfileOutsideOptions({"buyer": {low: 0.5, high: 2.0}})

    assert options.value("buyer", low) == pytest.approx(0.5)
    assert options.value("buyer", high) == pytest.approx(2.0)
    with pytest.raises(RequirementValidationError, match="missing an agent/profile") as missing:
        options.value("buyer", _profile("other"))
    assert missing.value.context == {"agent": "buyer", "profile": _profile("other")}

    with pytest.raises(RequirementValidationError, match="TypeProfile"):
        ProfileOutsideOptions({"buyer": {("low",): 0.0}})  # type: ignore[dict-item]


def test_profile_table_rejects_duplicate_agents_and_profiles() -> None:
    profile = _profile()
    duplicate_agents = _RepeatedKeys(("buyer", "buyer"), {profile: 0.0})
    duplicate_profiles = _RepeatedKeys((profile, profile), 0.0)

    with pytest.raises(RequirementValidationError, match="duplicate agents"):
        ProfileOutsideOptions(duplicate_agents)
    with pytest.raises(RequirementValidationError, match="duplicate profiles"):
        ProfileOutsideOptions({"buyer": duplicate_profiles})
