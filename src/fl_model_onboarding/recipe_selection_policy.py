"""Typed, versioned recipe selection policy for CPU INT4 quantization candidates.

This module is planning metadata only: it declares which quantization candidates
*may* be attempted for a recipe and under what declarative, allowlisted trigger a
non-default candidate becomes eligible. It does not execute recipes, does not
select models, and does not talk to Olive/Mobius/onnxruntime directly. Execution
wiring belongs to a later slice.
"""

from __future__ import annotations

import hashlib
import json

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .quality_validation import QualityRetryDisposition

# Single source of truth for the allowlisted retry trigger string: this is the exact
# value of `QualityRetryDisposition.RETRYABLE_OPTIMIZED_STRUCTURAL_REGRESSION`, imported
# rather than duplicated, so a future rename of either side cannot silently desynchronize
# the policy trigger from the disposition that is supposed to activate it. See also the
# cross-module equality assertion in `recipe_attempt_store.py`, which fails fast at import
# time if this invariant is ever violated despite the shared source.
RETRYABLE_OPTIMIZED_STRUCTURAL_REGRESSION_TRIGGER = (
    QualityRetryDisposition.RETRYABLE_OPTIMIZED_STRUCTURAL_REGRESSION.value
)

_ALLOWED_ELIGIBILITY_TRIGGERS = frozenset({RETRYABLE_OPTIMIZED_STRUCTURAL_REGRESSION_TRIGGER})
_ALLOWED_QUANTIZATION_OVERRIDE_KEYS = frozenset({"block_size"})
_ALLOWED_CANDIDATE_KEYS = frozenset(
    {
        "candidate_index",
        "candidate_id",
        "description",
        "quantization_override",
        "eligibility_trigger",
    }
)
_ALLOWED_POLICY_KEYS = frozenset(
    {
        "policy_id",
        "version",
        "target_device",
        "quantization",
        "max_candidates",
        "candidates",
    }
)
_ALLOWED_TOP_LEVEL_KEYS = frozenset({"schema_version", "policies"})


class RecipeSelectionTargetDevice(StrEnum):
    CPU = "cpu"


class RecipeSelectionQuantization(StrEnum):
    INT4 = "int4"


@dataclass(frozen=True)
class RecipeQuantizationOverride:
    block_size: int


@dataclass(frozen=True)
class RecipeSelectionCandidate:
    candidate_index: int
    candidate_id: str
    description: str
    quantization_override: RecipeQuantizationOverride | None
    eligibility_trigger: str | None

    @property
    def is_default(self) -> bool:
        return self.candidate_index == 0

    def is_eligible_for(self, trigger: str | None) -> bool:
        if self.eligibility_trigger is None:
            return True
        return trigger is not None and trigger == self.eligibility_trigger


@dataclass(frozen=True)
class RecipeSelectionPolicy:
    policy_id: str
    version: str
    target_device: RecipeSelectionTargetDevice
    quantization: RecipeSelectionQuantization
    max_candidates: int
    candidates: tuple[RecipeSelectionCandidate, ...]

    @property
    def fingerprint(self) -> str:
        payload = {
            "policy_id": self.policy_id,
            "version": self.version,
            "target_device": self.target_device.value,
            "quantization": self.quantization.value,
            "max_candidates": self.max_candidates,
            "candidates": [
                {
                    "candidate_index": candidate.candidate_index,
                    "candidate_id": candidate.candidate_id,
                    "description": candidate.description,
                    "quantization_override": (
                        {"block_size": candidate.quantization_override.block_size}
                        if candidate.quantization_override is not None
                        else None
                    ),
                    "eligibility_trigger": candidate.eligibility_trigger,
                }
                for candidate in self.candidates
            ],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def identity(self) -> str:
        return f"{self.policy_id}:{self.version}:{self.fingerprint}"

    def plan(self, *, trigger: str | None = None) -> tuple[RecipeSelectionCandidate, ...]:
        """Return the deterministic, ordered set of candidates eligible for planning.

        This is planning metadata only: it decides which candidates are eligible to
        be *attempted*, not whether any of them will actually run or succeed. The
        default candidate (index 0) is always eligible. Any other candidate is
        eligible only when ``trigger`` matches its declared ``eligibility_trigger``
        exactly.
        """
        return tuple(candidate for candidate in self.candidates if candidate.is_eligible_for(trigger))


class RecipeSelectionPolicyRegistry:
    def __init__(self, *, schema_version: str, policies: tuple[RecipeSelectionPolicy, ...]) -> None:
        if not policies:
            raise ValueError("Recipe selection policy registry is empty.")
        self.schema_version = schema_version
        self._policies = policies
        self._policies_by_id: dict[str, RecipeSelectionPolicy] = {}
        for policy in policies:
            if policy.policy_id in self._policies_by_id:
                raise ValueError(f"Duplicate recipe selection policy_id '{policy.policy_id}'.")
            self._policies_by_id[policy.policy_id] = policy

    def all(self) -> tuple[RecipeSelectionPolicy, ...]:
        return self._policies

    def get(self, policy_id: str) -> RecipeSelectionPolicy:
        if policy_id not in self._policies_by_id:
            raise ValueError(f"Unknown recipe selection policy_id '{policy_id}'.")
        return self._policies_by_id[policy_id]


def recipe_selection_policy_schema_path() -> Path:
    return Path(__file__).resolve().parents[2] / "contracts" / "recipe-selection-policy.schema.json"


def recipe_selection_policies_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "recipe-selection-policies.json"


def load_recipe_selection_policy_registry(
    data_path: Path | None = None,
    schema_path: Path | None = None,
) -> RecipeSelectionPolicyRegistry:
    effective_schema = schema_path or recipe_selection_policy_schema_path()
    effective_data = data_path or recipe_selection_policies_path()
    schema_raw = _load_json_file(effective_schema)
    payload = _load_json_file(effective_data)
    _validate_payload_against_schema_header(payload, schema_raw)
    policies = _parse_policies(payload)
    schema_version = _coerce_str(payload.get("schema_version"), "schema_version")
    return RecipeSelectionPolicyRegistry(schema_version=schema_version, policies=policies)


def _load_json_file(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file '{path}' must contain an object at top level.")
    return payload


def _validate_payload_against_schema_header(payload: dict[str, object], schema: dict[str, object]) -> None:
    required = _coerce_str_tuple(schema.get("required"), "schema.required")
    for key in required:
        if key not in payload:
            raise ValueError(f"Recipe selection policy data is missing required key '{key}'.")
    properties = _coerce_mapping(schema.get("properties"), "schema.properties")
    schema_version_prop = _coerce_mapping(
        properties.get("schema_version"),
        "schema.properties.schema_version",
    )
    expected = _coerce_optional_str(
        schema_version_prop.get("const"),
        "schema.properties.schema_version.const",
    )
    actual = _coerce_optional_str(payload.get("schema_version"), "schema_version")
    if expected and actual != expected:
        raise ValueError(
            "Recipe selection policy schema_version mismatch: "
            f"expected '{expected}', got '{actual}'."
        )


def _parse_policies(payload: dict[str, object]) -> tuple[RecipeSelectionPolicy, ...]:
    _reject_unknown_keys(payload, _ALLOWED_TOP_LEVEL_KEYS, "<root>")
    rows = _coerce_sequence(payload.get("policies"), "policies")
    if not rows:
        raise ValueError("Recipe selection policy data has no policies.")
    parsed = [_parse_policy(row, index) for index, row in enumerate(rows, start=1)]
    return tuple(parsed)


def _parse_policy(row: object, index: int) -> RecipeSelectionPolicy:
    path = f"policies[{index}]"
    value = _coerce_mapping(row, path)
    _reject_unknown_keys(value, _ALLOWED_POLICY_KEYS, path)

    target_device = RecipeSelectionTargetDevice(_coerce_str(value.get("target_device"), f"{path}.target_device"))
    quantization = RecipeSelectionQuantization(_coerce_str(value.get("quantization"), f"{path}.quantization"))
    max_candidates = _coerce_positive_int(value.get("max_candidates"), f"{path}.max_candidates")
    if max_candidates > 2:
        raise ValueError(f"{path}.max_candidates must not exceed 2 in this policy slice.")

    candidates_raw = _coerce_sequence(value.get("candidates"), f"{path}.candidates")
    if not candidates_raw:
        raise ValueError(f"{path}.candidates cannot be empty.")
    candidates = tuple(
        _parse_candidate(candidate_row, path=f"{path}.candidates[{candidate_index}]")
        for candidate_index, candidate_row in enumerate(candidates_raw, start=1)
    )

    if len(candidates) != max_candidates:
        raise ValueError(
            f"{path}.max_candidates ({max_candidates}) must equal the number of declared "
            f"candidates ({len(candidates)})."
        )

    expected_indices = tuple(range(len(candidates)))
    actual_indices = tuple(candidate.candidate_index for candidate in candidates)
    if actual_indices != expected_indices:
        raise ValueError(
            f"{path}.candidates must declare candidate_index values {expected_indices} in order; "
            f"got {actual_indices}."
        )

    default_candidate = candidates[0]
    if default_candidate.eligibility_trigger is not None:
        raise ValueError(f"{path}.candidates[0] (the default candidate) must not declare an eligibility_trigger.")
    if default_candidate.quantization_override is not None:
        raise ValueError(f"{path}.candidates[0] (the default candidate) must not declare a quantization_override.")

    for candidate in candidates[1:]:
        if candidate.eligibility_trigger is None:
            raise ValueError(
                f"{path}.candidates[{candidate.candidate_index}] must declare an eligibility_trigger; "
                "non-default candidates are never unconditionally eligible."
            )

    return RecipeSelectionPolicy(
        policy_id=_coerce_str(value.get("policy_id"), f"{path}.policy_id"),
        version=_coerce_str(value.get("version"), f"{path}.version"),
        target_device=target_device,
        quantization=quantization,
        max_candidates=max_candidates,
        candidates=candidates,
    )


def _parse_candidate(row: object, *, path: str) -> RecipeSelectionCandidate:
    value = _coerce_mapping(row, path)
    _reject_unknown_keys(value, _ALLOWED_CANDIDATE_KEYS, path)

    candidate_index = _coerce_non_negative_int(value.get("candidate_index"), f"{path}.candidate_index")
    if candidate_index > 1:
        raise ValueError(f"{path}.candidate_index must not exceed 1 in this policy slice.")

    eligibility_trigger = _coerce_optional_str(value.get("eligibility_trigger"), f"{path}.eligibility_trigger")
    if eligibility_trigger is not None and eligibility_trigger not in _ALLOWED_ELIGIBILITY_TRIGGERS:
        raise ValueError(
            f"{path}.eligibility_trigger must be one of {sorted(_ALLOWED_ELIGIBILITY_TRIGGERS)} or null; "
            f"got '{eligibility_trigger}'."
        )

    quantization_override = _parse_quantization_override(
        value.get("quantization_override"),
        path=f"{path}.quantization_override",
    )

    return RecipeSelectionCandidate(
        candidate_index=candidate_index,
        candidate_id=_coerce_str(value.get("candidate_id"), f"{path}.candidate_id"),
        description=_coerce_str(value.get("description"), f"{path}.description"),
        quantization_override=quantization_override,
        eligibility_trigger=eligibility_trigger,
    )


def _parse_quantization_override(value: object, *, path: str) -> RecipeQuantizationOverride | None:
    if value is None:
        return None
    mapping = _coerce_mapping(value, path)
    _reject_unknown_keys(mapping, _ALLOWED_QUANTIZATION_OVERRIDE_KEYS, path)
    if "block_size" not in mapping:
        raise ValueError(f"{path} must declare 'block_size' when not null.")
    block_size = _coerce_positive_int(mapping.get("block_size"), f"{path}.block_size")
    return RecipeQuantizationOverride(block_size=block_size)


def _reject_unknown_keys(value: dict[str, object], allowed: frozenset[str], path: str) -> None:
    unknown = sorted(set(value.keys()) - allowed)
    if unknown:
        raise ValueError(f"{path} has unknown field(s): {', '.join(unknown)}.")


def _coerce_mapping(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object.")
    return value


def _coerce_sequence(value: object, path: str) -> tuple[object, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be an array.")
    return tuple(value)


def _coerce_str(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string.")
    return value.strip()


def _coerce_optional_str(value: object, path: str) -> str | None:
    if value is None:
        return None
    return _coerce_str(value, path)


def _coerce_str_tuple(value: object, path: str) -> tuple[str, ...]:
    rows = _coerce_sequence(value, path)
    return tuple(_coerce_str(item, f"{path}[{index}]") for index, item in enumerate(rows, start=1))


def _coerce_positive_int(value: object, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{path} must be an integer.")
    if value <= 0:
        raise ValueError(f"{path} must be greater than zero.")
    return value


def _coerce_non_negative_int(value: object, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{path} must be an integer.")
    if value < 0:
        raise ValueError(f"{path} must be greater than or equal to zero.")
    return value


DEFAULT_RECIPE_SELECTION_POLICY_REGISTRY = load_recipe_selection_policy_registry()
DEFAULT_CPU_INT4_RECIPE_SELECTION_POLICY = DEFAULT_RECIPE_SELECTION_POLICY_REGISTRY.get(
    "cpu-int4-recipe-selection-v1"
)


__all__ = [
    "DEFAULT_CPU_INT4_RECIPE_SELECTION_POLICY",
    "DEFAULT_RECIPE_SELECTION_POLICY_REGISTRY",
    "RETRYABLE_OPTIMIZED_STRUCTURAL_REGRESSION_TRIGGER",
    "RecipeQuantizationOverride",
    "RecipeSelectionCandidate",
    "RecipeSelectionPolicy",
    "RecipeSelectionPolicyRegistry",
    "RecipeSelectionQuantization",
    "RecipeSelectionTargetDevice",
    "load_recipe_selection_policy_registry",
    "recipe_selection_policies_path",
    "recipe_selection_policy_schema_path",
]
