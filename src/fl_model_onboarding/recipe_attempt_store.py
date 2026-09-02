from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from .quality_validation import QualityRetryDisposition, QualityRetryEvaluation
from .recipe_compiler import (
    GeneratedRecipe,
    PromotionGateCheck,
    PromotionGateEvidence,
    RecipeCompilerToolchain,
    RecipeGenerationProvenance,
    RecipeInputMetadata,
)
from .recipe_selection_policy import (
    RETRYABLE_OPTIMIZED_STRUCTURAL_REGRESSION_TRIGGER,
    RecipeQuantizationOverride,
    RecipeSelectionPolicy,
)
from .recipes import ModelRecipe, RecipeStatus

# Cross-module boundary enforcement (Slice 2 reviewer follow-up): the policy trigger
# string and the QualityRetryDisposition value it must match are defined in two
# different modules for layering reasons (policy metadata vs. quality-gate evidence
# derivation). `recipe_selection_policy` already imports its constant directly from
# `QualityRetryDisposition`, so this can only fail if that import is ever removed or a
# constant is renamed inconsistently. Assert here too, at recipe-attempt-store import
# time, so any future drift fails fast and loudly instead of silently disabling the
# fallback candidate. See also `tests/test_recipe_attempt_store.py::
# test_retry_trigger_constant_matches_quality_retry_disposition_value` for an explicit
# regression test of this exact invariant.
assert (
    RETRYABLE_OPTIMIZED_STRUCTURAL_REGRESSION_TRIGGER
    == QualityRetryDisposition.RETRYABLE_OPTIMIZED_STRUCTURAL_REGRESSION.value
), (
    "recipe_selection_policy.RETRYABLE_OPTIMIZED_STRUCTURAL_REGRESSION_TRIGGER has "
    "diverged from QualityRetryDisposition.RETRYABLE_OPTIMIZED_STRUCTURAL_REGRESSION; "
    "the fallback candidate trigger would silently stop matching quality-retry disposition."
)

RECIPE_ATTEMPT_SCHEMA_VERSION = "1.0.0"
RECIPE_ATTEMPT_STORE_SCHEMA_VERSION = 3
LEGACY_PROFILE_FINGERPRINT = hashlib.sha256(b"legacy-profile-missing-v1").hexdigest()

_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_SECRET_INLINE_RE = re.compile(
    r"(?i)\b(api[-_ ]?key|access[-_ ]?token|refresh[-_ ]?token|authorization|secret)\s*[:=]\s*([^\s,;]+)"
)
_HF_TOKEN_RE = re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")
_BEARER_TOKEN_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{10,}\b")
_WINDOWS_ABS_RE = re.compile(r"(?i)^[a-z]:[\\/]")

_MAX_REFERENCE_LENGTH = 512
_MAX_MESSAGE_LENGTH = 2048
_MAX_FAILURE_EVIDENCE_REFS = 16


class RecipeAttemptStoreError(RuntimeError):
    """Base class for recipe-attempt store failures."""


class RecipeAttemptSchemaError(RecipeAttemptStoreError):
    """Raised when payloads fail schema or typed validation."""


class RecipeAttemptMigrationError(RecipeAttemptStoreError):
    """Raised when store schema migration cannot safely complete."""


class AttemptIdempotencyConflictError(RecipeAttemptStoreError):
    """Raised when an idempotency key is reused with a different request fingerprint."""


class AttemptStateTransitionError(RecipeAttemptStoreError):
    """Raised when attempt state transitions violate the contract."""


class AttemptGateSequenceError(RecipeAttemptStoreError):
    """Raised when gate order or per-attempt sequence monotonicity is violated."""


class RecipePromotionConflictError(RecipeAttemptStoreError):
    """Raised when verified promotion requirements are not fully satisfied."""


class RecipeAttemptSecurityError(RecipeAttemptStoreError):
    """Raised when evidence or failure payloads contain unsafe content."""


class CandidatePlanValidationError(RecipeAttemptStoreError):
    """Raised when candidate-plan registration is malformed, duplicated, out of
    order, exceeds the policy's max_candidates, or conflicts with an already
    registered policy/quality-profile identity for the same parent lineage."""


class CandidateSelectionConflictError(RecipeAttemptStoreError):
    """Raised when selecting or exhausting a candidate lineage would violate the
    single-winner, verified-only selection contract (for example: selecting a
    non-verified or unknown child, selecting a second winner, or exhausting a
    lineage that still has an unselected verified candidate)."""


class AttemptState(StrEnum):
    GENERATED = "generated"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_ATTEMPT_STATES = frozenset(
    {
        AttemptState.SUCCEEDED,
        AttemptState.FAILED,
        AttemptState.CANCELLED,
    }
)


class AttemptGate(StrEnum):
    MOBIUS_BUILD = "mobius_build"
    OLIVE_OPTIMIZE = "olive_optimize"
    ONNX_VALIDATION = "onnx_validation"
    ORT_VALIDATION = "ort_validation"
    OGA_VALIDATION = "oga_validation"
    FL_SDK_INFERENCE = "fl_sdk_inference"
    QUALITY_VALIDATION = "quality_validation"


ATTEMPT_GATE_ORDER: tuple[AttemptGate, ...] = (
    AttemptGate.MOBIUS_BUILD,
    AttemptGate.OLIVE_OPTIMIZE,
    AttemptGate.ONNX_VALIDATION,
    AttemptGate.ORT_VALIDATION,
    AttemptGate.OGA_VALIDATION,
    AttemptGate.FL_SDK_INFERENCE,
    AttemptGate.QUALITY_VALIDATION,
)


class AttemptGateStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_RUN = "not_run"
    UNAVAILABLE = "unavailable"


class AttemptFailureClassification(StrEnum):
    GATE_FAILED = "gate_failed"
    VALIDATION_FAILED = "validation_failed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"
    INTERNAL_ERROR = "internal_error"


class CandidateLineageSelectionState(StrEnum):
    """Lifecycle of a parent attempt's candidate-selection lineage.

    ``PENDING`` means at least one candidate slot is registered and no winner has
    been chosen yet. ``SELECTED`` and ``EXHAUSTED`` are terminal: exactly one of
    them applies once a lineage is finalized, and a finalized lineage never
    accepts new candidates or a different winner.
    """

    PENDING = "pending"
    SELECTED = "selected"
    EXHAUSTED = "exhausted"


TERMINAL_CANDIDATE_LINEAGE_SELECTION_STATES = frozenset(
    {
        CandidateLineageSelectionState.SELECTED,
        CandidateLineageSelectionState.EXHAUSTED,
    }
)


class CandidateWinnerStatus(StrEnum):
    """Per-candidate selection flag. At most one candidate per lineage may be
    ``SELECTED``, and only a candidate whose linked attempt has reached
    ``AttemptState.SUCCEEDED`` (i.e. it passed every required gate including
    quality validation) may ever transition to ``SELECTED``."""

    NOT_SELECTED = "not_selected"
    SELECTED = "selected"


@dataclass(frozen=True)
class CandidateInvocationCounters:
    """Nullable, typed placeholders for real Mobius/Olive invocation counts and
    cost/performance metrics.

    Every field defaults to ``None`` and stays ``None`` until a later slice
    actually instruments a real recipe run and supplies a measurement. ``None``
    must never be coerced to or serialized as ``0``: it means "not yet
    instrumented", never "zero invocations occurred".
    """

    mobius_build_invocation_count: int | None = None
    olive_optimize_invocation_count: int | None = None
    total_invocation_count: int | None = None
    wall_clock_seconds: float | None = None
    estimated_cost_usd: float | None = None


_NULL_CANDIDATE_INVOCATION_COUNTERS = CandidateInvocationCounters()


@dataclass(frozen=True)
class CandidateAttemptRecord:
    """A durable, immutable-once-terminal child candidate under a recipe-selection
    lineage. Each row reuses an existing :class:`RecipeAttempt` (via ``attempt_id``)
    for its own gate/state execution record instead of duplicating that state
    machine; this record only carries candidate-plan and selection metadata.
    """

    candidate_attempt_id: str
    parent_attempt_id: str
    attempt_id: str
    candidate_index: int
    candidate_id: str
    policy_id: str
    policy_version: str
    policy_fingerprint: str
    quality_profile_fingerprint: str
    quantization_override_block_size: int | None
    eligibility_trigger: str | None
    disposition: str | None
    disposition_reasons: tuple[str, ...]
    recipe_fingerprint: str
    candidate_fingerprint: str
    created_utc: datetime
    attempt_state: AttemptState
    artifact_ref: str | None
    package_ref: str | None
    invocation_counters: CandidateInvocationCounters
    selection_status: CandidateWinnerStatus
    selected_by: str | None
    selection_reason: str | None
    selected_utc: datetime | None
    validated_target_device: str | None
    validated_target_ep: str | None
    validated_toolchain_fingerprint: str | None
    validated_environment_scope: str | None

    @property
    def is_default(self) -> bool:
        return self.candidate_index == 0

    @property
    def is_verified(self) -> bool:
        """Whether this candidate's linked attempt reached the terminal
        succeeded state (the closest existing analog to "verified": it passed
        every required gate including quality validation)."""
        return self.attempt_state == AttemptState.SUCCEEDED

    @property
    def has_fully_validated_selection_scope(self) -> bool:
        """True only when this candidate is selected *and* every validated-scope
        field is present. Absence of any single validated-scope field must never
        be treated as an implicit match for that scope: callers must not assume
        "verified for all scopes" merely because a selection provenance field is
        missing/null (for example, before Slice 3 supplies real validated
        target/EP/toolchain/environment data)."""
        return (
            self.selection_status == CandidateWinnerStatus.SELECTED
            and self.validated_target_device is not None
            and self.validated_target_ep is not None
            and self.validated_toolchain_fingerprint is not None
            and self.validated_environment_scope is not None
        )


@dataclass(frozen=True)
class RecipeCandidateLineage:
    """The parent-level selection record for one recipe-attempt's candidate plan."""

    parent_attempt_id: str
    policy_id: str
    policy_version: str
    policy_fingerprint: str
    policy_max_candidates: int
    quality_profile_fingerprint: str
    created_utc: datetime
    selection_state: CandidateLineageSelectionState
    selected_candidate_attempt_id: str | None
    selection_reason: str | None
    finalized_utc: datetime | None


@dataclass(frozen=True)
class CandidateSelectionReuseQuery:
    """Complete policy-aware reuse identity for a selected candidate winner.

    Extends the existing generated-recipe identity fields (model/revision/device/
    precision/compiler/capability/toolchain/profile) with the two additional
    identities Slice 2 introduces: the quality-validation profile fingerprint
    (so a quality-profile version bump invalidates reuse) and the selection
    policy fingerprint (so a policy id/version/candidate-set change invalidates
    reuse). This is intentionally a separate type from the existing
    :class:`RecipeReuseQuery` so Slice 1 callers (`local_service.py`) are
    unaffected.
    """

    model_id: str
    revision_sha: str
    requested_device: str
    requested_precision: str
    compiler_version: str
    capability_fingerprint: str
    toolchain_fingerprint: str
    profile_fingerprint: str
    quality_profile_fingerprint: str
    policy_fingerprint: str


@dataclass(frozen=True)
class GeneratedRecipeRecord:
    recipe_fingerprint: str
    schema_version: str
    recipe_status: RecipeStatus
    model_id: str
    revision_sha: str
    requested_device: str
    requested_precision: str
    compiler_version: str
    capability_fingerprint: str
    toolchain_fingerprint: str
    profile_fingerprint: str
    canonical_json: str
    created_utc: datetime

    def payload(self) -> dict[str, object]:
        parsed = json.loads(self.canonical_json)
        if not isinstance(parsed, dict):
            raise RecipeAttemptSchemaError("Generated recipe canonical_json must be a JSON object.")
        return parsed


@dataclass(frozen=True)
class AttemptGateResult:
    sequence: int
    gate: AttemptGate
    status: AttemptGateStatus
    evidence_ref: str
    metrics_ref: str | None
    started_utc: datetime
    finished_utc: datetime


@dataclass(frozen=True)
class AttemptFailure:
    classification: AttemptFailureClassification
    stage: str
    message: str
    evidence_refs: tuple[str, ...]
    source_owner: str
    next_action: str


@dataclass(frozen=True)
class RecipeAttempt:
    attempt_id: str
    idempotency_key: str
    request_fingerprint: str
    recipe_fingerprint: str
    model_id: str
    revision_sha: str
    requested_device: str
    requested_precision: str
    compiler_version: str
    capability_fingerprint: str
    toolchain_fingerprint: str
    profile_fingerprint: str
    created_utc: datetime
    finished_utc: datetime | None
    state: AttemptState
    gate_results: tuple[AttemptGateResult, ...]
    failure: AttemptFailure | None


@dataclass(frozen=True)
class VerifiedRecipeRecord:
    verified_fingerprint: str
    source_recipe_fingerprint: str
    attempt_id: str
    schema_version: str
    model_id: str
    revision_sha: str
    requested_device: str
    requested_precision: str
    compiler_version: str
    capability_fingerprint: str
    toolchain_fingerprint: str
    profile_fingerprint: str
    promoted_utc: datetime
    promotion_evidence: PromotionGateEvidence
    canonical_json: str

    def payload(self) -> dict[str, object]:
        parsed = json.loads(self.canonical_json)
        if not isinstance(parsed, dict):
            raise RecipeAttemptSchemaError("Verified recipe canonical_json must be a JSON object.")
        return parsed


@dataclass(frozen=True)
class RecipeAttemptRequest:
    recipe_fingerprint: str
    model_id: str
    revision_sha: str
    requested_device: str
    requested_precision: str
    compiler_version: str
    capability_fingerprint: str
    toolchain_fingerprint: str
    profile_fingerprint: str


@dataclass(frozen=True)
class RecipeReuseQuery:
    model_id: str
    revision_sha: str
    requested_device: str
    requested_precision: str
    compiler_version: str
    capability_fingerprint: str
    toolchain_fingerprint: str
    profile_fingerprint: str


def recipe_attempt_schema_path() -> Path:
    return Path(__file__).resolve().parents[2] / "contracts" / "recipe-attempt.schema.json"


def load_recipe_attempt_schema(schema_path: Path | None = None) -> dict[str, object]:
    path = schema_path or recipe_attempt_schema_path()
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise RecipeAttemptSchemaError(f"Schema at '{path}' must be a JSON object.")
    return parsed


def build_toolchain_fingerprint(toolchain: RecipeCompilerToolchain) -> str:
    payload = {
        "mobius_version": _coerce_str(toolchain.mobius_version, "toolchain.mobius_version"),
        "olive_version": _coerce_str(toolchain.olive_version, "toolchain.olive_version"),
        "onnx_version": _coerce_str(toolchain.onnx_version, "toolchain.onnx_version"),
        "ort_version": _coerce_str(toolchain.ort_version, "toolchain.ort_version"),
        "oga_version": _coerce_str(toolchain.oga_version, "toolchain.oga_version"),
        "foundry_sdk_version": _coerce_str(toolchain.foundry_sdk_version, "toolchain.foundry_sdk_version"),
        "foundry_cli_version": _coerce_str(toolchain.foundry_cli_version, "toolchain.foundry_cli_version"),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def build_capability_fingerprint(provenance: RecipeGenerationProvenance) -> str:
    payload = {
        "compiler_version": _coerce_str(provenance.compiler_version, "provenance.compiler_version"),
        "generation_kind": _coerce_str(provenance.generation_kind, "provenance.generation_kind"),
        "capability_id": _coerce_str(provenance.capability_id, "provenance.capability_id"),
        "capability_version": _coerce_str(provenance.capability_version, "provenance.capability_version"),
        "capability_status": provenance.capability_status.value,
        "resolution_outcome": provenance.resolution_outcome.value,
        "resolution_reason_code": _coerce_str(
            provenance.resolution_reason_code,
            "provenance.resolution_reason_code",
        ),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def build_profile_fingerprint(recipe: ModelRecipe, input_metadata: RecipeInputMetadata) -> str:
    olive_payload: dict[str, object] | None
    if recipe.olive is None:
        olive_payload = None
    else:
        olive_payload = {
            "task": recipe.olive.task,
            "precision": recipe.olive.precision,
            "device": recipe.olive.device,
            "provider": recipe.olive.provider,
            "log_level": recipe.olive.log_level,
        }
    payload: dict[str, object] = {
        "recipe_id": _coerce_str(recipe.id, "recipe.id"),
        "task_profile": _coerce_str(recipe.task_profile, "recipe.task_profile"),
        "requested_task": _coerce_str(input_metadata.task, "input_metadata.task"),
        "requested_device": _coerce_str(input_metadata.requested_device, "input_metadata.requested_device").lower(),
        "requested_precision": _coerce_str(
            input_metadata.requested_precision,
            "input_metadata.requested_precision",
        ).lower(),
        "mobius": {
            "task": recipe.mobius.task,
            "ep": recipe.mobius.ep,
            "runtime": recipe.mobius.runtime,
            "dtype": recipe.mobius.dtype,
        },
        "olive": olive_payload,
        "optimization_choices": [
            {
                "strategy": row.strategy,
                "precision": row.precision,
                "task_profile": row.task_profile,
                "skip_olive": row.skip_olive,
                "default": row.default,
            }
            for row in recipe.optimization_choices
        ],
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def build_candidate_recipe_fingerprint(
    *,
    recipe_fingerprint: str,
    quantization_override: RecipeQuantizationOverride | None,
    policy_fingerprint: str,
) -> str:
    """Deterministic identity for one candidate's actual resolved recipe.

    Includes only the actual fully-resolved recipe fingerprint, the actual
    quantization override applied (or ``None`` for the default candidate), and
    the selection-policy identity fingerprint. Deliberately excludes
    timestamps, filesystem paths, and candidate-attempt UUIDs so the same
    candidate always produces the same fingerprint regardless of when or where
    it was registered.
    """
    normalized_recipe_fingerprint = _normalize_hex(
        recipe_fingerprint,
        expected_len=64,
        path="candidate.recipe_fingerprint",
    )
    normalized_policy_fingerprint = _normalize_hex(
        policy_fingerprint,
        expected_len=64,
        path="candidate.policy_fingerprint",
    )
    payload: dict[str, object] = {
        "recipe_fingerprint": normalized_recipe_fingerprint,
        "quantization_override": (
            {"block_size": quantization_override.block_size}
            if quantization_override is not None
            else None
        ),
        "policy_fingerprint": normalized_policy_fingerprint,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def build_attempt_request_from_generated(record: GeneratedRecipeRecord) -> RecipeAttemptRequest:
    return RecipeAttemptRequest(
        recipe_fingerprint=record.recipe_fingerprint,
        model_id=record.model_id,
        revision_sha=record.revision_sha,
        requested_device=record.requested_device,
        requested_precision=record.requested_precision,
        compiler_version=record.compiler_version,
        capability_fingerprint=record.capability_fingerprint,
        toolchain_fingerprint=record.toolchain_fingerprint,
        profile_fingerprint=record.profile_fingerprint,
    )


def build_reuse_query_from_generated(record: GeneratedRecipeRecord) -> RecipeReuseQuery:
    return RecipeReuseQuery(
        model_id=record.model_id,
        revision_sha=record.revision_sha,
        requested_device=record.requested_device,
        requested_precision=record.requested_precision,
        compiler_version=record.compiler_version,
        capability_fingerprint=record.capability_fingerprint,
        toolchain_fingerprint=record.toolchain_fingerprint,
        profile_fingerprint=record.profile_fingerprint,
    )


def build_attempt_request_fingerprint(request: RecipeAttemptRequest) -> str:
    normalized = _normalize_attempt_request(request)
    payload = {
        "recipe_fingerprint": normalized.recipe_fingerprint,
        "model_id": normalized.model_id,
        "revision_sha": normalized.revision_sha,
        "requested_device": normalized.requested_device,
        "requested_precision": normalized.requested_precision,
        "compiler_version": normalized.compiler_version,
        "capability_fingerprint": normalized.capability_fingerprint,
        "toolchain_fingerprint": normalized.toolchain_fingerprint,
        "profile_fingerprint": normalized.profile_fingerprint,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def serialize_generated_recipe_record(record: GeneratedRecipeRecord) -> dict[str, object]:
    return {
        "record_type": "generated_recipe_record",
        "recipe_fingerprint": record.recipe_fingerprint,
        "schema_version": record.schema_version,
        "recipe_status": record.recipe_status.value,
        "model_id": record.model_id,
        "revision_sha": record.revision_sha,
        "requested_device": record.requested_device,
        "requested_precision": record.requested_precision,
        "compiler_version": record.compiler_version,
        "capability_fingerprint": record.capability_fingerprint,
        "toolchain_fingerprint": record.toolchain_fingerprint,
        "profile_fingerprint": record.profile_fingerprint,
        "canonical_json": record.canonical_json,
        "created_utc": _format_datetime(record.created_utc),
    }


def serialize_attempt_gate_result(gate: AttemptGateResult) -> dict[str, object]:
    return {
        "sequence": gate.sequence,
        "gate": gate.gate.value,
        "status": gate.status.value,
        "evidence_ref": gate.evidence_ref,
        "metrics_ref": gate.metrics_ref,
        "started_utc": _format_datetime(gate.started_utc),
        "finished_utc": _format_datetime(gate.finished_utc),
    }


def serialize_attempt_failure(failure: AttemptFailure) -> dict[str, object]:
    return {
        "classification": failure.classification.value,
        "stage": failure.stage,
        "message": failure.message,
        "evidence_refs": list(failure.evidence_refs),
        "source_owner": failure.source_owner,
        "next_action": failure.next_action,
    }


def serialize_recipe_attempt(attempt: RecipeAttempt) -> dict[str, object]:
    return {
        "record_type": "recipe_attempt",
        "attempt_id": attempt.attempt_id,
        "idempotency_key": attempt.idempotency_key,
        "request_fingerprint": attempt.request_fingerprint,
        "recipe_fingerprint": attempt.recipe_fingerprint,
        "model_id": attempt.model_id,
        "revision_sha": attempt.revision_sha,
        "requested_device": attempt.requested_device,
        "requested_precision": attempt.requested_precision,
        "compiler_version": attempt.compiler_version,
        "capability_fingerprint": attempt.capability_fingerprint,
        "toolchain_fingerprint": attempt.toolchain_fingerprint,
        "profile_fingerprint": attempt.profile_fingerprint,
        "created_utc": _format_datetime(attempt.created_utc),
        "finished_utc": _format_datetime(attempt.finished_utc) if attempt.finished_utc is not None else None,
        "state": attempt.state.value,
        "gate_results": [serialize_attempt_gate_result(row) for row in attempt.gate_results],
        "failure": serialize_attempt_failure(attempt.failure) if attempt.failure is not None else None,
    }


def serialize_verified_recipe_record(record: VerifiedRecipeRecord) -> dict[str, object]:
    return {
        "record_type": "verified_recipe_record",
        "verified_fingerprint": record.verified_fingerprint,
        "source_recipe_fingerprint": record.source_recipe_fingerprint,
        "attempt_id": record.attempt_id,
        "schema_version": record.schema_version,
        "model_id": record.model_id,
        "revision_sha": record.revision_sha,
        "requested_device": record.requested_device,
        "requested_precision": record.requested_precision,
        "compiler_version": record.compiler_version,
        "capability_fingerprint": record.capability_fingerprint,
        "toolchain_fingerprint": record.toolchain_fingerprint,
        "profile_fingerprint": record.profile_fingerprint,
        "promoted_utc": _format_datetime(record.promoted_utc),
        "promotion_evidence": _promotion_evidence_to_payload(record.promotion_evidence),
        "canonical_json": record.canonical_json,
    }


def deserialize_generated_recipe_record(
    payload: dict[str, object],
    *,
    schema_root: dict[str, object] | None = None,
) -> GeneratedRecipeRecord:
    _validate_record_payload(payload, record_def="generated_recipe_record", schema_root=schema_root)
    record = GeneratedRecipeRecord(
        recipe_fingerprint=_normalize_hex(
            _coerce_str(payload.get("recipe_fingerprint"), "generated_recipe_record.recipe_fingerprint"),
            expected_len=64,
            path="generated_recipe_record.recipe_fingerprint",
        ),
        schema_version=_coerce_str(payload.get("schema_version"), "generated_recipe_record.schema_version"),
        recipe_status=RecipeStatus(
            _coerce_str(payload.get("recipe_status"), "generated_recipe_record.recipe_status")
        ),
        model_id=_coerce_str(payload.get("model_id"), "generated_recipe_record.model_id"),
        revision_sha=_normalize_hex(
            _coerce_str(payload.get("revision_sha"), "generated_recipe_record.revision_sha"),
            expected_len=40,
            path="generated_recipe_record.revision_sha",
        ),
        requested_device=_coerce_str(
            payload.get("requested_device"),
            "generated_recipe_record.requested_device",
        ).lower(),
        requested_precision=_coerce_str(
            payload.get("requested_precision"),
            "generated_recipe_record.requested_precision",
        ).lower(),
        compiler_version=_coerce_str(
            payload.get("compiler_version"),
            "generated_recipe_record.compiler_version",
        ),
        capability_fingerprint=_normalize_hex(
            _coerce_str(
                payload.get("capability_fingerprint"),
                "generated_recipe_record.capability_fingerprint",
            ),
            expected_len=64,
            path="generated_recipe_record.capability_fingerprint",
        ),
        toolchain_fingerprint=_normalize_hex(
            _coerce_str(
                payload.get("toolchain_fingerprint"),
                "generated_recipe_record.toolchain_fingerprint",
            ),
            expected_len=64,
            path="generated_recipe_record.toolchain_fingerprint",
        ),
        profile_fingerprint=_normalize_hex(
            _coerce_str(
                payload.get("profile_fingerprint"),
                "generated_recipe_record.profile_fingerprint",
            ),
            expected_len=64,
            path="generated_recipe_record.profile_fingerprint",
        ),
        canonical_json=_coerce_str(payload.get("canonical_json"), "generated_recipe_record.canonical_json"),
        created_utc=_parse_datetime(
            _coerce_str(payload.get("created_utc"), "generated_recipe_record.created_utc"),
            "generated_recipe_record.created_utc",
        ),
    )
    _validate_canonical_recipe_payload(
        canonical_json=record.canonical_json,
        expected_fingerprint=record.recipe_fingerprint,
        path="generated_recipe_record.canonical_json",
    )
    return record


def deserialize_attempt_gate_result(payload: dict[str, object]) -> AttemptGateResult:
    sequence = _coerce_int(payload.get("sequence"), "attempt_gate_result.sequence")
    if sequence <= 0:
        raise RecipeAttemptSchemaError("attempt_gate_result.sequence must be > 0.")
    started = _parse_datetime(
        _coerce_str(payload.get("started_utc"), "attempt_gate_result.started_utc"),
        "attempt_gate_result.started_utc",
    )
    finished = _parse_datetime(
        _coerce_str(payload.get("finished_utc"), "attempt_gate_result.finished_utc"),
        "attempt_gate_result.finished_utc",
    )
    if finished < started:
        raise RecipeAttemptSchemaError("attempt_gate_result.finished_utc cannot be earlier than started_utc.")
    return AttemptGateResult(
        sequence=sequence,
        gate=AttemptGate(_coerce_str(payload.get("gate"), "attempt_gate_result.gate")),
        status=AttemptGateStatus(_coerce_str(payload.get("status"), "attempt_gate_result.status")),
        evidence_ref=_sanitize_reference(
            _coerce_str(payload.get("evidence_ref"), "attempt_gate_result.evidence_ref"),
            path="attempt_gate_result.evidence_ref",
        ),
        metrics_ref=_sanitize_optional_reference(
            payload.get("metrics_ref"),
            path="attempt_gate_result.metrics_ref",
        ),
        started_utc=started,
        finished_utc=finished,
    )


def deserialize_attempt_failure(payload: dict[str, object]) -> AttemptFailure:
    evidence_refs_raw = _coerce_schema_array(payload.get("evidence_refs"), "attempt_failure.evidence_refs")
    if len(evidence_refs_raw) > _MAX_FAILURE_EVIDENCE_REFS:
        raise RecipeAttemptSecurityError(
            f"attempt_failure.evidence_refs cannot exceed {_MAX_FAILURE_EVIDENCE_REFS} entries."
        )
    evidence_refs = tuple(
        _sanitize_reference(
            _coerce_str(value, f"attempt_failure.evidence_refs[{idx}]"),
            path=f"attempt_failure.evidence_refs[{idx}]",
        )
        for idx, value in enumerate(evidence_refs_raw, start=1)
    )
    return AttemptFailure(
        classification=AttemptFailureClassification(
            _coerce_str(payload.get("classification"), "attempt_failure.classification")
        ),
        stage=_coerce_str(payload.get("stage"), "attempt_failure.stage"),
        message=_sanitize_message(
            _coerce_str(payload.get("message"), "attempt_failure.message"),
            path="attempt_failure.message",
        ),
        evidence_refs=evidence_refs,
        source_owner=_coerce_str(payload.get("source_owner"), "attempt_failure.source_owner"),
        next_action=_coerce_str(payload.get("next_action"), "attempt_failure.next_action"),
    )


def deserialize_recipe_attempt(
    payload: dict[str, object],
    *,
    schema_root: dict[str, object] | None = None,
) -> RecipeAttempt:
    _validate_record_payload(payload, record_def="recipe_attempt", schema_root=schema_root)
    gate_rows = _coerce_schema_array(payload.get("gate_results"), "recipe_attempt.gate_results")
    gates = tuple(
        deserialize_attempt_gate_result(
            _coerce_schema_object(row, f"recipe_attempt.gate_results[{idx}]"),
        )
        for idx, row in enumerate(gate_rows, start=1)
    )
    failure_raw = payload.get("failure")
    failure: AttemptFailure | None
    if failure_raw is None:
        failure = None
    else:
        failure = deserialize_attempt_failure(
            _coerce_schema_object(failure_raw, "recipe_attempt.failure")
        )
    attempt = RecipeAttempt(
        attempt_id=_coerce_str(payload.get("attempt_id"), "recipe_attempt.attempt_id"),
        idempotency_key=_coerce_str(payload.get("idempotency_key"), "recipe_attempt.idempotency_key"),
        request_fingerprint=_normalize_hex(
            _coerce_str(payload.get("request_fingerprint"), "recipe_attempt.request_fingerprint"),
            expected_len=64,
            path="recipe_attempt.request_fingerprint",
        ),
        recipe_fingerprint=_normalize_hex(
            _coerce_str(payload.get("recipe_fingerprint"), "recipe_attempt.recipe_fingerprint"),
            expected_len=64,
            path="recipe_attempt.recipe_fingerprint",
        ),
        model_id=_coerce_str(payload.get("model_id"), "recipe_attempt.model_id"),
        revision_sha=_normalize_hex(
            _coerce_str(payload.get("revision_sha"), "recipe_attempt.revision_sha"),
            expected_len=40,
            path="recipe_attempt.revision_sha",
        ),
        requested_device=_coerce_str(payload.get("requested_device"), "recipe_attempt.requested_device").lower(),
        requested_precision=_coerce_str(
            payload.get("requested_precision"),
            "recipe_attempt.requested_precision",
        ).lower(),
        compiler_version=_coerce_str(payload.get("compiler_version"), "recipe_attempt.compiler_version"),
        capability_fingerprint=_normalize_hex(
            _coerce_str(payload.get("capability_fingerprint"), "recipe_attempt.capability_fingerprint"),
            expected_len=64,
            path="recipe_attempt.capability_fingerprint",
        ),
        toolchain_fingerprint=_normalize_hex(
            _coerce_str(payload.get("toolchain_fingerprint"), "recipe_attempt.toolchain_fingerprint"),
            expected_len=64,
            path="recipe_attempt.toolchain_fingerprint",
        ),
        profile_fingerprint=_normalize_hex(
            _coerce_str(payload.get("profile_fingerprint"), "recipe_attempt.profile_fingerprint"),
            expected_len=64,
            path="recipe_attempt.profile_fingerprint",
        ),
        created_utc=_parse_datetime(
            _coerce_str(payload.get("created_utc"), "recipe_attempt.created_utc"),
            "recipe_attempt.created_utc",
        ),
        finished_utc=_parse_optional_datetime(payload.get("finished_utc"), "recipe_attempt.finished_utc"),
        state=AttemptState(_coerce_str(payload.get("state"), "recipe_attempt.state")),
        gate_results=gates,
        failure=failure,
    )
    _validate_attempt_gate_monotonicity(attempt.gate_results)
    if attempt.state in TERMINAL_ATTEMPT_STATES and attempt.finished_utc is None:
        raise RecipeAttemptSchemaError("Terminal attempt records must include finished_utc.")
    if attempt.state not in TERMINAL_ATTEMPT_STATES and attempt.finished_utc is not None:
        raise RecipeAttemptSchemaError("Non-terminal attempt records cannot include finished_utc.")
    return attempt


def deserialize_verified_recipe_record(
    payload: dict[str, object],
    *,
    schema_root: dict[str, object] | None = None,
) -> VerifiedRecipeRecord:
    _validate_record_payload(payload, record_def="verified_recipe_record", schema_root=schema_root)
    evidence_payload = _coerce_schema_object(
        payload.get("promotion_evidence"),
        "verified_recipe_record.promotion_evidence",
    )
    promotion_evidence = _promotion_evidence_from_payload(evidence_payload)
    record = VerifiedRecipeRecord(
        verified_fingerprint=_normalize_hex(
            _coerce_str(payload.get("verified_fingerprint"), "verified_recipe_record.verified_fingerprint"),
            expected_len=64,
            path="verified_recipe_record.verified_fingerprint",
        ),
        source_recipe_fingerprint=_normalize_hex(
            _coerce_str(
                payload.get("source_recipe_fingerprint"),
                "verified_recipe_record.source_recipe_fingerprint",
            ),
            expected_len=64,
            path="verified_recipe_record.source_recipe_fingerprint",
        ),
        attempt_id=_coerce_str(payload.get("attempt_id"), "verified_recipe_record.attempt_id"),
        schema_version=_coerce_str(payload.get("schema_version"), "verified_recipe_record.schema_version"),
        model_id=_coerce_str(payload.get("model_id"), "verified_recipe_record.model_id"),
        revision_sha=_normalize_hex(
            _coerce_str(payload.get("revision_sha"), "verified_recipe_record.revision_sha"),
            expected_len=40,
            path="verified_recipe_record.revision_sha",
        ),
        requested_device=_coerce_str(
            payload.get("requested_device"),
            "verified_recipe_record.requested_device",
        ).lower(),
        requested_precision=_coerce_str(
            payload.get("requested_precision"),
            "verified_recipe_record.requested_precision",
        ).lower(),
        compiler_version=_coerce_str(
            payload.get("compiler_version"),
            "verified_recipe_record.compiler_version",
        ),
        capability_fingerprint=_normalize_hex(
            _coerce_str(
                payload.get("capability_fingerprint"),
                "verified_recipe_record.capability_fingerprint",
            ),
            expected_len=64,
            path="verified_recipe_record.capability_fingerprint",
        ),
        toolchain_fingerprint=_normalize_hex(
            _coerce_str(
                payload.get("toolchain_fingerprint"),
                "verified_recipe_record.toolchain_fingerprint",
            ),
            expected_len=64,
            path="verified_recipe_record.toolchain_fingerprint",
        ),
        profile_fingerprint=_normalize_hex(
            _coerce_str(
                payload.get("profile_fingerprint"),
                "verified_recipe_record.profile_fingerprint",
            ),
            expected_len=64,
            path="verified_recipe_record.profile_fingerprint",
        ),
        promoted_utc=_parse_datetime(
            _coerce_str(payload.get("promoted_utc"), "verified_recipe_record.promoted_utc"),
            "verified_recipe_record.promoted_utc",
        ),
        promotion_evidence=promotion_evidence,
        canonical_json=_coerce_str(payload.get("canonical_json"), "verified_recipe_record.canonical_json"),
    )
    _validate_canonical_recipe_payload(
        canonical_json=record.canonical_json,
        expected_fingerprint=record.verified_fingerprint,
        path="verified_recipe_record.canonical_json",
    )
    return record


def serialize_candidate_attempt_record(record: CandidateAttemptRecord) -> dict[str, object]:
    return {
        "record_type": "candidate_attempt_record",
        "candidate_attempt_id": record.candidate_attempt_id,
        "parent_attempt_id": record.parent_attempt_id,
        "attempt_id": record.attempt_id,
        "candidate_index": record.candidate_index,
        "candidate_id": record.candidate_id,
        "policy_id": record.policy_id,
        "policy_version": record.policy_version,
        "policy_fingerprint": record.policy_fingerprint,
        "quality_profile_fingerprint": record.quality_profile_fingerprint,
        "quantization_override_block_size": record.quantization_override_block_size,
        "eligibility_trigger": record.eligibility_trigger,
        "disposition": record.disposition,
        "disposition_reasons": list(record.disposition_reasons),
        "recipe_fingerprint": record.recipe_fingerprint,
        "candidate_fingerprint": record.candidate_fingerprint,
        "created_utc": _format_datetime(record.created_utc),
        "attempt_state": record.attempt_state.value,
        "artifact_ref": record.artifact_ref,
        "package_ref": record.package_ref,
        "mobius_build_invocation_count": record.invocation_counters.mobius_build_invocation_count,
        "olive_optimize_invocation_count": record.invocation_counters.olive_optimize_invocation_count,
        "total_invocation_count": record.invocation_counters.total_invocation_count,
        "wall_clock_seconds": record.invocation_counters.wall_clock_seconds,
        "estimated_cost_usd": record.invocation_counters.estimated_cost_usd,
        "selection_status": record.selection_status.value,
        "selected_by": record.selected_by,
        "selection_reason": record.selection_reason,
        "selected_utc": _format_datetime(record.selected_utc) if record.selected_utc is not None else None,
        "validated_target_device": record.validated_target_device,
        "validated_target_ep": record.validated_target_ep,
        "validated_toolchain_fingerprint": record.validated_toolchain_fingerprint,
        "validated_environment_scope": record.validated_environment_scope,
    }


def deserialize_candidate_attempt_record(
    payload: dict[str, object],
    *,
    schema_root: dict[str, object] | None = None,
) -> CandidateAttemptRecord:
    _validate_record_payload(payload, record_def="candidate_attempt_record", schema_root=schema_root)
    disposition_reasons_raw = _coerce_schema_array(
        payload.get("disposition_reasons"),
        "candidate_attempt_record.disposition_reasons",
    )
    disposition_reasons = tuple(
        _sanitize_reference(
            _coerce_str(value, f"candidate_attempt_record.disposition_reasons[{idx}]"),
            path=f"candidate_attempt_record.disposition_reasons[{idx}]",
        )
        for idx, value in enumerate(disposition_reasons_raw, start=1)
    )
    quantization_override_block_size = payload.get("quantization_override_block_size")
    if quantization_override_block_size is not None:
        quantization_override_block_size = _coerce_int(
            quantization_override_block_size,
            "candidate_attempt_record.quantization_override_block_size",
        )
    invocation_counters = CandidateInvocationCounters(
        mobius_build_invocation_count=_coerce_optional_int(
            payload.get("mobius_build_invocation_count"),
            "candidate_attempt_record.mobius_build_invocation_count",
        ),
        olive_optimize_invocation_count=_coerce_optional_int(
            payload.get("olive_optimize_invocation_count"),
            "candidate_attempt_record.olive_optimize_invocation_count",
        ),
        total_invocation_count=_coerce_optional_int(
            payload.get("total_invocation_count"),
            "candidate_attempt_record.total_invocation_count",
        ),
        wall_clock_seconds=_coerce_optional_float(
            payload.get("wall_clock_seconds"),
            "candidate_attempt_record.wall_clock_seconds",
        ),
        estimated_cost_usd=_coerce_optional_float(
            payload.get("estimated_cost_usd"),
            "candidate_attempt_record.estimated_cost_usd",
        ),
    )
    record = CandidateAttemptRecord(
        candidate_attempt_id=_coerce_str(
            payload.get("candidate_attempt_id"), "candidate_attempt_record.candidate_attempt_id"
        ),
        parent_attempt_id=_coerce_str(
            payload.get("parent_attempt_id"), "candidate_attempt_record.parent_attempt_id"
        ),
        attempt_id=_coerce_str(payload.get("attempt_id"), "candidate_attempt_record.attempt_id"),
        candidate_index=_coerce_int(payload.get("candidate_index"), "candidate_attempt_record.candidate_index"),
        candidate_id=_coerce_str(payload.get("candidate_id"), "candidate_attempt_record.candidate_id"),
        policy_id=_coerce_str(payload.get("policy_id"), "candidate_attempt_record.policy_id"),
        policy_version=_coerce_str(payload.get("policy_version"), "candidate_attempt_record.policy_version"),
        policy_fingerprint=_normalize_hex(
            _coerce_str(payload.get("policy_fingerprint"), "candidate_attempt_record.policy_fingerprint"),
            expected_len=64,
            path="candidate_attempt_record.policy_fingerprint",
        ),
        quality_profile_fingerprint=_normalize_hex(
            _coerce_str(
                payload.get("quality_profile_fingerprint"),
                "candidate_attempt_record.quality_profile_fingerprint",
            ),
            expected_len=64,
            path="candidate_attempt_record.quality_profile_fingerprint",
        ),
        quantization_override_block_size=quantization_override_block_size,
        eligibility_trigger=_coerce_optional_str(
            payload.get("eligibility_trigger"), "candidate_attempt_record.eligibility_trigger"
        ),
        disposition=_coerce_optional_str(payload.get("disposition"), "candidate_attempt_record.disposition"),
        disposition_reasons=disposition_reasons,
        recipe_fingerprint=_normalize_hex(
            _coerce_str(payload.get("recipe_fingerprint"), "candidate_attempt_record.recipe_fingerprint"),
            expected_len=64,
            path="candidate_attempt_record.recipe_fingerprint",
        ),
        candidate_fingerprint=_normalize_hex(
            _coerce_str(payload.get("candidate_fingerprint"), "candidate_attempt_record.candidate_fingerprint"),
            expected_len=64,
            path="candidate_attempt_record.candidate_fingerprint",
        ),
        created_utc=_parse_datetime(
            _coerce_str(payload.get("created_utc"), "candidate_attempt_record.created_utc"),
            "candidate_attempt_record.created_utc",
        ),
        attempt_state=AttemptState(_coerce_str(payload.get("attempt_state"), "candidate_attempt_record.attempt_state")),
        artifact_ref=_sanitize_optional_reference(
            payload.get("artifact_ref"), path="candidate_attempt_record.artifact_ref"
        ),
        package_ref=_sanitize_optional_reference(
            payload.get("package_ref"), path="candidate_attempt_record.package_ref"
        ),
        invocation_counters=invocation_counters,
        selection_status=CandidateWinnerStatus(
            _coerce_str(payload.get("selection_status"), "candidate_attempt_record.selection_status")
        ),
        selected_by=_coerce_optional_str(payload.get("selected_by"), "candidate_attempt_record.selected_by"),
        selection_reason=_coerce_optional_str(
            payload.get("selection_reason"), "candidate_attempt_record.selection_reason"
        ),
        selected_utc=_parse_optional_datetime(
            payload.get("selected_utc"), "candidate_attempt_record.selected_utc"
        ),
        validated_target_device=_coerce_optional_str(
            payload.get("validated_target_device"), "candidate_attempt_record.validated_target_device"
        ),
        validated_target_ep=_coerce_optional_str(
            payload.get("validated_target_ep"), "candidate_attempt_record.validated_target_ep"
        ),
        validated_toolchain_fingerprint=_coerce_optional_str(
            payload.get("validated_toolchain_fingerprint"),
            "candidate_attempt_record.validated_toolchain_fingerprint",
        ),
        validated_environment_scope=_coerce_optional_str(
            payload.get("validated_environment_scope"),
            "candidate_attempt_record.validated_environment_scope",
        ),
    )
    if record.candidate_index == 0 and record.attempt_id != record.parent_attempt_id:
        raise RecipeAttemptSchemaError(
            "candidate_attempt_record.candidate_index 0 must use attempt_id == parent_attempt_id."
        )
    if (record.selection_status == CandidateWinnerStatus.SELECTED) != (record.selected_by is not None):
        raise RecipeAttemptSchemaError(
            "candidate_attempt_record.selected_by must be set if and only if selection_status is 'selected'."
        )
    return record


def serialize_recipe_candidate_lineage(lineage: RecipeCandidateLineage) -> dict[str, object]:
    return {
        "record_type": "recipe_candidate_lineage_record",
        "parent_attempt_id": lineage.parent_attempt_id,
        "policy_id": lineage.policy_id,
        "policy_version": lineage.policy_version,
        "policy_fingerprint": lineage.policy_fingerprint,
        "policy_max_candidates": lineage.policy_max_candidates,
        "quality_profile_fingerprint": lineage.quality_profile_fingerprint,
        "created_utc": _format_datetime(lineage.created_utc),
        "selection_state": lineage.selection_state.value,
        "selected_candidate_attempt_id": lineage.selected_candidate_attempt_id,
        "selection_reason": lineage.selection_reason,
        "finalized_utc": _format_datetime(lineage.finalized_utc) if lineage.finalized_utc is not None else None,
    }


def deserialize_recipe_candidate_lineage(
    payload: dict[str, object],
    *,
    schema_root: dict[str, object] | None = None,
) -> RecipeCandidateLineage:
    _validate_record_payload(payload, record_def="recipe_candidate_lineage_record", schema_root=schema_root)
    lineage = RecipeCandidateLineage(
        parent_attempt_id=_coerce_str(
            payload.get("parent_attempt_id"), "recipe_candidate_lineage_record.parent_attempt_id"
        ),
        policy_id=_coerce_str(payload.get("policy_id"), "recipe_candidate_lineage_record.policy_id"),
        policy_version=_coerce_str(payload.get("policy_version"), "recipe_candidate_lineage_record.policy_version"),
        policy_fingerprint=_normalize_hex(
            _coerce_str(payload.get("policy_fingerprint"), "recipe_candidate_lineage_record.policy_fingerprint"),
            expected_len=64,
            path="recipe_candidate_lineage_record.policy_fingerprint",
        ),
        policy_max_candidates=_coerce_int(
            payload.get("policy_max_candidates"), "recipe_candidate_lineage_record.policy_max_candidates"
        ),
        quality_profile_fingerprint=_normalize_hex(
            _coerce_str(
                payload.get("quality_profile_fingerprint"),
                "recipe_candidate_lineage_record.quality_profile_fingerprint",
            ),
            expected_len=64,
            path="recipe_candidate_lineage_record.quality_profile_fingerprint",
        ),
        created_utc=_parse_datetime(
            _coerce_str(payload.get("created_utc"), "recipe_candidate_lineage_record.created_utc"),
            "recipe_candidate_lineage_record.created_utc",
        ),
        selection_state=CandidateLineageSelectionState(
            _coerce_str(payload.get("selection_state"), "recipe_candidate_lineage_record.selection_state")
        ),
        selected_candidate_attempt_id=_coerce_optional_str(
            payload.get("selected_candidate_attempt_id"),
            "recipe_candidate_lineage_record.selected_candidate_attempt_id",
        ),
        selection_reason=_coerce_optional_str(
            payload.get("selection_reason"), "recipe_candidate_lineage_record.selection_reason"
        ),
        finalized_utc=_parse_optional_datetime(
            payload.get("finalized_utc"), "recipe_candidate_lineage_record.finalized_utc"
        ),
    )
    if lineage.selection_state == CandidateLineageSelectionState.PENDING:
        if lineage.selected_candidate_attempt_id is not None or lineage.finalized_utc is not None:
            raise RecipeAttemptSchemaError(
                "recipe_candidate_lineage_record.selected_candidate_attempt_id/finalized_utc must be null while pending."
            )
    if lineage.selection_state == CandidateLineageSelectionState.SELECTED:
        if lineage.selected_candidate_attempt_id is None or lineage.finalized_utc is None:
            raise RecipeAttemptSchemaError(
                "recipe_candidate_lineage_record.selected_candidate_attempt_id/finalized_utc are required once selected."
            )
    if lineage.selection_state == CandidateLineageSelectionState.EXHAUSTED:
        if lineage.selected_candidate_attempt_id is not None:
            raise RecipeAttemptSchemaError(
                "recipe_candidate_lineage_record.selected_candidate_attempt_id must be null when exhausted."
            )
        if lineage.finalized_utc is None:
            raise RecipeAttemptSchemaError(
                "recipe_candidate_lineage_record.finalized_utc is required once exhausted."
            )
    return lineage


class RecipeAttemptStore:
    def __init__(
        self,
        db_path: Path,
        *,
        schema_path: Path | None = None,
        _connection_factory: Callable[..., sqlite3.Connection] | None = None,
    ) -> None:
        self._db_path = db_path.resolve()
        self._schema_path = schema_path or recipe_attempt_schema_path()
        self._schema_root: dict[str, object] | None = None
        self._connection_factory = _connection_factory or sqlite3.connect
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL;")
            connection.execute("PRAGMA synchronous=NORMAL;")
            self._migrate(connection)
            self._recover_interrupted_attempts(connection)

    @property
    def db_path(self) -> Path:
        return self._db_path

    def upsert_generated_recipe(
        self,
        recipe: GeneratedRecipe,
        *,
        created_utc: datetime | None = None,
    ) -> GeneratedRecipeRecord:
        identity = _identity_from_generated_recipe(recipe)
        payload = recipe.payload()
        schema_version = _coerce_str(payload.get("schema_version"), "generated_recipe.schema_version")
        record = GeneratedRecipeRecord(
            recipe_fingerprint=_normalize_hex(
                recipe.fingerprint,
                expected_len=64,
                path="generated_recipe.fingerprint",
            ),
            schema_version=schema_version,
            recipe_status=recipe.recipe.status,
            model_id=identity.model_id,
            revision_sha=identity.revision_sha,
            requested_device=identity.requested_device,
            requested_precision=identity.requested_precision,
            compiler_version=identity.compiler_version,
            capability_fingerprint=identity.capability_fingerprint,
            toolchain_fingerprint=identity.toolchain_fingerprint,
            profile_fingerprint=identity.profile_fingerprint,
            canonical_json=_coerce_str(recipe.canonical_json, "generated_recipe.canonical_json"),
            created_utc=_ensure_utc(created_utc or datetime.now(timezone.utc), "generated_recipe.created_utc"),
        )
        _ = deserialize_generated_recipe_record(
            serialize_generated_recipe_record(record),
            schema_root=self._schema(),
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE;")
            existing = connection.execute(
                """
                SELECT recipe_fingerprint, schema_version, recipe_status, model_id, revision_sha,
                       requested_device, requested_precision, compiler_version, capability_fingerprint,
                       toolchain_fingerprint, profile_fingerprint, canonical_json, created_utc
                FROM generated_recipes
                WHERE recipe_fingerprint = ?
                """,
                (record.recipe_fingerprint,),
            ).fetchone()
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO generated_recipes (
                        recipe_fingerprint,
                        schema_version,
                        recipe_status,
                        model_id,
                        revision_sha,
                        requested_device,
                        requested_precision,
                        compiler_version,
                        capability_fingerprint,
                        toolchain_fingerprint,
                        profile_fingerprint,
                        canonical_json,
                        created_utc
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.recipe_fingerprint,
                        record.schema_version,
                        record.recipe_status.value,
                        record.model_id,
                        record.revision_sha,
                        record.requested_device,
                        record.requested_precision,
                        record.compiler_version,
                        record.capability_fingerprint,
                        record.toolchain_fingerprint,
                        record.profile_fingerprint,
                        record.canonical_json,
                        _format_datetime(record.created_utc),
                    ),
                )
            else:
                existing_record = self._generated_record_from_row(existing)
                if existing_record.canonical_json != record.canonical_json:
                    raise RecipeAttemptStoreError(
                        "Generated recipe fingerprint collision with different canonical payload."
                    )
            row = connection.execute(
                """
                SELECT recipe_fingerprint, schema_version, recipe_status, model_id, revision_sha,
                       requested_device, requested_precision, compiler_version, capability_fingerprint,
                       toolchain_fingerprint, profile_fingerprint, canonical_json, created_utc
                FROM generated_recipes
                WHERE recipe_fingerprint = ?
                """,
                (record.recipe_fingerprint,),
            ).fetchone()
            assert row is not None
            return self._generated_record_from_row(row)

    def get_generated_recipe(self, recipe_fingerprint: str) -> GeneratedRecipeRecord | None:
        normalized = _normalize_hex(recipe_fingerprint, expected_len=64, path="recipe_fingerprint")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT recipe_fingerprint, schema_version, recipe_status, model_id, revision_sha,
                       requested_device, requested_precision, compiler_version, capability_fingerprint,
                       toolchain_fingerprint, profile_fingerprint, canonical_json, created_utc
                FROM generated_recipes
                WHERE recipe_fingerprint = ?
                """,
                (normalized,),
            ).fetchone()
            if row is None:
                return None
            return self._generated_record_from_row(row)

    def create_attempt(
        self,
        *,
        idempotency_key: str,
        request: RecipeAttemptRequest,
        request_fingerprint: str,
        attempt_id: str | None = None,
        created_utc: datetime | None = None,
    ) -> tuple[RecipeAttempt, bool]:
        key = _coerce_str(idempotency_key, "idempotency_key")
        normalized_request = _normalize_attempt_request(request)
        expected_fingerprint = build_attempt_request_fingerprint(normalized_request)
        supplied_fingerprint = _normalize_hex(
            _coerce_str(request_fingerprint, "request_fingerprint"),
            expected_len=64,
            path="request_fingerprint",
        )
        if supplied_fingerprint != expected_fingerprint:
            raise ValueError(
                "request_fingerprint must match the canonical RecipeAttemptRequest fingerprint."
            )
        attempt_id_value = _coerce_str(attempt_id, "attempt_id") if attempt_id else str(uuid.uuid4())
        created_value = _ensure_utc(created_utc or datetime.now(timezone.utc), "created_utc")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE;")
            existing = connection.execute(
                """
                SELECT attempt_id, request_fingerprint
                FROM attempts
                WHERE idempotency_key = ?
                """,
                (key,),
            ).fetchone()
            if existing is not None:
                existing_fingerprint = _normalize_hex(
                    existing["request_fingerprint"],
                    expected_len=64,
                    path="attempts.request_fingerprint",
                )
                if existing_fingerprint != supplied_fingerprint:
                    raise AttemptIdempotencyConflictError(
                        "Idempotency key was reused with a different request fingerprint."
                    )
                return self._load_attempt(connection, existing["attempt_id"]), True

            generated = connection.execute(
                """
                SELECT recipe_fingerprint, schema_version, recipe_status, model_id, revision_sha,
                       requested_device, requested_precision, compiler_version, capability_fingerprint,
                       toolchain_fingerprint, profile_fingerprint, canonical_json, created_utc
                FROM generated_recipes
                WHERE recipe_fingerprint = ?
                """,
                (normalized_request.recipe_fingerprint,),
            ).fetchone()
            if generated is None:
                raise KeyError(
                    f"Generated recipe '{normalized_request.recipe_fingerprint}' does not exist."
                )
            generated_record = self._generated_record_from_row(generated)
            _assert_attempt_request_matches_generated(normalized_request, generated_record)
            try:
                connection.execute(
                    """
                    INSERT INTO attempts (
                        attempt_id,
                        idempotency_key,
                        request_fingerprint,
                        recipe_fingerprint,
                        model_id,
                        revision_sha,
                        requested_device,
                        requested_precision,
                        compiler_version,
                        capability_fingerprint,
                        toolchain_fingerprint,
                        profile_fingerprint,
                        created_utc,
                        finished_utc,
                        state,
                        failure_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        attempt_id_value,
                        key,
                        supplied_fingerprint,
                        normalized_request.recipe_fingerprint,
                        normalized_request.model_id,
                        normalized_request.revision_sha,
                        normalized_request.requested_device,
                        normalized_request.requested_precision,
                        normalized_request.compiler_version,
                        normalized_request.capability_fingerprint,
                        normalized_request.toolchain_fingerprint,
                        normalized_request.profile_fingerprint,
                        _format_datetime(created_value),
                        None,
                        AttemptState.GENERATED.value,
                        None,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                replay = connection.execute(
                    """
                    SELECT attempt_id, request_fingerprint
                    FROM attempts
                    WHERE idempotency_key = ?
                    """,
                    (key,),
                ).fetchone()
                if replay is not None:
                    existing_fingerprint = _normalize_hex(
                        replay["request_fingerprint"],
                        expected_len=64,
                        path="attempts.request_fingerprint",
                    )
                    if existing_fingerprint == supplied_fingerprint:
                        return self._load_attempt(connection, replay["attempt_id"]), True
                    raise AttemptIdempotencyConflictError(
                        "Idempotency key was reused with a different request fingerprint."
                    ) from exc
                raise
            return self._load_attempt(connection, attempt_id_value), False

    def get_attempt(self, attempt_id: str) -> RecipeAttempt:
        attempt_id_value = _coerce_str(attempt_id, "attempt_id")
        with self._connect() as connection:
            return self._load_attempt(connection, attempt_id_value)

    def list_attempts(self) -> tuple[RecipeAttempt, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT attempt_id
                FROM attempts
                ORDER BY created_utc ASC, attempt_id ASC
                """
            ).fetchall()
            return tuple(self._load_attempt(connection, row["attempt_id"]) for row in rows)

    def start_attempt(self, attempt_id: str) -> RecipeAttempt:
        attempt_id_value = _coerce_str(attempt_id, "attempt_id")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE;")
            row = self._load_attempt(connection, attempt_id_value)
            if row.state != AttemptState.GENERATED:
                raise AttemptStateTransitionError(
                    f"Attempt '{attempt_id_value}' cannot transition to running from '{row.state.value}'."
                )
            connection.execute(
                "UPDATE attempts SET state = ? WHERE attempt_id = ?",
                (AttemptState.RUNNING.value, attempt_id_value),
            )
            return self._load_attempt(connection, attempt_id_value)

    def record_attempt_gate(
        self,
        *,
        attempt_id: str,
        gate: AttemptGate,
        status: AttemptGateStatus,
        evidence_ref: str,
        metrics_ref: str | None = None,
        started_utc: datetime | None = None,
        finished_utc: datetime | None = None,
        expected_sequence: int | None = None,
    ) -> AttemptGateResult:
        attempt_id_value = _coerce_str(attempt_id, "attempt_id")
        normalized_evidence = _sanitize_reference(evidence_ref, path="evidence_ref")
        normalized_metrics = _sanitize_optional_reference(metrics_ref, path="metrics_ref")
        started_value = _ensure_utc(started_utc or datetime.now(timezone.utc), "started_utc")
        finished_value = _ensure_utc(finished_utc or datetime.now(timezone.utc), "finished_utc")
        if finished_value < started_value:
            raise ValueError("finished_utc cannot be earlier than started_utc.")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE;")
            attempt = self._load_attempt(connection, attempt_id_value)
            if attempt.state != AttemptState.RUNNING:
                raise AttemptStateTransitionError(
                    f"Attempt '{attempt_id_value}' must be running to record gate results."
                )
            existing = connection.execute(
                """
                SELECT sequence, gate_name
                FROM attempt_gates
                WHERE attempt_id = ?
                ORDER BY sequence ASC
                """,
                (attempt_id_value,),
            ).fetchall()
            next_sequence = len(existing) + 1
            if expected_sequence is not None and expected_sequence != next_sequence:
                raise AttemptGateSequenceError(
                    f"Expected sequence {expected_sequence}, but next monotonic sequence is {next_sequence}."
                )
            if next_sequence > len(ATTEMPT_GATE_ORDER):
                raise AttemptGateSequenceError(
                    f"All required gates are already recorded for attempt '{attempt_id_value}'."
                )
            expected_gate = ATTEMPT_GATE_ORDER[next_sequence - 1]
            if gate != expected_gate:
                raise AttemptGateSequenceError(
                    f"Gate '{gate.value}' is out of order. Expected '{expected_gate.value}'."
                )
            connection.execute(
                """
                INSERT INTO attempt_gates (
                    attempt_id,
                    sequence,
                    gate_name,
                    status,
                    evidence_ref,
                    metrics_ref,
                    started_utc,
                    finished_utc
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt_id_value,
                    next_sequence,
                    gate.value,
                    status.value,
                    normalized_evidence,
                    normalized_metrics,
                    _format_datetime(started_value),
                    _format_datetime(finished_value),
                ),
            )
            gate_row = AttemptGateResult(
                sequence=next_sequence,
                gate=gate,
                status=status,
                evidence_ref=normalized_evidence,
                metrics_ref=normalized_metrics,
                started_utc=started_value,
                finished_utc=finished_value,
            )
            _ = deserialize_attempt_gate_result(serialize_attempt_gate_result(gate_row))
            return gate_row

    def finish_attempt_succeeded(
        self,
        attempt_id: str,
        *,
        finished_utc: datetime | None = None,
    ) -> RecipeAttempt:
        attempt_id_value = _coerce_str(attempt_id, "attempt_id")
        finished_value = _ensure_utc(finished_utc or datetime.now(timezone.utc), "finished_utc")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE;")
            attempt = self._load_attempt(connection, attempt_id_value)
            if attempt.state != AttemptState.RUNNING:
                raise AttemptStateTransitionError(
                    f"Attempt '{attempt_id_value}' cannot transition to succeeded from '{attempt.state.value}'."
                )
            _require_complete_successful_gates(attempt.gate_results)
            connection.execute(
                """
                UPDATE attempts
                SET state = ?, finished_utc = ?, failure_json = NULL
                WHERE attempt_id = ?
                """,
                (AttemptState.SUCCEEDED.value, _format_datetime(finished_value), attempt_id_value),
            )
            return self._load_attempt(connection, attempt_id_value)

    def finish_attempt_failed(
        self,
        attempt_id: str,
        *,
        failure: AttemptFailure,
        finished_utc: datetime | None = None,
    ) -> RecipeAttempt:
        return self._finish_terminal_with_failure(
            attempt_id=attempt_id,
            target=AttemptState.FAILED,
            failure=failure,
            finished_utc=finished_utc,
        )

    def cancel_attempt(
        self,
        attempt_id: str,
        *,
        failure: AttemptFailure,
        finished_utc: datetime | None = None,
    ) -> RecipeAttempt:
        if failure.classification != AttemptFailureClassification.CANCELLED:
            raise ValueError("cancel_attempt requires failure.classification='cancelled'.")
        return self._finish_terminal_with_failure(
            attempt_id=attempt_id,
            target=AttemptState.CANCELLED,
            failure=failure,
            finished_utc=finished_utc,
        )

    def promote_verified_recipe(
        self,
        *,
        attempt_id: str,
        promoted_recipe: GeneratedRecipe,
        promoted_utc: datetime | None = None,
    ) -> VerifiedRecipeRecord:
        attempt_id_value = _coerce_str(attempt_id, "attempt_id")
        promotion = promoted_recipe.provenance.promotion
        if promotion is None:
            raise RecipePromotionConflictError(
                "Promotion requires compiler-provided promotion evidence."
            )
        if promoted_recipe.recipe.status != RecipeStatus.VERIFIED:
            raise RecipePromotionConflictError(
                f"Only verified recipes can be promoted for reuse; got '{promoted_recipe.recipe.status.value}'."
            )
        normalized_evidence = _normalize_promotion_gate_evidence(promotion.gate_evidence)
        source_fingerprint = _normalize_hex(
            promotion.promoted_from_fingerprint,
            expected_len=64,
            path="promotion.promoted_from_fingerprint",
        )
        identity = _identity_from_generated_recipe(promoted_recipe)
        promoted_value = _ensure_utc(promoted_utc or datetime.now(timezone.utc), "promoted_utc")
        canonical_json = _coerce_str(promoted_recipe.canonical_json, "promoted_recipe.canonical_json")
        verified_fingerprint = _normalize_hex(
            promoted_recipe.fingerprint,
            expected_len=64,
            path="promoted_recipe.fingerprint",
        )
        payload = promoted_recipe.payload()
        schema_version = _coerce_str(payload.get("schema_version"), "promoted_recipe.schema_version")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE;")
            attempt = self._load_attempt(connection, attempt_id_value)
            if attempt.state != AttemptState.SUCCEEDED:
                raise RecipePromotionConflictError(
                    f"Attempt '{attempt_id_value}' must be succeeded before promotion."
                )
            if attempt.recipe_fingerprint != source_fingerprint:
                raise RecipePromotionConflictError(
                    "Attempt recipe fingerprint does not match promotion source fingerprint."
                )
            _assert_attempt_identity_matches_generated(attempt, identity)

            source_generated = connection.execute(
                """
                SELECT recipe_fingerprint, schema_version, recipe_status, model_id, revision_sha,
                       requested_device, requested_precision, compiler_version, capability_fingerprint,
                       toolchain_fingerprint, profile_fingerprint, canonical_json, created_utc
                FROM generated_recipes
                WHERE recipe_fingerprint = ?
                """,
                (source_fingerprint,),
            ).fetchone()
            if source_generated is None:
                raise RecipePromotionConflictError(
                    f"Source generated recipe '{source_fingerprint}' is not persisted."
                )

            _require_complete_successful_gates(attempt.gate_results)
            _assert_promotion_evidence_matches_attempt(
                attempt=attempt,
                promotion_evidence=normalized_evidence,
            )

            existing = connection.execute(
                """
                SELECT verified_fingerprint, source_recipe_fingerprint, attempt_id, schema_version,
                       model_id, revision_sha, requested_device, requested_precision, compiler_version,
                       capability_fingerprint, toolchain_fingerprint, profile_fingerprint,
                       promoted_utc, promotion_evidence_json, canonical_json
                FROM verified_recipes
                WHERE attempt_id = ?
                """,
                (attempt_id_value,),
            ).fetchone()
            if existing is not None:
                current = self._verified_record_from_row(existing)
                if (
                    current.verified_fingerprint == verified_fingerprint
                    and current.canonical_json == canonical_json
                ):
                    return current
                raise RecipePromotionConflictError(
                    f"Attempt '{attempt_id_value}' is already promoted to a different verified recipe."
                )

            record = VerifiedRecipeRecord(
                verified_fingerprint=verified_fingerprint,
                source_recipe_fingerprint=source_fingerprint,
                attempt_id=attempt_id_value,
                schema_version=schema_version,
                model_id=identity.model_id,
                revision_sha=identity.revision_sha,
                requested_device=identity.requested_device,
                requested_precision=identity.requested_precision,
                compiler_version=identity.compiler_version,
                capability_fingerprint=identity.capability_fingerprint,
                toolchain_fingerprint=identity.toolchain_fingerprint,
                profile_fingerprint=identity.profile_fingerprint,
                promoted_utc=promoted_value,
                promotion_evidence=normalized_evidence,
                canonical_json=canonical_json,
            )
            _ = deserialize_verified_recipe_record(
                serialize_verified_recipe_record(record),
                schema_root=self._schema(),
            )
            connection.execute(
                """
                INSERT INTO verified_recipes (
                    verified_fingerprint,
                    source_recipe_fingerprint,
                    attempt_id,
                    schema_version,
                    model_id,
                    revision_sha,
                    requested_device,
                    requested_precision,
                    compiler_version,
                    capability_fingerprint,
                    toolchain_fingerprint,
                    profile_fingerprint,
                    promoted_utc,
                    promotion_evidence_json,
                    canonical_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.verified_fingerprint,
                    record.source_recipe_fingerprint,
                    record.attempt_id,
                    record.schema_version,
                    record.model_id,
                    record.revision_sha,
                    record.requested_device,
                    record.requested_precision,
                    record.compiler_version,
                    record.capability_fingerprint,
                    record.toolchain_fingerprint,
                    record.profile_fingerprint,
                    _format_datetime(record.promoted_utc),
                    json.dumps(_promotion_evidence_to_payload(record.promotion_evidence), separators=(",", ":")),
                    record.canonical_json,
                ),
            )
            row = connection.execute(
                """
                SELECT verified_fingerprint, source_recipe_fingerprint, attempt_id, schema_version,
                       model_id, revision_sha, requested_device, requested_precision, compiler_version,
                       capability_fingerprint, toolchain_fingerprint, profile_fingerprint,
                       promoted_utc, promotion_evidence_json, canonical_json
                FROM verified_recipes
                WHERE verified_fingerprint = ?
                """,
                (record.verified_fingerprint,),
            ).fetchone()
            assert row is not None
            return self._verified_record_from_row(row)

    def get_verified_recipe(self, verified_fingerprint: str) -> VerifiedRecipeRecord | None:
        normalized = _normalize_hex(
            verified_fingerprint,
            expected_len=64,
            path="verified_fingerprint",
        )
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT verified_fingerprint, source_recipe_fingerprint, attempt_id, schema_version,
                       model_id, revision_sha, requested_device, requested_precision, compiler_version,
                       capability_fingerprint, toolchain_fingerprint, profile_fingerprint,
                       promoted_utc, promotion_evidence_json, canonical_json
                FROM verified_recipes
                WHERE verified_fingerprint = ?
                """,
                (normalized,),
            ).fetchone()
            if row is None:
                return None
            return self._verified_record_from_row(row)

    def find_reusable_verified_recipe(self, query: RecipeReuseQuery) -> VerifiedRecipeRecord | None:
        normalized_query = _normalize_reuse_query(query)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT verified_fingerprint, source_recipe_fingerprint, attempt_id, schema_version,
                       model_id, revision_sha, requested_device, requested_precision, compiler_version,
                       capability_fingerprint, toolchain_fingerprint, profile_fingerprint,
                       promoted_utc, promotion_evidence_json, canonical_json
                FROM verified_recipes
                WHERE model_id = ?
                  AND revision_sha = ?
                  AND requested_device = ?
                  AND requested_precision = ?
                  AND compiler_version = ?
                  AND capability_fingerprint = ?
                  AND toolchain_fingerprint = ?
                  AND profile_fingerprint = ?
                ORDER BY promoted_utc DESC, verified_fingerprint DESC
                LIMIT 1
                """,
                (
                    normalized_query.model_id,
                    normalized_query.revision_sha,
                    normalized_query.requested_device,
                    normalized_query.requested_precision,
                    normalized_query.compiler_version,
                    normalized_query.capability_fingerprint,
                    normalized_query.toolchain_fingerprint,
                    normalized_query.profile_fingerprint,
                ),
            ).fetchone()
            if row is None:
                return None
            return self._verified_record_from_row(row)

    def list_verified_recipes(self) -> tuple[VerifiedRecipeRecord, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT verified_fingerprint, source_recipe_fingerprint, attempt_id, schema_version,
                       model_id, revision_sha, requested_device, requested_precision, compiler_version,
                       capability_fingerprint, toolchain_fingerprint, profile_fingerprint,
                       promoted_utc, promotion_evidence_json, canonical_json
                FROM verified_recipes
                ORDER BY promoted_utc ASC, verified_fingerprint ASC
                """
            ).fetchall()
            return tuple(self._verified_record_from_row(row) for row in rows)

    # -- Slice 2: candidate plan / selection ------------------------------------

    def register_candidate_attempt(
        self,
        *,
        parent_attempt_id: str,
        attempt_id: str,
        candidate_index: int,
        policy: RecipeSelectionPolicy,
        quality_profile_fingerprint: str,
        trigger: str | None = None,
        retry_evaluation: QualityRetryEvaluation | None = None,
        created_utc: datetime | None = None,
    ) -> CandidateAttemptRecord:
        """Create/validate one candidate-plan slot from an approved policy.

        Callable once per ``candidate_index`` for a given ``parent_attempt_id``.
        Candidate index 0 (the always-eligible default) must be registered
        first, using ``attempt_id == parent_attempt_id`` (the default candidate
        *is* the parent's own existing :class:`RecipeAttempt`). Any further
        candidate (index >= 1) must reuse a distinct, already-created attempt
        (via the existing ``create_attempt``), and must supply the exact
        ``trigger``/``retry_evaluation`` that the policy declares for that
        candidate index -- fail-closed on any mismatch, duplicate, ordering
        violation, or `max_candidates` overrun.
        """
        parent_id = _coerce_str(parent_attempt_id, "parent_attempt_id")
        child_id = _coerce_str(attempt_id, "attempt_id")
        normalized_quality_profile_fingerprint = _normalize_hex(
            quality_profile_fingerprint,
            expected_len=64,
            path="quality_profile_fingerprint",
        )
        if candidate_index < 0 or candidate_index >= len(policy.candidates):
            raise CandidatePlanValidationError(
                f"candidate_index {candidate_index} is out of range for policy '{policy.policy_id}'."
            )
        candidate = policy.candidates[candidate_index]
        if candidate.candidate_index != candidate_index:
            raise CandidatePlanValidationError(
                "Policy candidate_index does not match its declared position; malformed policy."
            )

        if candidate_index == 0:
            if child_id != parent_id:
                raise CandidatePlanValidationError(
                    "Candidate index 0 (the default) must be registered with attempt_id == parent_attempt_id."
                )
            if trigger is not None:
                raise CandidatePlanValidationError("Candidate index 0 (the default) cannot declare a trigger.")
            if retry_evaluation is not None:
                raise CandidatePlanValidationError(
                    "Candidate index 0 (the default) cannot carry a quality retry evaluation."
                )
        else:
            if child_id == parent_id:
                raise CandidatePlanValidationError(
                    f"Candidate index {candidate_index} must use a distinct attempt_id from the parent."
                )
            if candidate.eligibility_trigger is None:
                raise CandidatePlanValidationError(
                    f"Policy candidate at index {candidate_index} has no eligibility_trigger; malformed policy."
                )
            if trigger != candidate.eligibility_trigger:
                raise CandidatePlanValidationError(
                    f"Supplied trigger '{trigger}' does not match policy candidate eligibility_trigger "
                    f"'{candidate.eligibility_trigger}'."
                )
            if trigger != RETRYABLE_OPTIMIZED_STRUCTURAL_REGRESSION_TRIGGER:
                raise CandidatePlanValidationError(
                    f"Trigger '{trigger}' is not an allowlisted retry trigger."
                )
            if retry_evaluation is None:
                raise CandidatePlanValidationError(
                    f"Candidate index {candidate_index} requires a quality retry evaluation."
                )
            if retry_evaluation.disposition != QualityRetryDisposition.RETRYABLE_OPTIMIZED_STRUCTURAL_REGRESSION:
                raise CandidatePlanValidationError(
                    "Quality retry evaluation must be retryable to register a non-default candidate."
                )
            if retry_evaluation.disposition.value != trigger:
                # Cross-module boundary re-check: even though both constants share one
                # source today, never trust that silently -- verify the exact value used
                # to justify *this* registration matches the trigger actually supplied.
                raise CandidatePlanValidationError(
                    "Quality retry disposition value does not match the supplied policy eligibility "
                    "trigger; refusing to register a fallback candidate on a mismatched trigger."
                )

        disposition_value = retry_evaluation.disposition.value if retry_evaluation is not None else None
        disposition_reasons = retry_evaluation.reasons if retry_evaluation is not None else ()
        sanitized_reasons = tuple(
            _sanitize_reference(reason, path=f"candidate_attempt.disposition_reasons[{idx}]")
            for idx, reason in enumerate(disposition_reasons, start=1)
        )

        created_value = _ensure_utc(created_utc or datetime.now(timezone.utc), "created_utc")
        quantization_override = candidate.quantization_override
        policy_fingerprint = policy.fingerprint

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE;")
            child_attempt = self._load_attempt(connection, child_id)
            recipe_fingerprint = child_attempt.recipe_fingerprint

            lineage_row = connection.execute(
                """
                SELECT policy_id, policy_version, policy_fingerprint, policy_max_candidates,
                       quality_profile_fingerprint, selection_state
                FROM recipe_candidate_lineages
                WHERE parent_attempt_id = ?
                """,
                (parent_id,),
            ).fetchone()

            if lineage_row is None:
                if candidate_index != 0:
                    raise CandidatePlanValidationError(
                        "A candidate lineage must be created starting with candidate_index 0 (the default)."
                    )
                connection.execute(
                    """
                    INSERT INTO recipe_candidate_lineages (
                        parent_attempt_id, policy_id, policy_version, policy_fingerprint,
                        policy_max_candidates, quality_profile_fingerprint, created_utc,
                        selection_state, selected_candidate_attempt_id, selection_reason, finalized_utc
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL)
                    """,
                    (
                        parent_id,
                        policy.policy_id,
                        policy.version,
                        policy_fingerprint,
                        policy.max_candidates,
                        normalized_quality_profile_fingerprint,
                        _format_datetime(created_value),
                        CandidateLineageSelectionState.PENDING.value,
                    ),
                )
            else:
                if (
                    lineage_row["policy_id"] != policy.policy_id
                    or lineage_row["policy_version"] != policy.version
                    or lineage_row["policy_fingerprint"] != policy_fingerprint
                    or lineage_row["policy_max_candidates"] != policy.max_candidates
                ):
                    raise CandidatePlanValidationError(
                        f"Parent attempt '{parent_id}' is already registered under a different selection policy."
                    )
                if lineage_row["quality_profile_fingerprint"] != normalized_quality_profile_fingerprint:
                    raise CandidatePlanValidationError(
                        f"Parent attempt '{parent_id}' is already registered under a different quality profile."
                    )
                if lineage_row["selection_state"] != CandidateLineageSelectionState.PENDING.value:
                    raise CandidatePlanValidationError(
                        f"Parent attempt '{parent_id}' candidate lineage is already finalized "
                        f"('{lineage_row['selection_state']}') and cannot accept new candidates."
                    )

            existing_candidates = connection.execute(
                """
                SELECT candidate_index FROM candidate_attempts WHERE parent_attempt_id = ?
                """,
                (parent_id,),
            ).fetchall()
            if len(existing_candidates) >= policy.max_candidates:
                raise CandidatePlanValidationError(
                    f"Parent attempt '{parent_id}' already has the maximum of {policy.max_candidates} "
                    "planned candidates."
                )
            if candidate_index != 0 and not any(row["candidate_index"] == 0 for row in existing_candidates):
                raise CandidatePlanValidationError(
                    f"Candidate index {candidate_index} cannot be registered before candidate index 0."
                )

            candidate_fingerprint = build_candidate_recipe_fingerprint(
                recipe_fingerprint=recipe_fingerprint,
                quantization_override=quantization_override,
                policy_fingerprint=policy_fingerprint,
            )
            candidate_attempt_id = str(uuid.uuid4())
            try:
                connection.execute(
                    """
                    INSERT INTO candidate_attempts (
                        candidate_attempt_id, parent_attempt_id, attempt_id, candidate_index, candidate_id,
                        policy_id, policy_version, policy_fingerprint, quality_profile_fingerprint,
                        quantization_override_block_size, eligibility_trigger, disposition,
                        disposition_reasons_json, recipe_fingerprint, candidate_fingerprint, created_utc,
                        artifact_ref, package_ref,
                        mobius_build_invocation_count, olive_optimize_invocation_count,
                        total_invocation_count, wall_clock_seconds, estimated_cost_usd,
                        selection_status, selected_by, selection_reason, selected_utc,
                        validated_target_device, validated_target_ep, validated_toolchain_fingerprint,
                        validated_environment_scope
                    )
                    VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        candidate_attempt_id,
                        parent_id,
                        child_id,
                        candidate_index,
                        candidate.candidate_id,
                        policy.policy_id,
                        policy.version,
                        policy_fingerprint,
                        normalized_quality_profile_fingerprint,
                        quantization_override.block_size if quantization_override is not None else None,
                        candidate.eligibility_trigger,
                        disposition_value,
                        json.dumps(list(sanitized_reasons), separators=(",", ":")),
                        recipe_fingerprint,
                        candidate_fingerprint,
                        _format_datetime(created_value),
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        CandidateWinnerStatus.NOT_SELECTED.value,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise CandidatePlanValidationError(
                    f"Candidate index {candidate_index} or candidate_id '{candidate.candidate_id}' is "
                    f"already registered for parent attempt '{parent_id}'."
                ) from exc

            return self._load_candidate_attempt(connection, candidate_attempt_id)

    def finalize_candidate_attempt_evidence(
        self,
        candidate_attempt_id: str,
        *,
        artifact_ref: str | None = None,
        package_ref: str | None = None,
        invocation_counters: CandidateInvocationCounters | None = None,
    ) -> CandidateAttemptRecord:
        """Atomically attach terminal evidence/counters to a candidate slot.

        Requires the candidate's linked attempt to already be terminal
        (succeeded/failed/cancelled). Write-once: calling again with identical
        values is a no-op; calling again with different values raises, because
        terminal candidate evidence is immutable.
        """
        candidate_id_value = _coerce_str(candidate_attempt_id, "candidate_attempt_id")
        normalized_artifact_ref = _sanitize_optional_reference(artifact_ref, path="artifact_ref")
        normalized_package_ref = _sanitize_optional_reference(package_ref, path="package_ref")
        counters = invocation_counters or _NULL_CANDIDATE_INVOCATION_COUNTERS
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE;")
            existing = self._load_candidate_attempt(connection, candidate_id_value)
            if existing.attempt_state not in TERMINAL_ATTEMPT_STATES:
                raise CandidatePlanValidationError(
                    f"Candidate attempt '{candidate_id_value}' cannot record terminal evidence while its "
                    f"linked attempt is '{existing.attempt_state.value}'."
                )
            proposed = CandidateInvocationCounters(
                mobius_build_invocation_count=counters.mobius_build_invocation_count,
                olive_optimize_invocation_count=counters.olive_optimize_invocation_count,
                total_invocation_count=counters.total_invocation_count,
                wall_clock_seconds=counters.wall_clock_seconds,
                estimated_cost_usd=counters.estimated_cost_usd,
            )
            already_recorded = (
                existing.artifact_ref is not None
                or existing.package_ref is not None
                or existing.invocation_counters != _NULL_CANDIDATE_INVOCATION_COUNTERS
            )
            if already_recorded:
                if (
                    existing.artifact_ref != normalized_artifact_ref
                    or existing.package_ref != normalized_package_ref
                    or existing.invocation_counters != proposed
                ):
                    raise CandidatePlanValidationError(
                        f"Candidate attempt '{candidate_id_value}' already has terminal evidence recorded "
                        "and cannot be mutated with different values."
                    )
                return existing
            connection.execute(
                """
                UPDATE candidate_attempts
                SET artifact_ref = ?, package_ref = ?,
                    mobius_build_invocation_count = ?, olive_optimize_invocation_count = ?,
                    total_invocation_count = ?, wall_clock_seconds = ?, estimated_cost_usd = ?
                WHERE candidate_attempt_id = ?
                """,
                (
                    normalized_artifact_ref,
                    normalized_package_ref,
                    proposed.mobius_build_invocation_count,
                    proposed.olive_optimize_invocation_count,
                    proposed.total_invocation_count,
                    proposed.wall_clock_seconds,
                    proposed.estimated_cost_usd,
                    candidate_id_value,
                ),
            )
            return self._load_candidate_attempt(connection, candidate_id_value)

    def select_verified_candidate_attempt(
        self,
        *,
        parent_attempt_id: str,
        candidate_attempt_id: str,
        reason: str,
        selected_utc: datetime | None = None,
        validated_target_device: str | None = None,
        validated_target_ep: str | None = None,
        validated_toolchain_fingerprint: str | None = None,
        validated_environment_scope: str | None = None,
    ) -> tuple[RecipeCandidateLineage, CandidateAttemptRecord]:
        """Atomically select the single verified winner for a candidate lineage.

        Fails closed unless: the lineage exists and is still ``PENDING``; the
        candidate belongs to that lineage; and the candidate's linked attempt
        is ``SUCCEEDED`` (the only state that counts as "verified" here --
        it means every required gate, including quality validation, passed).
        ``selected_by`` is always persisted as the literal ``"validation"``:
        selection in this store is only ever performed once a candidate has
        passed quality validation. The validated target/EP/toolchain/
        environment-scope fields are nullable until a later slice actually
        supplies them; leaving them null never masquerades as a verified scope
        (see :attr:`CandidateAttemptRecord.has_fully_validated_selection_scope`).
        """
        parent_id = _coerce_str(parent_attempt_id, "parent_attempt_id")
        candidate_id_value = _coerce_str(candidate_attempt_id, "candidate_attempt_id")
        normalized_reason = _sanitize_reference(reason, path="selection_reason")
        selected_value = _ensure_utc(selected_utc or datetime.now(timezone.utc), "selected_utc")
        normalized_target_device = _coerce_optional_str(validated_target_device, "validated_target_device")
        normalized_target_ep = _coerce_optional_str(validated_target_ep, "validated_target_ep")
        normalized_toolchain_fingerprint = _coerce_optional_str(
            validated_toolchain_fingerprint, "validated_toolchain_fingerprint"
        )
        normalized_environment_scope = _coerce_optional_str(
            validated_environment_scope, "validated_environment_scope"
        )

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE;")
            lineage = self._load_candidate_lineage(connection, parent_id)
            if lineage is None:
                raise CandidateSelectionConflictError(
                    f"Parent attempt '{parent_id}' has no candidate lineage to select from."
                )
            if lineage.selection_state != CandidateLineageSelectionState.PENDING:
                raise CandidateSelectionConflictError(
                    f"Parent attempt '{parent_id}' lineage is already finalized "
                    f"('{lineage.selection_state.value}'); only one winner is ever allowed."
                )
            try:
                candidate = self._load_candidate_attempt(connection, candidate_id_value)
            except KeyError as exc:
                raise CandidateSelectionConflictError(
                    f"Candidate attempt '{candidate_id_value}' is unknown and cannot be selected."
                ) from exc
            if candidate.parent_attempt_id != parent_id:
                raise CandidateSelectionConflictError(
                    f"Candidate attempt '{candidate_id_value}' does not belong to parent '{parent_id}'."
                )
            if not candidate.is_verified:
                raise CandidateSelectionConflictError(
                    f"Candidate attempt '{candidate_id_value}' is not verified "
                    f"(linked attempt state is '{candidate.attempt_state.value}') and cannot be selected."
                )
            connection.execute(
                """
                UPDATE candidate_attempts
                SET selection_status = ?, selected_by = ?, selection_reason = ?, selected_utc = ?,
                    validated_target_device = ?, validated_target_ep = ?,
                    validated_toolchain_fingerprint = ?, validated_environment_scope = ?
                WHERE candidate_attempt_id = ?
                """,
                (
                    CandidateWinnerStatus.SELECTED.value,
                    "validation",
                    normalized_reason,
                    _format_datetime(selected_value),
                    normalized_target_device,
                    normalized_target_ep,
                    normalized_toolchain_fingerprint,
                    normalized_environment_scope,
                    candidate_id_value,
                ),
            )
            connection.execute(
                """
                UPDATE recipe_candidate_lineages
                SET selection_state = ?, selected_candidate_attempt_id = ?, selection_reason = ?,
                    finalized_utc = ?
                WHERE parent_attempt_id = ?
                """,
                (
                    CandidateLineageSelectionState.SELECTED.value,
                    candidate_id_value,
                    normalized_reason,
                    _format_datetime(selected_value),
                    parent_id,
                ),
            )
            refreshed_lineage = self._load_candidate_lineage(connection, parent_id)
            assert refreshed_lineage is not None
            refreshed_candidate = self._load_candidate_attempt(connection, candidate_id_value)
            return refreshed_lineage, refreshed_candidate

    def finalize_exhausted_candidate_lineage(
        self,
        parent_attempt_id: str,
        *,
        reason: str,
        finished_utc: datetime | None = None,
    ) -> RecipeCandidateLineage:
        """Atomically mark a lineage exhausted: no candidate is selected.

        Fails closed unless the lineage is still ``PENDING``, every registered
        candidate is terminal (its linked attempt is succeeded/failed/
        cancelled), and -- critically -- no candidate is verified-but-unselected
        (a verified candidate must be selected, never silently dropped by
        exhausting the lineage around it).
        """
        parent_id = _coerce_str(parent_attempt_id, "parent_attempt_id")
        normalized_reason = _sanitize_reference(reason, path="selection_reason")
        finished_value = _ensure_utc(finished_utc or datetime.now(timezone.utc), "finished_utc")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE;")
            lineage = self._load_candidate_lineage(connection, parent_id)
            if lineage is None:
                raise CandidateSelectionConflictError(
                    f"Parent attempt '{parent_id}' has no candidate lineage to exhaust."
                )
            if lineage.selection_state != CandidateLineageSelectionState.PENDING:
                raise CandidateSelectionConflictError(
                    f"Parent attempt '{parent_id}' lineage is already finalized "
                    f"('{lineage.selection_state.value}')."
                )
            candidates = self._list_candidate_attempts_locked(connection, parent_id)
            if not candidates:
                raise CandidateSelectionConflictError(
                    f"Parent attempt '{parent_id}' has no registered candidates to exhaust."
                )
            for row in candidates:
                if row.attempt_state not in TERMINAL_ATTEMPT_STATES:
                    raise CandidateSelectionConflictError(
                        f"Candidate '{row.candidate_attempt_id}' is still '{row.attempt_state.value}'; "
                        "a lineage cannot be exhausted while a candidate is not yet terminal."
                    )
                if row.is_verified:
                    raise CandidateSelectionConflictError(
                        f"Candidate '{row.candidate_attempt_id}' is verified but not selected; "
                        "select it instead of exhausting the lineage."
                    )
            connection.execute(
                """
                UPDATE recipe_candidate_lineages
                SET selection_state = ?, selected_candidate_attempt_id = NULL, selection_reason = ?,
                    finalized_utc = ?
                WHERE parent_attempt_id = ?
                """,
                (
                    CandidateLineageSelectionState.EXHAUSTED.value,
                    normalized_reason,
                    _format_datetime(finished_value),
                    parent_id,
                ),
            )
            refreshed = self._load_candidate_lineage(connection, parent_id)
            assert refreshed is not None
            return refreshed

    def list_candidate_attempts(self, parent_attempt_id: str) -> tuple[CandidateAttemptRecord, ...]:
        parent_id = _coerce_str(parent_attempt_id, "parent_attempt_id")
        with self._connect() as connection:
            return self._list_candidate_attempts_locked(connection, parent_id)

    def get_candidate_attempt(self, candidate_attempt_id: str) -> CandidateAttemptRecord:
        candidate_id_value = _coerce_str(candidate_attempt_id, "candidate_attempt_id")
        with self._connect() as connection:
            return self._load_candidate_attempt(connection, candidate_id_value)

    def get_candidate_lineage(self, parent_attempt_id: str) -> RecipeCandidateLineage | None:
        parent_id = _coerce_str(parent_attempt_id, "parent_attempt_id")
        with self._connect() as connection:
            return self._load_candidate_lineage(connection, parent_id)

    def find_reusable_candidate_selection(
        self,
        query: CandidateSelectionReuseQuery,
    ) -> CandidateAttemptRecord | None:
        """Read-only lookup of a previously-selected verified candidate winner
        by complete provenance identity. Never executes anything and never
        infers invocation counts: it only returns durable rows already
        persisted by :meth:`select_verified_candidate_attempt`.
        """
        normalized = _normalize_candidate_selection_reuse_query(query)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT winner.candidate_attempt_id
                FROM recipe_candidate_lineages AS lineage
                JOIN attempts AS parent_attempt ON parent_attempt.attempt_id = lineage.parent_attempt_id
                JOIN candidate_attempts AS winner
                    ON winner.candidate_attempt_id = lineage.selected_candidate_attempt_id
                WHERE lineage.selection_state = ?
                  AND parent_attempt.model_id = ?
                  AND parent_attempt.revision_sha = ?
                  AND parent_attempt.requested_device = ?
                  AND parent_attempt.requested_precision = ?
                  AND parent_attempt.compiler_version = ?
                  AND parent_attempt.capability_fingerprint = ?
                  AND parent_attempt.toolchain_fingerprint = ?
                  AND parent_attempt.profile_fingerprint = ?
                  AND lineage.quality_profile_fingerprint = ?
                  AND lineage.policy_fingerprint = ?
                ORDER BY lineage.finalized_utc DESC, winner.candidate_attempt_id DESC
                LIMIT 1
                """,
                (
                    CandidateLineageSelectionState.SELECTED.value,
                    normalized.model_id,
                    normalized.revision_sha,
                    normalized.requested_device,
                    normalized.requested_precision,
                    normalized.compiler_version,
                    normalized.capability_fingerprint,
                    normalized.toolchain_fingerprint,
                    normalized.profile_fingerprint,
                    normalized.quality_profile_fingerprint,
                    normalized.policy_fingerprint,
                ),
            ).fetchone()
            if row is None:
                return None
            return self._load_candidate_attempt(connection, row["candidate_attempt_id"])

    def _list_candidate_attempts_locked(
        self,
        connection: sqlite3.Connection,
        parent_attempt_id: str,
    ) -> tuple[CandidateAttemptRecord, ...]:
        rows = connection.execute(
            """
            SELECT candidate_attempt_id
            FROM candidate_attempts
            WHERE parent_attempt_id = ?
            ORDER BY candidate_index ASC
            """,
            (parent_attempt_id,),
        ).fetchall()
        return tuple(self._load_candidate_attempt(connection, row["candidate_attempt_id"]) for row in rows)

    def _finish_terminal_with_failure(
        self,
        *,
        attempt_id: str,
        target: AttemptState,
        failure: AttemptFailure,
        finished_utc: datetime | None,
    ) -> RecipeAttempt:
        attempt_id_value = _coerce_str(attempt_id, "attempt_id")
        if target not in {AttemptState.FAILED, AttemptState.CANCELLED}:
            raise ValueError("Terminal target must be failed or cancelled.")
        normalized_failure = _normalize_failure(failure)
        finished_value = _ensure_utc(finished_utc or datetime.now(timezone.utc), "finished_utc")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE;")
            attempt = self._load_attempt(connection, attempt_id_value)
            if attempt.state != AttemptState.RUNNING:
                raise AttemptStateTransitionError(
                    f"Attempt '{attempt_id_value}' cannot transition to '{target.value}' from '{attempt.state.value}'."
                )
            payload = serialize_attempt_failure(normalized_failure)
            _validate_record_payload(
                payload,
                record_def="attempt_failure",
                schema_root=self._schema(),
            )
            connection.execute(
                """
                UPDATE attempts
                SET state = ?, finished_utc = ?, failure_json = ?
                WHERE attempt_id = ?
                """,
                (
                    target.value,
                    _format_datetime(finished_value),
                    json.dumps(payload, separators=(",", ":")),
                    attempt_id_value,
                ),
            )
            return self._load_attempt(connection, attempt_id_value)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = self._connection_factory(
            str(self._db_path),
            timeout=30,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON;")
        connection.execute("PRAGMA busy_timeout=30000;")
        try:
            yield connection
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        else:
            if connection.in_transaction:
                connection.commit()
        finally:
            connection.close()

    def _schema(self) -> dict[str, object]:
        if self._schema_root is None:
            self._schema_root = load_recipe_attempt_schema(self._schema_path)
        return self._schema_root

    def _migrate(self, connection: sqlite3.Connection) -> None:
        current = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if current == 0 and self._table_exists(connection, "generated_recipes"):
            current = 1
        if current == 0:
            self._create_schema_v1(connection)
            connection.execute("PRAGMA user_version = 1;")
            current = 1
        if current < 2:
            self._migrate_v1_to_v2(connection)
            connection.execute("PRAGMA user_version = 2;")
            current = 2
        if current < 3:
            self._migrate_v2_to_v3(connection)
            connection.execute("PRAGMA user_version = 3;")
            current = 3
        if current != RECIPE_ATTEMPT_STORE_SCHEMA_VERSION:
            raise RecipeAttemptMigrationError(
                f"Unsupported recipe-attempt store schema version {current}; "
                f"expected {RECIPE_ATTEMPT_STORE_SCHEMA_VERSION}."
            )


    def _create_schema_v1(self, connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS generated_recipes (
                recipe_fingerprint TEXT PRIMARY KEY,
                schema_version TEXT NOT NULL,
                recipe_status TEXT NOT NULL,
                model_id TEXT NOT NULL,
                revision_sha TEXT NOT NULL,
                requested_device TEXT NOT NULL,
                requested_precision TEXT NOT NULL,
                compiler_version TEXT NOT NULL,
                capability_fingerprint TEXT NOT NULL,
                toolchain_fingerprint TEXT NOT NULL,
                canonical_json TEXT NOT NULL,
                created_utc TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS attempts (
                attempt_id TEXT PRIMARY KEY,
                idempotency_key TEXT NOT NULL UNIQUE,
                request_fingerprint TEXT NOT NULL,
                recipe_fingerprint TEXT NOT NULL,
                model_id TEXT NOT NULL,
                revision_sha TEXT NOT NULL,
                requested_device TEXT NOT NULL,
                requested_precision TEXT NOT NULL,
                compiler_version TEXT NOT NULL,
                capability_fingerprint TEXT NOT NULL,
                toolchain_fingerprint TEXT NOT NULL,
                created_utc TEXT NOT NULL,
                finished_utc TEXT,
                state TEXT NOT NULL,
                failure_json TEXT,
                FOREIGN KEY(recipe_fingerprint) REFERENCES generated_recipes(recipe_fingerprint)
            );

            CREATE TABLE IF NOT EXISTS attempt_gates (
                attempt_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                gate_name TEXT NOT NULL,
                status TEXT NOT NULL,
                evidence_ref TEXT NOT NULL,
                metrics_ref TEXT,
                started_utc TEXT NOT NULL,
                finished_utc TEXT NOT NULL,
                PRIMARY KEY (attempt_id, sequence),
                FOREIGN KEY(attempt_id) REFERENCES attempts(attempt_id) ON DELETE CASCADE
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_attempt_gates_gate_name
                ON attempt_gates (attempt_id, gate_name);
            CREATE INDEX IF NOT EXISTS idx_attempts_recipe_fingerprint
                ON attempts (recipe_fingerprint);

            CREATE TABLE IF NOT EXISTS verified_recipes (
                verified_fingerprint TEXT PRIMARY KEY,
                source_recipe_fingerprint TEXT NOT NULL,
                attempt_id TEXT NOT NULL UNIQUE,
                schema_version TEXT NOT NULL,
                model_id TEXT NOT NULL,
                revision_sha TEXT NOT NULL,
                requested_device TEXT NOT NULL,
                requested_precision TEXT NOT NULL,
                compiler_version TEXT NOT NULL,
                capability_fingerprint TEXT NOT NULL,
                toolchain_fingerprint TEXT NOT NULL,
                promoted_utc TEXT NOT NULL,
                promotion_evidence_json TEXT NOT NULL,
                canonical_json TEXT NOT NULL,
                FOREIGN KEY(source_recipe_fingerprint) REFERENCES generated_recipes(recipe_fingerprint),
                FOREIGN KEY(attempt_id) REFERENCES attempts(attempt_id)
            );
            """
        )

    def _migrate_v1_to_v2(self, connection: sqlite3.Connection) -> None:
        self._ensure_profile_fingerprint_column(
            connection,
            table_name="generated_recipes",
        )
        self._ensure_profile_fingerprint_column(
            connection,
            table_name="attempts",
        )
        self._ensure_profile_fingerprint_column(
            connection,
            table_name="verified_recipes",
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_verified_reuse_lookup
                ON verified_recipes (
                    model_id,
                    revision_sha,
                    requested_device,
                    requested_precision,
                    compiler_version,
                    capability_fingerprint,
                    toolchain_fingerprint,
                    profile_fingerprint,
                    promoted_utc DESC
                )
            """
        )

    def _migrate_v2_to_v3(self, connection: sqlite3.Connection) -> None:
        """Additive Slice 2 migration: new candidate-plan/selection tables only.

        No existing table (`generated_recipes`, `attempts`, `attempt_gates`,
        `verified_recipes`) is altered, recreated, or dropped. Legacy parent
        attempts created before this migration simply have no lineage row and
        no candidate rows, and behave exactly as before (no policy attached).
        Re-running this migration against an already-migrated database is a
        no-op because every statement uses `IF NOT EXISTS`.
        """
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS recipe_candidate_lineages (
                parent_attempt_id TEXT PRIMARY KEY,
                policy_id TEXT NOT NULL,
                policy_version TEXT NOT NULL,
                policy_fingerprint TEXT NOT NULL,
                policy_max_candidates INTEGER NOT NULL,
                quality_profile_fingerprint TEXT NOT NULL,
                created_utc TEXT NOT NULL,
                selection_state TEXT NOT NULL DEFAULT 'pending',
                selected_candidate_attempt_id TEXT,
                selection_reason TEXT,
                finalized_utc TEXT,
                FOREIGN KEY(parent_attempt_id) REFERENCES attempts(attempt_id)
            );

            CREATE TABLE IF NOT EXISTS candidate_attempts (
                candidate_attempt_id TEXT PRIMARY KEY,
                parent_attempt_id TEXT NOT NULL,
                attempt_id TEXT NOT NULL UNIQUE,
                candidate_index INTEGER NOT NULL,
                candidate_id TEXT NOT NULL,
                policy_id TEXT NOT NULL,
                policy_version TEXT NOT NULL,
                policy_fingerprint TEXT NOT NULL,
                quality_profile_fingerprint TEXT NOT NULL,
                quantization_override_block_size INTEGER,
                eligibility_trigger TEXT,
                disposition TEXT,
                disposition_reasons_json TEXT NOT NULL DEFAULT '[]',
                recipe_fingerprint TEXT NOT NULL,
                candidate_fingerprint TEXT NOT NULL,
                created_utc TEXT NOT NULL,
                artifact_ref TEXT,
                package_ref TEXT,
                mobius_build_invocation_count INTEGER,
                olive_optimize_invocation_count INTEGER,
                total_invocation_count INTEGER,
                wall_clock_seconds REAL,
                estimated_cost_usd REAL,
                selection_status TEXT NOT NULL DEFAULT 'not_selected',
                selected_by TEXT,
                selection_reason TEXT,
                selected_utc TEXT,
                validated_target_device TEXT,
                validated_target_ep TEXT,
                validated_toolchain_fingerprint TEXT,
                validated_environment_scope TEXT,
                UNIQUE (parent_attempt_id, candidate_index),
                UNIQUE (parent_attempt_id, candidate_id),
                FOREIGN KEY(parent_attempt_id) REFERENCES recipe_candidate_lineages(parent_attempt_id),
                FOREIGN KEY(attempt_id) REFERENCES attempts(attempt_id),
                FOREIGN KEY(recipe_fingerprint) REFERENCES generated_recipes(recipe_fingerprint)
            );

            CREATE INDEX IF NOT EXISTS idx_candidate_attempts_parent
                ON candidate_attempts (parent_attempt_id, candidate_index);

            CREATE INDEX IF NOT EXISTS idx_candidate_selection_reuse_lookup
                ON recipe_candidate_lineages (
                    policy_fingerprint,
                    quality_profile_fingerprint,
                    selection_state,
                    finalized_utc DESC
                );
            """
        )

    def _ensure_profile_fingerprint_column(
        self,
        connection: sqlite3.Connection,
        *,
        table_name: str,
    ) -> None:
        if not self._column_exists(connection, table_name, "profile_fingerprint"):
            connection.execute(
                f"""
                ALTER TABLE {table_name}
                ADD COLUMN profile_fingerprint TEXT NOT NULL
                DEFAULT '{LEGACY_PROFILE_FINGERPRINT}'
                """
            )
        connection.execute(
            f"""
            UPDATE {table_name}
            SET profile_fingerprint = ?
            WHERE profile_fingerprint IS NULL OR profile_fingerprint = ''
            """,
            (LEGACY_PROFILE_FINGERPRINT,),
        )

    def _recover_interrupted_attempts(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            """
            SELECT attempt_id
            FROM attempts
            WHERE state = ?
            """,
            (AttemptState.RUNNING.value,),
        ).fetchall()
        if not rows:
            return
        finished_utc = _format_datetime(datetime.now(timezone.utc))
        for row in rows:
            failure = AttemptFailure(
                classification=AttemptFailureClassification.INTERRUPTED,
                stage="attempt_store_recovery",
                message=(
                    "Attempt was running when the local service restarted and has been marked failed."
                ),
                evidence_refs=("store://recipe-attempt/restart-interrupted",),
                source_owner="recipe-attempt-store",
                next_action="Create a new idempotency key to retry with a fresh attempt.",
            )
            payload = json.dumps(
                serialize_attempt_failure(failure),
                separators=(",", ":"),
            )
            connection.execute(
                """
                UPDATE attempts
                SET state = ?, finished_utc = ?, failure_json = ?
                WHERE attempt_id = ?
                """,
                (
                    AttemptState.FAILED.value,
                    finished_utc,
                    payload,
                    row["attempt_id"],
                ),
            )

    def _table_exists(self, connection: sqlite3.Connection, table_name: str) -> bool:
        row = connection.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table' AND name = ?
            LIMIT 1
            """,
            (table_name,),
        ).fetchone()
        return row is not None

    def _column_exists(self, connection: sqlite3.Connection, table_name: str, column_name: str) -> bool:
        columns = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        return any(row["name"] == column_name for row in columns)

    def _generated_record_from_row(self, row: sqlite3.Row) -> GeneratedRecipeRecord:
        payload = {
            "record_type": "generated_recipe_record",
            "recipe_fingerprint": row["recipe_fingerprint"],
            "schema_version": row["schema_version"],
            "recipe_status": row["recipe_status"],
            "model_id": row["model_id"],
            "revision_sha": row["revision_sha"],
            "requested_device": row["requested_device"],
            "requested_precision": row["requested_precision"],
            "compiler_version": row["compiler_version"],
            "capability_fingerprint": row["capability_fingerprint"],
            "toolchain_fingerprint": row["toolchain_fingerprint"],
            "profile_fingerprint": row["profile_fingerprint"],
            "canonical_json": row["canonical_json"],
            "created_utc": row["created_utc"],
        }
        return deserialize_generated_recipe_record(payload, schema_root=self._schema())

    def _load_attempt(self, connection: sqlite3.Connection, attempt_id: str) -> RecipeAttempt:
        row = connection.execute(
            """
            SELECT attempt_id, idempotency_key, request_fingerprint, recipe_fingerprint, model_id,
                   revision_sha, requested_device, requested_precision, compiler_version,
                   capability_fingerprint, toolchain_fingerprint, profile_fingerprint,
                   created_utc, finished_utc, state, failure_json
            FROM attempts
            WHERE attempt_id = ?
            """,
            (attempt_id,),
        ).fetchone()
        if row is None:
            raise KeyError(attempt_id)

        gate_rows = connection.execute(
            """
            SELECT sequence, gate_name, status, evidence_ref, metrics_ref, started_utc, finished_utc
            FROM attempt_gates
            WHERE attempt_id = ?
            ORDER BY sequence ASC
            """,
            (attempt_id,),
        ).fetchall()
        failure_payload: dict[str, object] | None = None
        if row["failure_json"]:
            decoded = json.loads(row["failure_json"])
            if not isinstance(decoded, dict):
                raise RecipeAttemptSchemaError("failure_json must be a JSON object.")
            failure_payload = decoded

        payload = {
            "record_type": "recipe_attempt",
            "attempt_id": row["attempt_id"],
            "idempotency_key": row["idempotency_key"],
            "request_fingerprint": row["request_fingerprint"],
            "recipe_fingerprint": row["recipe_fingerprint"],
            "model_id": row["model_id"],
            "revision_sha": row["revision_sha"],
            "requested_device": row["requested_device"],
            "requested_precision": row["requested_precision"],
            "compiler_version": row["compiler_version"],
            "capability_fingerprint": row["capability_fingerprint"],
            "toolchain_fingerprint": row["toolchain_fingerprint"],
            "profile_fingerprint": row["profile_fingerprint"],
            "created_utc": row["created_utc"],
            "finished_utc": row["finished_utc"],
            "state": row["state"],
            "gate_results": [
                {
                    "sequence": gate["sequence"],
                    "gate": gate["gate_name"],
                    "status": gate["status"],
                    "evidence_ref": gate["evidence_ref"],
                    "metrics_ref": gate["metrics_ref"],
                    "started_utc": gate["started_utc"],
                    "finished_utc": gate["finished_utc"],
                }
                for gate in gate_rows
            ],
            "failure": failure_payload,
        }
        return deserialize_recipe_attempt(payload, schema_root=self._schema())

    def _verified_record_from_row(self, row: sqlite3.Row) -> VerifiedRecipeRecord:
        decoded = json.loads(row["promotion_evidence_json"])
        if not isinstance(decoded, dict):
            raise RecipeAttemptSchemaError("promotion_evidence_json must be a JSON object.")
        payload = {
            "record_type": "verified_recipe_record",
            "verified_fingerprint": row["verified_fingerprint"],
            "source_recipe_fingerprint": row["source_recipe_fingerprint"],
            "attempt_id": row["attempt_id"],
            "schema_version": row["schema_version"],
            "model_id": row["model_id"],
            "revision_sha": row["revision_sha"],
            "requested_device": row["requested_device"],
            "requested_precision": row["requested_precision"],
            "compiler_version": row["compiler_version"],
            "capability_fingerprint": row["capability_fingerprint"],
            "toolchain_fingerprint": row["toolchain_fingerprint"],
            "profile_fingerprint": row["profile_fingerprint"],
            "promoted_utc": row["promoted_utc"],
            "promotion_evidence": decoded,
            "canonical_json": row["canonical_json"],
        }
        return deserialize_verified_recipe_record(payload, schema_root=self._schema())

    def _load_candidate_attempt(
        self,
        connection: sqlite3.Connection,
        candidate_attempt_id: str,
    ) -> CandidateAttemptRecord:
        row = connection.execute(
            """
            SELECT candidate_attempt_id, parent_attempt_id, attempt_id, candidate_index, candidate_id,
                   policy_id, policy_version, policy_fingerprint, quality_profile_fingerprint,
                   quantization_override_block_size, eligibility_trigger, disposition,
                   disposition_reasons_json, recipe_fingerprint, candidate_fingerprint, created_utc,
                   artifact_ref, package_ref,
                   mobius_build_invocation_count, olive_optimize_invocation_count,
                   total_invocation_count, wall_clock_seconds, estimated_cost_usd,
                   selection_status, selected_by, selection_reason, selected_utc,
                   validated_target_device, validated_target_ep, validated_toolchain_fingerprint,
                   validated_environment_scope
            FROM candidate_attempts
            WHERE candidate_attempt_id = ?
            """,
            (candidate_attempt_id,),
        ).fetchone()
        if row is None:
            raise KeyError(candidate_attempt_id)
        attempt_state_row = connection.execute(
            "SELECT state FROM attempts WHERE attempt_id = ?",
            (row["attempt_id"],),
        ).fetchone()
        if attempt_state_row is None:
            raise RecipeAttemptSchemaError(
                f"Candidate attempt '{candidate_attempt_id}' references a missing linked attempt."
            )
        disposition_reasons = json.loads(row["disposition_reasons_json"])
        if not isinstance(disposition_reasons, list):
            raise RecipeAttemptSchemaError("disposition_reasons_json must be a JSON array.")
        payload = {
            "record_type": "candidate_attempt_record",
            "candidate_attempt_id": row["candidate_attempt_id"],
            "parent_attempt_id": row["parent_attempt_id"],
            "attempt_id": row["attempt_id"],
            "candidate_index": row["candidate_index"],
            "candidate_id": row["candidate_id"],
            "policy_id": row["policy_id"],
            "policy_version": row["policy_version"],
            "policy_fingerprint": row["policy_fingerprint"],
            "quality_profile_fingerprint": row["quality_profile_fingerprint"],
            "quantization_override_block_size": row["quantization_override_block_size"],
            "eligibility_trigger": row["eligibility_trigger"],
            "disposition": row["disposition"],
            "disposition_reasons": disposition_reasons,
            "recipe_fingerprint": row["recipe_fingerprint"],
            "candidate_fingerprint": row["candidate_fingerprint"],
            "created_utc": row["created_utc"],
            "attempt_state": attempt_state_row["state"],
            "artifact_ref": row["artifact_ref"],
            "package_ref": row["package_ref"],
            "mobius_build_invocation_count": row["mobius_build_invocation_count"],
            "olive_optimize_invocation_count": row["olive_optimize_invocation_count"],
            "total_invocation_count": row["total_invocation_count"],
            "wall_clock_seconds": row["wall_clock_seconds"],
            "estimated_cost_usd": row["estimated_cost_usd"],
            "selection_status": row["selection_status"],
            "selected_by": row["selected_by"],
            "selection_reason": row["selection_reason"],
            "selected_utc": row["selected_utc"],
            "validated_target_device": row["validated_target_device"],
            "validated_target_ep": row["validated_target_ep"],
            "validated_toolchain_fingerprint": row["validated_toolchain_fingerprint"],
            "validated_environment_scope": row["validated_environment_scope"],
        }
        return deserialize_candidate_attempt_record(payload, schema_root=self._schema())

    def _load_candidate_lineage(
        self,
        connection: sqlite3.Connection,
        parent_attempt_id: str,
    ) -> RecipeCandidateLineage | None:
        row = connection.execute(
            """
            SELECT parent_attempt_id, policy_id, policy_version, policy_fingerprint,
                   policy_max_candidates, quality_profile_fingerprint, created_utc,
                   selection_state, selected_candidate_attempt_id, selection_reason, finalized_utc
            FROM recipe_candidate_lineages
            WHERE parent_attempt_id = ?
            """,
            (parent_attempt_id,),
        ).fetchone()
        if row is None:
            return None
        payload = {
            "record_type": "recipe_candidate_lineage_record",
            "parent_attempt_id": row["parent_attempt_id"],
            "policy_id": row["policy_id"],
            "policy_version": row["policy_version"],
            "policy_fingerprint": row["policy_fingerprint"],
            "policy_max_candidates": row["policy_max_candidates"],
            "quality_profile_fingerprint": row["quality_profile_fingerprint"],
            "created_utc": row["created_utc"],
            "selection_state": row["selection_state"],
            "selected_candidate_attempt_id": row["selected_candidate_attempt_id"],
            "selection_reason": row["selection_reason"],
            "finalized_utc": row["finalized_utc"],
        }
        return deserialize_recipe_candidate_lineage(payload, schema_root=self._schema())


@dataclass(frozen=True)
class _GeneratedIdentity:
    model_id: str
    revision_sha: str
    requested_device: str
    requested_precision: str
    compiler_version: str
    capability_fingerprint: str
    toolchain_fingerprint: str
    profile_fingerprint: str


def _identity_from_generated_recipe(recipe: GeneratedRecipe) -> _GeneratedIdentity:
    model_id_recipe = _coerce_str(recipe.recipe.huggingface_model_id, "recipe.recipe.huggingface_model_id")
    model_id_input = _coerce_str(
        recipe.provenance.input_metadata.model_id,
        "recipe.provenance.input_metadata.model_id",
    )
    if model_id_recipe != model_id_input:
        raise RecipeAttemptSchemaError(
            "Generated recipe model identity mismatch between recipe and provenance metadata."
        )
    revision_sha = _normalize_hex(recipe.pinned_revision, expected_len=40, path="recipe.pinned_revision")
    input_revision = _normalize_hex(
        recipe.provenance.input_metadata.revision_sha,
        expected_len=40,
        path="recipe.provenance.input_metadata.revision_sha",
    )
    if revision_sha != input_revision:
        raise RecipeAttemptSchemaError(
            "Generated recipe pinned revision mismatch between recipe and provenance metadata."
        )
    return _GeneratedIdentity(
        model_id=model_id_recipe,
        revision_sha=revision_sha,
        requested_device=_coerce_str(
            recipe.provenance.input_metadata.requested_device,
            "recipe.provenance.input_metadata.requested_device",
        ).lower(),
        requested_precision=_coerce_str(
            recipe.provenance.input_metadata.requested_precision,
            "recipe.provenance.input_metadata.requested_precision",
        ).lower(),
        compiler_version=_coerce_str(
            recipe.provenance.compiler_version,
            "recipe.provenance.compiler_version",
        ),
        capability_fingerprint=build_capability_fingerprint(recipe.provenance),
        toolchain_fingerprint=build_toolchain_fingerprint(recipe.provenance.toolchain),
        profile_fingerprint=build_profile_fingerprint(recipe.recipe, recipe.provenance.input_metadata),
    )


def _normalize_attempt_request(request: RecipeAttemptRequest) -> RecipeAttemptRequest:
    return RecipeAttemptRequest(
        recipe_fingerprint=_normalize_hex(
            _coerce_str(request.recipe_fingerprint, "request.recipe_fingerprint"),
            expected_len=64,
            path="request.recipe_fingerprint",
        ),
        model_id=_coerce_str(request.model_id, "request.model_id"),
        revision_sha=_normalize_hex(
            _coerce_str(request.revision_sha, "request.revision_sha"),
            expected_len=40,
            path="request.revision_sha",
        ),
        requested_device=_coerce_str(request.requested_device, "request.requested_device").lower(),
        requested_precision=_coerce_str(request.requested_precision, "request.requested_precision").lower(),
        compiler_version=_coerce_str(request.compiler_version, "request.compiler_version"),
        capability_fingerprint=_normalize_hex(
            _coerce_str(request.capability_fingerprint, "request.capability_fingerprint"),
            expected_len=64,
            path="request.capability_fingerprint",
        ),
        toolchain_fingerprint=_normalize_hex(
            _coerce_str(request.toolchain_fingerprint, "request.toolchain_fingerprint"),
            expected_len=64,
            path="request.toolchain_fingerprint",
        ),
        profile_fingerprint=_normalize_hex(
            _coerce_str(request.profile_fingerprint, "request.profile_fingerprint"),
            expected_len=64,
            path="request.profile_fingerprint",
        ),
    )


def _normalize_reuse_query(query: RecipeReuseQuery) -> RecipeReuseQuery:
    return RecipeReuseQuery(
        model_id=_coerce_str(query.model_id, "query.model_id"),
        revision_sha=_normalize_hex(
            _coerce_str(query.revision_sha, "query.revision_sha"),
            expected_len=40,
            path="query.revision_sha",
        ),
        requested_device=_coerce_str(query.requested_device, "query.requested_device").lower(),
        requested_precision=_coerce_str(query.requested_precision, "query.requested_precision").lower(),
        compiler_version=_coerce_str(query.compiler_version, "query.compiler_version"),
        capability_fingerprint=_normalize_hex(
            _coerce_str(query.capability_fingerprint, "query.capability_fingerprint"),
            expected_len=64,
            path="query.capability_fingerprint",
        ),
        toolchain_fingerprint=_normalize_hex(
            _coerce_str(query.toolchain_fingerprint, "query.toolchain_fingerprint"),
            expected_len=64,
            path="query.toolchain_fingerprint",
        ),
        profile_fingerprint=_normalize_hex(
            _coerce_str(query.profile_fingerprint, "query.profile_fingerprint"),
            expected_len=64,
            path="query.profile_fingerprint",
        ),
    )


def _normalize_candidate_selection_reuse_query(
    query: CandidateSelectionReuseQuery,
) -> CandidateSelectionReuseQuery:
    return CandidateSelectionReuseQuery(
        model_id=_coerce_str(query.model_id, "query.model_id"),
        revision_sha=_normalize_hex(
            _coerce_str(query.revision_sha, "query.revision_sha"),
            expected_len=40,
            path="query.revision_sha",
        ),
        requested_device=_coerce_str(query.requested_device, "query.requested_device").lower(),
        requested_precision=_coerce_str(query.requested_precision, "query.requested_precision").lower(),
        compiler_version=_coerce_str(query.compiler_version, "query.compiler_version"),
        capability_fingerprint=_normalize_hex(
            _coerce_str(query.capability_fingerprint, "query.capability_fingerprint"),
            expected_len=64,
            path="query.capability_fingerprint",
        ),
        toolchain_fingerprint=_normalize_hex(
            _coerce_str(query.toolchain_fingerprint, "query.toolchain_fingerprint"),
            expected_len=64,
            path="query.toolchain_fingerprint",
        ),
        profile_fingerprint=_normalize_hex(
            _coerce_str(query.profile_fingerprint, "query.profile_fingerprint"),
            expected_len=64,
            path="query.profile_fingerprint",
        ),
        quality_profile_fingerprint=_normalize_hex(
            _coerce_str(query.quality_profile_fingerprint, "query.quality_profile_fingerprint"),
            expected_len=64,
            path="query.quality_profile_fingerprint",
        ),
        policy_fingerprint=_normalize_hex(
            _coerce_str(query.policy_fingerprint, "query.policy_fingerprint"),
            expected_len=64,
            path="query.policy_fingerprint",
        ),
    )


def _assert_attempt_request_matches_generated(
    request: RecipeAttemptRequest,
    record: GeneratedRecipeRecord,
) -> None:
    normalized = _normalize_attempt_request(request)
    if normalized.recipe_fingerprint != record.recipe_fingerprint:
        raise RecipeAttemptStoreError("Attempt request recipe fingerprint does not match generated record.")
    mismatches: list[str] = []
    for field_name in (
        "model_id",
        "revision_sha",
        "requested_device",
        "requested_precision",
        "compiler_version",
        "capability_fingerprint",
        "toolchain_fingerprint",
        "profile_fingerprint",
    ):
        if getattr(normalized, field_name) != getattr(record, field_name):
            mismatches.append(field_name)
    if mismatches:
        raise RecipeAttemptStoreError(
            "Attempt request identity mismatch against generated recipe for field(s): "
            + ", ".join(mismatches)
        )


def _assert_attempt_identity_matches_generated(
    attempt: RecipeAttempt,
    identity: _GeneratedIdentity,
) -> None:
    comparisons = {
        "model_id": (attempt.model_id, identity.model_id),
        "revision_sha": (attempt.revision_sha, identity.revision_sha),
        "requested_device": (attempt.requested_device, identity.requested_device),
        "requested_precision": (attempt.requested_precision, identity.requested_precision),
        "compiler_version": (attempt.compiler_version, identity.compiler_version),
        "capability_fingerprint": (attempt.capability_fingerprint, identity.capability_fingerprint),
        "toolchain_fingerprint": (attempt.toolchain_fingerprint, identity.toolchain_fingerprint),
        "profile_fingerprint": (attempt.profile_fingerprint, identity.profile_fingerprint),
    }
    mismatched = [name for name, values in comparisons.items() if values[0] != values[1]]
    if mismatched:
        raise RecipePromotionConflictError(
            "Promotion identity mismatch between attempt and generated recipe for field(s): "
            + ", ".join(mismatched)
        )


def _validate_attempt_gate_monotonicity(gates: tuple[AttemptGateResult, ...]) -> None:
    expected_sequence = 1
    for row in gates:
        if row.sequence != expected_sequence:
            raise RecipeAttemptSchemaError(
                f"Attempt gate sequence must be monotonic. Expected {expected_sequence}, got {row.sequence}."
            )
        expected_sequence += 1


def _require_complete_successful_gates(gates: tuple[AttemptGateResult, ...]) -> None:
    _validate_attempt_gate_monotonicity(gates)
    if len(gates) != len(ATTEMPT_GATE_ORDER):
        raise AttemptStateTransitionError(
            "Attempt must record the full ordered gate sequence before succeeding."
        )
    for expected, actual in zip(ATTEMPT_GATE_ORDER, gates, strict=True):
        if actual.gate != expected:
            raise AttemptStateTransitionError(
                f"Attempt gate order mismatch. Expected '{expected.value}', got '{actual.gate.value}'."
            )
        if actual.status != AttemptGateStatus.PASSED:
            raise AttemptStateTransitionError(
                f"Attempt gate '{actual.gate.value}' did not pass and cannot promote or succeed."
            )


def _normalize_failure(failure: AttemptFailure) -> AttemptFailure:
    evidence = tuple(
        _sanitize_reference(row, path=f"attempt_failure.evidence_refs[{idx}]")
        for idx, row in enumerate(failure.evidence_refs, start=1)
    )
    if len(evidence) > _MAX_FAILURE_EVIDENCE_REFS:
        raise RecipeAttemptSecurityError(
            f"attempt_failure.evidence_refs cannot exceed {_MAX_FAILURE_EVIDENCE_REFS} entries."
        )
    normalized = AttemptFailure(
        classification=failure.classification,
        stage=_coerce_str(failure.stage, "attempt_failure.stage"),
        message=_sanitize_message(
            _coerce_str(failure.message, "attempt_failure.message"),
            path="attempt_failure.message",
        ),
        evidence_refs=evidence,
        source_owner=_coerce_str(failure.source_owner, "attempt_failure.source_owner"),
        next_action=_coerce_str(failure.next_action, "attempt_failure.next_action"),
    )
    _ = deserialize_attempt_failure(serialize_attempt_failure(normalized))
    return normalized


def _normalize_promotion_gate_evidence(gates: PromotionGateEvidence) -> PromotionGateEvidence:
    normalized: dict[str, PromotionGateCheck] = {}
    for gate in ATTEMPT_GATE_ORDER:
        check = getattr(gates, gate.value)
        if not check.passed:
            raise RecipePromotionConflictError(
                f"Promotion gate '{gate.value}' must pass."
            )
        evidence = _sanitize_reference(check.evidence, path=f"promotion_evidence.{gate.value}.evidence")
        metrics_ref = _sanitize_optional_reference(
            check.metrics_ref,
            path=f"promotion_evidence.{gate.value}.metrics_ref",
        )
        normalized[gate.value] = PromotionGateCheck(
            passed=True,
            evidence=evidence,
            metrics_ref=metrics_ref,
        )
    return PromotionGateEvidence(
        mobius_build=normalized[AttemptGate.MOBIUS_BUILD.value],
        olive_optimize=normalized[AttemptGate.OLIVE_OPTIMIZE.value],
        onnx_validation=normalized[AttemptGate.ONNX_VALIDATION.value],
        ort_validation=normalized[AttemptGate.ORT_VALIDATION.value],
        oga_validation=normalized[AttemptGate.OGA_VALIDATION.value],
        fl_sdk_inference=normalized[AttemptGate.FL_SDK_INFERENCE.value],
        quality_validation=normalized[AttemptGate.QUALITY_VALIDATION.value],
    )


def _assert_promotion_evidence_matches_attempt(
    *,
    attempt: RecipeAttempt,
    promotion_evidence: PromotionGateEvidence,
) -> None:
    gate_by_name = {row.gate: row for row in attempt.gate_results}
    for gate in ATTEMPT_GATE_ORDER:
        if gate not in gate_by_name:
            raise RecipePromotionConflictError(
                f"Attempt is missing required gate '{gate.value}'."
            )
        attempt_gate = gate_by_name[gate]
        if attempt_gate.status != AttemptGateStatus.PASSED:
            raise RecipePromotionConflictError(
                f"Attempt gate '{gate.value}' did not pass."
            )
        evidence = getattr(promotion_evidence, gate.value).evidence
        if attempt_gate.evidence_ref != evidence:
            raise RecipePromotionConflictError(
                f"Promotion evidence mismatch for gate '{gate.value}'."
            )
        expected_metrics_ref = getattr(promotion_evidence, gate.value).metrics_ref
        if expected_metrics_ref is not None and attempt_gate.metrics_ref != expected_metrics_ref:
            raise RecipePromotionConflictError(
                f"Promotion metrics evidence mismatch for gate '{gate.value}'."
            )


def _promotion_evidence_to_payload(gates: PromotionGateEvidence) -> dict[str, object]:
    return {
        "mobius_build": _promotion_gate_to_payload(gates.mobius_build),
        "olive_optimize": _promotion_gate_to_payload(gates.olive_optimize),
        "onnx_validation": _promotion_gate_to_payload(gates.onnx_validation),
        "ort_validation": _promotion_gate_to_payload(gates.ort_validation),
        "oga_validation": _promotion_gate_to_payload(gates.oga_validation),
        "fl_sdk_inference": _promotion_gate_to_payload(gates.fl_sdk_inference),
        "quality_validation": _promotion_gate_to_payload(gates.quality_validation),
    }


def _promotion_evidence_from_payload(payload: dict[str, object]) -> PromotionGateEvidence:
    return _normalize_promotion_gate_evidence(
        PromotionGateEvidence(
            mobius_build=_promotion_gate_from_payload(payload, AttemptGate.MOBIUS_BUILD),
            olive_optimize=_promotion_gate_from_payload(payload, AttemptGate.OLIVE_OPTIMIZE),
            onnx_validation=_promotion_gate_from_payload(payload, AttemptGate.ONNX_VALIDATION),
            ort_validation=_promotion_gate_from_payload(payload, AttemptGate.ORT_VALIDATION),
            oga_validation=_promotion_gate_from_payload(payload, AttemptGate.OGA_VALIDATION),
            fl_sdk_inference=_promotion_gate_from_payload(payload, AttemptGate.FL_SDK_INFERENCE),
            quality_validation=_promotion_gate_from_payload(payload, AttemptGate.QUALITY_VALIDATION),
        )
    )


def _promotion_gate_from_payload(payload: dict[str, object], gate: AttemptGate) -> PromotionGateCheck:
    check = _coerce_schema_object(payload.get(gate.value), f"promotion_evidence.{gate.value}")
    passed = _coerce_bool(check.get("passed"), f"promotion_evidence.{gate.value}.passed")
    if not passed:
        raise RecipePromotionConflictError(f"Promotion gate '{gate.value}' must pass.")
    evidence = _sanitize_reference(
        _coerce_str(check.get("evidence"), f"promotion_evidence.{gate.value}.evidence"),
        path=f"promotion_evidence.{gate.value}.evidence",
    )
    metrics_ref = _sanitize_optional_reference(
        check.get("metrics_ref"),
        path=f"promotion_evidence.{gate.value}.metrics_ref",
    )
    return PromotionGateCheck(passed=True, evidence=evidence, metrics_ref=metrics_ref)


def _promotion_gate_to_payload(gate: PromotionGateCheck) -> dict[str, object]:
    payload: dict[str, object] = {
        "passed": gate.passed,
        "evidence": gate.evidence,
    }
    if gate.metrics_ref is not None:
        payload["metrics_ref"] = gate.metrics_ref
    return payload


def _validate_record_payload(
    payload: dict[str, object],
    *,
    record_def: str,
    schema_root: dict[str, object] | None = None,
) -> None:
    schema = schema_root or load_recipe_attempt_schema()
    if "$defs" not in schema:
        raise RecipeAttemptSchemaError("Recipe-attempt schema must define $defs.")
    _validate_json_value(
        payload,
        schema={"$ref": f"#/$defs/{record_def}"},
        schema_root=schema,
        path="$",
    )


def _validate_canonical_recipe_payload(
    *,
    canonical_json: str,
    expected_fingerprint: str,
    path: str,
) -> None:
    parsed = json.loads(canonical_json)
    if not isinstance(parsed, dict):
        raise RecipeAttemptSchemaError(f"{path} must encode a JSON object.")
    fingerprint = _coerce_str(parsed.get("fingerprint"), f"{path}.fingerprint")
    normalized = _normalize_hex(fingerprint, expected_len=64, path=f"{path}.fingerprint")
    if normalized != expected_fingerprint:
        raise RecipeAttemptSchemaError(
            f"{path}.fingerprint does not match expected immutable fingerprint."
        )


def _sanitize_optional_reference(value: object, *, path: str) -> str | None:
    if value is None:
        return None
    return _sanitize_reference(_coerce_str(value, path), path=path)


def _sanitize_reference(value: str, *, path: str) -> str:
    normalized = _coerce_str(value, path)
    if len(normalized) > _MAX_REFERENCE_LENGTH:
        raise RecipeAttemptSecurityError(
            f"{path} exceeds {_MAX_REFERENCE_LENGTH} characters; large logs/artifacts are not persisted."
        )
    if "\n" in normalized or "\r" in normalized:
        raise RecipeAttemptSecurityError(
            f"{path} must be a compact structured reference, not multiline log content."
        )
    if _contains_secret_like_token(normalized):
        raise RecipeAttemptSecurityError(
            f"{path} contains token-like credentials and cannot be persisted."
        )
    if _contains_private_absolute_path(normalized):
        raise RecipeAttemptSecurityError(
            f"{path} contains an unsafe absolute private path and cannot be persisted."
        )
    return normalized


def _sanitize_message(value: str, *, path: str) -> str:
    normalized = _coerce_str(value, path)
    if len(normalized) > _MAX_MESSAGE_LENGTH:
        raise RecipeAttemptSecurityError(
            f"{path} exceeds {_MAX_MESSAGE_LENGTH} characters; raw unbounded logs are not persisted."
        )
    redacted = _SECRET_INLINE_RE.sub(lambda match: f"{match.group(1)}=[REDACTED]", normalized)
    if _contains_secret_like_token(redacted):
        raise RecipeAttemptSecurityError(
            f"{path} still contains token-like credentials after redaction."
        )
    if _contains_private_absolute_path(redacted):
        raise RecipeAttemptSecurityError(
            f"{path} contains an unsafe absolute private path and cannot be persisted."
        )
    return redacted


def _contains_secret_like_token(value: str) -> bool:
    return _HF_TOKEN_RE.search(value) is not None or _BEARER_TOKEN_RE.search(value) is not None


def _contains_private_absolute_path(value: str) -> bool:
    candidate = value.strip()
    if candidate.lower().startswith("file://"):
        candidate = candidate[7:]
    if _WINDOWS_ABS_RE.match(candidate):
        return True
    if candidate.startswith("/"):
        return True
    return False


def _canonical_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _coerce_bool(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise RecipeAttemptSchemaError(f"{path} must be a boolean.")
    return value


def _coerce_int(value: object, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise RecipeAttemptSchemaError(f"{path} must be an integer.")
    return value


def _coerce_optional_int(value: object, path: str) -> int | None:
    if value is None:
        return None
    return _coerce_int(value, path)


def _coerce_optional_float(value: object, path: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecipeAttemptSchemaError(f"{path} must be a number or null.")
    return float(value)


def _coerce_str(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecipeAttemptSchemaError(f"{path} must be a non-empty string.")
    return value.strip()


def _coerce_optional_str(value: object, path: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RecipeAttemptSchemaError(f"{path} must be a string or null.")
    stripped = value.strip()
    if not stripped:
        raise RecipeAttemptSchemaError(f"{path} cannot be empty when provided.")
    return stripped


def _normalize_hex(value: str, *, expected_len: int, path: str) -> str:
    lowered = _coerce_str(value, path).lower()
    if expected_len == 40 and _HEX40_RE.fullmatch(lowered) is None:
        raise RecipeAttemptSchemaError(f"{path} must be a 40-character lowercase hex string.")
    if expected_len == 64 and _HEX64_RE.fullmatch(lowered) is None:
        raise RecipeAttemptSchemaError(f"{path} must be a 64-character lowercase hex string.")
    if len(lowered) != expected_len:
        raise RecipeAttemptSchemaError(f"{path} must be exactly {expected_len} characters.")
    return lowered


def _ensure_utc(value: datetime, path: str) -> datetime:
    if value.tzinfo is None:
        raise RecipeAttemptSchemaError(f"{path} must include timezone information.")
    return value.astimezone(timezone.utc)


def _format_datetime(value: datetime) -> str:
    return _ensure_utc(value, "datetime").isoformat()


def _parse_datetime(value: str, path: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise RecipeAttemptSchemaError(f"{path} must be ISO-8601 datetime.") from exc
    if parsed.tzinfo is None:
        raise RecipeAttemptSchemaError(f"{path} must include timezone information.")
    return parsed.astimezone(timezone.utc)


def _parse_optional_datetime(value: object, path: str) -> datetime | None:
    normalized = _coerce_optional_str(value, path)
    if normalized is None:
        return None
    return _parse_datetime(normalized, path)


def _validate_json_value(
    value: object,
    *,
    schema: dict[str, object],
    schema_root: dict[str, object],
    path: str,
) -> None:
    if "$ref" in schema:
        ref = _coerce_str(schema["$ref"], f"{path}.$ref")
        resolved = _resolve_schema_ref(schema_root, ref)
        _validate_json_value(value, schema=resolved, schema_root=schema_root, path=path)
        return

    if "allOf" in schema:
        rows = _coerce_schema_array(schema["allOf"], f"{path}.allOf")
        for idx, row in enumerate(rows, start=1):
            child = _coerce_schema_object(row, f"{path}.allOf[{idx}]")
            _validate_json_value(value, schema=child, schema_root=schema_root, path=path)

    if "type" in schema:
        _validate_type(value, schema["type"], path=path)

    if "const" in schema and value != schema["const"]:
        raise RecipeAttemptSchemaError(f"{path} must equal constant value '{schema['const']}'.")

    if "enum" in schema:
        enum_values = _coerce_schema_array(schema["enum"], f"{path}.enum")
        if value not in enum_values:
            raise RecipeAttemptSchemaError(
                f"{path} must be one of {enum_values}; got '{value}'."
            )

    if "pattern" in schema and isinstance(value, str):
        pattern = _coerce_str(schema["pattern"], f"{path}.pattern")
        if re.fullmatch(pattern, value) is None:
            raise RecipeAttemptSchemaError(f"{path} must match pattern '{pattern}'.")

    if isinstance(value, dict):
        required = _coerce_optional_schema_array(schema.get("required"), f"{path}.required")
        for key in required:
            if not isinstance(key, str):
                raise RecipeAttemptSchemaError(f"{path}.required must contain only strings.")
            if key not in value:
                raise RecipeAttemptSchemaError(f"{path} is missing required key '{key}'.")
        properties = _coerce_optional_schema_object(schema.get("properties"), f"{path}.properties")
        additional_properties = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in properties:
                child = _coerce_schema_object(properties[key], f"{path}.properties.{key}")
                _validate_json_value(
                    item,
                    schema=child,
                    schema_root=schema_root,
                    path=f"{path}.{key}",
                )
                continue
            if additional_properties is False:
                raise RecipeAttemptSchemaError(f"{path} does not allow additional key '{key}'.")

    if isinstance(value, list):
        min_items = schema.get("minItems")
        if isinstance(min_items, int) and len(value) < min_items:
            raise RecipeAttemptSchemaError(f"{path} must have at least {min_items} items.")
        if "items" in schema:
            item_schema = _coerce_schema_object(schema["items"], f"{path}.items")
            for idx, item in enumerate(value, start=1):
                _validate_json_value(
                    item,
                    schema=item_schema,
                    schema_root=schema_root,
                    path=f"{path}[{idx}]",
                )


def _validate_type(value: object, schema_type: object, *, path: str) -> None:
    if isinstance(schema_type, str):
        if not _value_matches_type(value, schema_type):
            raise RecipeAttemptSchemaError(f"{path} must be of type '{schema_type}'.")
        return
    if isinstance(schema_type, list):
        if any(_value_matches_type(value, item) for item in schema_type if isinstance(item, str)):
            return
        raise RecipeAttemptSchemaError(f"{path} does not match any allowed type {schema_type}.")
    raise RecipeAttemptSchemaError(f"{path}.type must be a string or an array of strings.")


def _value_matches_type(value: object, schema_type: str) -> bool:
    if schema_type == "object":
        return isinstance(value, dict)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if schema_type == "null":
        return value is None
    return False


def _resolve_schema_ref(schema_root: dict[str, object], ref: str) -> dict[str, object]:
    if not ref.startswith("#/"):
        raise RecipeAttemptSchemaError(f"Only local schema refs are supported; got '{ref}'.")
    node: object = schema_root
    for part in ref[2:].split("/"):
        if not isinstance(node, dict) or part not in node:
            raise RecipeAttemptSchemaError(f"Unable to resolve schema ref '{ref}'.")
        node = node[part]
    if not isinstance(node, dict):
        raise RecipeAttemptSchemaError(f"Schema ref '{ref}' does not resolve to an object.")
    return node


def _coerce_optional_schema_object(value: object, path: str) -> dict[str, object]:
    if value is None:
        return {}
    return _coerce_schema_object(value, path)


def _coerce_schema_object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RecipeAttemptSchemaError(f"{path} must be an object.")
    return value


def _coerce_optional_schema_array(value: object, path: str) -> list[object]:
    if value is None:
        return []
    return _coerce_schema_array(value, path)


def _coerce_schema_array(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        raise RecipeAttemptSchemaError(f"{path} must be an array.")
    return value


__all__ = [
    "ATTEMPT_GATE_ORDER",
    "TERMINAL_ATTEMPT_STATES",
    "TERMINAL_CANDIDATE_LINEAGE_SELECTION_STATES",
    "AttemptFailure",
    "AttemptFailureClassification",
    "AttemptGate",
    "AttemptGateResult",
    "AttemptGateSequenceError",
    "AttemptGateStatus",
    "AttemptIdempotencyConflictError",
    "AttemptState",
    "AttemptStateTransitionError",
    "CandidateAttemptRecord",
    "CandidateInvocationCounters",
    "CandidateLineageSelectionState",
    "CandidatePlanValidationError",
    "CandidateSelectionConflictError",
    "CandidateSelectionReuseQuery",
    "CandidateWinnerStatus",
    "GeneratedRecipeRecord",
    "LEGACY_PROFILE_FINGERPRINT",
    "RECIPE_ATTEMPT_SCHEMA_VERSION",
    "RECIPE_ATTEMPT_STORE_SCHEMA_VERSION",
    "RecipeAttempt",
    "RecipeAttemptMigrationError",
    "RecipeAttemptRequest",
    "RecipeAttemptSchemaError",
    "RecipeAttemptSecurityError",
    "RecipeAttemptStore",
    "RecipeAttemptStoreError",
    "RecipeCandidateLineage",
    "RecipePromotionConflictError",
    "RecipeReuseQuery",
    "VerifiedRecipeRecord",
    "build_attempt_request_fingerprint",
    "build_attempt_request_from_generated",
    "build_candidate_recipe_fingerprint",
    "build_capability_fingerprint",
    "build_profile_fingerprint",
    "build_reuse_query_from_generated",
    "build_toolchain_fingerprint",
    "deserialize_attempt_failure",
    "deserialize_attempt_gate_result",
    "deserialize_candidate_attempt_record",
    "deserialize_generated_recipe_record",
    "deserialize_recipe_attempt",
    "deserialize_recipe_candidate_lineage",
    "deserialize_verified_recipe_record",
    "load_recipe_attempt_schema",
    "recipe_attempt_schema_path",
    "serialize_attempt_failure",
    "serialize_attempt_gate_result",
    "serialize_candidate_attempt_record",
    "serialize_generated_recipe_record",
    "serialize_recipe_attempt",
    "serialize_recipe_candidate_lineage",
    "serialize_verified_recipe_record",
]
