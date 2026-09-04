from __future__ import annotations

import json
import sqlite3
import threading
import time

from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from fl_model_onboarding.architecture_capabilities import (
    load_architecture_capability_registry,
    normalize_huggingface_metadata,
)
from fl_model_onboarding.recipe_attempt_store import (
    ATTEMPT_GATE_ORDER,
    LEGACY_PROFILE_FINGERPRINT,
    RECIPE_ATTEMPT_STORE_SCHEMA_VERSION,
    AttemptFailure,
    AttemptFailureClassification,
    AttemptGate,
    AttemptGateSequenceError,
    AttemptGateStatus,
    AttemptIdempotencyConflictError,
    AttemptState,
    AttemptStateTransitionError,
    RecipeAttemptMigrationError,
    RecipeAttemptRequest,
    RecipeAttemptSchemaError,
    RecipeAttemptSecurityError,
    RecipeAttemptStore,
    RecipePromotionConflictError,
    RecipeReuseQuery,
    build_attempt_request_fingerprint,
    build_attempt_request_from_generated,
    build_reuse_query_from_generated,
    deserialize_generated_recipe_record,
    deserialize_recipe_attempt,
    serialize_generated_recipe_record,
    serialize_recipe_attempt,
)
from fl_model_onboarding.recipe_compiler import (
    PromotionGateCheck,
    PromotionGateEvidence,
    RecipeCompilerInput,
    RecipeCompilerToolchain,
    compile_generated_recipe,
    promote_generated_recipe,
)

_REVISION_SHA = "0123456789abcdef0123456789abcdef01234567"
_ALT_REVISION_SHA = "89abcdef0123456789abcdef0123456789abcdef"


@dataclass
class _ConnectionLifecycleTracker:
    opened: int = 0
    closed: int = 0


def _tracked_connection_factory(tracker: _ConnectionLifecycleTracker):
    class _TrackedConnection(sqlite3.Connection):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            tracker.opened += 1
            self._tracker_closed = False

        def close(self) -> None:
            if not self._tracker_closed:
                tracker.closed += 1
                self._tracker_closed = True
            super().close()

    def connect(*args, **kwargs) -> sqlite3.Connection:
        return sqlite3.connect(*args, factory=_TrackedConnection, **kwargs)

    return connect


def _toolchain(*, ort_version: str = "1.29.0") -> RecipeCompilerToolchain:
    return RecipeCompilerToolchain(
        mobius_version="0.1.0",
        olive_version="0.13.0",
        onnx_version="1.22.0",
        ort_version=ort_version,
        oga_version="0.15.2",
        foundry_sdk_version="1.2.4",
        foundry_cli_version="0.11.0",
    )


def _resolve_capability(*, model_id: str, model_type: str, architecture: str, requested_precision: str = "auto"):
    registry = load_architecture_capability_registry()
    metadata = normalize_huggingface_metadata(
        model_id=model_id,
        config={
            "model_type": model_type,
            "architectures": [architecture],
        },
        is_gated=False,
        is_private=False,
    )
    return registry.resolve(
        metadata=metadata,
        task="llm",
        device="cpu",
        requested_precision=requested_precision,
    )


def _compiler_input(
    *,
    model_id: str = "example-org/new-text-model",
    revision_sha: str = _REVISION_SHA,
    requested_precision: str = "auto",
    ort_version: str = "1.29.0",
) -> RecipeCompilerInput:
    model_type = "llama"
    architecture = "LlamaForCausalLM"
    resolution = _resolve_capability(
        model_id=model_id,
        model_type=model_type,
        architecture=architecture,
        requested_precision=requested_precision,
    )
    return RecipeCompilerInput(
        model_id=model_id,
        revision_sha=revision_sha,
        model_type=model_type,
        architectures=(architecture,),
        task="llm",
        requested_device="cpu",
        requested_precision=requested_precision,
        is_gated=False,
        requires_remote_code=False,
        config_files=("config.json",),
        tokenizer_files=("tokenizer.json",),
        available_files=("config.json", "tokenizer.json", "model.safetensors"),
        capability_resolution=resolution,
        toolchain=_toolchain(ort_version=ort_version),
    )


def _candidate(
    *,
    model_id: str = "example-org/new-text-model",
    revision_sha: str = _REVISION_SHA,
    ort_version: str = "1.29.0",
) -> object:
    return compile_generated_recipe(
        _compiler_input(
            model_id=model_id,
            revision_sha=revision_sha,
            ort_version=ort_version,
        )
    )


def _promotion_evidence_for_token(token: str) -> PromotionGateEvidence:
    return PromotionGateEvidence(
        mobius_build=PromotionGateCheck(passed=True, evidence=f"mobius://{token}"),
        olive_optimize=PromotionGateCheck(passed=True, evidence=f"olive://{token}"),
        onnx_validation=PromotionGateCheck(passed=True, evidence=f"onnx://{token}"),
        ort_validation=PromotionGateCheck(passed=True, evidence=f"ort://{token}"),
        oga_validation=PromotionGateCheck(passed=True, evidence=f"oga://{token}"),
        fl_sdk_inference=PromotionGateCheck(passed=True, evidence=f"flsdk://{token}"),
        quality_validation=PromotionGateCheck(passed=True, evidence=f"quality://{token}"),
    )


def _record_success_gates(store: RecipeAttemptStore, attempt_id: str, *, token: str) -> None:
    evidence = _promotion_evidence_for_token(token)
    for idx, gate in enumerate(ATTEMPT_GATE_ORDER, start=1):
        store.record_attempt_gate(
            attempt_id=attempt_id,
            gate=gate,
            status=AttemptGateStatus.PASSED,
            evidence_ref=getattr(evidence, gate.value).evidence,
            metrics_ref=f"metrics://{token}/{idx}",
        )


def _create_attempt(
    store: RecipeAttemptStore,
    *,
    idempotency_key: str = "idem-1",
    model_id: str = "example-org/new-text-model",
    revision_sha: str = _REVISION_SHA,
    ort_version: str = "1.29.0",
) -> tuple[object, object]:
    generated = _candidate(model_id=model_id, revision_sha=revision_sha, ort_version=ort_version)
    generated_record = store.upsert_generated_recipe(generated)
    request = build_attempt_request_from_generated(generated_record)
    request_fingerprint = build_attempt_request_fingerprint(request)
    attempt, replay = store.create_attempt(
        idempotency_key=idempotency_key,
        request=request,
        request_fingerprint=request_fingerprint,
    )
    assert replay is False
    return generated, attempt


def _legacy_promotion_payload(tag: str) -> str:
    payload = {
        "mobius_build": {"passed": True, "evidence": f"mobius://{tag}"},
        "olive_optimize": {"passed": True, "evidence": f"olive://{tag}"},
        "onnx_validation": {"passed": True, "evidence": f"onnx://{tag}"},
        "ort_validation": {"passed": True, "evidence": f"ort://{tag}"},
        "oga_validation": {"passed": True, "evidence": f"oga://{tag}"},
        "fl_sdk_inference": {"passed": True, "evidence": f"flsdk://{tag}"},
        "quality_validation": {"passed": True, "evidence": f"quality://{tag}"},
    }
    return json.dumps(payload, separators=(",", ":"))


def test_generated_recipe_upsert_is_idempotent_and_preserves_full_identity(tmp_path: Path) -> None:
    store = RecipeAttemptStore(tmp_path / "recipe-attempt.sqlite3")
    generated = _candidate()

    first = store.upsert_generated_recipe(generated)
    second = store.upsert_generated_recipe(generated)

    assert first == second
    assert first.revision_sha == _REVISION_SHA
    assert first.recipe_fingerprint == generated.fingerprint
    assert len(first.revision_sha) == 40
    assert first.model_id == generated.recipe.huggingface_model_id


def test_attempt_creation_idempotency_and_request_fingerprint_conflict(tmp_path: Path) -> None:
    store = RecipeAttemptStore(tmp_path / "recipe-attempt.sqlite3")
    generated = _candidate()
    generated_record = store.upsert_generated_recipe(generated)
    request = build_attempt_request_from_generated(generated_record)
    request_fingerprint = build_attempt_request_fingerprint(request)

    first, replay_first = store.create_attempt(
        idempotency_key="idem-key",
        request=request,
        request_fingerprint=request_fingerprint,
    )
    second, replay_second = store.create_attempt(
        idempotency_key="idem-key",
        request=request,
        request_fingerprint=request_fingerprint,
    )
    assert replay_first is False
    assert replay_second is True
    assert first.attempt_id == second.attempt_id

    changed_request = replace(request, requested_precision="int4")
    changed_fingerprint = build_attempt_request_fingerprint(changed_request)
    with pytest.raises(AttemptIdempotencyConflictError):
        store.create_attempt(
            idempotency_key="idem-key",
            request=changed_request,
            request_fingerprint=changed_fingerprint,
        )

    with pytest.raises(ValueError):
        store.create_attempt(
            idempotency_key="idem-key-2",
            request=request,
            request_fingerprint="f" * 64,
        )


def test_attempt_state_contract_and_terminal_immutability(tmp_path: Path) -> None:
    store = RecipeAttemptStore(tmp_path / "recipe-attempt.sqlite3")
    _, attempt = _create_attempt(store)

    running = store.start_attempt(attempt.attempt_id)
    assert running.state == AttemptState.RUNNING

    _record_success_gates(store, attempt.attempt_id, token=attempt.attempt_id)
    succeeded = store.finish_attempt_succeeded(attempt.attempt_id)
    assert succeeded.state == AttemptState.SUCCEEDED
    assert succeeded.finished_utc is not None

    with pytest.raises(AttemptStateTransitionError):
        store.start_attempt(attempt.attempt_id)
    with pytest.raises(AttemptStateTransitionError):
        store.record_attempt_gate(
            attempt_id=attempt.attempt_id,
            gate=AttemptGate.QUALITY_VALIDATION,
            status=AttemptGateStatus.PASSED,
            evidence_ref="quality://again",
        )


def test_gate_order_and_monotonic_sequence_are_enforced(tmp_path: Path) -> None:
    store = RecipeAttemptStore(tmp_path / "recipe-attempt.sqlite3")
    _, attempt = _create_attempt(store)
    store.start_attempt(attempt.attempt_id)

    with pytest.raises(AttemptGateSequenceError):
        store.record_attempt_gate(
            attempt_id=attempt.attempt_id,
            gate=AttemptGate.OLIVE_OPTIMIZE,
            status=AttemptGateStatus.PASSED,
            evidence_ref="olive://too-early",
        )

    first = store.record_attempt_gate(
        attempt_id=attempt.attempt_id,
        gate=AttemptGate.MOBIUS_BUILD,
        status=AttemptGateStatus.PASSED,
        evidence_ref="mobius://ok",
    )
    assert first.sequence == 1

    with pytest.raises(AttemptGateSequenceError):
        store.record_attempt_gate(
            attempt_id=attempt.attempt_id,
            gate=AttemptGate.OLIVE_OPTIMIZE,
            status=AttemptGateStatus.PASSED,
            evidence_ref="olive://wrong-seq",
            expected_sequence=3,
        )


def test_verified_promotion_requires_succeeded_attempt_and_matching_identity(tmp_path: Path) -> None:
    store = RecipeAttemptStore(tmp_path / "recipe-attempt.sqlite3")
    generated, attempt = _create_attempt(store, idempotency_key="promote-me")
    store.start_attempt(attempt.attempt_id)
    _record_success_gates(store, attempt.attempt_id, token=attempt.attempt_id)
    succeeded = store.finish_attempt_succeeded(attempt.attempt_id)
    assert succeeded.state == AttemptState.SUCCEEDED

    promoted = promote_generated_recipe(
        generated,
        _promotion_evidence_for_token(attempt.attempt_id),
        new_version="1.0.1",
        status_reason="All required gates passed.",
    )
    verified = store.promote_verified_recipe(
        attempt_id=attempt.attempt_id,
        promoted_recipe=promoted,
    )
    assert verified.source_recipe_fingerprint == generated.fingerprint
    assert verified.verified_fingerprint == promoted.fingerprint

    generated_record = store.get_generated_recipe(generated.fingerprint)
    assert generated_record is not None
    exact_query = build_reuse_query_from_generated(generated_record)
    reusable = store.find_reusable_verified_recipe(exact_query)
    assert reusable is not None
    assert reusable.verified_fingerprint == promoted.fingerprint

    stale_toolchain_query = replace(
        exact_query,
        toolchain_fingerprint="a" * 64,
    )
    assert store.find_reusable_verified_recipe(stale_toolchain_query) is None

    stale_profile_query = replace(
        exact_query,
        profile_fingerprint="b" * 64,
    )
    assert store.find_reusable_verified_recipe(stale_profile_query) is None


def test_promotion_rejects_failed_attempt_and_identity_mismatch(tmp_path: Path) -> None:
    store = RecipeAttemptStore(tmp_path / "recipe-attempt.sqlite3")
    generated, attempt = _create_attempt(store, idempotency_key="failed-promo")
    store.start_attempt(attempt.attempt_id)
    failed = store.finish_attempt_failed(
        attempt.attempt_id,
        failure=AttemptFailure(
            classification=AttemptFailureClassification.GATE_FAILED,
            stage="mobius_build",
            message="Mobius failed without valid output.",
            evidence_refs=("mobius://failed",),
            source_owner="recipe-agent",
            next_action="Inspect mobius diagnostics and retry with a new idempotency key.",
        ),
    )
    assert failed.state == AttemptState.FAILED

    promoted = promote_generated_recipe(
        generated,
        _promotion_evidence_for_token("failed-promo"),
        new_version="1.0.1",
        status_reason="Should not be accepted for failed attempt.",
    )
    with pytest.raises(RecipePromotionConflictError):
        store.promote_verified_recipe(
            attempt_id=attempt.attempt_id,
            promoted_recipe=promoted,
        )

    alt_generated = _candidate(revision_sha=_ALT_REVISION_SHA)
    alt_record = store.upsert_generated_recipe(alt_generated)
    alt_request = build_attempt_request_from_generated(alt_record)
    alt_fp = build_attempt_request_fingerprint(alt_request)
    alt_attempt, _ = store.create_attempt(
        idempotency_key="alt-attempt",
        request=alt_request,
        request_fingerprint=alt_fp,
    )
    store.start_attempt(alt_attempt.attempt_id)
    _record_success_gates(store, alt_attempt.attempt_id, token=alt_attempt.attempt_id)
    store.finish_attempt_succeeded(alt_attempt.attempt_id)

    mismatch_promoted = promote_generated_recipe(
        generated,
        _promotion_evidence_for_token(alt_attempt.attempt_id),
        new_version="1.0.2",
        status_reason="Mismatched source should be rejected.",
    )
    with pytest.raises(RecipePromotionConflictError):
        store.promote_verified_recipe(
            attempt_id=alt_attempt.attempt_id,
            promoted_recipe=mismatch_promoted,
        )


def test_security_guards_reject_secret_like_and_private_path_evidence(tmp_path: Path) -> None:
    store = RecipeAttemptStore(tmp_path / "recipe-attempt.sqlite3")
    _, attempt = _create_attempt(store)
    store.start_attempt(attempt.attempt_id)

    with pytest.raises(RecipeAttemptSecurityError):
        store.record_attempt_gate(
            attempt_id=attempt.attempt_id,
            gate=AttemptGate.MOBIUS_BUILD,
            status=AttemptGateStatus.PASSED,
            evidence_ref="Bearer abcdefghijklmnopqrstuvwxyz0123456789",
        )
    with pytest.raises(RecipeAttemptSecurityError):
        store.record_attempt_gate(
            attempt_id=attempt.attempt_id,
            gate=AttemptGate.MOBIUS_BUILD,
            status=AttemptGateStatus.PASSED,
            evidence_ref=r"C:\Users\alice\private\mobius.log",
        )

    store.record_attempt_gate(
        attempt_id=attempt.attempt_id,
        gate=AttemptGate.MOBIUS_BUILD,
        status=AttemptGateStatus.FAILED,
        evidence_ref="mobius://job-1",
    )
    failed = store.finish_attempt_failed(
        attempt.attempt_id,
        failure=AttemptFailure(
            classification=AttemptFailureClassification.GATE_FAILED,
            stage="mobius_build",
            message="Gate failed due to api_key=abcd1234efgh5678",
            evidence_refs=("mobius://job-1",),
            source_owner="recipe-agent",
            next_action="Rotate credential, then retry with a new idempotency key.",
        ),
    )
    assert failed.failure is not None
    assert "[REDACTED]" in failed.failure.message
    assert "abcd1234efgh5678" not in failed.failure.message


def test_schema_roundtrip_and_fail_closed_record_loader(tmp_path: Path) -> None:
    store = RecipeAttemptStore(tmp_path / "recipe-attempt.sqlite3")
    generated = _candidate()
    generated_record = store.upsert_generated_recipe(generated)

    serialized_generated = serialize_generated_recipe_record(generated_record)
    restored_generated = deserialize_generated_recipe_record(serialized_generated)
    assert restored_generated == generated_record

    with pytest.raises(RecipeAttemptSchemaError):
        deserialize_generated_recipe_record(
            {
                **serialized_generated,
                "unexpected_key": "should-fail-closed",
            }
        )

    request = build_attempt_request_from_generated(generated_record)
    request_fp = build_attempt_request_fingerprint(request)
    attempt, _ = store.create_attempt(
        idempotency_key="schema-roundtrip",
        request=request,
        request_fingerprint=request_fp,
    )
    roundtrip_attempt = deserialize_recipe_attempt(serialize_recipe_attempt(attempt))
    assert roundtrip_attempt == attempt


def test_restart_recovery_marks_running_attempt_failed_with_evidence(tmp_path: Path) -> None:
    db_path = tmp_path / "recipe-attempt.sqlite3"
    first = RecipeAttemptStore(db_path)
    _, attempt = _create_attempt(first, idempotency_key="restart-running")
    first.start_attempt(attempt.attempt_id)

    resumed = RecipeAttemptStore(db_path)
    recovered = resumed.get_attempt(attempt.attempt_id)
    assert recovered.state == AttemptState.FAILED
    assert recovered.failure is not None
    assert recovered.failure.classification == AttemptFailureClassification.INTERRUPTED
    assert recovered.failure.evidence_refs
    assert recovered.finished_utc is not None


def test_verified_records_survive_restart(tmp_path: Path) -> None:
    db_path = tmp_path / "recipe-attempt.sqlite3"
    first = RecipeAttemptStore(db_path)
    generated, attempt = _create_attempt(first, idempotency_key="verify-restart")
    first.start_attempt(attempt.attempt_id)
    _record_success_gates(first, attempt.attempt_id, token=attempt.attempt_id)
    first.finish_attempt_succeeded(attempt.attempt_id)
    promoted = promote_generated_recipe(
        generated,
        _promotion_evidence_for_token(attempt.attempt_id),
        new_version="1.0.1",
        status_reason="All checks passed.",
    )
    first.promote_verified_recipe(
        attempt_id=attempt.attempt_id,
        promoted_recipe=promoted,
    )
    generated_record = first.get_generated_recipe(generated.fingerprint)
    assert generated_record is not None

    resumed = RecipeAttemptStore(db_path)
    reusable = resumed.find_reusable_verified_recipe(
        build_reuse_query_from_generated(generated_record)
    )
    assert reusable is not None
    assert reusable.verified_fingerprint == promoted.fingerprint


def test_migration_v1_to_v2_adds_profile_fingerprint_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.sqlite3"
    generated_fingerprint = "1" * 64
    verified_fingerprint = "2" * 64
    revision = "a" * 40
    request_fingerprint = "3" * 64
    capability_fingerprint = "4" * 64
    toolchain_fingerprint = "5" * 64

    generated_canonical = json.dumps({"fingerprint": generated_fingerprint}, separators=(",", ":"))
    verified_canonical = json.dumps({"fingerprint": verified_fingerprint}, separators=(",", ":"))
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE generated_recipes (
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
            CREATE TABLE attempts (
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
                failure_json TEXT
            );
            CREATE TABLE attempt_gates (
                attempt_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                gate_name TEXT NOT NULL,
                status TEXT NOT NULL,
                evidence_ref TEXT NOT NULL,
                metrics_ref TEXT,
                started_utc TEXT NOT NULL,
                finished_utc TEXT NOT NULL,
                PRIMARY KEY (attempt_id, sequence)
            );
            CREATE TABLE verified_recipes (
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
                canonical_json TEXT NOT NULL
            );
            """
        )
        connection.execute(
            """
            INSERT INTO generated_recipes (
                recipe_fingerprint, schema_version, recipe_status, model_id, revision_sha,
                requested_device, requested_precision, compiler_version, capability_fingerprint,
                toolchain_fingerprint, canonical_json, created_utc
            )
            VALUES (?, '1.0.0', 'experimental', 'legacy/model', ?, 'cpu', 'int4', '1.0.0', ?, ?, ?, '2026-08-31T00:00:00+00:00')
            """,
            (
                generated_fingerprint,
                revision,
                capability_fingerprint,
                toolchain_fingerprint,
                generated_canonical,
            ),
        )
        connection.execute(
            """
            INSERT INTO attempts (
                attempt_id, idempotency_key, request_fingerprint, recipe_fingerprint,
                model_id, revision_sha, requested_device, requested_precision, compiler_version,
                capability_fingerprint, toolchain_fingerprint, created_utc, finished_utc, state, failure_json
            )
            VALUES (
                'attempt-legacy', 'legacy-idem', ?, ?, 'legacy/model', ?, 'cpu', 'int4', '1.0.0',
                ?, ?, '2026-08-31T00:00:00+00:00', '2026-08-31T00:01:00+00:00', 'failed',
                '{"classification":"interrupted","stage":"legacy","message":"legacy","evidence_refs":["legacy://e"],"source_owner":"legacy","next_action":"retry"}'
            )
            """,
            (
                request_fingerprint,
                generated_fingerprint,
                revision,
                capability_fingerprint,
                toolchain_fingerprint,
            ),
        )
        connection.execute(
            """
            INSERT INTO verified_recipes (
                verified_fingerprint, source_recipe_fingerprint, attempt_id, schema_version, model_id,
                revision_sha, requested_device, requested_precision, compiler_version,
                capability_fingerprint, toolchain_fingerprint, promoted_utc, promotion_evidence_json, canonical_json
            )
            VALUES (?, ?, 'attempt-legacy', '1.0.0', 'legacy/model', ?, 'cpu', 'int4', '1.0.0', ?, ?, '2026-08-31T00:02:00+00:00', ?, ?)
            """,
            (
                verified_fingerprint,
                generated_fingerprint,
                revision,
                capability_fingerprint,
                toolchain_fingerprint,
                _legacy_promotion_payload("legacy"),
                verified_canonical,
            ),
        )
        connection.execute("PRAGMA user_version = 1;")

    migrated = RecipeAttemptStore(db_path)
    with sqlite3.connect(db_path) as check:
        check.row_factory = sqlite3.Row
        user_version = int(check.execute("PRAGMA user_version").fetchone()[0])
        assert user_version == RECIPE_ATTEMPT_STORE_SCHEMA_VERSION
        generated_columns = {
            row["name"] for row in check.execute("PRAGMA table_info(generated_recipes)").fetchall()
        }
        attempt_columns = {
            row["name"] for row in check.execute("PRAGMA table_info(attempts)").fetchall()
        }
        verified_columns = {
            row["name"] for row in check.execute("PRAGMA table_info(verified_recipes)").fetchall()
        }
        assert "profile_fingerprint" in generated_columns
        assert "profile_fingerprint" in attempt_columns
        assert "profile_fingerprint" in verified_columns
        candidate_tables = {
            row["name"]
            for row in check.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert "candidate_attempts" in candidate_tables
        assert "recipe_candidate_lineages" in candidate_tables

    attempt = migrated.get_attempt("attempt-legacy")
    assert attempt.profile_fingerprint == LEGACY_PROFILE_FINGERPRINT
    # Legacy attempts predate Slice 2 and carry no candidate lineage/plan at all;
    # they remain fully readable and retain their old (pre-policy) behavior.
    assert migrated.get_candidate_lineage("attempt-legacy") is None
    assert migrated.list_candidate_attempts("attempt-legacy") == ()
    reusable = migrated.find_reusable_verified_recipe(
        RecipeReuseQuery(
            model_id="legacy/model",
            revision_sha=revision,
            requested_device="cpu",
            requested_precision="int4",
            compiler_version="1.0.0",
            capability_fingerprint=capability_fingerprint,
            toolchain_fingerprint=toolchain_fingerprint,
            profile_fingerprint=LEGACY_PROFILE_FINGERPRINT,
        )
    )
    assert reusable is not None
    assert (
        migrated.find_reusable_verified_recipe(
            replace(
                RecipeReuseQuery(
                    model_id="legacy/model",
                    revision_sha=revision,
                    requested_device="cpu",
                    requested_precision="int4",
                    compiler_version="1.0.0",
                    capability_fingerprint=capability_fingerprint,
                    toolchain_fingerprint=toolchain_fingerprint,
                    profile_fingerprint=LEGACY_PROFILE_FINGERPRINT,
                ),
                profile_fingerprint="9" * 64,
            )
        )
        is None
    )

    # Reopening (re-running migration) repeatedly against an already-migrated
    # database is idempotent: no error, same schema version, same data.
    reopened = RecipeAttemptStore(db_path)
    with sqlite3.connect(db_path) as check_again:
        assert int(check_again.execute("PRAGMA user_version").fetchone()[0]) == RECIPE_ATTEMPT_STORE_SCHEMA_VERSION
    assert reopened.get_attempt("attempt-legacy").attempt_id == "attempt-legacy"
    assert reopened.get_candidate_lineage("attempt-legacy") is None


def test_concurrent_duplicate_idempotency_returns_single_attempt(tmp_path: Path) -> None:
    store = RecipeAttemptStore(tmp_path / "recipe-attempt.sqlite3")
    generated = _candidate()
    generated_record = store.upsert_generated_recipe(generated)
    request = build_attempt_request_from_generated(generated_record)
    request_fingerprint = build_attempt_request_fingerprint(request)

    barrier = threading.Barrier(8)
    results: list[tuple[str, bool]] = []
    errors: list[Exception] = []

    def worker() -> None:
        try:
            barrier.wait(timeout=5)
            attempt, replay = store.create_attempt(
                idempotency_key="concurrent-idem",
                request=request,
                request_fingerprint=request_fingerprint,
            )
            results.append((attempt.attempt_id, replay))
        except Exception as exc:  # pragma: no cover - explicit assertion below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not errors
    assert len(results) == 8
    assert len({attempt_id for attempt_id, _ in results}) == 1
    assert sum(1 for _, replay in results if replay is False) == 1
    assert sum(1 for _, replay in results if replay is True) == 7


def test_concurrent_promotion_transaction_is_atomic(tmp_path: Path) -> None:
    store = RecipeAttemptStore(tmp_path / "recipe-attempt.sqlite3")
    generated, attempt = _create_attempt(store, idempotency_key="concurrent-promotion")
    store.start_attempt(attempt.attempt_id)
    _record_success_gates(store, attempt.attempt_id, token=attempt.attempt_id)
    store.finish_attempt_succeeded(attempt.attempt_id)
    promoted = promote_generated_recipe(
        generated,
        _promotion_evidence_for_token(attempt.attempt_id),
        new_version="1.0.1",
        status_reason="Concurrent promotion check.",
    )

    barrier = threading.Barrier(2)
    results: list[str] = []
    errors: list[Exception] = []

    def promote_worker() -> None:
        try:
            barrier.wait(timeout=5)
            row = store.promote_verified_recipe(
                attempt_id=attempt.attempt_id,
                promoted_recipe=promoted,
            )
            results.append(row.verified_fingerprint)
        except Exception as exc:  # pragma: no cover - explicit assertion below
            errors.append(exc)

    left = threading.Thread(target=promote_worker)
    right = threading.Thread(target=promote_worker)
    left.start()
    right.start()
    left.join(timeout=5)
    right.join(timeout=5)

    assert not errors
    assert results == [promoted.fingerprint, promoted.fingerprint]
    with sqlite3.connect(store.db_path) as connection:
        count = int(
            connection.execute(
                "SELECT COUNT(*) FROM verified_recipes WHERE attempt_id = ?",
                (attempt.attempt_id,),
            ).fetchone()[0]
        )
    assert count == 1


def test_store_closes_connections_after_success_and_exception_paths(tmp_path: Path) -> None:
    tracker = _ConnectionLifecycleTracker()
    store = RecipeAttemptStore(
        tmp_path / "recipe-attempt.sqlite3",
        _connection_factory=_tracked_connection_factory(tracker),
    )
    assert tracker.opened == 1
    assert tracker.closed == 1

    generated = _candidate()
    store.upsert_generated_recipe(generated)
    assert tracker.opened == tracker.closed

    with pytest.raises(KeyError):
        store.start_attempt("missing-attempt")
    assert tracker.opened == tracker.closed


def test_multiple_readers_can_poll_while_writer_advances_attempt(tmp_path: Path) -> None:
    store = RecipeAttemptStore(tmp_path / "recipe-attempt.sqlite3")
    _, attempt = _create_attempt(store, idempotency_key="reader-writer")
    store.start_attempt(attempt.attempt_id)

    stop = threading.Event()
    errors: list[Exception] = []

    def reader() -> None:
        while not stop.is_set():
            try:
                _ = store.get_attempt(attempt.attempt_id)
            except Exception as exc:  # pragma: no cover - explicit assertion below
                errors.append(exc)
                stop.set()
                return
            time.sleep(0.005)

    readers = [threading.Thread(target=reader) for _ in range(4)]
    for thread in readers:
        thread.start()
    try:
        _record_success_gates(store, attempt.attempt_id, token=attempt.attempt_id)
        store.finish_attempt_succeeded(attempt.attempt_id)
    finally:
        stop.set()
        for thread in readers:
            thread.join(timeout=5)

    assert not errors
    final = store.get_attempt(attempt.attempt_id)
    assert final.state == AttemptState.SUCCEEDED


def test_store_releases_database_file_for_rename_after_repeated_operations(tmp_path: Path) -> None:
    db_path = tmp_path / "recipe-attempt.sqlite3"
    store = RecipeAttemptStore(db_path)
    generated, attempt = _create_attempt(store, idempotency_key="rename-database")

    for _ in range(20):
        assert store.get_generated_recipe(generated.fingerprint) is not None
        _ = store.get_attempt(attempt.attempt_id)
        _ = store.list_attempts()

    renamed = tmp_path / "recipe-attempt-renamed.sqlite3"
    db_path.rename(renamed)
    assert renamed.exists()

    with sqlite3.connect(renamed) as connection:
        count = int(connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0])
    assert count == 1


def test_store_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    db_path = tmp_path / "bad-version.sqlite3"
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA user_version = 999;")
    with pytest.raises(RecipeAttemptMigrationError):
        RecipeAttemptStore(db_path)


def test_attempt_request_record_type_is_typed_and_immutable_shape(tmp_path: Path) -> None:
    store = RecipeAttemptStore(tmp_path / "recipe-attempt.sqlite3")
    generated = _candidate()
    record = store.upsert_generated_recipe(generated)
    request = build_attempt_request_from_generated(record)
    assert isinstance(request, RecipeAttemptRequest)
    with pytest.raises(AttributeError):
        request.model_id = "changed"  # type: ignore[misc]
