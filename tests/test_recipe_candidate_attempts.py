"""Slice 2: durable candidate persistence, migration, fingerprints, and reuse identity.

These tests cover the additive `RecipeAttemptStore` candidate-plan/selection APIs on
top of the existing (Slice 1, approved) generated-recipe/attempt/verified-recipe
persistence, without modifying any of that existing behavior. No real model runs
are performed anywhere in this file: every "recipe" is a deterministically compiled
`GeneratedRecipe` and every candidate outcome is driven directly through the store's
typed APIs.
"""

from __future__ import annotations

import sqlite3

from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from fl_model_onboarding.architecture_capabilities import (
    load_architecture_capability_registry,
    normalize_huggingface_metadata,
)
from fl_model_onboarding.quality_validation import (
    DEFAULT_TEXT_GENERATION_QUALITY_PROFILE,
    QualityRetryDisposition,
    QualityRetryEvaluation,
)
from fl_model_onboarding.recipe_attempt_store import (
    ATTEMPT_GATE_ORDER,
    RECIPE_ATTEMPT_STORE_SCHEMA_VERSION,
    AttemptFailure,
    AttemptFailureClassification,
    AttemptGate,
    AttemptGateStatus,
    AttemptState,
    CandidateAttemptRecord,
    CandidateInvocationCounters,
    CandidateLineageSelectionState,
    CandidatePlanValidationError,
    CandidateReuseIntegrityError,
    CandidateSelectionConflictError,
    CandidateSelectionReuseQuery,
    CandidateWinnerStatus,
    RecipeAttemptMigrationError,
    RecipeAttemptStore,
    build_attempt_request_fingerprint,
    build_attempt_request_from_generated,
    build_candidate_recipe_fingerprint,
    deserialize_candidate_attempt_record,
    serialize_candidate_attempt_record,
)
from fl_model_onboarding.recipe_compiler import (
    RecipeCompilerInput,
    RecipeCompilerToolchain,
    compile_generated_recipe,
)
from fl_model_onboarding.recipe_selection_policy import (
    DEFAULT_CPU_INT4_RECIPE_SELECTION_POLICY,
    RETRYABLE_OPTIMIZED_STRUCTURAL_REGRESSION_TRIGGER,
    RecipeQuantizationOverride,
)

_REVISION_SHA = "0123456789abcdef0123456789abcdef01234567"
_QUALITY_PROFILE_FINGERPRINT = DEFAULT_TEXT_GENERATION_QUALITY_PROFILE.fingerprint
_POLICY = DEFAULT_CPU_INT4_RECIPE_SELECTION_POLICY
_RETRYABLE_EVALUATION = QualityRetryEvaluation(
    disposition=QualityRetryDisposition.RETRYABLE_OPTIMIZED_STRUCTURAL_REGRESSION,
    reasons=("optimized_structural_regression:prompt-1:invalid_json_output",),
)
_NOT_RETRYABLE_EVALUATION = QualityRetryEvaluation(
    disposition=QualityRetryDisposition.NOT_RETRYABLE,
    reasons=("baseline_unavailable",),
)


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


def _resolve_capability(*, model_id: str):
    registry = load_architecture_capability_registry()
    metadata = normalize_huggingface_metadata(
        model_id=model_id,
        config={"model_type": "llama", "architectures": ["LlamaForCausalLM"]},
        is_gated=False,
        is_private=False,
    )
    return registry.resolve(metadata=metadata, task="llm", device="cpu", requested_precision="auto")


def _generated_recipe(
    *,
    model_id: str = "example-org/candidate-model",
    ort_version: str = "1.29.0",
    extra_available_files: tuple[str, ...] = (),
):
    resolution = _resolve_capability(model_id=model_id)
    return compile_generated_recipe(
        RecipeCompilerInput(
            model_id=model_id,
            revision_sha=_REVISION_SHA,
            model_type="llama",
            architectures=("LlamaForCausalLM",),
            task="llm",
            requested_device="cpu",
            requested_precision="auto",
            is_gated=False,
            requires_remote_code=False,
            config_files=("config.json",),
            tokenizer_files=("tokenizer.json",),
            available_files=("config.json", "tokenizer.json", "model.safetensors") + extra_available_files,
            capability_resolution=resolution,
            toolchain=_toolchain(ort_version=ort_version),
        )
    )


def _create_and_start_attempt(
    store: RecipeAttemptStore,
    *,
    idempotency_key: str,
    model_id: str = "example-org/candidate-model",
    ort_version: str = "1.29.0",
    extra_available_files: tuple[str, ...] = (),
) -> tuple[object, object]:
    """Create + start a plain attempt (candidate_index 0 style setup), independent
    of any candidate-plan registration."""
    generated = _generated_recipe(model_id=model_id, ort_version=ort_version, extra_available_files=extra_available_files)
    generated_record = store.upsert_generated_recipe(generated)
    request = build_attempt_request_from_generated(generated_record)
    request_fingerprint = build_attempt_request_fingerprint(request)
    attempt, _replay = store.create_attempt(
        idempotency_key=idempotency_key,
        request=request,
        request_fingerprint=request_fingerprint,
    )
    store.start_attempt(attempt.attempt_id)
    return generated, attempt


def _fail_attempt_at_first_gate(store: RecipeAttemptStore, attempt_id: str) -> None:
    store.record_attempt_gate(
        attempt_id=attempt_id,
        gate=AttemptGate.MOBIUS_BUILD,
        status=AttemptGateStatus.FAILED,
        evidence_ref=f"mobius://{attempt_id}/failed",
    )
    store.finish_attempt_failed(
        attempt_id,
        failure=AttemptFailure(
            classification=AttemptFailureClassification.GATE_FAILED,
            stage="mobius_build",
            message="Mobius build failed for the default candidate.",
            evidence_refs=(f"mobius://{attempt_id}/failed",),
            source_owner="recipe-agent",
            next_action="Evaluate quality retry disposition.",
        ),
    )


def _succeed_attempt(store: RecipeAttemptStore, attempt_id: str) -> None:
    for gate in ATTEMPT_GATE_ORDER:
        store.record_attempt_gate(
            attempt_id=attempt_id,
            gate=gate,
            status=AttemptGateStatus.PASSED,
            evidence_ref=f"{gate.value}://{attempt_id}/ok",
        )
    store.finish_attempt_succeeded(attempt_id)


def _register_default(store: RecipeAttemptStore, attempt_id: str) -> CandidateAttemptRecord:
    return store.register_candidate_attempt(
        parent_attempt_id=attempt_id,
        attempt_id=attempt_id,
        candidate_index=0,
        policy=_POLICY,
        quality_profile_fingerprint=_QUALITY_PROFILE_FINGERPRINT,
    )


def _register_fallback(
    store: RecipeAttemptStore,
    *,
    parent_attempt_id: str,
    attempt_id: str,
    retry_evaluation: QualityRetryEvaluation = _RETRYABLE_EVALUATION,
) -> CandidateAttemptRecord:
    return store.register_candidate_attempt(
        parent_attempt_id=parent_attempt_id,
        attempt_id=attempt_id,
        candidate_index=1,
        policy=_POLICY,
        quality_profile_fingerprint=_QUALITY_PROFILE_FINGERPRINT,
        trigger=RETRYABLE_OPTIMIZED_STRUCTURAL_REGRESSION_TRIGGER,
        retry_evaluation=retry_evaluation,
    )


def _clone_attempt_row_with_overrides(
    store: RecipeAttemptStore,
    *,
    template_attempt_id: str,
    new_attempt_id: str,
    new_idempotency_key: str,
    overrides: dict[str, str],
) -> None:
    """Insert a new ``attempts`` row that is an exact clone of
    ``template_attempt_id`` except for ``overrides``, entirely bypassing
    ``create_attempt``/gate recording.

    Used only to exercise the store's fail-closed generation-identity checks with
    single-field mismatches the deterministic recipe compiler cannot itself
    produce today (for example, it only ever compiles for one
    ``requested_device``). This talks to the sqlite file directly through a
    plain ``sqlite3.connect`` (not ``RecipeAttemptStore._connect``), so no
    foreign-key enforcement applies to the cloned ``recipe_fingerprint``
    reference.
    """
    with sqlite3.connect(store.db_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM attempts WHERE attempt_id = ?", (template_attempt_id,)
        ).fetchone()
        assert row is not None
        columns = list(row.keys())
        values = {column: row[column] for column in columns}
        values["attempt_id"] = new_attempt_id
        values["idempotency_key"] = new_idempotency_key
        values.update(overrides)
        connection.execute(
            f"INSERT INTO attempts ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})",
            tuple(values[column] for column in columns),
        )
        connection.commit()


def _repoint_candidate_attempt_to(
    store: RecipeAttemptStore,
    *,
    candidate_attempt_id: str,
    attempt_id: str,
) -> None:
    """Directly rewrite a persisted candidate's linked ``attempt_id`` via raw SQL.

    Simulates a corrupted/tampered row (or one written through some unsupported,
    non-``register_candidate_attempt`` path before the generation-identity
    invariant existed): something the store's own write APIs would never
    produce, but which read paths must still fail closed against instead of
    silently serving.
    """
    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            "UPDATE candidate_attempts SET attempt_id = ? WHERE candidate_attempt_id = ?",
            (attempt_id, candidate_attempt_id),
        )
        connection.commit()


# --------------------------------------------------------------------------
# Fresh DB: plan / default + fallback persistence / order
# --------------------------------------------------------------------------


def test_fresh_db_candidate_plan_persists_default_and_fallback_in_order(tmp_path: Path) -> None:
    store = RecipeAttemptStore(tmp_path / "recipe-attempt.sqlite3")
    _default_generated, default_attempt = _create_and_start_attempt(store, idempotency_key="default")
    # Same full generation identity as the parent (model/revision/device/precision/
    # compiler/capability/toolchain/profile): only the extra available file differs,
    # which changes recipe_fingerprint without changing identity -- the legitimate
    # shape of a fallback candidate.
    _fallback_generated, fallback_attempt = _create_and_start_attempt(
        store, idempotency_key="fallback", extra_available_files=("generation_config.json",)
    )

    default_candidate = _register_default(store, default_attempt.attempt_id)
    assert default_candidate.candidate_index == 0
    assert default_candidate.candidate_id == "default-int4"
    assert default_candidate.quantization_override_block_size is None
    assert default_candidate.eligibility_trigger is None
    assert default_candidate.disposition is None
    assert default_candidate.disposition_reasons == ()
    assert default_candidate.selection_status == CandidateWinnerStatus.NOT_SELECTED

    fallback_candidate = _register_fallback(
        store,
        parent_attempt_id=default_attempt.attempt_id,
        attempt_id=fallback_attempt.attempt_id,
    )
    assert fallback_candidate.candidate_index == 1
    assert fallback_candidate.candidate_id == "int4-block-size-64"
    assert fallback_candidate.quantization_override_block_size == 64
    assert fallback_candidate.eligibility_trigger == RETRYABLE_OPTIMIZED_STRUCTURAL_REGRESSION_TRIGGER
    assert fallback_candidate.disposition == QualityRetryDisposition.RETRYABLE_OPTIMIZED_STRUCTURAL_REGRESSION.value
    assert fallback_candidate.disposition_reasons == _RETRYABLE_EVALUATION.reasons

    ordered = store.list_candidate_attempts(default_attempt.attempt_id)
    assert [row.candidate_attempt_id for row in ordered] == [
        default_candidate.candidate_attempt_id,
        fallback_candidate.candidate_attempt_id,
    ]
    assert [row.candidate_index for row in ordered] == [0, 1]

    lineage = store.get_candidate_lineage(default_attempt.attempt_id)
    assert lineage is not None
    assert lineage.policy_id == _POLICY.policy_id
    assert lineage.policy_version == _POLICY.version
    assert lineage.policy_fingerprint == _POLICY.fingerprint
    assert lineage.policy_max_candidates == _POLICY.max_candidates
    assert lineage.quality_profile_fingerprint == _QUALITY_PROFILE_FINGERPRINT
    assert lineage.selection_state == CandidateLineageSelectionState.PENDING
    assert lineage.selected_candidate_attempt_id is None


def test_candidate_index_0_must_use_attempt_id_equal_to_parent(tmp_path: Path) -> None:
    store = RecipeAttemptStore(tmp_path / "recipe-attempt.sqlite3")
    _generated, default_attempt = _create_and_start_attempt(store, idempotency_key="default")
    _generated2, other_attempt = _create_and_start_attempt(store, idempotency_key="other", ort_version="1.31.0")

    with pytest.raises(CandidatePlanValidationError):
        store.register_candidate_attempt(
            parent_attempt_id=default_attempt.attempt_id,
            attempt_id=other_attempt.attempt_id,
            candidate_index=0,
            policy=_POLICY,
            quality_profile_fingerprint=_QUALITY_PROFILE_FINGERPRINT,
        )


def test_fallback_cannot_be_registered_before_default(tmp_path: Path) -> None:
    store = RecipeAttemptStore(tmp_path / "recipe-attempt.sqlite3")
    _default_generated, default_attempt = _create_and_start_attempt(store, idempotency_key="default")
    _fallback_generated, fallback_attempt = _create_and_start_attempt(
        store, idempotency_key="fallback", ort_version="1.30.0"
    )

    with pytest.raises(CandidatePlanValidationError):
        _register_fallback(
            store,
            parent_attempt_id=default_attempt.attempt_id,
            attempt_id=fallback_attempt.attempt_id,
        )


def test_duplicate_candidate_index_and_max_candidates_are_enforced(tmp_path: Path) -> None:
    store = RecipeAttemptStore(tmp_path / "recipe-attempt.sqlite3")
    _default_generated, default_attempt = _create_and_start_attempt(store, idempotency_key="default")
    _register_default(store, default_attempt.attempt_id)

    with pytest.raises(CandidatePlanValidationError):
        _register_default(store, default_attempt.attempt_id)

    _fallback_generated, fallback_attempt = _create_and_start_attempt(
        store, idempotency_key="fallback", extra_available_files=("generation_config.json",)
    )
    _register_fallback(
        store,
        parent_attempt_id=default_attempt.attempt_id,
        attempt_id=fallback_attempt.attempt_id,
    )

    # Policy declares only 2 candidates; a third slot is out of range entirely.
    _third_generated, third_attempt = _create_and_start_attempt(
        store, idempotency_key="third", ort_version="1.31.0"
    )
    with pytest.raises(CandidatePlanValidationError):
        store.register_candidate_attempt(
            parent_attempt_id=default_attempt.attempt_id,
            attempt_id=third_attempt.attempt_id,
            candidate_index=2,
            policy=_POLICY,
            quality_profile_fingerprint=_QUALITY_PROFILE_FINGERPRINT,
        )


def test_mismatched_trigger_and_non_retryable_disposition_are_rejected(tmp_path: Path) -> None:
    store = RecipeAttemptStore(tmp_path / "recipe-attempt.sqlite3")
    _default_generated, default_attempt = _create_and_start_attempt(store, idempotency_key="default")
    _register_default(store, default_attempt.attempt_id)
    _fallback_generated, fallback_attempt = _create_and_start_attempt(
        store, idempotency_key="fallback", ort_version="1.30.0"
    )

    with pytest.raises(CandidatePlanValidationError):
        store.register_candidate_attempt(
            parent_attempt_id=default_attempt.attempt_id,
            attempt_id=fallback_attempt.attempt_id,
            candidate_index=1,
            policy=_POLICY,
            quality_profile_fingerprint=_QUALITY_PROFILE_FINGERPRINT,
            trigger="some-other-trigger",
            retry_evaluation=_RETRYABLE_EVALUATION,
        )

    with pytest.raises(CandidatePlanValidationError):
        _register_fallback(
            store,
            parent_attempt_id=default_attempt.attempt_id,
            attempt_id=fallback_attempt.attempt_id,
            retry_evaluation=_NOT_RETRYABLE_EVALUATION,
        )

    with pytest.raises(CandidatePlanValidationError):
        # Default candidate cannot carry a trigger/disposition at all.
        store.register_candidate_attempt(
            parent_attempt_id=default_attempt.attempt_id,
            attempt_id=fallback_attempt.attempt_id,
            candidate_index=0,
            policy=_POLICY,
            quality_profile_fingerprint=_QUALITY_PROFILE_FINGERPRINT,
            trigger=RETRYABLE_OPTIMIZED_STRUCTURAL_REGRESSION_TRIGGER,
        )


def test_register_candidate_attempt_rejects_non_allowlisted_trigger(tmp_path: Path) -> None:
    """A policy candidate that (hypothetically) declared a non-canonical trigger
    string must still be rejected by the store, independent of policy-file
    validation -- this is the store-side half of the cross-module trigger
    boundary enforcement."""
    store = RecipeAttemptStore(tmp_path / "recipe-attempt.sqlite3")
    _default_generated, default_attempt = _create_and_start_attempt(store, idempotency_key="default")
    _register_default(store, default_attempt.attempt_id)
    _fallback_generated, fallback_attempt = _create_and_start_attempt(
        store, idempotency_key="fallback", ort_version="1.30.0"
    )

    rogue_policy = replace(
        _POLICY,
        candidates=(
            _POLICY.candidates[0],
            replace(_POLICY.candidates[1], eligibility_trigger="totally-different-trigger"),
        ),
    )
    with pytest.raises(CandidatePlanValidationError):
        store.register_candidate_attempt(
            parent_attempt_id=default_attempt.attempt_id,
            attempt_id=fallback_attempt.attempt_id,
            candidate_index=1,
            policy=rogue_policy,
            quality_profile_fingerprint=_QUALITY_PROFILE_FINGERPRINT,
            trigger="totally-different-trigger",
            retry_evaluation=_RETRYABLE_EVALUATION,
        )


# --------------------------------------------------------------------------
# Migration / reopen legacy DB / idempotency
# --------------------------------------------------------------------------


def test_migration_v2_to_v3_is_additive_and_idempotent_on_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy-v2.sqlite3"
    # Build a v2 store the normal way, then hand-roll it back down to
    # user_version=2 with the Slice 2 tables removed, to simulate a database
    # that predates this migration but already has the v1->v2 profile_fingerprint
    # columns.
    seed_store = RecipeAttemptStore(db_path)
    _generated, attempt = _create_and_start_attempt(seed_store, idempotency_key="seed")
    _succeed_attempt(seed_store, attempt.attempt_id)
    del seed_store

    with sqlite3.connect(db_path) as connection:
        connection.execute("DROP TABLE IF EXISTS candidate_attempts;")
        connection.execute("DROP TABLE IF EXISTS recipe_candidate_lineages;")
        connection.execute("PRAGMA user_version = 2;")

    migrated = RecipeAttemptStore(db_path)
    with sqlite3.connect(db_path) as check:
        assert int(check.execute("PRAGMA user_version").fetchone()[0]) == RECIPE_ATTEMPT_STORE_SCHEMA_VERSION
        tables = {
            row[0] for row in check.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }
        assert "candidate_attempts" in tables
        assert "recipe_candidate_lineages" in tables

    # The pre-existing succeeded attempt is untouched and still has no lineage.
    assert migrated.get_attempt(attempt.attempt_id).state == AttemptState.SUCCEEDED
    assert migrated.get_candidate_lineage(attempt.attempt_id) is None

    # A brand-new candidate plan can now be registered against it.
    registered = _register_default(migrated, attempt.attempt_id)
    assert registered.candidate_index == 0

    # Reopening again (migration re-run) is idempotent: no error, same version,
    # existing lineage/candidate data untouched.
    reopened = RecipeAttemptStore(db_path)
    with sqlite3.connect(db_path) as check_again:
        assert int(check_again.execute("PRAGMA user_version").fetchone()[0]) == RECIPE_ATTEMPT_STORE_SCHEMA_VERSION
    assert reopened.get_candidate_lineage(attempt.attempt_id) is not None
    assert len(reopened.list_candidate_attempts(attempt.attempt_id)) == 1


def test_store_rejects_unsupported_schema_version_still_works(tmp_path: Path) -> None:
    db_path = tmp_path / "bad-version.sqlite3"
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA user_version = 999;")
    with pytest.raises(RecipeAttemptMigrationError):
        RecipeAttemptStore(db_path)


# --------------------------------------------------------------------------
# Immutable failed default + verified fallback + atomic one-winner selection
# --------------------------------------------------------------------------


def test_failed_default_and_verified_fallback_atomic_selection(tmp_path: Path) -> None:
    store = RecipeAttemptStore(tmp_path / "recipe-attempt.sqlite3")
    _default_generated, default_attempt = _create_and_start_attempt(store, idempotency_key="default")
    default_candidate = _register_default(store, default_attempt.attempt_id)
    _fail_attempt_at_first_gate(store, default_attempt.attempt_id)

    _fallback_generated, fallback_attempt = _create_and_start_attempt(
        store, idempotency_key="fallback", extra_available_files=("generation_config.json",)
    )
    fallback_candidate = _register_fallback(
        store,
        parent_attempt_id=default_attempt.attempt_id,
        attempt_id=fallback_attempt.attempt_id,
    )
    _succeed_attempt(store, fallback_attempt.attempt_id)

    refreshed_default = store.get_candidate_attempt(default_candidate.candidate_attempt_id)
    assert refreshed_default.attempt_state == AttemptState.FAILED
    assert refreshed_default.is_verified is False

    refreshed_fallback = store.get_candidate_attempt(fallback_candidate.candidate_attempt_id)
    assert refreshed_fallback.attempt_state == AttemptState.SUCCEEDED
    assert refreshed_fallback.is_verified is True

    # Selecting the failed default must be rejected (non-verified child).
    with pytest.raises(CandidateSelectionConflictError):
        store.select_verified_candidate_attempt(
            parent_attempt_id=default_attempt.attempt_id,
            candidate_attempt_id=default_candidate.candidate_attempt_id,
            reason="attempted incorrect selection",
        )

    lineage, winner = store.select_verified_candidate_attempt(
        parent_attempt_id=default_attempt.attempt_id,
        candidate_attempt_id=fallback_candidate.candidate_attempt_id,
        reason="fallback passed quality validation after optimized-only structural regression",
    )
    assert lineage.selection_state == CandidateLineageSelectionState.SELECTED
    assert lineage.selected_candidate_attempt_id == fallback_candidate.candidate_attempt_id
    assert winner.selection_status == CandidateWinnerStatus.SELECTED
    assert winner.selected_by == "validation"
    assert winner.has_fully_validated_selection_scope is False  # no Slice 3 scope supplied yet

    # Never overwrite the failed default: it remains failed/not-selected and
    # fully discoverable alongside the winner.
    all_candidates = store.list_candidate_attempts(default_attempt.attempt_id)
    assert len(all_candidates) == 2
    by_index = {row.candidate_index: row for row in all_candidates}
    assert by_index[0].attempt_state == AttemptState.FAILED
    assert by_index[0].selection_status == CandidateWinnerStatus.NOT_SELECTED
    assert by_index[1].selection_status == CandidateWinnerStatus.SELECTED

    # Only one winner is ever allowed, even for the same candidate again.
    with pytest.raises(CandidateSelectionConflictError):
        store.select_verified_candidate_attempt(
            parent_attempt_id=default_attempt.attempt_id,
            candidate_attempt_id=fallback_candidate.candidate_attempt_id,
            reason="duplicate selection attempt",
        )

    # A finalized lineage cannot accept new candidates either.
    _third_generated, third_attempt = _create_and_start_attempt(
        store, idempotency_key="post-selection", ort_version="1.32.0"
    )
    with pytest.raises(CandidatePlanValidationError):
        store.register_candidate_attempt(
            parent_attempt_id=default_attempt.attempt_id,
            attempt_id=third_attempt.attempt_id,
            candidate_index=1,
            policy=_POLICY,
            quality_profile_fingerprint=_QUALITY_PROFILE_FINGERPRINT,
            trigger=RETRYABLE_OPTIMIZED_STRUCTURAL_REGRESSION_TRIGGER,
            retry_evaluation=_RETRYABLE_EVALUATION,
        )


def test_finalize_candidate_attempt_evidence_is_write_once(tmp_path: Path) -> None:
    store = RecipeAttemptStore(tmp_path / "recipe-attempt.sqlite3")
    _default_generated, default_attempt = _create_and_start_attempt(store, idempotency_key="default")
    default_candidate = _register_default(store, default_attempt.attempt_id)

    # Cannot finalize evidence while the linked attempt is still running.
    with pytest.raises(CandidatePlanValidationError):
        store.finalize_candidate_attempt_evidence(
            default_candidate.candidate_attempt_id,
            artifact_ref="artifact://too-early",
        )

    _succeed_attempt(store, default_attempt.attempt_id)
    counters = CandidateInvocationCounters(mobius_build_invocation_count=1, olive_optimize_invocation_count=1)
    finalized = store.finalize_candidate_attempt_evidence(
        default_candidate.candidate_attempt_id,
        artifact_ref="artifact://default/onnx",
        package_ref="package://default/oga",
        invocation_counters=counters,
    )
    assert finalized.artifact_ref == "artifact://default/onnx"
    assert finalized.package_ref == "package://default/oga"
    assert finalized.invocation_counters.mobius_build_invocation_count == 1
    # Untouched counters remain None -- never inferred/coerced to 0.
    assert finalized.invocation_counters.total_invocation_count is None
    assert finalized.invocation_counters.wall_clock_seconds is None
    assert finalized.invocation_counters.estimated_cost_usd is None

    # Idempotent no-op with identical values.
    same_again = store.finalize_candidate_attempt_evidence(
        default_candidate.candidate_attempt_id,
        artifact_ref="artifact://default/onnx",
        package_ref="package://default/oga",
        invocation_counters=counters,
    )
    assert same_again == finalized

    # Immutable: different values after terminal evidence is recorded raise.
    with pytest.raises(CandidatePlanValidationError):
        store.finalize_candidate_attempt_evidence(
            default_candidate.candidate_attempt_id,
            artifact_ref="artifact://different",
        )


# --------------------------------------------------------------------------
# Fallback exhausted / no selection / no third candidate
# --------------------------------------------------------------------------


def test_lineage_can_be_exhausted_when_every_candidate_fails(tmp_path: Path) -> None:
    store = RecipeAttemptStore(tmp_path / "recipe-attempt.sqlite3")
    _default_generated, default_attempt = _create_and_start_attempt(store, idempotency_key="default")
    _register_default(store, default_attempt.attempt_id)
    _fail_attempt_at_first_gate(store, default_attempt.attempt_id)

    _fallback_generated, fallback_attempt = _create_and_start_attempt(
        store, idempotency_key="fallback", extra_available_files=("generation_config.json",)
    )
    _register_fallback(
        store,
        parent_attempt_id=default_attempt.attempt_id,
        attempt_id=fallback_attempt.attempt_id,
    )
    _fail_attempt_at_first_gate(store, fallback_attempt.attempt_id)

    lineage = store.finalize_exhausted_candidate_lineage(
        default_attempt.attempt_id,
        reason="both candidates failed; policy exhausted with no verified winner",
    )
    assert lineage.selection_state == CandidateLineageSelectionState.EXHAUSTED
    assert lineage.selected_candidate_attempt_id is None

    # No selection remains possible, and no third candidate can ever be added
    # (policy only declares 2, and the lineage is finalized regardless).
    with pytest.raises(CandidateSelectionConflictError):
        store.select_verified_candidate_attempt(
            parent_attempt_id=default_attempt.attempt_id,
            candidate_attempt_id=store.list_candidate_attempts(default_attempt.attempt_id)[0].candidate_attempt_id,
            reason="cannot select after exhaustion",
        )
    _third_generated, third_attempt = _create_and_start_attempt(
        store, idempotency_key="third", ort_version="1.33.0"
    )
    with pytest.raises(CandidatePlanValidationError):
        store.register_candidate_attempt(
            parent_attempt_id=default_attempt.attempt_id,
            attempt_id=third_attempt.attempt_id,
            candidate_index=1,
            policy=_POLICY,
            quality_profile_fingerprint=_QUALITY_PROFILE_FINGERPRINT,
            trigger=RETRYABLE_OPTIMIZED_STRUCTURAL_REGRESSION_TRIGGER,
            retry_evaluation=_RETRYABLE_EVALUATION,
        )


def test_cannot_exhaust_lineage_while_a_verified_candidate_is_unselected(tmp_path: Path) -> None:
    store = RecipeAttemptStore(tmp_path / "recipe-attempt.sqlite3")
    _default_generated, default_attempt = _create_and_start_attempt(store, idempotency_key="default")
    _register_default(store, default_attempt.attempt_id)
    _succeed_attempt(store, default_attempt.attempt_id)

    with pytest.raises(CandidateSelectionConflictError):
        store.finalize_exhausted_candidate_lineage(
            default_attempt.attempt_id,
            reason="should not be allowed while candidate 0 is verified but unselected",
        )


def test_cannot_exhaust_lineage_with_a_still_running_candidate(tmp_path: Path) -> None:
    store = RecipeAttemptStore(tmp_path / "recipe-attempt.sqlite3")
    _default_generated, default_attempt = _create_and_start_attempt(store, idempotency_key="default")
    _register_default(store, default_attempt.attempt_id)

    with pytest.raises(CandidateSelectionConflictError):
        store.finalize_exhausted_candidate_lineage(
            default_attempt.attempt_id,
            reason="candidate is still running",
        )


# --------------------------------------------------------------------------
# Fingerprint stability / invalidation
# --------------------------------------------------------------------------


def test_candidate_fingerprint_is_stable_and_invalidated_by_content_changes() -> None:
    recipe_fingerprint = "a" * 64
    policy_fingerprint = "b" * 64
    override = RecipeQuantizationOverride(block_size=64)

    first = build_candidate_recipe_fingerprint(
        recipe_fingerprint=recipe_fingerprint,
        quantization_override=override,
        policy_fingerprint=policy_fingerprint,
    )
    second = build_candidate_recipe_fingerprint(
        recipe_fingerprint=recipe_fingerprint,
        quantization_override=override,
        policy_fingerprint=policy_fingerprint,
    )
    assert first == second

    different_recipe = build_candidate_recipe_fingerprint(
        recipe_fingerprint="c" * 64,
        quantization_override=override,
        policy_fingerprint=policy_fingerprint,
    )
    assert different_recipe != first

    different_override = build_candidate_recipe_fingerprint(
        recipe_fingerprint=recipe_fingerprint,
        quantization_override=RecipeQuantizationOverride(block_size=32),
        policy_fingerprint=policy_fingerprint,
    )
    assert different_override != first

    no_override = build_candidate_recipe_fingerprint(
        recipe_fingerprint=recipe_fingerprint,
        quantization_override=None,
        policy_fingerprint=policy_fingerprint,
    )
    assert no_override != first

    different_policy = build_candidate_recipe_fingerprint(
        recipe_fingerprint=recipe_fingerprint,
        quantization_override=override,
        policy_fingerprint="d" * 64,
    )
    assert different_policy != first


def test_candidate_fingerprints_differ_across_a_real_default_and_fallback_pair(tmp_path: Path) -> None:
    store = RecipeAttemptStore(tmp_path / "recipe-attempt.sqlite3")
    _default_generated, default_attempt = _create_and_start_attempt(store, idempotency_key="default")
    default_candidate = _register_default(store, default_attempt.attempt_id)
    _fallback_generated, fallback_attempt = _create_and_start_attempt(
        store, idempotency_key="fallback", extra_available_files=("generation_config.json",)
    )
    fallback_candidate = _register_fallback(
        store,
        parent_attempt_id=default_attempt.attempt_id,
        attempt_id=fallback_attempt.attempt_id,
    )
    assert default_candidate.candidate_fingerprint != fallback_candidate.candidate_fingerprint
    assert default_candidate.recipe_fingerprint != fallback_candidate.recipe_fingerprint


def test_candidate_fingerprint_excludes_timestamps_paths_and_uuids(tmp_path: Path) -> None:
    """Registering the same candidate slot content at two different real times,
    under two different candidate_attempt_id UUIDs, must still produce an
    identical candidate_fingerprint -- it must never depend on created_utc,
    candidate_attempt_id, or any filesystem path."""
    store_a = RecipeAttemptStore(tmp_path / "a" / "recipe-attempt.sqlite3")
    store_b = RecipeAttemptStore(tmp_path / "b" / "recipe-attempt.sqlite3")
    _generated_a, attempt_a = _create_and_start_attempt(store_a, idempotency_key="default")
    _generated_b, attempt_b = _create_and_start_attempt(store_b, idempotency_key="default")

    import time as _time

    candidate_a = _register_default(store_a, attempt_a.attempt_id)
    _time.sleep(0.01)
    candidate_b = _register_default(store_b, attempt_b.attempt_id)

    assert candidate_a.candidate_attempt_id != candidate_b.candidate_attempt_id
    assert candidate_a.created_utc != candidate_b.created_utc
    assert candidate_a.candidate_fingerprint == candidate_b.candidate_fingerprint


# --------------------------------------------------------------------------
# Reuse lookup by complete provenance identity
# --------------------------------------------------------------------------


def _reuse_query_for(
    *,
    default_attempt,
    fallback_attempt,
    quality_profile_fingerprint: str = _QUALITY_PROFILE_FINGERPRINT,
    policy_fingerprint: str | None = None,
) -> CandidateSelectionReuseQuery:
    return CandidateSelectionReuseQuery(
        model_id=default_attempt.model_id,
        revision_sha=default_attempt.revision_sha,
        requested_device=default_attempt.requested_device,
        requested_precision=default_attempt.requested_precision,
        compiler_version=default_attempt.compiler_version,
        capability_fingerprint=default_attempt.capability_fingerprint,
        toolchain_fingerprint=default_attempt.toolchain_fingerprint,
        profile_fingerprint=default_attempt.profile_fingerprint,
        quality_profile_fingerprint=quality_profile_fingerprint,
        policy_fingerprint=policy_fingerprint or _POLICY.fingerprint,
    )


def test_reuse_lookup_returns_selected_verified_child_with_full_provenance(tmp_path: Path) -> None:
    store = RecipeAttemptStore(tmp_path / "recipe-attempt.sqlite3")
    _default_generated, default_attempt = _create_and_start_attempt(store, idempotency_key="default")
    default_candidate = _register_default(store, default_attempt.attempt_id)
    _fail_attempt_at_first_gate(store, default_attempt.attempt_id)

    _fallback_generated, fallback_attempt = _create_and_start_attempt(
        store, idempotency_key="fallback", extra_available_files=("generation_config.json",)
    )
    fallback_candidate = _register_fallback(
        store,
        parent_attempt_id=default_attempt.attempt_id,
        attempt_id=fallback_attempt.attempt_id,
    )
    _succeed_attempt(store, fallback_attempt.attempt_id)

    query = _reuse_query_for(default_attempt=default_attempt, fallback_attempt=fallback_attempt)

    # Not yet selected: no reusable winner exists.
    assert store.find_reusable_candidate_selection(query) is None

    store.select_verified_candidate_attempt(
        parent_attempt_id=default_attempt.attempt_id,
        candidate_attempt_id=fallback_candidate.candidate_attempt_id,
        reason="fallback verified after retryable structural regression",
    )

    reusable = store.find_reusable_candidate_selection(query)
    assert reusable is not None
    assert reusable.candidate_attempt_id == fallback_candidate.candidate_attempt_id
    assert reusable.selection_status == CandidateWinnerStatus.SELECTED
    assert reusable.candidate_id == "int4-block-size-64"
    assert reusable.quantization_override_block_size == 64
    assert reusable.policy_fingerprint == _POLICY.fingerprint

    # The negative (failed) default candidate remains discoverable alongside
    # the winner via the ordinary listing API.
    siblings = store.list_candidate_attempts(default_attempt.attempt_id)
    sibling_ids = {row.candidate_attempt_id for row in siblings}
    assert default_candidate.candidate_attempt_id in sibling_ids
    assert fallback_candidate.candidate_attempt_id in sibling_ids

    # A stale toolchain identity misses, exactly like the Slice 1 verified-recipe
    # reuse lookup.
    stale_toolchain = replace(query, toolchain_fingerprint="f" * 64)
    assert store.find_reusable_candidate_selection(stale_toolchain) is None

    # Quality-profile version changes invalidate reuse.
    stale_quality_profile = replace(query, quality_profile_fingerprint="e" * 64)
    assert store.find_reusable_candidate_selection(stale_quality_profile) is None

    # Policy identity changes invalidate reuse.
    stale_policy = replace(query, policy_fingerprint="9" * 64)
    assert store.find_reusable_candidate_selection(stale_policy) is None


def _selected_reuse_fixture(store: RecipeAttemptStore) -> tuple[Any, Any, CandidateAttemptRecord]:
    """Shared Slice 3B2b fixture: a default candidate that fails at its first
    gate, and a verified+selected block64 fallback candidate winner -- the
    same shape as `test_reuse_lookup_returns_selected_verified_child_with_full_provenance`,
    factored out for reuse by the full invalidation-matrix parametrization and
    the measured-zero dispatch evidence tests."""
    _default_generated, default_attempt = _create_and_start_attempt(store, idempotency_key="default")
    _register_default(store, default_attempt.attempt_id)
    _fail_attempt_at_first_gate(store, default_attempt.attempt_id)

    _fallback_generated, fallback_attempt = _create_and_start_attempt(
        store, idempotency_key="fallback", extra_available_files=("generation_config.json",)
    )
    fallback_candidate = _register_fallback(
        store,
        parent_attempt_id=default_attempt.attempt_id,
        attempt_id=fallback_attempt.attempt_id,
    )
    _succeed_attempt(store, fallback_attempt.attempt_id)
    store.select_verified_candidate_attempt(
        parent_attempt_id=default_attempt.attempt_id,
        candidate_attempt_id=fallback_candidate.candidate_attempt_id,
        reason="fallback verified after retryable structural regression",
        validated_target_device="cpu",
        validated_target_ep="CPUExecutionProvider",
        validated_toolchain_fingerprint=fallback_attempt.toolchain_fingerprint,
        validated_environment_scope="foundry-local-onboarding",
    )
    winner = store.get_candidate_attempt(fallback_candidate.candidate_attempt_id)
    return default_attempt, fallback_attempt, winner


@pytest.mark.parametrize(
    ("field_name", "override_value"),
    [
        ("model_id", "totally-different-model"),
        ("revision_sha", "f" * 40),
        ("requested_device", "gpu"),
        ("requested_precision", "int8"),
        ("compiler_version", "9.9.9"),
        ("capability_fingerprint", "a" * 64),
        ("toolchain_fingerprint", "b" * 64),
        ("profile_fingerprint", "c" * 64),
        ("quality_profile_fingerprint", "e" * 64),
        ("policy_fingerprint", "9" * 64),
    ],
)
def test_find_reusable_candidate_selection_rejects_every_reuse_identity_field_mismatch(
    tmp_path: Path,
    field_name: str,
    override_value: str,
) -> None:
    """Slice 3B2b complete exact invalidation matrix: a selected-candidate
    reuse lookup must safely miss (return `None`, never serve the wrong
    winner) whenever the requesting query differs from the winner's
    provenance on *any single one* of the full set of reuse-identity fields
    -- the eight generation-identity fields (model/revision/device/precision/
    compiler/capability/toolchain/profile) plus the quality-validation
    profile fingerprint and the selection-policy fingerprint. Parameterized
    over every field independently, proving no field is ever wildcarded."""
    store = RecipeAttemptStore(tmp_path / "recipe-attempt.sqlite3")
    default_attempt, fallback_attempt, _winner = _selected_reuse_fixture(store)

    query = _reuse_query_for(default_attempt=default_attempt, fallback_attempt=fallback_attempt)
    # Sanity: the exact, unmodified query is a genuine hit before mismatching it.
    assert store.find_reusable_candidate_selection(query) is not None

    mismatched = replace(query, **{field_name: override_value})
    assert store.find_reusable_candidate_selection(mismatched) is None


# --------------------------------------------------------------------------
# Slice 3B2b: persisted measured-zero candidate-selection-reuse dispatch evidence
# --------------------------------------------------------------------------


def test_record_reuse_dispatch_evidence_persists_measured_zero_and_is_idempotent(tmp_path: Path) -> None:
    store = RecipeAttemptStore(tmp_path / "recipe-attempt.sqlite3")
    _default_attempt, _fallback_attempt, winner = _selected_reuse_fixture(store)

    # `reused_attempt_id` must be a real, distinct attempt row (the consumer's
    # own materialized reuse attempt) -- the FK to `attempts` is enforced.
    _reused_generated, reused_attempt = _create_and_start_attempt(
        store, idempotency_key="reuse-consumer-1"
    )

    winner_before = store.get_attempt(winner.attempt_id)
    winner_counters_before = winner.invocation_counters

    reused_attempt_id = reused_attempt.attempt_id
    evidence = store.record_reuse_dispatch_evidence(
        reused_attempt_id=reused_attempt_id,
        source_attempt_id=winner.attempt_id,
        source_candidate_attempt_id=winner.candidate_attempt_id,
        parent_attempt_id=winner.parent_attempt_id,
        policy_id=winner.policy_id,
        policy_version=winner.policy_version,
        policy_fingerprint=winner.policy_fingerprint,
        quality_profile_fingerprint=winner.quality_profile_fingerprint,
    )
    assert evidence.reused_attempt_id == reused_attempt_id
    assert evidence.source_attempt_id == winner.attempt_id
    assert evidence.source_candidate_attempt_id == winner.candidate_attempt_id
    assert evidence.parent_attempt_id == winner.parent_attempt_id
    assert evidence.policy_id == winner.policy_id
    assert evidence.policy_version == winner.policy_version
    assert evidence.policy_fingerprint == winner.policy_fingerprint
    assert evidence.quality_profile_fingerprint == winner.quality_profile_fingerprint
    assert evidence.reused_without_build is True
    assert evidence.runner_dispatch_count == 0
    assert evidence.mobius_invocation_count == 0
    assert evidence.olive_invocation_count == 0

    read_back = store.get_reuse_dispatch_evidence(reused_attempt_id)
    assert read_back is not None
    assert read_back == evidence

    # The winner's own recorded attempt/candidate history -- including its real
    # (non-zero) invocation counters -- is never touched by recording evidence
    # for a completely different reused_attempt_id.
    winner_after = store.get_attempt(winner.attempt_id)
    winner_candidate_after = store.get_candidate_attempt(winner.candidate_attempt_id)
    assert winner_after == winner_before
    assert winner_candidate_after.invocation_counters == winner_counters_before

    # Idempotent: recording again for the SAME reused_attempt_id (e.g. a
    # resumed materialization) never overwrites the first persisted row, even
    # with a different recorded_utc.
    again = store.record_reuse_dispatch_evidence(
        reused_attempt_id=reused_attempt_id,
        source_attempt_id=winner.attempt_id,
        source_candidate_attempt_id=winner.candidate_attempt_id,
        parent_attempt_id=winner.parent_attempt_id,
        policy_id=winner.policy_id,
        policy_version=winner.policy_version,
        policy_fingerprint=winner.policy_fingerprint,
        quality_profile_fingerprint=winner.quality_profile_fingerprint,
        recorded_utc=evidence.recorded_utc + timedelta(hours=1),
    )
    assert again == evidence


def test_get_reuse_dispatch_evidence_returns_none_for_non_reuse_attempt(tmp_path: Path) -> None:
    store = RecipeAttemptStore(tmp_path / "recipe-attempt.sqlite3")
    _generated, attempt = _create_and_start_attempt(store, idempotency_key="plain")
    _succeed_attempt(store, attempt.attempt_id)
    assert store.get_reuse_dispatch_evidence(attempt.attempt_id) is None


# --------------------------------------------------------------------------
# Slice 3B2b: additive migration v4 -> v5 (candidate_reuse_dispatch_evidence)
# --------------------------------------------------------------------------


def test_migration_v4_to_v5_is_additive_and_idempotent_on_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy-v4.sqlite3"
    seed_store = RecipeAttemptStore(db_path)
    _generated, attempt = _create_and_start_attempt(seed_store, idempotency_key="seed")
    _succeed_attempt(seed_store, attempt.attempt_id)
    del seed_store

    with sqlite3.connect(db_path) as connection:
        connection.execute("DROP TABLE IF EXISTS candidate_reuse_dispatch_evidence;")
        connection.execute("PRAGMA user_version = 4;")

    migrated = RecipeAttemptStore(db_path)
    with sqlite3.connect(db_path) as check:
        assert int(check.execute("PRAGMA user_version").fetchone()[0]) == RECIPE_ATTEMPT_STORE_SCHEMA_VERSION
        tables = {
            row[0] for row in check.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }
        assert "candidate_reuse_dispatch_evidence" in tables

    # Pre-existing (legacy, pre-migration) attempt is untouched and has no
    # reuse-dispatch evidence.
    assert migrated.get_attempt(attempt.attempt_id).state == AttemptState.SUCCEEDED
    assert migrated.get_reuse_dispatch_evidence(attempt.attempt_id) is None

    # Reopening again (migration re-run) is idempotent: no error, same version.
    reopened = RecipeAttemptStore(db_path)
    with sqlite3.connect(db_path) as check_again:
        assert int(check_again.execute("PRAGMA user_version").fetchone()[0]) == RECIPE_ATTEMPT_STORE_SCHEMA_VERSION
    assert reopened.get_attempt(attempt.attempt_id).state == AttemptState.SUCCEEDED


# --------------------------------------------------------------------------
# Nullable counters remain null, not inferred zero
# --------------------------------------------------------------------------


def test_nullable_invocation_counters_are_never_inferred_as_zero(tmp_path: Path) -> None:
    store = RecipeAttemptStore(tmp_path / "recipe-attempt.sqlite3")
    _default_generated, default_attempt = _create_and_start_attempt(store, idempotency_key="default")
    default_candidate = _register_default(store, default_attempt.attempt_id)

    assert default_candidate.invocation_counters == CandidateInvocationCounters()
    payload = serialize_candidate_attempt_record(default_candidate)
    assert payload["mobius_build_invocation_count"] is None
    assert payload["olive_optimize_invocation_count"] is None
    assert payload["total_invocation_count"] is None
    assert payload["wall_clock_seconds"] is None
    assert payload["estimated_cost_usd"] is None

    round_tripped = deserialize_candidate_attempt_record(payload)
    assert round_tripped.invocation_counters == CandidateInvocationCounters()


# --------------------------------------------------------------------------
# Cross-module trigger identity enforcement
# --------------------------------------------------------------------------


def test_retry_trigger_constant_matches_quality_retry_disposition_value() -> None:
    """Explicit boundary regression test: if either constant is ever renamed or
    redefined independently, this test (not just the module-level assertion in
    recipe_attempt_store.py) fails, so CI catches the drift even if the
    assertion import path changes."""
    assert (
        RETRYABLE_OPTIMIZED_STRUCTURAL_REGRESSION_TRIGGER
        == QualityRetryDisposition.RETRYABLE_OPTIMIZED_STRUCTURAL_REGRESSION.value
    )


def test_unknown_or_non_verified_candidate_selection_is_rejected(tmp_path: Path) -> None:
    store = RecipeAttemptStore(tmp_path / "recipe-attempt.sqlite3")
    _default_generated, default_attempt = _create_and_start_attempt(store, idempotency_key="default")
    _register_default(store, default_attempt.attempt_id)

    with pytest.raises(KeyError):
        store.get_candidate_attempt("unknown-candidate-attempt-id")

    with pytest.raises(CandidateSelectionConflictError):
        store.select_verified_candidate_attempt(
            parent_attempt_id=default_attempt.attempt_id,
            candidate_attempt_id="unknown-candidate-attempt-id",
            reason="unknown child should never be selectable",
        )


# --------------------------------------------------------------------------
# Reviewer defect: a candidate must share its parent's full generation identity
# --------------------------------------------------------------------------


def test_reviewer_repro_fully_separate_model_b_attempt_cannot_register_as_model_a_fallback(
    tmp_path: Path,
) -> None:
    """Exact reviewer reproduction: a fully successful, completely separate
    (different model_id, and therefore different generation identity) attempt
    must never be registrable as a fallback candidate of another model's
    attempt, even though it independently passed every gate on its own."""
    store = RecipeAttemptStore(tmp_path / "recipe-attempt.sqlite3")
    _model_a_generated, model_a_default = _create_and_start_attempt(
        store, idempotency_key="model-a-default", model_id="example-org/model-a"
    )
    _register_default(store, model_a_default.attempt_id)

    _model_b_generated, model_b_attempt = _create_and_start_attempt(
        store, idempotency_key="model-b-fully-separate", model_id="example-org/model-b"
    )
    _succeed_attempt(store, model_b_attempt.attempt_id)

    with pytest.raises(CandidatePlanValidationError, match="model_id"):
        store.register_candidate_attempt(
            parent_attempt_id=model_a_default.attempt_id,
            attempt_id=model_b_attempt.attempt_id,
            candidate_index=1,
            policy=_POLICY,
            quality_profile_fingerprint=_QUALITY_PROFILE_FINGERPRINT,
            trigger=RETRYABLE_OPTIMIZED_STRUCTURAL_REGRESSION_TRIGGER,
            retry_evaluation=_RETRYABLE_EVALUATION,
        )

    # Model A's lineage remains untouched: still pending, with only its own
    # default candidate -- the Model B attempt was never inserted as a row.
    lineage = store.get_candidate_lineage(model_a_default.attempt_id)
    assert lineage is not None
    assert lineage.selection_state == CandidateLineageSelectionState.PENDING
    assert len(store.list_candidate_attempts(model_a_default.attempt_id)) == 1


@pytest.mark.parametrize(
    ("field_name", "override_value"),
    [
        ("model_id", "totally-different-model"),
        ("revision_sha", "f" * 40),
        ("requested_device", "gpu"),
        ("requested_precision", "int8"),
        ("compiler_version", "9.9.9"),
        ("capability_fingerprint", "a" * 64),
        ("toolchain_fingerprint", "b" * 64),
        ("profile_fingerprint", "c" * 64),
    ],
)
def test_register_candidate_attempt_rejects_every_generation_identity_field_mismatch(
    tmp_path: Path,
    field_name: str,
    override_value: str,
) -> None:
    """Parameterized over every field register_candidate_attempt must compare:
    model_id, revision_sha, requested_device, requested_precision,
    compiler_version, capability_fingerprint, toolchain_fingerprint, and
    profile_fingerprint. A candidate whose linked attempt differs from its
    parent on *any single one* of these must be rejected, and the raised error
    must name the mismatched field."""
    store = RecipeAttemptStore(tmp_path / "recipe-attempt.sqlite3")
    _default_generated, default_attempt = _create_and_start_attempt(store, idempotency_key="default")
    _register_default(store, default_attempt.attempt_id)
    _succeed_attempt(store, default_attempt.attempt_id)

    mismatched_attempt_id = f"mismatched-{field_name}"
    _clone_attempt_row_with_overrides(
        store,
        template_attempt_id=default_attempt.attempt_id,
        new_attempt_id=mismatched_attempt_id,
        new_idempotency_key=mismatched_attempt_id,
        overrides={field_name: override_value},
    )

    with pytest.raises(CandidatePlanValidationError, match=field_name):
        store.register_candidate_attempt(
            parent_attempt_id=default_attempt.attempt_id,
            attempt_id=mismatched_attempt_id,
            candidate_index=1,
            policy=_POLICY,
            quality_profile_fingerprint=_QUALITY_PROFILE_FINGERPRINT,
            trigger=RETRYABLE_OPTIMIZED_STRUCTURAL_REGRESSION_TRIGGER,
            retry_evaluation=_RETRYABLE_EVALUATION,
        )
    # Never inserted: the lineage still has only its default candidate.
    assert len(store.list_candidate_attempts(default_attempt.attempt_id)) == 1


def test_register_candidate_attempt_accepts_same_identity_with_different_recipe_and_block_size(
    tmp_path: Path,
) -> None:
    """Same generation identity as the parent, but a different recipe_fingerprint
    and a different candidate quantization block_size, must be accepted --
    those are the only two things allowed to differ between a parent and its
    fallback candidate."""
    store = RecipeAttemptStore(tmp_path / "recipe-attempt.sqlite3")
    _default_generated, default_attempt = _create_and_start_attempt(store, idempotency_key="default")
    default_candidate = _register_default(store, default_attempt.attempt_id)

    _fallback_generated, fallback_attempt = _create_and_start_attempt(
        store, idempotency_key="fallback", extra_available_files=("generation_config.json",)
    )
    assert fallback_attempt.model_id == default_attempt.model_id
    assert fallback_attempt.revision_sha == default_attempt.revision_sha
    assert fallback_attempt.requested_device == default_attempt.requested_device
    assert fallback_attempt.requested_precision == default_attempt.requested_precision
    assert fallback_attempt.compiler_version == default_attempt.compiler_version
    assert fallback_attempt.capability_fingerprint == default_attempt.capability_fingerprint
    assert fallback_attempt.toolchain_fingerprint == default_attempt.toolchain_fingerprint
    assert fallback_attempt.profile_fingerprint == default_attempt.profile_fingerprint
    assert fallback_attempt.recipe_fingerprint != default_attempt.recipe_fingerprint

    fallback_candidate = _register_fallback(
        store,
        parent_attempt_id=default_attempt.attempt_id,
        attempt_id=fallback_attempt.attempt_id,
    )
    assert fallback_candidate.quantization_override_block_size == 64
    assert default_candidate.quantization_override_block_size is None
    assert fallback_candidate.recipe_fingerprint != default_candidate.recipe_fingerprint


def test_select_verified_candidate_attempt_refuses_corrupt_cross_identity_row(tmp_path: Path) -> None:
    """Defense in depth: even if a candidate row's linked attempt somehow points
    at a different generation identity than its parent (e.g. a row written
    through an unsupported/direct path, or corrupted after the fact),
    ``select_verified_candidate_attempt`` must refuse to select it rather than
    silently letting a cross-identity attempt win."""
    store = RecipeAttemptStore(tmp_path / "recipe-attempt.sqlite3")
    _default_generated, default_attempt = _create_and_start_attempt(store, idempotency_key="default")
    _register_default(store, default_attempt.attempt_id)
    _fail_attempt_at_first_gate(store, default_attempt.attempt_id)

    _fallback_generated, fallback_attempt = _create_and_start_attempt(
        store, idempotency_key="fallback", extra_available_files=("generation_config.json",)
    )
    fallback_candidate = _register_fallback(
        store,
        parent_attempt_id=default_attempt.attempt_id,
        attempt_id=fallback_attempt.attempt_id,
    )
    _succeed_attempt(store, fallback_attempt.attempt_id)

    _cross_generated, cross_attempt = _create_and_start_attempt(
        store, idempotency_key="cross-model", model_id="totally-different-model"
    )
    _succeed_attempt(store, cross_attempt.attempt_id)

    # Simulate a corrupt/tampered row by repointing the already-registered
    # fallback candidate at the fully separate, successful, different-identity
    # attempt -- register_candidate_attempt would never allow this, but a
    # preexisting/corrupt row must still be refused at selection time.
    _repoint_candidate_attempt_to(
        store,
        candidate_attempt_id=fallback_candidate.candidate_attempt_id,
        attempt_id=cross_attempt.attempt_id,
    )

    with pytest.raises(CandidateSelectionConflictError, match="model_id"):
        store.select_verified_candidate_attempt(
            parent_attempt_id=default_attempt.attempt_id,
            candidate_attempt_id=fallback_candidate.candidate_attempt_id,
            reason="attempted selection of a corrupted cross-identity row",
        )

    # No selection was ever committed: the lineage is still pending.
    lineage = store.get_candidate_lineage(default_attempt.attempt_id)
    assert lineage is not None
    assert lineage.selection_state == CandidateLineageSelectionState.PENDING


def test_find_reusable_candidate_selection_refuses_corrupt_selected_row(tmp_path: Path) -> None:
    """Defense in depth: if an already-*selected* winner row is later found to
    point at a different generation identity than its parent (corrupted after
    selection, or written through an unsupported path), the reuse lookup must
    fail closed rather than ever silently returning that winner as reusable."""
    store = RecipeAttemptStore(tmp_path / "recipe-attempt.sqlite3")
    _default_generated, default_attempt = _create_and_start_attempt(store, idempotency_key="default")
    _register_default(store, default_attempt.attempt_id)
    _fail_attempt_at_first_gate(store, default_attempt.attempt_id)

    _fallback_generated, fallback_attempt = _create_and_start_attempt(
        store, idempotency_key="fallback", extra_available_files=("generation_config.json",)
    )
    fallback_candidate = _register_fallback(
        store,
        parent_attempt_id=default_attempt.attempt_id,
        attempt_id=fallback_attempt.attempt_id,
    )
    _succeed_attempt(store, fallback_attempt.attempt_id)
    store.select_verified_candidate_attempt(
        parent_attempt_id=default_attempt.attempt_id,
        candidate_attempt_id=fallback_candidate.candidate_attempt_id,
        reason="fallback verified after retryable structural regression",
    )

    _cross_generated, cross_attempt = _create_and_start_attempt(
        store, idempotency_key="cross-model", model_id="totally-different-model"
    )
    _succeed_attempt(store, cross_attempt.attempt_id)

    # Corrupt the *already-selected* winner row after the fact.
    _repoint_candidate_attempt_to(
        store,
        candidate_attempt_id=fallback_candidate.candidate_attempt_id,
        attempt_id=cross_attempt.attempt_id,
    )

    query = _reuse_query_for(default_attempt=default_attempt, fallback_attempt=fallback_attempt)
    with pytest.raises(CandidateReuseIntegrityError, match="model_id"):
        store.find_reusable_candidate_selection(query)


# --------------------------------------------------------------------------
# Existing attempt-store behavior is unaffected (spot check; full suite also run)
# --------------------------------------------------------------------------


def test_plain_attempts_without_any_candidate_plan_are_unaffected(tmp_path: Path) -> None:
    store = RecipeAttemptStore(tmp_path / "recipe-attempt.sqlite3")
    _generated, attempt = _create_and_start_attempt(store, idempotency_key="no-policy")
    _succeed_attempt(store, attempt.attempt_id)
    assert store.get_attempt(attempt.attempt_id).state == AttemptState.SUCCEEDED
    assert store.get_candidate_lineage(attempt.attempt_id) is None
    assert store.list_candidate_attempts(attempt.attempt_id) == ()


# --------------------------------------------------------------------------
# Slice 3B1: reverse lookup by attempt_id (parent or child)
# --------------------------------------------------------------------------


def test_find_candidate_attempt_by_attempt_id_resolves_parent_and_child(tmp_path: Path) -> None:
    store = RecipeAttemptStore(tmp_path / "recipe-attempt.sqlite3")
    _generated, default_attempt = _create_and_start_attempt(store, idempotency_key="reverse-lookup-default")
    _register_default(store, default_attempt.attempt_id)
    _fail_attempt_at_first_gate(store, default_attempt.attempt_id)

    fallback_generated, fallback_attempt = _create_and_start_attempt(
        store,
        idempotency_key="reverse-lookup-fallback",
        model_id="example-org/candidate-model",
    )
    _succeed_attempt(store, fallback_attempt.attempt_id)
    fallback_candidate = _register_fallback(
        store,
        parent_attempt_id=default_attempt.attempt_id,
        attempt_id=fallback_attempt.attempt_id,
    )

    by_parent = store.find_candidate_attempt_by_attempt_id(default_attempt.attempt_id)
    assert by_parent is not None
    assert by_parent.candidate_index == 0
    assert by_parent.parent_attempt_id == default_attempt.attempt_id
    assert by_parent.attempt_id == default_attempt.attempt_id

    by_child = store.find_candidate_attempt_by_attempt_id(fallback_attempt.attempt_id)
    assert by_child is not None
    assert by_child.candidate_attempt_id == fallback_candidate.candidate_attempt_id
    assert by_child.candidate_index == 1
    assert by_child.parent_attempt_id == default_attempt.attempt_id
    assert by_child.attempt_id == fallback_attempt.attempt_id


def test_find_candidate_attempt_by_attempt_id_returns_none_for_untracked_attempt(tmp_path: Path) -> None:
    store = RecipeAttemptStore(tmp_path / "recipe-attempt.sqlite3")
    _generated, attempt = _create_and_start_attempt(store, idempotency_key="reverse-lookup-none")
    _succeed_attempt(store, attempt.attempt_id)
    assert store.find_candidate_attempt_by_attempt_id(attempt.attempt_id) is None

