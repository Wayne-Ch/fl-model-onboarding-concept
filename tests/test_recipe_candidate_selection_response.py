"""Slice 3C1: public backend serialization for candidate plan / timeline /
selection / counters / reuse evidence.

These tests exercise the *real* candidate orchestration fixtures already
established by `test_local_service_candidate_orchestration.py` (fully faked
`ProcessRunner`/`TextInferenceBackend`, no real Mobius/Olive/onnxruntime/
Foundry Local tooling, no real models) and assert on the additive public
response shape `LocalOnboardingService` now serializes for:

  * `generated_recipe_preview()` / `model_detail()` -> `generated_recipe.candidate_plan`
  * `get_recipe_attempt()` -> `workflow_outcome` + `candidate_selection`

Nothing here runs a real model or touches the network.
"""

from __future__ import annotations

import re
import sqlite3
import sys

from pathlib import Path

import jsonschema
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_local_service_candidate_orchestration as tlco  # noqa: E402

from fl_model_onboarding.contracts import CandidateModality, JobState  # noqa: E402
from fl_model_onboarding.local_service import LocalOnboardingService, ServiceError  # noqa: E402
from fl_model_onboarding.recipe_attempt_store import AttemptState  # noqa: E402
from fl_model_onboarding.recipe_selection_policy import (  # noqa: E402
    DEFAULT_CPU_INT4_RECIPE_SELECTION_POLICY,
)

_JOB_REF_RE = re.compile(r"^job://[0-9a-fA-F-]+/(artifact/[0-9a-fA-F-]+|package)$")
_OPENAPI_PATH = Path(__file__).resolve().parent.parent / "contracts" / "openapi.yaml"


def _assert_no_path_leakage(value: str | None) -> None:
    if value is None:
        return
    assert re.match(r"^[A-Za-z]:[\\/]", value) is None, f"absolute path leaked: {value!r}"
    assert "\\" not in value, f"path separator leaked: {value!r}"
    assert _JOB_REF_RE.match(value), f"unexpected reference shape: {value!r}"


def _openapi_nullable_to_json_schema(node: object) -> object:
    """Recursively rewrite OpenAPI 3.0's `nullable: true` sibling-keyword
    convention into the plain-JSON-Schema `oneOf: [<constraint>, {"type":
    "null"}]` shape that a real validator (`jsonschema`) understands.
    `nullable` is an OpenAPI-only keyword, not part of any JSON Schema
    draft, so a strict validator otherwise rejects every genuinely-null
    field the runtime emits (`policy_id`, `max_candidates`, ...). Doing this
    once here -- instead of relying on a hand-parsed, schema-shaped-only
    test -- lets a single test assert the *actual* runtime response payload
    against the *actual* documented contract, `$ref` resolution and all,
    so a real mismatch (like the `lineage_selection_state`/`enum` one this
    revision fixes) cannot slip past review again.
    """
    if isinstance(node, dict):
        rewritten = {key: _openapi_nullable_to_json_schema(value) for key, value in node.items()}
        if rewritten.pop("nullable", False) is True:
            return {"oneOf": [rewritten, {"type": "null"}]}
        return rewritten
    if isinstance(node, list):
        return [_openapi_nullable_to_json_schema(item) for item in node]
    return node


def _candidate_selection_schema_validator() -> jsonschema.Draft7Validator:
    spec = yaml.safe_load(_OPENAPI_PATH.read_text(encoding="utf-8"))
    spec = _openapi_nullable_to_json_schema(spec)
    schema = spec["components"]["schemas"]["RecipeAttemptCandidateSelection"]
    resolver = jsonschema.RefResolver.from_schema(spec)
    return jsonschema.Draft7Validator(schema, resolver=resolver)


_CANDIDATE_SELECTION_VALIDATOR = _candidate_selection_schema_validator()


def _assert_matches_candidate_selection_schema(candidate_selection: dict[str, object]) -> None:
    """Validate an *actual* `candidate_selection` response payload against
    the *actual* `contracts/openapi.yaml` `RecipeAttemptCandidateSelection`
    schema (full `$ref` resolution, `oneOf`/`enum`/nullable semantics --
    not a hand-parsed reimplementation of those rules)."""
    errors = sorted(_CANDIDATE_SELECTION_VALIDATOR.iter_errors(candidate_selection), key=str)
    assert not errors, "\n".join(str(error) for error in errors)


# --- 1. Preview candidate_plan -------------------------------------------


def test_candidate_plan_payload_matches_policy_shape() -> None:
    plan = LocalOnboardingService._candidate_plan_payload(DEFAULT_CPU_INT4_RECIPE_SELECTION_POLICY)
    assert plan["policy_id"] == DEFAULT_CPU_INT4_RECIPE_SELECTION_POLICY.policy_id
    assert plan["policy_version"] == DEFAULT_CPU_INT4_RECIPE_SELECTION_POLICY.version
    assert plan["policy_fingerprint"] == DEFAULT_CPU_INT4_RECIPE_SELECTION_POLICY.fingerprint
    assert plan["max_candidates"] == 2
    candidates = plan["candidates"]
    assert [c["candidate_index"] for c in candidates] == [0, 1]

    default_entry, fallback_entry = candidates
    assert default_entry["candidate_id"] == "default-int4"
    assert default_entry["role"] == "default"
    assert default_entry["quantization_override"] is None
    assert default_entry["eligibility_trigger"] is None

    assert fallback_entry["candidate_id"] == "int4-block-size-64"
    assert fallback_entry["role"] == "quality_retry"
    assert fallback_entry["quantization_override"] == {"block_size": 64}
    assert fallback_entry["eligibility_trigger"] == "retryable_optimized_structural_regression"


def test_generated_recipe_preview_includes_candidate_plan_when_cpu_int4_eligible(tmp_path: Path) -> None:
    service = tlco._service(tmp_path)
    try:
        preview = service.generated_recipe_preview(model_id=tlco._model(), task=CandidateModality.LLM)
        generated = preview["generated_recipe"]
        assert generated["eligible_for_automatic_recipe_attempt"] is True
        plan = generated["candidate_plan"]
        assert plan is not None
        assert plan["max_candidates"] == 2
        assert len(plan["candidates"]) == 2

        # Same shape/wiring is also reachable from model_detail().
        detail = service.model_detail(model_id=tlco._model())
        assert detail["generated_recipe"]["candidate_plan"] == plan
    finally:
        service.close()


def test_generated_recipe_preview_candidate_plan_null_when_not_cpu_int4_eligible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = tlco._service(tmp_path)
    try:
        monkeypatch.setattr(service, "_generated_record_is_cpu_int4_eligible", lambda record: False)
        preview = service.generated_recipe_preview(model_id=tlco._model(), task=CandidateModality.LLM)
        generated = preview["generated_recipe"]
        # A compiled recipe still exists (compile itself is unaffected)...
        assert generated["recipe"] is not None
        # ...but the candidate plan is safely null, never a stale/wrong plan.
        assert generated["candidate_plan"] is None
    finally:
        service.close()


# --- 2. Attempt/timeline candidate_selection ------------------------------


def test_attempt_candidate_selection_default_selected_single_candidate(tmp_path: Path) -> None:
    service = tlco._service(tmp_path)
    try:
        job, attempt, _fp = tlco._create_default_attempt(service, idempotency_key="resp-default-1")
        attempt_id = attempt["attempt_id"]
        assert tlco._wait_for_job_terminal(service, job.job_id).state == JobState.SUCCEEDED
        service.get_recipe_attempt(attempt_id=attempt_id)  # force lazy sync
        tlco._wait_for_lineage_finalized(service, attempt_id)

        status = service.get_recipe_attempt(attempt_id=attempt_id)
        assert status["state"] == "succeeded"
        assert status["workflow_outcome"] == "selected"

        selection = status["candidate_selection"]
        assert selection is not None
        assert selection["policy_id"] == DEFAULT_CPU_INT4_RECIPE_SELECTION_POLICY.policy_id
        assert selection["max_candidates"] == 2
        assert selection["lineage_selection_state"] == "selected"

        candidates = selection["candidates"]
        assert len(candidates) == 1
        entry = candidates[0]
        assert entry["candidate_index"] == 0
        assert entry["role"] == "default"
        assert entry["attempt_state"] == "succeeded"
        assert entry["selection_status"] == "selected"
        assert entry["quantization_override"] is None
        assert entry["eligibility_trigger"] is None
        _assert_no_path_leakage(entry["artifact_ref"])
        _assert_no_path_leakage(entry["package_ref"])

        selected = selection["selected_candidate"]
        assert selected is not None
        assert selected["candidate_index"] == 0
        assert selected["selected_by"] == "validation"
        assert selected["selection_reason"] is not None
        assert selected["selected_utc"] is not None

        # "1/1": a single real candidate contributed both real Mobius and
        # Olive invocations; aggregate is derived, never a hardcoded constant.
        aggregate = selection["aggregate_invocation_counters"]
        assert aggregate["mobius_build_invocation_count"] == 1
        assert aggregate["olive_optimize_invocation_count"] == 1
        assert aggregate["total_invocation_count"] == 2

        _assert_matches_candidate_selection_schema(selection)
    finally:
        service.close()


def test_attempt_candidate_selection_fallback_selected_default_remains_failed(tmp_path: Path) -> None:
    backend = tlco.DeterministicQualityTextBackend()
    service = tlco._service(tmp_path, text_backend=backend)
    try:
        preview = service.generated_recipe_preview(model_id=tlco._model(), task=CandidateModality.LLM)
        default_fingerprint = str(preview["generated_recipe"]["fingerprint"])
        backend.regress_recipe_fingerprints.add(default_fingerprint)

        job, attempt, _fp = tlco._create_default_attempt(service, idempotency_key="resp-fallback-1")
        default_attempt_id = attempt["attempt_id"]
        assert tlco._wait_for_job_terminal(service, job.job_id).state == JobState.SUCCEEDED
        tlco._wait_for_lineage_finalized(service, default_attempt_id)

        default_status = service.get_recipe_attempt(attempt_id=default_attempt_id)
        # The default candidate's own attempt never gets rewritten/relabeled,
        # even though the *workflow* overall succeeded via its fallback.
        assert default_status["state"] == "failed"
        assert default_status["workflow_outcome"] == "selected"

        selection = default_status["candidate_selection"]
        assert selection is not None
        assert selection["lineage_selection_state"] == "selected"
        candidates = {row["candidate_index"]: row for row in selection["candidates"]}
        assert set(candidates) == {0, 1}

        default_entry = candidates[0]
        assert default_entry["attempt_state"] == "failed"
        assert default_entry["selection_status"] == "not_selected"
        assert default_entry["quantization_override"] is None

        fallback_entry = candidates[1]
        assert fallback_entry["attempt_state"] == "succeeded"
        assert fallback_entry["selection_status"] == "selected"
        assert fallback_entry["role"] == "quality_retry"
        assert fallback_entry["quantization_override"] == {"block_size": 64}
        assert fallback_entry["eligibility_trigger"] == "retryable_optimized_structural_regression"
        assert fallback_entry["disposition"] == "retryable_optimized_structural_regression"
        for entry in (default_entry, fallback_entry):
            _assert_no_path_leakage(entry["artifact_ref"])
            _assert_no_path_leakage(entry["package_ref"])

        selected = selection["selected_candidate"]
        assert selected is not None
        assert selected["candidate_index"] == 1

        # Querying the *fallback's own* attempt id resolves the exact same
        # lineage-derived workflow_outcome and candidate set.
        fallback_status = service.get_recipe_attempt(attempt_id=fallback_entry["attempt_id"])
        assert fallback_status["state"] == "succeeded"
        assert fallback_status["workflow_outcome"] == "selected"
        assert {row["candidate_index"] for row in fallback_status["candidate_selection"]["candidates"]} == {0, 1}

        # "1/2": Mobius only ran once (default candidate; fallback reused the
        # pre-Olive artifact), Olive ran once per candidate -- derived from
        # real, persisted per-candidate counters, never hardcoded constants.
        candidate_rows = service._recipe_attempt_store.list_candidate_attempts(default_attempt_id)
        expected_mobius = sum(
            row.invocation_counters.mobius_build_invocation_count or 0 for row in candidate_rows
        )
        expected_olive = sum(
            row.invocation_counters.olive_optimize_invocation_count or 0 for row in candidate_rows
        )
        aggregate = selection["aggregate_invocation_counters"]
        assert aggregate["mobius_build_invocation_count"] == expected_mobius == 1
        assert aggregate["olive_optimize_invocation_count"] == expected_olive == 2

        _assert_matches_candidate_selection_schema(selection)
        _assert_matches_candidate_selection_schema(fallback_status["candidate_selection"])
    finally:
        service.close()


def test_attempt_candidate_selection_exhausted_both_regress(tmp_path: Path) -> None:
    backend = tlco.AlwaysRegressedTextBackend()
    service = tlco._service(tmp_path, text_backend=backend)
    try:
        job, attempt, _fp = tlco._create_default_attempt(service, idempotency_key="resp-exhausted-1")
        default_attempt_id = attempt["attempt_id"]
        assert tlco._wait_for_job_terminal(service, job.job_id).state == JobState.SUCCEEDED
        tlco._wait_for_lineage_finalized(service, default_attempt_id, timeout_seconds=15.0)

        status = service.get_recipe_attempt(attempt_id=default_attempt_id)
        assert status["state"] == "failed"
        assert status["workflow_outcome"] == "exhausted"

        selection = status["candidate_selection"]
        assert selection is not None
        assert selection["lineage_selection_state"] == "exhausted"
        assert selection["selected_candidate"] is None
        assert len(selection["candidates"]) == 2
        assert all(row["selection_status"] == "not_selected" for row in selection["candidates"])
        assert all(row["attempt_state"] == "failed" for row in selection["candidates"])

        _assert_matches_candidate_selection_schema(selection)
    finally:
        service.close()


def test_attempt_candidate_selection_pending_mid_flight(tmp_path: Path) -> None:
    runner = tlco.BlockingNthOccurrenceProcessRunner(
        match=lambda argv: argv[:2] == ("mobius", "build"), occurrence=1
    )
    service = tlco._service(tmp_path, process_runner=runner)
    try:
        job, attempt, _fp = tlco._create_default_attempt(service, idempotency_key="resp-pending-1")
        attempt_id = attempt["attempt_id"]
        assert runner.started.wait(timeout=5.0)
        try:
            status = service.get_recipe_attempt(attempt_id=attempt_id)
            assert status["state"] == "running"
            assert status["workflow_outcome"] == "pending"

            selection = status["candidate_selection"]
            assert selection is not None
            assert selection["lineage_selection_state"] == "pending"
            assert selection["selected_candidate"] is None
            assert len(selection["candidates"]) == 1
            entry = selection["candidates"][0]
            assert entry["candidate_index"] == 0
            assert entry["attempt_state"] == "running"

            # Never-yet-measured evidence stays null -- never coerced to 0.
            counters = entry["invocation_counters"]
            assert counters["mobius_build_invocation_count"] is None
            assert counters["olive_optimize_invocation_count"] is None
            assert counters["total_invocation_count"] is None
            aggregate = selection["aggregate_invocation_counters"]
            assert aggregate["mobius_build_invocation_count"] is None
            assert aggregate["olive_optimize_invocation_count"] is None
            assert aggregate["total_invocation_count"] is None

            _assert_matches_candidate_selection_schema(selection)
        finally:
            runner.release.set()
        assert tlco._wait_for_job_terminal(service, job.job_id).state == JobState.SUCCEEDED
    finally:
        service.close()


def test_attempt_candidate_selection_reused_default_winner_measured_zero_evidence(tmp_path: Path) -> None:
    service = tlco._service(tmp_path)
    try:
        source_job, source_attempt, source_fingerprint = tlco._create_default_attempt(
            service, idempotency_key="resp-reuse-default-source-1"
        )
        assert tlco._wait_for_job_terminal(service, source_job.job_id).state == JobState.SUCCEEDED
        source_attempt_id = source_attempt["attempt_id"]
        service.get_recipe_attempt(attempt_id=source_attempt_id)
        tlco._wait_for_lineage_finalized(service, source_attempt_id)
        source_candidate = service._recipe_attempt_store.list_candidate_attempts(source_attempt_id)[0]

        _reused_job, replay, reused_attempt = service.create_generated_recipe_attempt(
            recipe_fingerprint=source_fingerprint,
            idempotency_key="resp-reuse-default-consumer-1",
            model_id=tlco._model(),
        )
        assert replay is False
        assert reused_attempt["state"] == "succeeded"
        assert reused_attempt["workflow_outcome"] == "reused"

        reuse = reused_attempt["candidate_selection"]["reuse"]
        assert reuse is not None
        assert reuse["reused_without_build"] is True
        assert reuse["source_attempt_id"] == source_candidate.attempt_id
        assert reuse["source_candidate_attempt_id"] == source_candidate.candidate_attempt_id
        assert reuse["source_parent_attempt_id"] == source_attempt_id
        assert reuse["runner_dispatch_count"] == 0
        assert reuse["mobius_invocation_count"] == 0
        assert reuse["olive_invocation_count"] == 0
        assert reuse["recorded_utc"] is not None
        assert reused_attempt["candidate_selection"]["candidates"] == []
        assert reused_attempt["candidate_selection"]["selected_candidate"] is None
        # A candidate-selection-reuse materialization has no lineage of its
        # own to report a selection state for: the *actual* runtime payload
        # emits `lineage_selection_state: None`, which must validate as a
        # real `null` against the OpenAPI schema (not silently rejected by
        # an enum that never listed `null`).
        assert reused_attempt["candidate_selection"]["lineage_selection_state"] is None
        _assert_matches_candidate_selection_schema(reused_attempt["candidate_selection"])

        # Poll path (get_recipe_attempt) reports the exact same summary.
        polled = service.get_recipe_attempt(attempt_id=reused_attempt["attempt_id"])
        assert polled["workflow_outcome"] == "reused"
        assert polled["candidate_selection"]["reuse"] == reuse
        assert polled["candidate_selection"]["lineage_selection_state"] is None
        _assert_matches_candidate_selection_schema(polled["candidate_selection"])
    finally:
        service.close()


def test_attempt_candidate_selection_reused_block64_winner_measured_zero_evidence(tmp_path: Path) -> None:
    backend = tlco.DeterministicQualityTextBackend()
    service = tlco._service(tmp_path, text_backend=backend)
    try:
        default_preview = service.generated_recipe_preview(model_id=tlco._model(), task=CandidateModality.LLM)
        default_fingerprint = str(default_preview["generated_recipe"]["fingerprint"])
        backend.regress_recipe_fingerprints.add(default_fingerprint)

        source_job, source_attempt, _fp = tlco._create_default_attempt(
            service, idempotency_key="resp-reuse-block64-source-1"
        )
        assert tlco._wait_for_job_terminal(service, source_job.job_id).state == JobState.SUCCEEDED
        default_attempt_id = source_attempt["attempt_id"]
        tlco._wait_for_lineage_finalized(service, default_attempt_id, timeout_seconds=15.0)
        fallback_candidate = next(
            row
            for row in service._recipe_attempt_store.list_candidate_attempts(default_attempt_id)
            if row.candidate_index == 1
        )

        _reused_job, replay, reused_attempt = service.create_generated_recipe_attempt(
            recipe_fingerprint=default_fingerprint,
            idempotency_key="resp-reuse-block64-consumer-1",
            model_id=tlco._model(),
        )
        assert replay is False
        assert reused_attempt["state"] == "succeeded"
        assert reused_attempt["workflow_outcome"] == "reused"

        reuse = reused_attempt["candidate_selection"]["reuse"]
        assert reuse is not None
        assert reuse["source_attempt_id"] == fallback_candidate.attempt_id
        assert reuse["source_candidate_attempt_id"] == fallback_candidate.candidate_attempt_id
        assert reuse["mobius_invocation_count"] == 0
        assert reuse["olive_invocation_count"] == 0
        assert reuse["runner_dispatch_count"] == 0
        # Same invariant as the default-winner reuse case: no lineage of
        # its own, so the actual runtime response is `None` here too.
        assert reused_attempt["candidate_selection"]["lineage_selection_state"] is None
        _assert_matches_candidate_selection_schema(reused_attempt["candidate_selection"])
    finally:
        service.close()


def test_attempt_candidate_selection_not_applicable_for_legacy_generated_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A generated attempt that predates/falls outside candidate-selection
    eligibility (no lineage ever registered) omits the richer summary
    safely: `workflow_outcome` is the explicit `"not_applicable"` code and
    `candidate_selection` is `null`, while every pre-3C1 field is completely
    unaffected -- exercising backward compatibility for legacy/static
    attempt payloads.
    """
    service = tlco._service(tmp_path)
    try:
        monkeypatch.setattr(service, "_generated_record_is_cpu_int4_eligible", lambda record: False)
        job, attempt, _fp = tlco._create_default_attempt(service, idempotency_key="resp-not-applicable-1")
        attempt_id = attempt["attempt_id"]
        assert tlco._wait_for_job_terminal(service, job.job_id).state == JobState.SUCCEEDED
        assert service._recipe_attempt_store.get_candidate_lineage(attempt_id) is None

        status = service.get_recipe_attempt(attempt_id=attempt_id)
        assert status["state"] == "succeeded"
        assert status["workflow_outcome"] == "not_applicable"
        assert status["candidate_selection"] is None
        # Every legacy/pre-3C1 field remains present and correct.
        assert status["attempt_id"] == attempt_id
        assert status["quality_validation"] is not None
        assert isinstance(status["gates"], list) and status["gates"]
    finally:
        service.close()


def test_attempt_candidate_selection_integrity_error_surfaces_on_corrupt_lineage(tmp_path: Path) -> None:
    service = tlco._service(tmp_path)
    try:
        job, attempt, _fp = tlco._create_default_attempt(service, idempotency_key="resp-corrupt-1")
        attempt_id = attempt["attempt_id"]
        assert tlco._wait_for_job_terminal(service, job.job_id).state == JobState.SUCCEEDED
        service.get_recipe_attempt(attempt_id=attempt_id)
        tlco._wait_for_lineage_finalized(service, attempt_id)

        # Simulate on-disk corruption: a candidate row survives without its
        # parent lineage row, violating the store's own invariant.
        db_path = service._recipe_attempt_store.db_path
        connection = sqlite3.connect(db_path)
        try:
            connection.execute(
                "DELETE FROM recipe_candidate_lineages WHERE parent_attempt_id = ?", (attempt_id,)
            )
            connection.commit()
        finally:
            connection.close()

        with pytest.raises(ServiceError) as exc_info:
            service.get_recipe_attempt(attempt_id=attempt_id)
        assert exc_info.value.status_code == 500
        assert exc_info.value.code == "RECIPE_ATTEMPT_STORE_ERROR"
    finally:
        service.close()


def test_attempt_candidate_selection_list_candidate_attempts_store_error_is_structured(
    tmp_path: Path,
) -> None:
    """`_candidate_selection_summary_for_attempt` must wrap a corrupt
    `list_candidate_attempts(...)` read (a candidate row whose own linked
    attempt row has gone missing -- the store's own invariant is that every
    registered candidate always has a live linked attempt) in the exact
    same structured `RECIPE_ATTEMPT_STORE_ERROR` 500 `ServiceError` handling
    as the adjacent `get_candidate_lineage(...)` call, instead of letting
    the raw `RecipeAttemptStoreError` escape uncaught -- and without
    exposing the on-disk sqlite path or any raw row data in the surfaced
    message."""
    backend = tlco.DeterministicQualityTextBackend()
    service = tlco._service(tmp_path, text_backend=backend)
    try:
        default_preview = service.generated_recipe_preview(model_id=tlco._model(), task=CandidateModality.LLM)
        default_fingerprint = str(default_preview["generated_recipe"]["fingerprint"])
        backend.regress_recipe_fingerprints.add(default_fingerprint)

        job, attempt, _fp = tlco._create_default_attempt(service, idempotency_key="resp-list-corrupt-1")
        default_attempt_id = attempt["attempt_id"]
        assert tlco._wait_for_job_terminal(service, job.job_id).state == JobState.SUCCEEDED
        tlco._wait_for_lineage_finalized(service, default_attempt_id, timeout_seconds=15.0)

        fallback_candidate = next(
            row
            for row in service._recipe_attempt_store.list_candidate_attempts(default_attempt_id)
            if row.candidate_index == 1
        )

        # Simulate on-disk corruption downstream of the lineage row itself:
        # the fallback candidate's own linked attempt row goes missing
        # while its `candidate_attempts` row (and the parent lineage row)
        # survive untouched -- this only ever surfaces inside
        # `list_candidate_attempts`, never `get_candidate_lineage`.
        db_path = service._recipe_attempt_store.db_path
        connection = sqlite3.connect(db_path)
        try:
            connection.execute("DELETE FROM attempts WHERE attempt_id = ?", (fallback_candidate.attempt_id,))
            connection.commit()
        finally:
            connection.close()

        # The lineage row itself is untouched, so this proves the failure
        # is specifically from `list_candidate_attempts`, not the earlier
        # `get_candidate_lineage` call in the same method.
        assert service._recipe_attempt_store.get_candidate_lineage(default_attempt_id) is not None

        with pytest.raises(ServiceError) as exc_info:
            service.get_recipe_attempt(attempt_id=default_attempt_id)
        assert exc_info.value.status_code == 500
        assert exc_info.value.code == "RECIPE_ATTEMPT_STORE_ERROR"
        message = str(exc_info.value)
        assert str(db_path) not in message
        assert "\\" not in message
    finally:
        service.close()

