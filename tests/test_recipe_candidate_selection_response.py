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

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_local_service_candidate_orchestration as tlco  # noqa: E402

from fl_model_onboarding.contracts import CandidateModality, JobState  # noqa: E402
from fl_model_onboarding.local_service import LocalOnboardingService, ServiceError  # noqa: E402
from fl_model_onboarding.recipe_attempt_store import AttemptState  # noqa: E402
from fl_model_onboarding.recipe_selection_policy import (  # noqa: E402
    DEFAULT_CPU_INT4_RECIPE_SELECTION_POLICY,
)

_JOB_REF_RE = re.compile(r"^job://[0-9a-fA-F-]+/(artifact/[0-9a-fA-F-]+|package)$")


def _assert_no_path_leakage(value: str | None) -> None:
    if value is None:
        return
    assert re.match(r"^[A-Za-z]:[\\/]", value) is None, f"absolute path leaked: {value!r}"
    assert "\\" not in value, f"path separator leaked: {value!r}"
    assert _JOB_REF_RE.match(value), f"unexpected reference shape: {value!r}"


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

        # Poll path (get_recipe_attempt) reports the exact same summary.
        polled = service.get_recipe_attempt(attempt_id=reused_attempt["attempt_id"])
        assert polled["workflow_outcome"] == "reused"
        assert polled["candidate_selection"]["reuse"] == reuse
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
