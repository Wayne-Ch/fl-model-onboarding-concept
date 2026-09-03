"""Slice 3B1 revision (Basher): regression tests proving `_safe_sync_generated_attempt`
serializes duplicate syncs of the *same* generated attempt through a narrow
per-attempt guard instead of holding the service-wide lock across unbounded
I/O (manifest hashing, quality validation, fallback compilation/launch).

These tests reuse the exact same fake `ProcessRunner`/`TextInferenceBackend`
fixtures as `test_local_service_candidate_orchestration.py` (no real model,
network, or Mobius/Olive tooling involved anywhere).
"""

from __future__ import annotations

import json
import sys
import threading
import time

from pathlib import Path
from threading import Event

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_local_service_candidate_orchestration import (  # noqa: E402
    AlwaysRegressedTextBackend,
    DeterministicQualityTextBackend,
    _create_default_attempt,
    _model,
    _service,
    _wait_for_job_terminal,
    _wait_for_lineage_finalized,
)

import fl_model_onboarding.local_service as local_service_module  # noqa: E402
from fl_model_onboarding.contracts import CandidateModality, JobState  # noqa: E402
from fl_model_onboarding.local_service import _QUALITY_EVIDENCE_FILENAME  # noqa: E402
from fl_model_onboarding.recipe_attempt_store import CandidateLineageSelectionState  # noqa: E402

_REAL_REVALIDATE = local_service_module.revalidate_pre_olive_source


def test_unrelated_calls_stay_responsive_while_fallback_revalidation_is_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default candidate's own *job* already reached a terminal state
    (Mobius/Olive/runtime all succeeded); only the slow candidate-
    orchestration post-processing (revalidating the captured pre-Olive
    artifact before trusting a fallback launch) is still in flight. Under the
    pre-fix global-lock design, that post-processing held `self._lock` for
    its entire duration, so every other `self._lock` user -- an unrelated
    `get_build`, a `cancel_build` on the very same already-terminal job, a
    brand new submission, and `health()` -- would all block until the
    revalidation finished. None of them may block now."""
    entered = Event()
    release = Event()

    def _blocking_revalidate(descriptor):  # noqa: ANN001
        entered.set()
        if not release.wait(timeout=10):
            raise TimeoutError("test never released the blocked revalidation")
        return _REAL_REVALIDATE(descriptor)

    monkeypatch.setattr(local_service_module, "revalidate_pre_olive_source", _blocking_revalidate)

    backend = DeterministicQualityTextBackend()
    service = _service(tmp_path, text_backend=backend)
    try:
        preview = service.generated_recipe_preview(model_id=_model(), task=CandidateModality.LLM)
        fingerprint = str(preview["generated_recipe"]["fingerprint"])
        backend.regress_recipe_fingerprints.add(fingerprint)

        job, attempt, _fp = _create_default_attempt(service, idempotency_key="responsive-1")

        assert entered.wait(timeout=10.0), "fallback revalidation never started"

        # `get_build` never triggers a sync at all; it must always be fast,
        # but it is included here as the simplest possible baseline.
        started = time.monotonic()
        completed = service.get_build(job.job_id)
        assert completed.state == JobState.SUCCEEDED
        assert time.monotonic() - started < 2.0

        # `cancel_build` on the same, already-terminal job: under the old
        # global-lock design this would block for the entire revalidation
        # (cancel_build held `self._lock` across its own call to
        # `_safe_sync_generated_attempt`, and the worker thread was already
        # holding that same lock for the whole sync). It must now return
        # (rejecting the request, since the job is no longer cancellable)
        # almost immediately instead.
        started = time.monotonic()
        with pytest.raises(Exception):
            service.cancel_build(job.job_id)
        assert time.monotonic() - started < 2.0

        # A brand new submission (a different generated-attempt idempotency
        # key) must be accepted promptly -- it only needs a brief
        # `self._lock` section of its own, never the busy per-attempt guard.
        started = time.monotonic()
        new_job, _replay, _new_attempt = service.create_generated_recipe_attempt(
            recipe_fingerprint=fingerprint,
            idempotency_key="responsive-new-submission-1",
            model_id=_model(),
        )
        assert time.monotonic() - started < 2.0
        assert new_job is not None

        started = time.monotonic()
        service.health()
        assert time.monotonic() - started < 2.0
    finally:
        release.set()
        service.close()


def test_duplicate_sync_for_same_attempt_is_serialized_and_launches_fallback_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Firing several concurrent duplicate syncs for the *same* already-
    terminal job/attempt (simulating the worker's own end-of-job sync racing
    a `get_recipe_attempt` poll's lazy re-sync) must never let two threads
    execute the (slow, unbounded) revalidation for that attempt at once, and
    must launch the fallback candidate exactly once."""
    concurrency_lock = threading.Lock()
    state = {"current": 0, "max": 0}

    def _tracking_revalidate(descriptor):  # noqa: ANN001
        with concurrency_lock:
            state["current"] += 1
            state["max"] = max(state["max"], state["current"])
        try:
            time.sleep(0.05)
            return _REAL_REVALIDATE(descriptor)
        finally:
            with concurrency_lock:
                state["current"] -= 1

    monkeypatch.setattr(local_service_module, "revalidate_pre_olive_source", _tracking_revalidate)

    backend = DeterministicQualityTextBackend()
    service = _service(tmp_path, text_backend=backend)
    try:
        preview = service.generated_recipe_preview(model_id=_model(), task=CandidateModality.LLM)
        backend.regress_recipe_fingerprints.add(str(preview["generated_recipe"]["fingerprint"]))

        job, attempt, _fp = _create_default_attempt(service, idempotency_key="duplicate-sync-race-1")
        default_attempt_id = attempt["attempt_id"]
        completed = _wait_for_job_terminal(service, job.job_id)
        assert completed.state == JobState.SUCCEEDED

        # Fire several additional duplicate syncs concurrently to widen the
        # race window regardless of how far the worker's own automatic sync
        # has already gotten.
        threads = [
            threading.Thread(target=lambda: service._safe_sync_generated_attempt(job=completed))
            for _ in range(6)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
            assert not t.is_alive(), "duplicate sync thread never returned (possible deadlock)"

        lineage = _wait_for_lineage_finalized(service, default_attempt_id, timeout_seconds=15.0)
        assert lineage.selection_state == CandidateLineageSelectionState.SELECTED

        candidates = service._recipe_attempt_store.list_candidate_attempts(default_attempt_id)
        assert len(candidates) == 2
        assert sum(1 for row in candidates if row.candidate_index == 1) == 1

        assert state["max"] <= 1, "revalidate_pre_olive_source ran concurrently for the same attempt"
    finally:
        service.close()


def test_cancellation_during_outside_lock_revalidation_prevents_fallback_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancellation signal that lands *while* the outside-lock
    revalidation is in flight must prevent the fallback candidate from ever
    being launched/promoted once the revalidation returns."""
    entered = Event()
    release = Event()

    def _blocking_revalidate(descriptor):  # noqa: ANN001
        entered.set()
        if not release.wait(timeout=10):
            raise TimeoutError("test never released the blocked revalidation")
        return _REAL_REVALIDATE(descriptor)

    monkeypatch.setattr(local_service_module, "revalidate_pre_olive_source", _blocking_revalidate)

    backend = DeterministicQualityTextBackend()
    service = _service(tmp_path, text_backend=backend)
    try:
        preview = service.generated_recipe_preview(model_id=_model(), task=CandidateModality.LLM)
        backend.regress_recipe_fingerprints.add(str(preview["generated_recipe"]["fingerprint"]))

        job, attempt, _fp = _create_default_attempt(service, idempotency_key="cancel-during-revalidate-1")
        default_attempt_id = attempt["attempt_id"]

        assert entered.wait(timeout=10.0), "fallback revalidation never started"

        # Simulate a cancellation landing while revalidation is outstanding --
        # directly set the job's own cancellation event, exactly as
        # `cancel_build` would.
        cancellation_event = service._cancel_events.get(job.job_id)
        assert cancellation_event is not None
        cancellation_event.set()

        release.set()

        completed = _wait_for_job_terminal(service, job.job_id)
        # The default candidate's own build already succeeded before
        # revalidation began; this post-hoc cancellation signal never rewinds
        # that job-level state.
        assert completed.state == JobState.SUCCEEDED

        lineage = _wait_for_lineage_finalized(service, default_attempt_id, timeout_seconds=15.0)
        assert lineage.selection_state == CandidateLineageSelectionState.EXHAUSTED
        assert lineage.selected_candidate_attempt_id is None

        candidates = service._recipe_attempt_store.list_candidate_attempts(default_attempt_id)
        assert len(candidates) == 1  # no fallback candidate ever registered/launched
        assert len(service._recipe_attempt_store.list_verified_recipes()) == 0
    finally:
        service.close()


def test_concurrent_different_attempt_syncs_progress_without_evidence_file_race(tmp_path: Path) -> None:
    """Duplicate/concurrent syncs across *two different* generated attempts
    must never block each other (independent per-attempt guards) and must
    never interleave writes to each attempt's own quality-validation evidence
    file."""
    backend = AlwaysRegressedTextBackend()
    service = _service(tmp_path, text_backend=backend)
    try:
        job_a, attempt_a, _fp_a = _create_default_attempt(service, idempotency_key="concurrent-attempt-a-1")
        attempt_a_id = attempt_a["attempt_id"]
        completed_a = _wait_for_job_terminal(service, job_a.job_id)
        assert completed_a.state == JobState.SUCCEEDED
        _wait_for_lineage_finalized(service, attempt_a_id, timeout_seconds=15.0)

        job_b, attempt_b, _fp_b = _create_default_attempt(service, idempotency_key="concurrent-attempt-b-1")
        attempt_b_id = attempt_b["attempt_id"]
        completed_b = _wait_for_job_terminal(service, job_b.job_id)
        assert completed_b.state == JobState.SUCCEEDED
        _wait_for_lineage_finalized(service, attempt_b_id, timeout_seconds=15.0)

        errors: list[BaseException] = []
        errors_lock = threading.Lock()

        def _sync(job) -> None:  # noqa: ANN001
            try:
                for _ in range(5):
                    service._safe_sync_generated_attempt(job=job)
            except BaseException as exc:  # noqa: BLE001 - captured for the assertion below
                with errors_lock:
                    errors.append(exc)

        threads = [threading.Thread(target=_sync, args=(completed_a,)) for _ in range(4)]
        threads += [threading.Thread(target=_sync, args=(completed_b,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
            assert not t.is_alive(), "concurrent duplicate-sync thread never returned (possible deadlock)"
        assert not errors, f"unexpected exceptions during concurrent sync: {errors}"

        for attempt_id, job in ((attempt_a_id, job_a), (attempt_b_id, job_b)):
            lineage = service._recipe_attempt_store.get_candidate_lineage(attempt_id)
            assert lineage is not None
            assert lineage.selection_state == CandidateLineageSelectionState.EXHAUSTED
            candidates = service._recipe_attempt_store.list_candidate_attempts(attempt_id)
            assert len(candidates) == 2  # default + fallback, never duplicated

            evidence_path = job.request.workspace_root / _QUALITY_EVIDENCE_FILENAME
            assert evidence_path.is_file()
            # A torn/interleaved concurrent write would fail to parse as JSON.
            payload = json.loads(evidence_path.read_text(encoding="utf-8"))
            assert payload["job_id"] == job.job_id
    finally:
        service.close()


def test_repeated_concurrent_sync_stress_no_deadlock(tmp_path: Path) -> None:
    """Many repeated cycles of concurrent duplicate syncs for the same
    attempt must never deadlock."""
    backend = DeterministicQualityTextBackend()
    service = _service(tmp_path, text_backend=backend)
    try:
        job, attempt, _fp = _create_default_attempt(service, idempotency_key="stress-1")
        attempt_id = attempt["attempt_id"]
        completed = _wait_for_job_terminal(service, job.job_id)
        assert completed.state == JobState.SUCCEEDED
        _wait_for_lineage_finalized(service, attempt_id)

        for _ in range(20):
            threads = [
                threading.Thread(target=lambda: service._safe_sync_generated_attempt(job=completed))
                for _ in range(8)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)
                assert not t.is_alive(), "duplicate sync thread never returned (possible deadlock)"

        lineage = service._recipe_attempt_store.get_candidate_lineage(attempt_id)
        assert lineage is not None
        assert lineage.selection_state == CandidateLineageSelectionState.SELECTED
        candidates = service._recipe_attempt_store.list_candidate_attempts(attempt_id)
        assert len(candidates) == 1
    finally:
        service.close()
