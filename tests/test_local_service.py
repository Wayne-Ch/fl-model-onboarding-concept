from __future__ import annotations

import time
import sqlite3

from datetime import datetime, timezone
from pathlib import Path

import pytest

from fl_model_onboarding.contracts import (
    ArtifactKind,
    BuildArtifact,
    BuildJob,
    BuildRequest,
    CandidateModality,
    FailureClassification,
    FailureInfo,
    JobState,
    PreflightResult,
    ToolAvailability,
)
from fl_model_onboarding.local_service import (
    BuildSubmission,
    LocalOnboardingService,
    ServiceError,
    enforce_loopback_host,
    is_loopback_host,
)
from fl_model_onboarding.production_runner import production_package_paths
from fl_model_onboarding.state_machine import transition

_TOOLS = (
    ToolAvailability("foundry", "command", True, "0.11.0"),
    ToolAvailability("mobius", "command", True, "0.1.0"),
    ToolAvailability("olive", "command", True, "0.13.0"),
    ToolAvailability("onnxruntime", "python-package", True, "1.29.0"),
    ToolAvailability("onnxruntime-genai", "python-package", True, "0.15.2"),
    ToolAvailability("foundry-local-sdk", "python-package", True, "1.2.4"),
    ToolAvailability("huggingface_hub", "python-package", True, "1.22.0"),
)


class PassingPreflightInspector:
    def inspect(self, request: BuildRequest) -> PreflightResult:
        return PreflightResult(
            candidate=request.candidate,
            workspace_root=request.workspace_root,
            model_cache_dir=request.model_cache_dir,
            output_dir=request.output_dir,
            disk_free_gb_workspace=100.0,
            disk_free_gb_cache=100.0,
            tools=_TOOLS,
            foundry_catalog_matches=(),
            huggingface_revision=request.hf_revision or "rev-1",
            huggingface_sha="1234567890abcdef1234567890abcdef12345678",
            huggingface_private=False,
            huggingface_gated=False,
            cache_key=f"cache::{request.candidate.huggingface_model_id}::{request.task_profile}",
            blockers=(),
            warnings=(),
        )


class BlockingPreflightInspector(PassingPreflightInspector):
    def inspect(self, request: BuildRequest) -> PreflightResult:
        base = super().inspect(request)
        return PreflightResult(
            candidate=base.candidate,
            workspace_root=base.workspace_root,
            model_cache_dir=base.model_cache_dir,
            output_dir=base.output_dir,
            disk_free_gb_workspace=base.disk_free_gb_workspace,
            disk_free_gb_cache=base.disk_free_gb_cache,
            tools=base.tools,
            foundry_catalog_matches=base.foundry_catalog_matches,
            huggingface_revision=base.huggingface_revision,
            huggingface_sha=base.huggingface_sha,
            huggingface_private=base.huggingface_private,
            huggingface_gated=base.huggingface_gated,
            cache_key=base.cache_key,
            blockers=(
                FailureInfo(
                    stage=JobState.MOBIUS_BUILDING,
                    classification=FailureClassification.TOOL_UNAVAILABLE,
                    message="mobius adapter unavailable in this environment",
                ),
            ),
            warnings=base.warnings,
        )


class DeterministicSuccessRunner:
    def run(self, job: BuildJob, *, persist, cancellation_event):  # noqa: ANN001
        if cancellation_event.is_set():
            return
        for stage in (
            JobState.DOWNLOADING,
            JobState.MOBIUS_BUILDING,
            JobState.MOBIUS_VALIDATING,
            JobState.OLIVE_OPTIMIZING,
            JobState.PACKAGING,
            JobState.RUNTIME_VALIDATING,
            JobState.FL_LOADING,
            JobState.INFERENCING,
        ):
            transition(job, stage, f"Reached '{stage.value}'")
            persist()
            if cancellation_event.is_set():
                return
        artifact_path = job.request.output_dir / "artifact.json"
        artifact_path.write_text("{}", encoding="utf-8")
        job.register_artifact(
            BuildArtifact(
                artifact_id=f"artifact-{job.job_id}",
                kind=ArtifactKind.MODEL,
                path=artifact_path,
                description="test artifact",
                size_bytes=artifact_path.stat().st_size,
            )
        )
        transition(job, JobState.SUCCEEDED, "Build succeeded.")
        job.finished_utc = datetime.now(timezone.utc)
        persist()


class SlowCancellableRunner:
    def run(self, job: BuildJob, *, persist, cancellation_event):  # noqa: ANN001
        if cancellation_event.is_set():
            return
        transition(job, JobState.DOWNLOADING, "Downloading model data.")
        partial = job.request.output_dir / "partial.bin"
        partial.write_bytes(b"x")
        persist()
        for _ in range(100):
            if cancellation_event.is_set():
                return
            time.sleep(0.02)


def _submission(model_id: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct") -> BuildSubmission:
    return BuildSubmission(
        model_id=model_id,
        task=CandidateModality.LLM,
        task_profile="llm-cpu-int4",
    )


def _service(
    tmp_path: Path,
    *,
    preflight_inspector=None,
    build_stage_runner=None,
) -> LocalOnboardingService:
    return LocalOnboardingService(
        db_path=tmp_path / "state.sqlite3",
        workspace_base=tmp_path / "w",
        model_cache_dir=tmp_path / "cache",
        preflight_inspector=preflight_inspector or PassingPreflightInspector(),  # type: ignore[arg-type]
        build_stage_runner=build_stage_runner,  # type: ignore[arg-type]
    )


def _wait_for_terminal(service: LocalOnboardingService, job_id: str, timeout_seconds: float = 5.0) -> BuildJob:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        job = service.get_build(job_id)
        if job.state in {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}:
            return job
        time.sleep(0.02)
    raise AssertionError(f"Timed out waiting for terminal state for job {job_id}")


def test_loopback_guard_blocks_non_loopback_without_opt_in() -> None:
    assert is_loopback_host("127.0.0.1")
    assert is_loopback_host("localhost")
    assert enforce_loopback_host("127.0.0.1", allow_non_loopback=False) is None
    with pytest.raises(ValueError):
        enforce_loopback_host("0.0.0.0", allow_non_loopback=False)
    warning = enforce_loopback_host("0.0.0.0", allow_non_loopback=True)
    assert warning is not None
    assert "WARNING" in warning


def test_preflight_cache_reports_cached_flag(tmp_path: Path) -> None:
    service = _service(tmp_path)
    try:
        first = service.preflight(_submission())
        second = service.preflight(_submission())
        assert first["ok"] is True
        assert first["cached"] is False
        assert second["cached"] is True
        assert first["cache_key"] == second["cache_key"]
    finally:
        service.close()


def test_idempotency_replay_and_conflict(tmp_path: Path) -> None:
    service = _service(tmp_path, build_stage_runner=SlowCancellableRunner())
    try:
        body = _submission()
        first, replay1 = service.create_build(body, idempotency_key="k-1")
        second, replay2 = service.create_build(body, idempotency_key="k-1")
        assert replay1 is False
        assert replay2 is True
        assert first.job_id == second.job_id

        changed = BuildSubmission(
            model_id=body.model_id,
            task=body.task,
            task_profile="llm-cpu-fp16",
            hf_revision=body.hf_revision,
            skip_olive=body.skip_olive,
        )
        with pytest.raises(ServiceError) as exc:
            service.create_build(changed, idempotency_key="k-1")
        assert exc.value.status_code == 409
        assert exc.value.code == "IDEMPOTENCY_BODY_CONFLICT"
    finally:
        service.close()


def test_sqlite_restart_replays_build_and_events(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    workspace_base = tmp_path / "w"
    cache = tmp_path / "cache"
    service = LocalOnboardingService(
        db_path=db_path,
        workspace_base=workspace_base,
        model_cache_dir=cache,
        preflight_inspector=PassingPreflightInspector(),  # type: ignore[arg-type]
    )
    try:
        job, _ = service.create_build(_submission(), idempotency_key="k-restart")
        completed = _wait_for_terminal(service, job.job_id)
        assert completed.events
        total_events = len(completed.events)
    finally:
        service.close()

    resumed = LocalOnboardingService(
        db_path=db_path,
        workspace_base=workspace_base,
        model_cache_dir=cache,
        preflight_inspector=PassingPreflightInspector(),  # type: ignore[arg-type]
    )
    try:
        reloaded = resumed.get_build(job.job_id)
        assert reloaded.state in {JobState.FAILED, JobState.CANCELLED, JobState.SUCCEEDED}
        replay = resumed.get_events(job.job_id, after=0)
        assert len(replay) == total_events
        assert resumed.get_events(job.job_id, after=replay[-1].sequence) == ()
    finally:
        resumed.close()


def test_background_build_reports_not_verified_failure(tmp_path: Path) -> None:
    service = _service(tmp_path)
    try:
        job, _ = service.create_build(_submission(), idempotency_key="k-fail")
        completed = _wait_for_terminal(service, job.job_id)
        assert completed.state == JobState.FAILED
        assert completed.failure is not None
        assert completed.failure.stage == JobState.MOBIUS_BUILDING
        assert completed.failure.classification in {
            FailureClassification.NOT_VERIFIED,
            FailureClassification.TOOL_UNAVAILABLE,
        }
    finally:
        service.close()


def test_cancellation_quarantines_partial_output(tmp_path: Path) -> None:
    service = _service(tmp_path, build_stage_runner=SlowCancellableRunner())
    try:
        job, _ = service.create_build(_submission(), idempotency_key="k-cancel")
        partial_path = job.request.output_dir / "partial.bin"
        deadline = time.time() + 5.0
        while time.time() < deadline and not partial_path.exists():
            time.sleep(0.02)
        cancelled, quarantine = service.cancel_build(job.job_id, reason="cancelled by test")
        assert cancelled.state == JobState.CANCELLED
        assert quarantine is not None
        assert quarantine.exists()
        assert (quarantine / "partial.bin").exists()
    finally:
        service.close()


def test_artifact_task_gating_and_failed_artifact_block(tmp_path: Path) -> None:
    service = _service(tmp_path, build_stage_runner=DeterministicSuccessRunner())
    try:
        job, _ = service.create_build(_submission(), idempotency_key="k-success")
        completed = _wait_for_terminal(service, job.job_id)
        assert completed.state == JobState.SUCCEEDED
        assert completed.result_artifact_id is not None

        with pytest.raises(ServiceError) as mismatch:
            service.infer_asr(
                artifact_id=completed.result_artifact_id,
                audio_bytes=b"\x00\x01",
                filename="clip.wav",
            )
        assert mismatch.value.status_code == 409
        assert mismatch.value.code == "ARTIFACT_TASK_MISMATCH"

        with pytest.raises(ServiceError) as missing_backend:
            service.infer_text(
                artifact_id=completed.result_artifact_id,
                prompt="hello",
                max_tokens=32,
            )
        assert missing_backend.value.status_code == 501
        assert missing_backend.value.code == "INFERENCE_NOT_IMPLEMENTED"
    finally:
        service.close()

    service_failed = _service(tmp_path / "failed")
    try:
        failed_job, _ = service_failed.create_build(_submission(), idempotency_key="k-failed")
        failed = _wait_for_terminal(service_failed, failed_job.job_id)
        assert failed.state == JobState.FAILED
        service_failed.record_artifact(
            failed.job_id,
            BuildArtifact(
                artifact_id=f"artifact-{failed.job_id}",
                kind=ArtifactKind.MODEL,
                path=Path("C:\\artifact"),
                description="failed artifact",
            ),
        )
        with pytest.raises(ServiceError) as not_ready:
            service_failed.infer_text(
                artifact_id=f"artifact-{failed.job_id}",
                prompt="hello",
                max_tokens=32,
            )
        assert not_ready.value.status_code == 409
        assert not_ready.value.code == "ARTIFACT_NOT_READY"
    finally:
        service_failed.close()


def test_blocking_preflight_records_tool_unavailable(tmp_path: Path) -> None:
    service = _service(tmp_path, preflight_inspector=BlockingPreflightInspector())
    try:
        job, _ = service.create_build(_submission(), idempotency_key="k-tool-unavailable")
        completed = _wait_for_terminal(service, job.job_id)
        assert completed.state == JobState.FAILED
        assert completed.failure is not None
        assert completed.failure.classification == FailureClassification.TOOL_UNAVAILABLE
    finally:
        service.close()


def test_restart_marks_interrupted_job_failed_instead_of_requeueing_invalid_state(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path, build_stage_runner=SlowCancellableRunner())
    job, _ = service.create_build(_submission(), idempotency_key="interrupted")
    deadline = time.time() + 5
    while time.time() < deadline and service.get_build(job.job_id).state == JobState.QUEUED:
        time.sleep(0.02)
    interrupted = service.get_build(job.job_id)
    staging, package = production_package_paths(interrupted)
    staging.mkdir(parents=True)
    package.mkdir(parents=True)
    service.close()

    resumed = _service(tmp_path)
    try:
        recovered = resumed.get_build(job.job_id)
        assert recovered.state == JobState.FAILED
        assert recovered.failure is not None
        assert recovered.failure.classification == FailureClassification.NOT_VERIFIED
        assert "not resumed" in recovered.failure.message
        assert not staging.exists()
        assert not package.exists()
    finally:
        resumed.close()


def test_legacy_tested_index_is_migrated_to_profile_table(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE tested_models (
                model_id TEXT PRIMARY KEY,
                task TEXT NOT NULL,
                artifact_id TEXT NOT NULL,
                verified_utc TEXT NOT NULL,
                evidence TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO tested_models
                (model_id, task, artifact_id, verified_utc, evidence)
            VALUES
                ('owner/model', 'llm', 'artifact-1', '2026-08-31T00:00:00+00:00',
                 'successful_fl_inference')
            """
        )

    service = LocalOnboardingService(
        db_path=db_path,
        workspace_base=tmp_path / "w",
        model_cache_dir=tmp_path / "cache",
        preflight_inspector=PassingPreflightInspector(),  # type: ignore[arg-type]
    )
    try:
        tested = service.health()["compatibility_index"]
        assert tested[0]["model_id"] == "owner/model"  # type: ignore[index]
        assert tested[0]["revision"] == ""  # type: ignore[index]
        assert tested[0]["task_profile"] == ""  # type: ignore[index]
    finally:
        service.close()
