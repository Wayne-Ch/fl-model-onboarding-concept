from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from fl_model_onboarding.build_service import (
    ArtifactCapabilityError,
    ArtifactNotReadyError,
    BuildCreateBody,
    BuildService,
    IdempotencyConflictError,
)
from fl_model_onboarding.contracts import (
    ArtifactKind,
    BuildArtifact,
    BuildRequest,
    JobState,
    PreflightResult,
    ToolAvailability,
)
from fl_model_onboarding.job_runner import LocalJobRunner


class FakeInspector:
    def inspect(self, request: BuildRequest) -> PreflightResult:
        return PreflightResult(
            candidate=request.candidate,
            workspace_root=request.workspace_root,
            model_cache_dir=request.model_cache_dir,
            output_dir=request.output_dir,
            disk_free_gb_workspace=100.0,
            disk_free_gb_cache=100.0,
            tools=(
                ToolAvailability("foundry", "command", True, "0.11.0"),
                ToolAvailability("mobius", "command", True, "0.1.0"),
                ToolAvailability("olive", "command", True, "0.13.0"),
                ToolAvailability("onnxruntime", "python-package", True, "1.29.0"),
                ToolAvailability("onnxruntime-genai", "python-package", True, "0.15.2"),
                ToolAvailability("foundry-local-sdk", "python-package", True, "1.2.4"),
                ToolAvailability("huggingface_hub", "python-package", True, "1.22.0"),
            ),
            foundry_catalog_matches=(),
            huggingface_revision="abc",
            huggingface_sha="1234567890abcdef1234567890abcdef12345678",
            huggingface_private=False,
            huggingface_gated=False,
            cache_key="cache-key",
            blockers=(),
            warnings=(),
        )


def _service(tmp_path: Path) -> BuildService:
    return BuildService(
        job_runner=LocalJobRunner(FakeInspector()),  # type: ignore[arg-type]
        model_cache_dir=tmp_path / "cache",
        workspace_base=tmp_path / "w",
    )


def test_idempotency_replays_same_job(tmp_path: Path) -> None:
    service = _service(tmp_path)
    body = BuildCreateBody(candidate_key="smollm2-1.7b-instruct", task_profile="llm-cpu-int4")
    first, replay1 = service.create_build(body, idempotency_key="k1")
    second, replay2 = service.create_build(body, idempotency_key="k1")
    assert replay1 is False
    assert replay2 is True
    assert first.job_id == second.job_id


def test_idempotency_conflict_on_body_change(tmp_path: Path) -> None:
    service = _service(tmp_path)
    body = BuildCreateBody(candidate_key="smollm2-1.7b-instruct", task_profile="llm-cpu-int4")
    service.create_build(body, idempotency_key="k1")
    changed = replace(body, task_profile="llm-cpu-fp16")
    with pytest.raises(IdempotencyConflictError):
        service.create_build(changed, idempotency_key="k1")


def test_event_polling_returns_sequences_after_cursor(tmp_path: Path) -> None:
    service = _service(tmp_path)
    body = BuildCreateBody(candidate_key="smollm2-1.7b-instruct", task_profile="llm-cpu-int4")
    job, _ = service.create_build(body, idempotency_key="k1")
    all_events = service.get_events(job.job_id, after=0)
    assert all_events
    replay = service.get_events(job.job_id, after=all_events[-1].sequence)
    assert replay == ()


def test_inference_requires_succeeded_job_and_capability_match(tmp_path: Path) -> None:
    service = _service(tmp_path)
    body = BuildCreateBody(candidate_key="smollm2-1.7b-instruct", task_profile="llm-cpu-int4")
    job, _ = service.create_build(body, idempotency_key="k1")
    artifact = BuildArtifact(
        artifact_id="artifact-1",
        kind=ArtifactKind.MODEL,
        path=Path("C:\\model"),
        description="model",
    )
    service.record_artifact(job.job_id, artifact)
    assert service.ensure_inference_target("artifact-1", "text").job_id == job.job_id
    with pytest.raises(ArtifactCapabilityError):
        service.ensure_inference_target("artifact-1", "asr")
    job.state = JobState.FAILED
    with pytest.raises(ArtifactNotReadyError):
        service.ensure_inference_target("artifact-1", "text")


def test_record_artifact_is_idempotent_for_replay(tmp_path: Path) -> None:
    service = _service(tmp_path)
    body = BuildCreateBody(candidate_key="smollm2-1.7b-instruct", task_profile="llm-cpu-int4")
    job, _ = service.create_build(body, idempotency_key="k1")
    artifact = BuildArtifact(
        artifact_id="artifact-1",
        kind=ArtifactKind.MODEL,
        path=Path("C:\\model"),
        description="model",
    )
    service.record_artifact(job.job_id, artifact)
    service.record_artifact(job.job_id, artifact)
    assert len(service.get_build(job.job_id).artifacts) == 1
