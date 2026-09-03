from __future__ import annotations

import json
import sqlite3
import time

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from threading import Event

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
from fl_model_onboarding.recipes import (
    DEFAULT_MODEL_RECIPES,
    GRANITE_MODEL_ID,
    RecipeRegistry,
    RecipeStatus,
)
from fl_model_onboarding.state_machine import fail_job, transition

_TOOLS = (
    ToolAvailability("foundry", "command", True, "0.11.0"),
    ToolAvailability("mobius", "command", True, "0.1.0"),
    ToolAvailability("olive", "command", True, "0.13.0"),
    ToolAvailability("onnxruntime", "python-package", True, "1.29.0"),
    ToolAvailability("onnxruntime-genai", "python-package", True, "0.15.2"),
    ToolAvailability("foundry-local-sdk", "python-package", True, "1.2.4"),
    ToolAvailability("huggingface_hub", "python-package", True, "1.22.0"),
)


class FakeHFMetadata:
    def search_models(self, query: str, limit: int = 20, sort: str = "downloads"):  # noqa: ANN001
        from fl_model_onboarding.adapters.interfaces import HuggingFaceSearchResult

        return (
            HuggingFaceSearchResult(
                model_id=f"{query}-one",
                downloads=100,
                likes=20,
                last_modified="2026-01-01T00:00:00Z",
            ),
            HuggingFaceSearchResult(
                model_id=f"{query}-two",
                downloads=50,
                likes=10,
                last_modified="2026-01-02T00:00:00Z",
            ),
        )[:limit]

    def get_metadata(self, model_id: str, revision: str | None = None, files_metadata: bool = False):  # noqa: ANN001
        from fl_model_onboarding.adapters.interfaces import HuggingFaceMetadata

        config: dict[str, object] = {"model_type": "llama"}
        if "whisper" in model_id:
            config = {"model_type": "whisper"}
        return HuggingFaceMetadata(
            model_id=model_id,
            revision=revision or "rev-1",
            sha="1234567890abcdef1234567890abcdef12345678",
            is_private=False,
            is_gated=False,
            last_modified="2026-01-01T00:00:00Z",
            config=config,
            safetensors_total_bytes=1024,
            safetensors_parameter_count=256,
            card_data={"license": "apache-2.0"},
            sibling_count=3,
            sibling_files=("config.json", "tokenizer.json", "model.safetensors"),
        )


class FakeFoundryCatalog:
    def list_matches(self, search_query: str):  # noqa: ARG002
        return ()


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


class BlockingUntilCancelledPreflightInspector(PassingPreflightInspector):
    def __init__(self) -> None:
        self.started = Event()
        self.seen_cancellation_event: Event | None = None

    def inspect(  # type: ignore[override]
        self,
        request: BuildRequest,
        *,
        cancellation_event: Event | None = None,
    ) -> PreflightResult:
        self.started.set()
        self.seen_cancellation_event = cancellation_event
        if cancellation_event is not None:
            cancellation_event.wait(timeout=5.0)
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
                    stage=JobState.PREFLIGHT,
                    classification=FailureClassification.TOOL_UNAVAILABLE,
                    message="cancelled preflight probe",
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
        baseline_dir = job.request.workspace_root / "mobius"
        baseline_dir.mkdir(parents=True, exist_ok=False)
        (baseline_dir / "model.onnx").write_text("baseline", encoding="utf-8")
        (baseline_dir / "inference_model.json").write_text(
            json.dumps({"Name": f"baseline-{job.job_id}"}, indent=2),
            encoding="utf-8",
        )
        artifact_path = job.request.output_dir / "optimized-package"
        artifact_path.mkdir(parents=True, exist_ok=False)
        (artifact_path / "model.onnx").write_text("optimized", encoding="utf-8")
        (artifact_path / "inference_model.json").write_text(
            json.dumps({"Name": f"optimized-{job.job_id}"}, indent=2),
            encoding="utf-8",
        )
        job.register_artifact(
            BuildArtifact(
                artifact_id=f"artifact-{job.job_id}",
                kind=ArtifactKind.MODEL,
                path=artifact_path,
                description="test artifact",
                size_bytes=(artifact_path / "model.onnx").stat().st_size,
            )
        )
        transition(job, JobState.SUCCEEDED, "Build succeeded.")
        job.finished_utc = datetime.now(timezone.utc)
        persist()


class MissingBaselineSuccessRunner:
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
        artifact_path = job.request.output_dir / "optimized-package"
        artifact_path.mkdir(parents=True, exist_ok=False)
        (artifact_path / "model.onnx").write_text("optimized", encoding="utf-8")
        (artifact_path / "inference_model.json").write_text(
            json.dumps({"Name": f"optimized-{job.job_id}"}, indent=2),
            encoding="utf-8",
        )
        job.register_artifact(
            BuildArtifact(
                artifact_id=f"artifact-{job.job_id}",
                kind=ArtifactKind.MODEL,
                path=artifact_path,
                description="test artifact",
                size_bytes=(artifact_path / "model.onnx").stat().st_size,
            )
        )
        transition(job, JobState.SUCCEEDED, "Build succeeded.")
        job.finished_utc = datetime.now(timezone.utc)
        persist()


class SelfComparingSuccessRunner:
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
        baseline_dir = job.request.workspace_root / "mobius"
        baseline_dir.mkdir(parents=True, exist_ok=False)
        (baseline_dir / "model.onnx").write_text("baseline", encoding="utf-8")
        (baseline_dir / "inference_model.json").write_text(
            json.dumps({"Name": f"baseline-{job.job_id}"}, indent=2),
            encoding="utf-8",
        )
        job.register_artifact(
            BuildArtifact(
                artifact_id=f"artifact-{job.job_id}",
                kind=ArtifactKind.MODEL,
                path=baseline_dir,
                description="test artifact",
                size_bytes=(baseline_dir / "model.onnx").stat().st_size,
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


class OversizedFailureRunner:
    def run(self, job: BuildJob, *, persist, cancellation_event):  # noqa: ANN001
        if cancellation_event.is_set():
            return
        transition(job, JobState.DOWNLOADING, "Downloading model data.")
        persist()
        if cancellation_event.is_set():
            return
        transition(job, JobState.MOBIUS_BUILDING, "Running oversized failure runner.")
        persist()
        fail_job(
            job,
            FailureInfo(
                stage=JobState.MOBIUS_BUILDING,
                classification=FailureClassification.PROCESS_FAILED,
                message="x" * 10000,
            ),
        )
        job.finished_utc = datetime.now(timezone.utc)
        persist()


class EchoTextBackend:
    _QUALITY_RESPONSES = {
        "What is 17 + 28? Reply using only digits.": "45",
        "Which planet is known as the Red Planet? Reply with one word.": "Mars",
        "Output exactly two words: blue river": "blue river",
        "Return valid JSON object with keys answer and unit, where answer is 12 and unit is cm.": (
            '{"answer":12,"unit":"cm"}'
        ),
    }

    def infer(self, *, artifact, job, prompt: str, max_tokens: int) -> str:  # noqa: ANN001
        quality_response = self._QUALITY_RESPONSES.get(prompt)
        if quality_response is not None:
            return quality_response
        return f"{artifact.artifact_id}:{job.job_id}:{prompt}:{max_tokens}"


class BaselineUnavailableTextBackend(EchoTextBackend):
    def infer(self, *, artifact, job, prompt: str, max_tokens: int) -> str:  # noqa: ANN001
        if str(artifact.artifact_id).startswith("baseline-"):
            raise RuntimeError("baseline runner unavailable")
        return super().infer(artifact=artifact, job=job, prompt=prompt, max_tokens=max_tokens)


class RegressedOptimizedTextBackend(EchoTextBackend):
    def infer(self, *, artifact, job, prompt: str, max_tokens: int) -> str:  # noqa: ANN001
        if (
            not str(artifact.artifact_id).startswith("baseline-")
            and prompt == "What is 17 + 28? Reply using only digits."
        ):
            return "forty five"
        return super().infer(artifact=artifact, job=job, prompt=prompt, max_tokens=max_tokens)


class BatchEchoTextBackend(EchoTextBackend):
    def __init__(self) -> None:
        self.infer_calls = 0
        self.batch_calls: list[dict[str, object]] = []

    def infer(self, *, artifact, job, prompt: str, max_tokens: int) -> str:  # noqa: ANN001
        self.infer_calls += 1
        return super().infer(artifact=artifact, job=job, prompt=prompt, max_tokens=max_tokens)

    def infer_batch(self, *, artifact, job, prompts, max_tokens: int):  # noqa: ANN001
        prompt_rows = list(prompts)
        self.batch_calls.append(
            {
                "artifact_id": artifact.artifact_id,
                "prompt_ids": [prompt_id for prompt_id, _ in prompt_rows],
                "max_tokens": max_tokens,
            }
        )
        outputs: list[str] = []
        for _, prompt in prompt_rows:
            quality_response = self._QUALITY_RESPONSES.get(prompt)
            if quality_response is not None:
                outputs.append(quality_response)
            else:
                outputs.append(f"{artifact.artifact_id}:{job.job_id}:{prompt}:{max_tokens}")
        return tuple(outputs)


class BatchBaselineTimeoutTextBackend(BatchEchoTextBackend):
    def infer_batch(self, *, artifact, job, prompts, max_tokens: int):  # noqa: ANN001
        if str(artifact.artifact_id).startswith("baseline-"):
            raise RuntimeError("stage=prompt_timeout prompt_id=factual-red-planet")
        return super().infer_batch(
            artifact=artifact,
            job=job,
            prompts=prompts,
            max_tokens=max_tokens,
        )


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
    text_backend=None,
    recipe_registry: RecipeRegistry | None = None,
    hf_metadata=None,
    foundry_catalog=None,
) -> LocalOnboardingService:
    return LocalOnboardingService(
        db_path=tmp_path / "state.sqlite3",
        workspace_base=tmp_path / "w",
        model_cache_dir=tmp_path / "cache",
        hf_metadata=hf_metadata or FakeHFMetadata(),  # type: ignore[arg-type]
        foundry_catalog=foundry_catalog or FakeFoundryCatalog(),  # type: ignore[arg-type]
        preflight_inspector=preflight_inspector or PassingPreflightInspector(),  # type: ignore[arg-type]
        build_stage_runner=build_stage_runner,  # type: ignore[arg-type]
        text_inference_backend=text_backend,  # type: ignore[arg-type]
        recipe_registry=recipe_registry,
    )


def _experimental_granite_registry() -> RecipeRegistry:
    granite = next(recipe for recipe in DEFAULT_MODEL_RECIPES if recipe.id == "granite-3.3-2b-cpu-int4")
    experimental = replace(
        granite,
        status=RecipeStatus.EXPERIMENTAL,
        status_reason="Recipe requires explicit experimental opt-in.",
    )
    return RecipeRegistry(
        tuple(
            experimental if recipe.id == experimental.id else recipe
            for recipe in DEFAULT_MODEL_RECIPES
        )
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


def test_unknown_model_preflight_and_build_are_blocked_without_recipe(tmp_path: Path) -> None:
    service = _service(tmp_path)
    try:
        blocked = service.preflight(
            BuildSubmission(
                model_id="owner/unregistered-model",
                task=CandidateModality.LLM,
                task_profile="llm-cpu-int4",
            )
        )
        assert blocked["ok"] is False
        assert blocked["recipe_status"] == "unregistered"
        assert blocked["result"]["blockers"][0]["message"].startswith("No recipe is registered")

        with pytest.raises(ServiceError) as exc:
            service.create_build(
                BuildSubmission(
                    model_id="owner/unregistered-model",
                    task=CandidateModality.LLM,
                    task_profile="llm-cpu-int4",
                ),
                idempotency_key="unknown-model",
            )
        assert exc.value.code == "MODEL_RECIPE_NOT_FOUND"
    finally:
        service.close()


def test_experimental_recipe_requires_explicit_opt_in(tmp_path: Path) -> None:
    service = _service(
        tmp_path,
        build_stage_runner=DeterministicSuccessRunner(),
        recipe_registry=_experimental_granite_registry(),
    )
    try:
        blocked = service.preflight(
            BuildSubmission(
                model_id=GRANITE_MODEL_ID,
                task=CandidateModality.LLM,
                task_profile="llm-cpu-int4",
            )
        )
        assert blocked["ok"] is False
        assert blocked["recipe_status"] == "experimental"
        assert blocked["requires_experimental_opt_in"] is True

        allowed = service.preflight(
            BuildSubmission(
                model_id=GRANITE_MODEL_ID,
                task=CandidateModality.LLM,
                task_profile="llm-cpu-int4",
                allow_experimental=True,
            )
        )
        assert allowed["recipe_status"] == "experimental"
        assert allowed["requires_experimental_opt_in"] is False
        assert allowed["ok"] is True

        with pytest.raises(ServiceError) as exc:
            service.create_build(
                BuildSubmission(
                    model_id=GRANITE_MODEL_ID,
                    task=CandidateModality.LLM,
                    task_profile="llm-cpu-int4",
                ),
                idempotency_key="granite-without-opt-in",
            )
        assert exc.value.code == "EXPERIMENTAL_RECIPE_OPT_IN_REQUIRED"

        _, replay = service.create_build(
            BuildSubmission(
                model_id=GRANITE_MODEL_ID,
                task=CandidateModality.LLM,
                task_profile="llm-cpu-int4",
                allow_experimental=True,
            ),
            idempotency_key="granite-with-opt-in",
        )
        assert replay is False
    finally:
        service.close()


def test_generated_recipe_preview_and_attempt_failure_sync(tmp_path: Path) -> None:
    service = _service(tmp_path)
    try:
        preview = service.generated_recipe_preview(
            model_id="owner/unregistered-model",
            task=CandidateModality.LLM,
        )
        generated = preview["generated_recipe"]
        assert generated["eligible_for_automatic_recipe_attempt"] is True
        fingerprint = generated["fingerprint"]
        assert isinstance(fingerprint, str) and len(fingerprint) == 64

        job, replay, attempt = service.create_generated_recipe_attempt(
            recipe_fingerprint=fingerprint,
            idempotency_key="generated-failure-1",
            model_id="owner/unregistered-model",
        )
        assert replay is False
        assert attempt["state"] == "running"
        completed = _wait_for_terminal(service, job.job_id)
        assert completed.state == JobState.FAILED

        attempt_status = service.get_recipe_attempt(attempt_id=attempt["attempt_id"])
        assert attempt_status["state"] == "failed"
        assert attempt_status["build_job_id"] == job.job_id
        assert attempt_status["failure"]["classification"] == "gate_failed"
        assert any(row["status"] == "failed" for row in attempt_status["gates"])
    finally:
        service.close()


def test_generated_recipe_attempt_sync_truncates_oversized_failure_messages(tmp_path: Path) -> None:
    service = _service(tmp_path, build_stage_runner=OversizedFailureRunner())
    try:
        preview = service.generated_recipe_preview(
            model_id="owner/unregistered-model",
            task=CandidateModality.LLM,
        )
        fingerprint = str(preview["generated_recipe"]["fingerprint"])
        job, _, attempt = service.create_generated_recipe_attempt(
            recipe_fingerprint=fingerprint,
            idempotency_key="generated-oversized-failure-1",
            model_id="owner/unregistered-model",
        )
        completed = _wait_for_terminal(service, job.job_id)
        assert completed.state == JobState.FAILED
        assert all(
            "Recipe attempt synchronization error" not in event.message
            for event in completed.events
        )

        attempt_status = service.get_recipe_attempt(attempt_id=attempt["attempt_id"])
        assert attempt_status["state"] == "failed"
        assert attempt_status["failure"]["classification"] == "gate_failed"
        assert len(str(attempt_status["failure"]["message"])) <= 2048
    finally:
        service.close()


def test_generated_recipe_attempt_success_promotes_and_enables_verified_reuse(tmp_path: Path) -> None:
    service = _service(
        tmp_path,
        build_stage_runner=DeterministicSuccessRunner(),
        text_backend=EchoTextBackend(),
    )
    try:
        preview = service.generated_recipe_preview(
            model_id="owner/unregistered-model",
            task=CandidateModality.LLM,
        )
        generated = preview["generated_recipe"]
        fingerprint = str(generated["fingerprint"])
        assert generated["eligible_for_automatic_recipe_attempt"] is True

        job, _, attempt = service.create_generated_recipe_attempt(
            recipe_fingerprint=fingerprint,
            idempotency_key="generated-success-1",
            model_id="owner/unregistered-model",
        )
        completed = _wait_for_terminal(service, job.job_id)
        assert completed.state == JobState.SUCCEEDED
        attempt_status = service.get_recipe_attempt(attempt_id=attempt["attempt_id"])
        assert attempt_status["state"] == "succeeded"
        assert [row["gate"] for row in attempt_status["gates"]] == [
            "mobius_build",
            "olive_optimize",
            "onnx_validation",
            "ort_validation",
            "oga_validation",
            "fl_sdk_inference",
            "quality_validation",
        ]
        assert all(row["status"] == "passed" for row in attempt_status["gates"])
        quality_gate = next(row for row in attempt_status["gates"] if row["gate"] == "quality_validation")
        assert "baseline-passed" in quality_gate["evidence_ref"]

        reused = service.generated_recipe_preview(
            model_id="owner/unregistered-model",
            task=CandidateModality.LLM,
        )["generated_recipe"]
        assert reused["eligible_for_automatic_recipe_attempt"] is False
        assert reused["verified_reuse"]["available"] is True
        assert reused["verified_reuse"]["source_recipe_fingerprint"] == fingerprint
        assert reused["verified_reuse"]["attempt_id"] == attempt["attempt_id"]

        tested = service.health()["compatibility_index"]
        assert any(row["model_id"] == "owner/unregistered-model" for row in tested)  # type: ignore[index]
    finally:
        service.close()


def test_generated_recipe_attempt_uses_batch_quality_inference_and_records_split_summary(
    tmp_path: Path,
) -> None:
    backend = BatchEchoTextBackend()
    service = _service(
        tmp_path,
        build_stage_runner=DeterministicSuccessRunner(),
        text_backend=backend,
    )
    try:
        preview = service.generated_recipe_preview(
            model_id="owner/unregistered-model",
            task=CandidateModality.LLM,
        )
        fingerprint = str(preview["generated_recipe"]["fingerprint"])
        job, _, attempt = service.create_generated_recipe_attempt(
            recipe_fingerprint=fingerprint,
            idempotency_key="generated-batch-quality-1",
            model_id="owner/unregistered-model",
        )
        completed = _wait_for_terminal(service, job.job_id)
        assert completed.state == JobState.SUCCEEDED

        expected_prompt_ids = [
            "arithmetic-addition-17-plus-28",
            "factual-red-planet",
            "instruction-two-words-blue-river",
            "format-json-answer-unit",
        ]
        deadline = time.time() + 5.0
        while len(backend.batch_calls) < 2 and time.time() < deadline:
            time.sleep(0.02)
        assert backend.infer_calls == 0
        assert len(backend.batch_calls) == 2
        assert backend.batch_calls[0]["prompt_ids"] == expected_prompt_ids
        assert backend.batch_calls[1]["prompt_ids"] == expected_prompt_ids
        assert str(backend.batch_calls[0]["artifact_id"]).startswith("baseline-")
        assert not str(backend.batch_calls[1]["artifact_id"]).startswith("baseline-")

        attempt_status = service.get_recipe_attempt(attempt_id=attempt["attempt_id"])
        assert attempt_status["state"] == "succeeded"
        quality_gate = next(row for row in attempt_status["gates"] if row["gate"] == "quality_validation")
        assert quality_gate["metrics_ref"] is not None
        quality_validation = attempt_status["quality_validation"]
        assert quality_validation["recipe_integrity"]["status"] == "verified"
        capability = quality_validation["model_capability"]
        assert capability["checks_passed"] == 4
        assert capability["total_checks"] == 4
        assert capability["confidence"]["level"] == "low"

        evidence = json.loads(
            (completed.request.workspace_root / "quality-validation-evidence.json").read_text(
                encoding="utf-8"
            )
        )
        prompt_rows = evidence["per_prompt"]
        assert [row["prompt_id"] for row in prompt_rows] == expected_prompt_ids
        assert all(isinstance(row.get("baseline"), dict) for row in prompt_rows)
        assert all(isinstance(row.get("optimized"), dict) for row in prompt_rows)
    finally:
        service.close()


def test_generated_recipe_attempt_requires_distinct_baseline_package(tmp_path: Path) -> None:
    service = _service(
        tmp_path,
        build_stage_runner=SelfComparingSuccessRunner(),
        text_backend=EchoTextBackend(),
    )
    try:
        preview = service.generated_recipe_preview(
            model_id="owner/unregistered-model",
            task=CandidateModality.LLM,
        )
        fingerprint = str(preview["generated_recipe"]["fingerprint"])
        job, _, attempt = service.create_generated_recipe_attempt(
            recipe_fingerprint=fingerprint,
            idempotency_key="generated-self-compare-1",
            model_id="owner/unregistered-model",
        )
        completed = _wait_for_terminal(service, job.job_id)
        assert completed.state == JobState.SUCCEEDED

        attempt_status = service.get_recipe_attempt(attempt_id=attempt["attempt_id"])
        assert attempt_status["state"] == "failed"
        quality_gate = next(row for row in attempt_status["gates"] if row["gate"] == "quality_validation")
        assert quality_gate["status"] == "not_run"
        assert "self-comparison" in quality_gate["evidence_ref"]
        assert attempt_status["failure"]["classification"] == "validation_failed"
        assert "same package identity" in attempt_status["failure"]["message"]

        reuse = service.generated_recipe_preview(
            model_id="owner/unregistered-model",
            task=CandidateModality.LLM,
        )["generated_recipe"]
        assert reuse["eligible_for_automatic_recipe_attempt"] is True
        assert reuse["verified_reuse"] is None
    finally:
        service.close()


def test_generated_recipe_attempt_without_baseline_cannot_promote(tmp_path: Path) -> None:
    service = _service(
        tmp_path,
        build_stage_runner=MissingBaselineSuccessRunner(),
        text_backend=EchoTextBackend(),
    )
    try:
        preview = service.generated_recipe_preview(
            model_id="owner/unregistered-model",
            task=CandidateModality.LLM,
        )
        fingerprint = str(preview["generated_recipe"]["fingerprint"])
        job, _, attempt = service.create_generated_recipe_attempt(
            recipe_fingerprint=fingerprint,
            idempotency_key="generated-missing-baseline-1",
            model_id="owner/unregistered-model",
        )
        completed = _wait_for_terminal(service, job.job_id)
        assert completed.state == JobState.SUCCEEDED

        attempt_status = service.get_recipe_attempt(attempt_id=attempt["attempt_id"])
        assert attempt_status["state"] == "failed"
        quality_gate = next(row for row in attempt_status["gates"] if row["gate"] == "quality_validation")
        assert quality_gate["status"] == "not_run"
        assert "baseline-not-run" in quality_gate["evidence_ref"]
        assert attempt_status["failure"]["classification"] == "validation_failed"
        assert "Quality baseline not run" in attempt_status["failure"]["message"]
    finally:
        service.close()


def test_generated_recipe_attempt_baseline_regression_blocks_promotion(tmp_path: Path) -> None:
    service = _service(
        tmp_path,
        build_stage_runner=DeterministicSuccessRunner(),
        text_backend=RegressedOptimizedTextBackend(),
    )
    try:
        preview = service.generated_recipe_preview(
            model_id="owner/unregistered-model",
            task=CandidateModality.LLM,
        )
        fingerprint = str(preview["generated_recipe"]["fingerprint"])
        job, _, attempt = service.create_generated_recipe_attempt(
            recipe_fingerprint=fingerprint,
            idempotency_key="generated-baseline-regression-1",
            model_id="owner/unregistered-model",
        )
        completed = _wait_for_terminal(service, job.job_id)
        assert completed.state == JobState.SUCCEEDED

        attempt_status = service.get_recipe_attempt(attempt_id=attempt["attempt_id"])
        assert attempt_status["state"] == "failed"
        quality_gate = next(row for row in attempt_status["gates"] if row["gate"] == "quality_validation")
        assert quality_gate["status"] == "failed"
        assert "baseline-regression" in quality_gate["evidence_ref"]
        quality_validation = attempt_status["quality_validation"]
        assert quality_validation["recipe_integrity"]["status"] == "blocked"
        assert quality_validation["model_capability"]["checks_passed"] == 3
        assert quality_validation["model_capability"]["total_checks"] == 4
        assert attempt_status["failure"]["classification"] == "validation_failed"
        assert "baseline_passed_optimized_failed:arithmetic-addition-17-plus-28" in attempt_status["failure"]["message"]
    finally:
        service.close()


def test_generated_recipe_attempt_baseline_execution_unavailable_is_structured_failure(tmp_path: Path) -> None:
    service = _service(
        tmp_path,
        build_stage_runner=DeterministicSuccessRunner(),
        text_backend=BaselineUnavailableTextBackend(),
    )
    try:
        preview = service.generated_recipe_preview(
            model_id="owner/unregistered-model",
            task=CandidateModality.LLM,
        )
        fingerprint = str(preview["generated_recipe"]["fingerprint"])
        job, _, attempt = service.create_generated_recipe_attempt(
            recipe_fingerprint=fingerprint,
            idempotency_key="generated-baseline-unavailable-1",
            model_id="owner/unregistered-model",
        )
        completed = _wait_for_terminal(service, job.job_id)
        assert completed.state == JobState.SUCCEEDED

        attempt_status = service.get_recipe_attempt(attempt_id=attempt["attempt_id"])
        assert attempt_status["state"] == "failed"
        quality_gate = next(row for row in attempt_status["gates"] if row["gate"] == "quality_validation")
        assert quality_gate["status"] == "unavailable"
        assert "baseline-unavailable" in quality_gate["evidence_ref"]
        quality_validation = attempt_status["quality_validation"]
        assert quality_validation["recipe_integrity"]["status"] == "inconclusive"
        assert quality_validation["model_capability"] is None
        assert attempt_status["failure"]["classification"] == "validation_failed"
        assert "Baseline quality prompt" in attempt_status["failure"]["message"]
        assert attempt_status["failure"]["source_owner"] == "fl-onboarding"
    finally:
        service.close()


def test_generated_recipe_attempt_batch_timeout_is_attributed_to_baseline_prompt(
    tmp_path: Path,
) -> None:
    service = _service(
        tmp_path,
        build_stage_runner=DeterministicSuccessRunner(),
        text_backend=BatchBaselineTimeoutTextBackend(),
    )
    try:
        preview = service.generated_recipe_preview(
            model_id="owner/unregistered-model",
            task=CandidateModality.LLM,
        )
        fingerprint = str(preview["generated_recipe"]["fingerprint"])
        job, _, attempt = service.create_generated_recipe_attempt(
            recipe_fingerprint=fingerprint,
            idempotency_key="generated-baseline-batch-timeout-1",
            model_id="owner/unregistered-model",
        )
        completed = _wait_for_terminal(service, job.job_id)
        assert completed.state == JobState.SUCCEEDED

        attempt_status = service.get_recipe_attempt(attempt_id=attempt["attempt_id"])
        assert attempt_status["state"] == "failed"
        quality_gate = next(row for row in attempt_status["gates"] if row["gate"] == "quality_validation")
        assert quality_gate["status"] == "unavailable"
        assert "baseline-unavailable" in quality_gate["evidence_ref"]
        assert attempt_status["failure"]["classification"] == "validation_failed"
        assert "prompt_timeout" in attempt_status["failure"]["message"]
        assert "factual-red-planet" in attempt_status["failure"]["message"]
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


def test_service_close_cancels_inflight_preflight_probe(tmp_path: Path) -> None:
    inspector = BlockingUntilCancelledPreflightInspector()
    service = _service(tmp_path, preflight_inspector=inspector)
    closed = False
    try:
        service.create_build(_submission(), idempotency_key="k-close-during-preflight")
        assert inspector.started.wait(timeout=5.0)
        service.close()
        closed = True
        assert inspector.seen_cancellation_event is not None
        assert inspector.seen_cancellation_event.is_set()
    finally:
        if not closed:
            try:
                service.close()
            except Exception:
                pass


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
    while time.time() < deadline and service.get_build(job.job_id).state in {
        JobState.QUEUED,
        JobState.PREFLIGHT,
    }:
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
