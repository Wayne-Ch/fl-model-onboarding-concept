from __future__ import annotations

import hashlib
import json
import shutil
import sys
import uuid

from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Callable

from .adapters.interfaces import CommandResult, CommandSpec, ProcessRunner
from .adapters.huggingface_acquisition import HuggingFaceAcquisitionAdapter
from .adapters.interfaces import HuggingFaceAcquisitionClient
from .contracts import (
    ArtifactKind,
    BuildArtifact,
    BuildJob,
    CandidateModality,
    FailureClassification,
    FailureInfo,
    JobState,
    ValidationResult,
    ValidationStatus,
)
from .state_machine import fail_job, transition

SMOLLM2_MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
SMOLLM2_REVISION = "31b70e2e869a7173562077fd711b654946d38674"


def _result_payload(result: CommandResult) -> dict[str, object]:
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {"ok": False, "error": result.stderr or "Command returned no JSON result."}


class FoundrySdkTextInferenceBackend:
    def __init__(
        self,
        process_runner: ProcessRunner,
        *,
        timeout_seconds: int = 900,
        cancellation_event: Event | None = None,
    ) -> None:
        self._process_runner = process_runner
        self._timeout_seconds = timeout_seconds
        self._cancellation_event = cancellation_event

    def infer(
        self,
        *,
        artifact: BuildArtifact,
        job: BuildJob,
        prompt: str,
        max_tokens: int,
    ) -> str:
        if len(prompt) > 8192:
            raise ValueError("Inference prompt exceeds the 8192 character limit.")
        model_dir = artifact.path.resolve()
        descriptor = json.loads((model_dir / "inference_model.json").read_text(encoding="utf-8"))
        model_name = descriptor["Name"]
        request_file = job.request.workspace_root / f"inference-{uuid.uuid4().hex}.json"
        request_file.write_text(
            json.dumps({"prompt": prompt, "max_tokens": max_tokens}),
            encoding="utf-8",
        )
        try:
            result = self._process_runner.run(
                CommandSpec(
                    argv=(
                        sys.executable,
                        "-m",
                        "fl_model_onboarding.runtime_worker",
                        "foundry-infer",
                        "--model-dir",
                        str(model_dir),
                        "--model-name",
                        str(model_name),
                        "--request-file",
                        str(request_file),
                    ),
                    cwd=job.request.workspace_root,
                    timeout_seconds=self._timeout_seconds,
                ),
                cancel_event=self._cancellation_event,
            )
        finally:
            request_file.unlink(missing_ok=True)
        payload = _result_payload(result)
        if not result.ok or payload.get("ok") is not True:
            raise RuntimeError(str(payload.get("error") or "Foundry Local inference failed."))
        return str(payload["output"])


class ProductionBuildStageRunner:
    def __init__(
        self,
        process_runner: ProcessRunner,
        *,
        build_timeout_seconds: int = 7200,
        olive_timeout_seconds: int = 5400,
        runtime_timeout_seconds: int = 900,
        model_acquisition: HuggingFaceAcquisitionClient | None = None,
    ) -> None:
        self._process_runner = process_runner
        self._build_timeout_seconds = build_timeout_seconds
        self._olive_timeout_seconds = olive_timeout_seconds
        self._runtime_timeout_seconds = runtime_timeout_seconds
        self._model_acquisition = model_acquisition or HuggingFaceAcquisitionAdapter()

    def run(
        self,
        job: BuildJob,
        *,
        persist: Callable[[], None],
        cancellation_event: Event,
    ) -> None:
        staging_dir, package_dir = production_package_paths(job)
        staging_preexisting = staging_dir.exists()
        package_preexisting = package_dir.exists()
        try:
            self._run(job, persist=persist, cancellation_event=cancellation_event)
        except Exception as exc:
            if not staging_preexisting and staging_dir.exists():
                shutil.rmtree(staging_dir)
            if not package_preexisting and package_dir.exists():
                shutil.rmtree(package_dir)
            if job.state == JobState.CANCELLED:
                return
            classification = (
                FailureClassification.MISSING_DEPENDENCY
                if isinstance(exc, FileNotFoundError)
                else FailureClassification.PROCESS_FAILED
            )
            fail_job(
                job,
                FailureInfo(
                    stage=job.state,
                    classification=classification,
                    message=str(exc),
                ),
            )
            job.finished_utc = datetime.now(timezone.utc)
            persist()

    def _run(
        self,
        job: BuildJob,
        *,
        persist: Callable[[], None],
        cancellation_event: Event,
    ) -> None:
        request = job.request
        if request.candidate.modality != CandidateModality.LLM or (
            request.candidate.huggingface_model_id != SMOLLM2_MODEL_ID
        ):
            raise RuntimeError(
                "Production execution is verified only for HuggingFaceTB/SmolLM2-1.7B-Instruct."
            )
        if request.hf_revision != SMOLLM2_REVISION:
            raise RuntimeError(
                f"Production execution requires pinned revision {SMOLLM2_REVISION}; "
                f"received {request.hf_revision or 'none'}."
            )
        if request.task_profile != "llm-cpu-int4" or request.skip_olive:
            raise RuntimeError(
                "Production execution requires task_profile=llm-cpu-int4 with Olive enabled."
            )

        snapshot_dir = request.workspace_root / "snapshot"
        mobius_dir = request.workspace_root / "mobius"
        olive_dir = request.workspace_root / "olive"
        mobius_dir.mkdir(parents=True, exist_ok=False)
        olive_dir.mkdir(parents=True, exist_ok=False)

        transition(job, JobState.DOWNLOADING, f"Pinned Hugging Face revision {request.hf_revision}.")
        persist()
        snapshot_path = self._model_acquisition.acquire_snapshot(
            SMOLLM2_MODEL_ID,
            snapshot_dir,
            revision=SMOLLM2_REVISION,
        )
        transition(job, JobState.MOBIUS_BUILDING, "Running verified Mobius CPU ort-genai f32 build.")
        persist()
        self._run_command(
            CommandSpec(
                argv=(
                    "mobius",
                    "build",
                    "--model",
                    str(snapshot_path),
                    "--ep",
                    "cpu",
                    "--runtime",
                    "ort-genai",
                    "--dtype",
                    "f32",
                    str(mobius_dir),
                ),
                cwd=request.workspace_root,
                timeout_seconds=self._build_timeout_seconds,
            ),
            cancellation_event,
            "Mobius build",
        )

        transition(job, JobState.MOBIUS_VALIDATING, "Mobius output created; ONNX validation follows Olive.")
        persist()
        transition(job, JobState.OLIVE_OPTIMIZING, "Running verified Olive existing-ONNX INT4 optimization.")
        persist()
        self._run_command(
            CommandSpec(
                argv=(
                    "olive",
                    "optimize",
                    "--model_name_or_path",
                    str(mobius_dir),
                    "--task",
                    "text-generation-with-past",
                    "--device",
                    "cpu",
                    "--provider",
                    "CPUExecutionProvider",
                    "--precision",
                    "int4",
                    "--output_path",
                    str(olive_dir),
                    "--log_level",
                    "1",
                ),
                cwd=request.workspace_root,
                timeout_seconds=self._olive_timeout_seconds,
            ),
            cancellation_event,
            "Olive optimize",
        )
        source_dir = olive_dir

        transition(job, JobState.PACKAGING, "Creating immutable Foundry Local BYOM package.")
        persist()
        artifact_id = self._artifact_id(job)
        model_name = f"smollm2-onboarding-{artifact_id[:12]}:1"
        staging_dir, package_dir = production_package_paths(job)
        if package_dir.exists():
            raise FileExistsError(f"Immutable artifact path already exists: {package_dir}")
        if staging_dir.exists():
            raise FileExistsError(f"Partial artifact path already exists: {staging_dir}")
        shutil.copytree(source_dir, staging_dir)
        (staging_dir / "inference_model.json").write_text(
            json.dumps({"Name": model_name}, indent=2),
            encoding="utf-8",
        )

        transition(job, JobState.RUNTIME_VALIDATING, "Validating ONNX, ORT CPU, and OGA generation.")
        persist()
        runtime_result = self._run_command(
            CommandSpec(
                argv=(
                    sys.executable,
                    "-m",
                    "fl_model_onboarding.runtime_worker",
                    "validate-runtime",
                    "--model-dir",
                    str(staging_dir),
                ),
                cwd=request.workspace_root,
                timeout_seconds=self._runtime_timeout_seconds,
            ),
            cancellation_event,
            "Runtime validation",
        )
        runtime_payload = _result_payload(runtime_result)
        checks = tuple(str(item) for item in runtime_payload.get("checks", []))
        job.validations.append(
            ValidationResult(
                stage=JobState.RUNTIME_VALIDATING,
                status=ValidationStatus.PASSED,
                checks=checks,
            )
        )

        staging_dir.rename(package_dir)
        transition(job, JobState.FL_LOADING, "Foundry Local SDK discovered and loaded the BYOM package.")
        persist()
        transition(job, JobState.INFERENCING, "Running bounded Foundry Local SDK chat inference.")
        persist()
        inference_backend = FoundrySdkTextInferenceBackend(
            self._process_runner,
            timeout_seconds=self._runtime_timeout_seconds,
            cancellation_event=cancellation_event,
        )
        output = inference_backend.infer(
            artifact=BuildArtifact(
                artifact_id=artifact_id,
                kind=ArtifactKind.MODEL,
                path=package_dir,
                description="Immutable Foundry Local BYOM model package",
            ),
            job=job,
            prompt="Reply with: OK",
            max_tokens=64,
        )
        if not output.strip():
            raise RuntimeError("Foundry Local SDK inference returned empty output.")
        job.validations.append(
            ValidationResult(
                stage=JobState.INFERENCING,
                status=ValidationStatus.PASSED,
                checks=("foundry_local_sdk_chat=passed",),
            )
        )
        job.register_artifact(
            BuildArtifact(
                artifact_id=artifact_id,
                kind=ArtifactKind.MODEL,
                path=package_dir,
                description="Immutable Foundry Local BYOM model package",
            )
        )
        transition(job, JobState.SUCCEEDED, "Verified SmolLM2 Foundry Local build and inference succeeded.")
        job.finished_utc = datetime.now(timezone.utc)
        persist()

    def _run_command(
        self,
        spec: CommandSpec,
        cancellation_event: Event,
        label: str,
    ) -> CommandResult:
        result = self._process_runner.run(spec, cancel_event=cancellation_event)
        if not result.ok:
            detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.exit_code}"
            raise RuntimeError(f"{label} failed: {detail}")
        return result

    @staticmethod
    def _artifact_id(job: BuildJob) -> str:
        request = job.request
        return hashlib.sha256(
            f"{SMOLLM2_MODEL_ID}:{request.hf_revision}:{request.task_profile}:{job.job_id}".encode()
        ).hexdigest()


def production_package_paths(job: BuildJob) -> tuple[Path, Path]:
    artifact_id = ProductionBuildStageRunner._artifact_id(job)
    cache = job.request.model_cache_dir
    return (
        cache / f".partial-smollm2-{artifact_id[:12]}",
        cache / f"smollm2-{artifact_id[:12]}",
    )
