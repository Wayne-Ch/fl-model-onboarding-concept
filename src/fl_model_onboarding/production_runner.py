from __future__ import annotations

import hashlib
import json
import re
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
from .recipes import (
    DEFAULT_RECIPE_REGISTRY,
    SMOLLM2_MODEL_ID as VERIFIED_SMOLLM2_MODEL_ID,
    SMOLLM2_VERIFIED_REVISION,
    ModelRecipe,
    RecipeRegistry,
    RecipeStatus,
)

SMOLLM2_MODEL_ID = VERIFIED_SMOLLM2_MODEL_ID
SMOLLM2_REVISION = SMOLLM2_VERIFIED_REVISION


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
        recipe_registry: RecipeRegistry | None = None,
    ) -> None:
        self._process_runner = process_runner
        self._build_timeout_seconds = build_timeout_seconds
        self._olive_timeout_seconds = olive_timeout_seconds
        self._runtime_timeout_seconds = runtime_timeout_seconds
        self._model_acquisition = model_acquisition or HuggingFaceAcquisitionAdapter()
        self._recipe_registry = recipe_registry or DEFAULT_RECIPE_REGISTRY

    def run(
        self,
        job: BuildJob,
        *,
        persist: Callable[[], None],
        cancellation_event: Event,
    ) -> None:
        staging_dir, package_dir = production_package_paths(job, recipe_registry=self._recipe_registry)
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
        recipe_match = self._recipe_registry.resolve(
            model_id=request.candidate.huggingface_model_id,
            modality=request.candidate.modality,
            task_profile=request.task_profile,
            allow_experimental=True,
        )
        recipe = recipe_match.recipe
        if recipe is None:
            raise RuntimeError(recipe_match.reason)
        if recipe.status != RecipeStatus.VERIFIED:
            raise RuntimeError(
                f"Production execution is verified only for recipe status 'verified'; "
                f"received '{recipe.id}' ({recipe.status.value})."
            )
        if recipe.verified_revision is None:
            raise RuntimeError(
                f"Verified recipe '{recipe.id}' is missing a pinned verified revision."
            )
        if request.hf_revision != recipe.verified_revision:
            raise RuntimeError(
                f"Production execution requires pinned revision {recipe.verified_revision}; "
                f"received {request.hf_revision or 'none'}."
            )
        optimization = recipe.choice_for_profile(request.task_profile, request.skip_olive)
        if optimization is None:
            supported = ", ".join(
                f"{choice.task_profile}/skip_olive={choice.skip_olive}"
                for choice in recipe.optimization_choices
            )
            raise RuntimeError(
                f"Recipe '{recipe.id}' does not support task_profile={request.task_profile} "
                f"with skip_olive={request.skip_olive}. Supported: {supported or 'none'}."
            )
        if request.candidate.modality != CandidateModality.LLM:
            raise RuntimeError("Production execution currently supports LLM runtime validation only.")
        if recipe.olive is None:
            raise RuntimeError(f"Recipe '{recipe.id}' requires Olive settings for production packaging.")

        snapshot_dir = request.workspace_root / "snapshot"
        mobius_dir = request.workspace_root / "mobius"
        olive_dir = request.workspace_root / "olive"
        mobius_dir.mkdir(parents=True, exist_ok=False)
        olive_dir.mkdir(parents=True, exist_ok=False)

        transition(job, JobState.DOWNLOADING, f"Pinned Hugging Face revision {request.hf_revision}.")
        persist()
        snapshot_path = self._model_acquisition.acquire_snapshot(
            recipe.huggingface_model_id,
            snapshot_dir,
            revision=recipe.verified_revision,
        )
        mobius_dtype = recipe.mobius.dtype or "default"
        transition(
            job,
            JobState.MOBIUS_BUILDING,
            (
                f"Running verified Mobius {recipe.mobius.ep} {recipe.mobius.runtime} "
                f"{mobius_dtype} build."
            ),
        )
        persist()
        mobius_argv: list[str] = [
            "mobius",
            "build",
            "--model",
            str(snapshot_path),
            "--ep",
            recipe.mobius.ep,
            "--runtime",
            recipe.mobius.runtime,
        ]
        if recipe.mobius.task:
            mobius_argv.extend(["--task", recipe.mobius.task])
        if recipe.mobius.dtype:
            mobius_argv.extend(["--dtype", recipe.mobius.dtype])
        mobius_argv.append(str(mobius_dir))
        self._run_command(
            CommandSpec(
                argv=tuple(mobius_argv),
                cwd=request.workspace_root,
                timeout_seconds=self._build_timeout_seconds,
            ),
            cancellation_event,
            "Mobius build",
        )
        baseline_model_name = f"{recipe.model_name_prefix}-{job.job_id[:12]}-mobius-baseline:1"
        (mobius_dir / "inference_model.json").write_text(
            json.dumps({"Name": baseline_model_name}, indent=2),
            encoding="utf-8",
        )

        transition(job, JobState.MOBIUS_VALIDATING, "Mobius output created; ONNX validation follows Olive.")
        persist()
        transition(
            job,
            JobState.OLIVE_OPTIMIZING,
            (
                f"Running verified Olive {recipe.olive.input_source} "
                f"{recipe.olive.precision or 'default'} optimization."
            ),
        )
        persist()
        olive_argv: list[str] = [
            "olive",
            "optimize",
            "--model_name_or_path",
            str(mobius_dir),
            "--task",
            recipe.olive.task,
            "--device",
            recipe.olive.device,
            "--provider",
            recipe.olive.provider,
        ]
        if recipe.olive.precision:
            olive_argv.extend(["--precision", recipe.olive.precision])
        olive_argv.extend(
            [
                "--output_path",
                str(olive_dir),
                "--log_level",
                recipe.olive.log_level,
            ]
        )
        self._run_command(
            CommandSpec(
                argv=tuple(olive_argv),
                cwd=request.workspace_root,
                timeout_seconds=self._olive_timeout_seconds,
            ),
            cancellation_event,
            "Olive optimize",
        )
        source_dir = olive_dir
        self._ensure_required_ancillary_files(source_dir=source_dir, recipe=recipe)

        transition(job, JobState.PACKAGING, "Creating immutable Foundry Local BYOM package.")
        persist()
        artifact_id = self._artifact_id(job)
        model_name = f"{recipe.model_name_prefix}-{artifact_id[:12]}:1"
        staging_dir, package_dir = production_package_paths(job, recipe_registry=self._recipe_registry)
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
        transition(job, JobState.SUCCEEDED, recipe.success_message)
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
            f"{request.candidate.huggingface_model_id}:{request.hf_revision}:{request.task_profile}:{job.job_id}".encode()
        ).hexdigest()

    @staticmethod
    def _ensure_required_ancillary_files(*, source_dir: Path, recipe: ModelRecipe) -> None:
        missing = [
            rule.relative_path
            for rule in recipe.ancillary_files
            if rule.required and not (source_dir / rule.relative_path).exists()
        ]
        if missing:
            joined = ", ".join(sorted(missing))
            raise RuntimeError(
                f"Recipe '{recipe.id}' packaging is missing required ancillary files: {joined}."
            )


def production_package_paths(
    job: BuildJob,
    *,
    recipe_registry: RecipeRegistry = DEFAULT_RECIPE_REGISTRY,
) -> tuple[Path, Path]:
    artifact_id = ProductionBuildStageRunner._artifact_id(job)
    cache = job.request.model_cache_dir
    recipe = recipe_registry.resolve(
        model_id=job.request.candidate.huggingface_model_id,
        modality=job.request.candidate.modality,
        task_profile=job.request.task_profile,
        allow_experimental=True,
    ).recipe
    prefix = recipe.artifact_cache_prefix if recipe else _fallback_cache_prefix(job.request.candidate.huggingface_model_id)
    return (
        cache / f".partial-{prefix}-{artifact_id[:12]}",
        cache / f"{prefix}-{artifact_id[:12]}",
    )


def _fallback_cache_prefix(model_id: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", model_id.strip().split("/")[-1].lower())
    return slug or "model"
