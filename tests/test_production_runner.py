from __future__ import annotations

import json
import threading
import time

from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest

import fl_model_onboarding.production_runner as production_runner_module
from fl_model_onboarding.adapters.interfaces import CommandResult, CommandSpec
from fl_model_onboarding.architecture_capabilities import (
    load_architecture_capability_registry,
    normalize_huggingface_metadata,
)
from fl_model_onboarding.candidates import PHASE0_CANDIDATES
from fl_model_onboarding.contracts import (
    ArtifactKind,
    BuildArtifact,
    BuildJob,
    BuildRequest,
    CandidateModality,
    FailureClassification,
    GeneratedRecipeAttemptBinding,
    JobState,
    ModelCandidate,
    PreflightResult,
    ToolAvailability,
    ToolInvocationTerminalStage,
    ValidationStatus,
)
from fl_model_onboarding.local_service import BuildSubmission, LocalOnboardingService
from fl_model_onboarding.production_runner import (
    FoundrySdkTextInferenceBackend,
    ProductionBuildStageRunner,
    SMOLLM2_REVISION,
    production_package_paths,
)
from fl_model_onboarding.recipe_attempt_store import (
    AttemptState,
    GeneratedRecipeRecord,
    RecipeAttempt,
    build_capability_fingerprint,
    build_profile_fingerprint,
    build_toolchain_fingerprint,
)
from fl_model_onboarding.recipe_compiler import (
    RecipeCompilerInput,
    RecipeCompilerToolchain,
    compile_generated_recipe,
    compile_trusted_candidate_recipe,
)
from fl_model_onboarding.recipe_selection_policy import DEFAULT_CPU_INT4_RECIPE_SELECTION_POLICY
from fl_model_onboarding.recipes import (
    DEFAULT_MODEL_RECIPES,
    DISTIL_WHISPER_MODEL_ID,
    RecipeRegistry,
    RecipeStatus,
)
from fl_model_onboarding.state_machine import transition


class ContractProcessRunner:
    def __init__(self) -> None:
        self.specs: list[CommandSpec] = []
        self.cancel_events = []

    def run(self, spec: CommandSpec, cancel_event=None) -> CommandResult:  # noqa: ANN001
        self.specs.append(spec)
        self.cancel_events.append(cancel_event)
        argv = spec.argv
        stdout = ""
        if argv[:2] == ("mobius", "build"):
            output = Path(argv[-1])
            (output / "model.onnx").write_bytes(b"onnx")
            (output / "genai_config.json").write_text("{}", encoding="utf-8")
            (output / "tokenizer.json").write_text("{}", encoding="utf-8")
        elif argv[:2] == ("olive", "optimize"):
            output = Path(argv[argv.index("--output_path") + 1])
            (output / "model.onnx").write_bytes(b"optimized")
            (output / "genai_config.json").write_text("{}", encoding="utf-8")
            (output / "tokenizer.json").write_text("{}", encoding="utf-8")
        elif "validate-runtime" in argv:
            stdout = json.dumps(
                {
                    "ok": True,
                    "checks": [
                        "onnx_checker=1",
                        "ort_cpu_load=passed",
                        "oga_generation=passed",
                    ],
                }
            )
        elif "foundry-infer" in argv:
            stdout = json.dumps({"ok": True, "output": "OK"})
        return CommandResult(spec=spec, exit_code=0, stdout=stdout, stderr="")


class BatchInferenceProcessRunner:
    def __init__(self, *, fail_payload: dict[str, object] | None = None) -> None:
        self.specs: list[CommandSpec] = []
        self.cancel_events = []
        self.request_payloads: list[dict[str, object]] = []
        self.request_files: list[Path] = []
        self._fail_payload = fail_payload

    def run(self, spec: CommandSpec, cancel_event=None) -> CommandResult:  # noqa: ANN001
        self.specs.append(spec)
        self.cancel_events.append(cancel_event)
        argv = spec.argv
        if "foundry-infer-batch" not in argv:
            raise AssertionError(f"Unexpected command in batch runner: {argv!r}")
        request_file = Path(argv[argv.index("--request-file") + 1])
        self.request_files.append(request_file)
        payload = json.loads(request_file.read_text(encoding="utf-8"))
        self.request_payloads.append(payload)
        if self._fail_payload is not None:
            return CommandResult(
                spec=spec,
                exit_code=1,
                stdout=json.dumps(self._fail_payload),
                stderr="",
            )
        prompts = payload.get("prompts")
        assert isinstance(prompts, list)
        results = [
            {"prompt_id": str(row.get("prompt_id")), "output": f"out:{row.get('prompt_id')}"}
            for row in prompts
            if isinstance(row, dict)
        ]
        return CommandResult(
            spec=spec,
            exit_code=0,
            stdout=json.dumps({"ok": True, "results": results}),
            stderr="",
        )


class PinnedSnapshot:
    def acquire_snapshot(
        self,
        model_id: str,
        local_dir: Path,
        revision: str | None = None,
        allow_patterns=None,  # noqa: ANN001
    ) -> Path:
        assert model_id == "HuggingFaceTB/SmolLM2-1.7B-Instruct"
        assert revision == SMOLLM2_REVISION
        local_dir.mkdir(parents=True)
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        return local_dir


class GenericSnapshot:
    def acquire_snapshot(
        self,
        model_id: str,  # noqa: ARG002
        local_dir: Path,
        revision: str | None = None,  # noqa: ARG002
        allow_patterns=None,  # noqa: ANN001, ARG002
    ) -> Path:
        local_dir.mkdir(parents=True)
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        (local_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        return local_dir


class CapturingSnapshot:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path, str | None]] = []

    def acquire_snapshot(
        self,
        model_id: str,
        local_dir: Path,
        revision: str | None = None,
        allow_patterns=None,  # noqa: ANN001, ARG002
    ) -> Path:
        self.calls.append((model_id, local_dir, revision))
        local_dir.mkdir(parents=True)
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        (local_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        return local_dir


class InMemoryAttemptStore:
    def __init__(
        self,
        *,
        attempt: RecipeAttempt | None,
        generated: GeneratedRecipeRecord | None,
    ) -> None:
        self._attempt = attempt
        self._generated = generated

    def get_attempt(self, attempt_id: str) -> RecipeAttempt:
        if self._attempt is None or self._attempt.attempt_id != attempt_id:
            raise KeyError(attempt_id)
        return self._attempt

    def get_generated_recipe(self, recipe_fingerprint: str) -> GeneratedRecipeRecord | None:
        if self._generated is None:
            return None
        if self._generated.recipe_fingerprint != recipe_fingerprint:
            return None
        return self._generated


_GENERATED_TOOLCHAIN = RecipeCompilerToolchain(
    mobius_version="0.1.0",
    olive_version="0.13.0",
    onnx_version="1.22.0",
    ort_version="1.29.0",
    oga_version="0.15.2",
    foundry_sdk_version="1.2.4",
    foundry_cli_version="0.11.0",
)

_FROZEN_FIVE_MODELS: tuple[tuple[str, str], ...] = (
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "fe8a4ea1ffedaf415f4da2f062534de366a451e6"),
    ("HuggingFaceTB/SmolLM2-360M-Instruct", "a10cc1512eabd3dde888204e902eca88bddb4951"),
    ("Qwen/Qwen2-1.5B-Instruct", "ba1cf1846d7df0a0591d6c00649f57e798519da8"),
    ("Qwen/Qwen2-0.5B-Instruct", "c540970f9e29518b1d8f06ab8b24cba66ad77b6d"),
    ("ibm-granite/granite-3.2-2b-instruct", "641593c3b25bec0b1efe9f0f7d7a67f7243f86a3"),
)


def _compile_generated_candidate(model_id: str, revision_sha: str):
    capability_registry = load_architecture_capability_registry()
    normalized_metadata = normalize_huggingface_metadata(
        model_id=model_id,
        config={"model_type": "llama", "architectures": ["LlamaForCausalLM"]},
        is_gated=False,
        is_private=False,
    )
    resolution = capability_registry.resolve(
        metadata=normalized_metadata,
        task="llm",
        device="cpu",
        requested_precision="auto",
    )
    return compile_generated_recipe(
        RecipeCompilerInput(
            model_id=model_id,
            revision_sha=revision_sha,
            model_type="llama",
            architectures=("LlamaForCausalLM",),
            task="llm",
            requested_device="cpu",
            requested_precision="auto",
            is_gated=False,
            requires_remote_code=False,
            config_files=("config.json",),
            tokenizer_files=("tokenizer.json",),
            available_files=("config.json", "tokenizer.json"),
            capability_resolution=resolution,
            toolchain=_GENERATED_TOOLCHAIN,
        )
    )


def _generated_record_for(candidate) -> GeneratedRecipeRecord:
    return GeneratedRecipeRecord(
        recipe_fingerprint=candidate.fingerprint,
        schema_version=str(candidate.payload()["schema_version"]),
        recipe_status=candidate.recipe.status,
        model_id=candidate.recipe.huggingface_model_id,
        revision_sha=candidate.pinned_revision,
        requested_device=candidate.provenance.input_metadata.requested_device,
        requested_precision=candidate.provenance.input_metadata.requested_precision,
        compiler_version=candidate.provenance.compiler_version,
        capability_fingerprint=build_capability_fingerprint(candidate.provenance),
        toolchain_fingerprint=build_toolchain_fingerprint(candidate.provenance.toolchain),
        profile_fingerprint=build_profile_fingerprint(
            candidate.recipe,
            candidate.provenance.input_metadata,
        ),
        canonical_json=candidate.canonical_json,
        created_utc=datetime.now(timezone.utc),
    )


def _attempt_for_generated(
    *,
    attempt_id: str,
    record: GeneratedRecipeRecord,
    state: AttemptState = AttemptState.RUNNING,
) -> RecipeAttempt:
    finished = None if state == AttemptState.RUNNING else datetime.now(timezone.utc)
    return RecipeAttempt(
        attempt_id=attempt_id,
        idempotency_key="generated-attempt-test",
        request_fingerprint="f" * 64,
        recipe_fingerprint=record.recipe_fingerprint,
        model_id=record.model_id,
        revision_sha=record.revision_sha,
        requested_device=record.requested_device,
        requested_precision=record.requested_precision,
        compiler_version=record.compiler_version,
        capability_fingerprint=record.capability_fingerprint,
        toolchain_fingerprint=record.toolchain_fingerprint,
        profile_fingerprint=record.profile_fingerprint,
        created_utc=datetime.now(timezone.utc),
        finished_utc=finished,
        state=state,
        gate_results=(),
        failure=None,
    )


def _generated_request(tmp_path: Path, candidate, *, attempt_id: str) -> BuildRequest:
    selected = candidate.recipe.default_optimization()
    assert selected is not None
    return BuildRequest(
        candidate=ModelCandidate(
            key=candidate.recipe.id,
            huggingface_model_id=candidate.recipe.huggingface_model_id,
            modality=candidate.recipe.modality,
            recommended_mobius_dtype=candidate.recipe.mobius.dtype,
            recommended_olive_precision=(None if selected.skip_olive else selected.precision),
            notes="generated attempt",
        ),
        workspace_root=tmp_path / "w-generated",
        model_cache_dir=tmp_path / "cache-generated",
        output_dir=tmp_path / "w-generated" / "output",
        task_profile=selected.task_profile,
        hf_revision=candidate.pinned_revision,
        skip_olive=selected.skip_olive,
        dry_run=False,
        recipe_id=candidate.recipe.id,
        recipe_version=candidate.recipe.version,
        recipe_status=candidate.recipe.status.value,
        recipe_reason="generated-attempt-confirmed",
        generated_recipe_attempt=GeneratedRecipeAttemptBinding(
            attempt_id=attempt_id,
            recipe_fingerprint=candidate.fingerprint,
            confirmed=True,
            confirmation_provenance="api.confirm_automatic_recipe_attempt",
        ),
        recipe_artifact_cache_prefix=candidate.recipe.artifact_cache_prefix,
        recipe_model_name_prefix=candidate.recipe.model_name_prefix,
        allow_experimental=True,
        optimization_strategy=selected.strategy,
        optimization_precision=selected.precision,
    )


class PinnedPreflight:
    def inspect(self, request: BuildRequest) -> PreflightResult:
        return PreflightResult(
            candidate=request.candidate,
            workspace_root=request.workspace_root,
            model_cache_dir=request.model_cache_dir,
            output_dir=request.output_dir,
            disk_free_gb_workspace=100.0,
            disk_free_gb_cache=100.0,
            tools=(
                ToolAvailability("mobius", "command", True, "0.1.0"),
                ToolAvailability("olive", "command", True, "0.13.0"),
                ToolAvailability("onnxruntime-genai", "python-package", True, "0.15.2"),
                ToolAvailability("foundry-local-sdk", "python-package", True, "1.2.4"),
            ),
            foundry_catalog_matches=(),
            huggingface_revision=SMOLLM2_REVISION,
            huggingface_sha=SMOLLM2_REVISION,
            huggingface_private=False,
            huggingface_gated=False,
            cache_key="production-smollm2",
            blockers=(),
            warnings=("mobius revision pin is recorded because its CLI has no --revision flag",),
        )


def _request(tmp_path: Path) -> BuildRequest:
    return BuildRequest(
        candidate=PHASE0_CANDIDATES["smollm2-1.7b-instruct"],
        workspace_root=tmp_path / "w",
        model_cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "w" / "output",
        task_profile="llm-cpu-int4",
        hf_revision=SMOLLM2_REVISION,
    )


def test_production_runner_uses_verified_contract_and_registers_artifact(tmp_path: Path) -> None:
    request = _request(tmp_path)
    request.workspace_root.mkdir(parents=True)
    request.model_cache_dir.mkdir(parents=True)
    job = BuildJob(job_id="12345678-production", request=request)
    transition(job, JobState.PREFLIGHT, "Preflight passed.")
    runner = ContractProcessRunner()
    production = ProductionBuildStageRunner(runner, model_acquisition=PinnedSnapshot())  # type: ignore[arg-type]

    production.run(job, persist=lambda: None, cancellation_event=Event())

    assert job.state == JobState.SUCCEEDED
    assert job.result_artifact_id is not None
    assert job.artifacts[0].path.is_dir()
    assert (job.artifacts[0].path / "inference_model.json").exists()
    assert any(
        validation.stage == JobState.INFERENCING
        and validation.status == ValidationStatus.PASSED
        for validation in job.validations
    )
    mobius = runner.specs[0]
    assert mobius.argv[:3] == ("mobius", "build", "--config")
    config_path = Path(mobius.argv[3])
    assert config_path.parent == request.model_cache_dir
    assert config_path.name.startswith("snapshot-huggingfacetb-smollm2-1.7b-instruct-")
    assert Path(mobius.argv[-1]).name == "mobius"
    olive = runner.specs[1]
    assert olive.argv[:2] == ("olive", "optimize")
    assert "text-generation-with-past" in olive.argv
    assert all(spec.timeout_seconds <= 7200 for spec in runner.specs)
    assert runner.cancel_events[-1] is not None

    evidence = job.production_invocation_evidence
    assert evidence is not None
    assert evidence.mobius.invocation_count == 1
    assert evidence.mobius.terminal_stage == ToolInvocationTerminalStage.COMPLETED
    assert evidence.mobius.success is True
    assert evidence.mobius.wall_seconds is not None
    assert evidence.olive.invocation_count == 1
    assert evidence.olive.terminal_stage == ToolInvocationTerminalStage.COMPLETED
    assert evidence.olive.success is True
    assert evidence.olive.wall_seconds is not None
    sanitized = evidence.sanitized_payload()
    assert sanitized["mobius_invocation_count"] == 1
    assert sanitized["olive_invocation_count"] == 1
    counters = production_runner_module.production_invocation_evidence_to_candidate_counters(evidence)
    assert counters.mobius_build_invocation_count == 1
    assert counters.olive_optimize_invocation_count == 1
    assert counters.total_invocation_count == 2


def test_validation_rejection_inside_run_leaves_tool_evidence_not_run(tmp_path: Path) -> None:
    """A validation failure reached *inside* `_run` (after the recipe/pinned
    revision resolves, but before any process is ever launched) must leave
    both tools' invocation evidence at their not-run defaults: no process was
    ever attempted, so the count must stay `0`/`NOT_RUN`, never be inferred as
    a real run of zero invocations."""
    request = replace(_request(tmp_path), skip_olive=True)
    request.workspace_root.mkdir(parents=True)
    request.model_cache_dir.mkdir(parents=True)
    job = BuildJob(job_id="preflight-reject", request=request)
    transition(job, JobState.PREFLIGHT, "Preflight passed.")
    runner = ContractProcessRunner()
    production = ProductionBuildStageRunner(runner, model_acquisition=PinnedSnapshot())  # type: ignore[arg-type]

    production.run(job, persist=lambda: None, cancellation_event=Event())

    assert job.state == JobState.FAILED
    assert not runner.specs
    evidence = job.production_invocation_evidence
    assert evidence is not None
    assert evidence.mobius.invocation_count == 0
    assert evidence.mobius.terminal_stage == ToolInvocationTerminalStage.NOT_RUN
    assert evidence.mobius.success is None
    assert evidence.olive.invocation_count == 0
    assert evidence.olive.terminal_stage == ToolInvocationTerminalStage.NOT_RUN
    assert evidence.olive.success is None
    counters = production_runner_module.production_invocation_evidence_to_candidate_counters(evidence)
    assert counters.mobius_build_invocation_count is None
    assert counters.olive_optimize_invocation_count is None
    assert counters.total_invocation_count is None


def test_resolution_failure_before_run_leaves_no_invocation_evidence(tmp_path: Path) -> None:
    """A resolution failure that never even enters `_run` (e.g. an unknown
    task profile) must leave `production_invocation_evidence` at its job-level
    `None` default -- not merely at not-run tool defaults."""
    request = replace(_request(tmp_path), task_profile="not-a-real-profile")
    request.workspace_root.mkdir(parents=True)
    request.model_cache_dir.mkdir(parents=True)
    job = BuildJob(job_id="resolution-reject", request=request)
    transition(job, JobState.PREFLIGHT, "Preflight passed.")
    runner = ContractProcessRunner()
    production = ProductionBuildStageRunner(runner, model_acquisition=PinnedSnapshot())  # type: ignore[arg-type]

    production.run(job, persist=lambda: None, cancellation_event=Event())

    assert job.state == JobState.FAILED
    assert not runner.specs
    assert job.production_invocation_evidence is None


class _MobiusTimeoutProcessRunner(ContractProcessRunner):
    def run(self, spec: CommandSpec, cancel_event=None) -> CommandResult:  # noqa: ANN001
        if spec.argv[:2] == ("mobius", "build"):
            self.specs.append(spec)
            self.cancel_events.append(cancel_event)
            raise TimeoutError("synthetic mobius timeout")
        return super().run(spec, cancel_event=cancel_event)


def test_mobius_timeout_counts_once_and_records_timed_out_stage(tmp_path: Path) -> None:
    request = _request(tmp_path)
    request.workspace_root.mkdir(parents=True)
    request.model_cache_dir.mkdir(parents=True)
    job = BuildJob(job_id="mobius-timeout", request=request)
    transition(job, JobState.PREFLIGHT, "Preflight passed.")
    runner = _MobiusTimeoutProcessRunner()
    production = ProductionBuildStageRunner(runner, model_acquisition=PinnedSnapshot())  # type: ignore[arg-type]

    production.run(job, persist=lambda: None, cancellation_event=Event())

    assert job.state == JobState.FAILED
    evidence = job.production_invocation_evidence
    assert evidence is not None
    assert evidence.mobius.invocation_count == 1
    assert evidence.mobius.terminal_stage == ToolInvocationTerminalStage.TIMED_OUT
    assert evidence.mobius.success is False
    assert evidence.mobius.finished_utc is not None
    # Olive was never reached because Mobius failed first.
    assert evidence.olive.invocation_count == 0
    assert evidence.olive.terminal_stage == ToolInvocationTerminalStage.NOT_RUN


class _OliveFailureProcessRunner(ContractProcessRunner):
    def run(self, spec: CommandSpec, cancel_event=None) -> CommandResult:  # noqa: ANN001
        if spec.argv[:2] == ("olive", "optimize"):
            self.specs.append(spec)
            self.cancel_events.append(cancel_event)
            return CommandResult(spec=spec, exit_code=1, stdout="", stderr="synthetic olive failure")
        return super().run(spec, cancel_event=cancel_event)


def test_olive_process_failure_counts_once_and_records_failed_stage(tmp_path: Path) -> None:
    request = _request(tmp_path)
    request.workspace_root.mkdir(parents=True)
    request.model_cache_dir.mkdir(parents=True)
    job = BuildJob(job_id="olive-failure", request=request)
    transition(job, JobState.PREFLIGHT, "Preflight passed.")
    runner = _OliveFailureProcessRunner()
    production = ProductionBuildStageRunner(runner, model_acquisition=PinnedSnapshot())  # type: ignore[arg-type]

    production.run(job, persist=lambda: None, cancellation_event=Event())

    assert job.state == JobState.FAILED
    evidence = job.production_invocation_evidence
    assert evidence is not None
    # Mobius still ran and succeeded before Olive failed.
    assert evidence.mobius.invocation_count == 1
    assert evidence.mobius.terminal_stage == ToolInvocationTerminalStage.COMPLETED
    assert evidence.olive.invocation_count == 1
    assert evidence.olive.terminal_stage == ToolInvocationTerminalStage.FAILED
    assert evidence.olive.success is False


class _OliveCancelProcessRunner(ContractProcessRunner):
    def run(self, spec: CommandSpec, cancel_event=None) -> CommandResult:  # noqa: ANN001
        if spec.argv[:2] == ("olive", "optimize"):
            self.specs.append(spec)
            self.cancel_events.append(cancel_event)
            if cancel_event is not None:
                cancel_event.set()
            raise RuntimeError("synthetic cancellation after launch")
        return super().run(spec, cancel_event=cancel_event)


def test_olive_cancellation_after_launch_counts_once_and_records_cancelled_stage(tmp_path: Path) -> None:
    request = _request(tmp_path)
    request.workspace_root.mkdir(parents=True)
    request.model_cache_dir.mkdir(parents=True)
    job = BuildJob(job_id="olive-cancel", request=request)
    transition(job, JobState.PREFLIGHT, "Preflight passed.")
    runner = _OliveCancelProcessRunner()
    production = ProductionBuildStageRunner(runner, model_acquisition=PinnedSnapshot())  # type: ignore[arg-type]

    production.run(job, persist=lambda: None, cancellation_event=Event())

    evidence = job.production_invocation_evidence
    assert evidence is not None
    assert evidence.olive.invocation_count == 1
    assert evidence.olive.terminal_stage == ToolInvocationTerminalStage.CANCELLED
    assert evidence.olive.success is False


def test_concurrent_runs_have_isolated_invocation_counters(tmp_path: Path) -> None:
    """The same `ProductionBuildStageRunner` (and the same underlying process
    runner) handling several jobs concurrently must never leak invocation
    counts between jobs: counters live on each `BuildJob`, not on the runner
    instance."""
    runner = ContractProcessRunner()
    production = ProductionBuildStageRunner(runner, model_acquisition=PinnedSnapshot())  # type: ignore[arg-type]

    jobs: list[BuildJob] = []
    for index in range(4):
        request = _request(tmp_path / f"job-{index}")
        request.workspace_root.mkdir(parents=True)
        request.model_cache_dir.mkdir(parents=True)
        job = BuildJob(job_id=f"concurrent-{index}", request=request)
        transition(job, JobState.PREFLIGHT, "Preflight passed.")
        jobs.append(job)

    def _run_job(job: BuildJob) -> None:
        production.run(job, persist=lambda: None, cancellation_event=Event())

    threads = [threading.Thread(target=_run_job, args=(job,)) for job in jobs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    for job in jobs:
        assert job.state == JobState.SUCCEEDED
        evidence = job.production_invocation_evidence
        assert evidence is not None
        assert evidence.mobius.invocation_count == 1
        assert evidence.olive.invocation_count == 1


def test_production_runner_uses_explicit_runtime_interpreter_for_worker_commands(tmp_path: Path) -> None:
    request = _request(tmp_path)
    request.workspace_root.mkdir(parents=True)
    request.model_cache_dir.mkdir(parents=True)
    job = BuildJob(job_id="explicit-runtime-python", request=request)
    transition(job, JobState.PREFLIGHT, "Preflight passed.")
    runner = ContractProcessRunner()
    explicit_python = (tmp_path / "runtime-venv" / "Scripts" / "python.exe").resolve()

    ProductionBuildStageRunner(
        runner,
        model_acquisition=PinnedSnapshot(),  # type: ignore[arg-type]
        runtime_python_executable=explicit_python,
    ).run(job, persist=lambda: None, cancellation_event=Event())

    assert job.state == JobState.SUCCEEDED
    validate_runtime = next(spec for spec in runner.specs if "validate-runtime" in spec.argv)
    foundry_infer = next(spec for spec in runner.specs if "foundry-infer" in spec.argv)
    assert validate_runtime.argv[0] == str(explicit_python)
    assert foundry_infer.argv[0] == str(explicit_python)


def test_foundry_inference_backend_batches_prompts_with_single_worker_call(tmp_path: Path) -> None:
    request = _request(tmp_path)
    request.workspace_root.mkdir(parents=True)
    model_dir = tmp_path / "batch-model"
    model_dir.mkdir(parents=True)
    (model_dir / "inference_model.json").write_text(
        json.dumps({"Name": "batch-model"}),
        encoding="utf-8",
    )
    artifact = BuildArtifact(
        artifact_id="artifact-batch",
        kind=ArtifactKind.MODEL,
        path=model_dir,
        description="batch test artifact",
    )
    job = BuildJob(job_id="batch-job", request=request)
    runner = BatchInferenceProcessRunner()
    backend = FoundrySdkTextInferenceBackend(runner, timeout_seconds=90)

    outputs = backend.infer_batch(
        artifact=artifact,
        job=job,
        prompts=(
            ("prompt-a", "What is 17 + 28? Reply using only digits."),
            ("prompt-b", "Which planet is known as the Red Planet? Reply with one word."),
        ),
        max_tokens=64,
    )

    assert outputs == ("out:prompt-a", "out:prompt-b")
    assert len(runner.specs) == 1
    spec = runner.specs[0]
    assert "foundry-infer-batch" in spec.argv
    assert spec.timeout_seconds == 195
    payload = runner.request_payloads[0]
    assert payload["per_prompt_timeout_seconds"] == 90
    assert payload["batch_timeout_seconds"] == 180
    prompts = payload["prompts"]
    assert isinstance(prompts, list)
    assert [row["prompt_id"] for row in prompts] == ["prompt-a", "prompt-b"]
    assert [row["max_tokens"] for row in prompts] == [64, 64]
    assert all(not path.exists() for path in runner.request_files)


def test_foundry_inference_backend_batch_failure_surfaces_stage_and_prompt(tmp_path: Path) -> None:
    request = _request(tmp_path)
    request.workspace_root.mkdir(parents=True)
    model_dir = tmp_path / "batch-model"
    model_dir.mkdir(parents=True)
    (model_dir / "inference_model.json").write_text(
        json.dumps({"Name": "batch-model"}),
        encoding="utf-8",
    )
    artifact = BuildArtifact(
        artifact_id="artifact-batch-fail",
        kind=ArtifactKind.MODEL,
        path=model_dir,
        description="batch failure artifact",
    )
    job = BuildJob(job_id="batch-fail", request=request)
    runner = BatchInferenceProcessRunner(
        fail_payload={
            "ok": False,
            "error": "Batch quality inference prompt timed out.",
            "failure_stage": "prompt_timeout",
            "failed_prompt_id": "prompt-b",
        }
    )
    backend = FoundrySdkTextInferenceBackend(runner, timeout_seconds=90)

    with pytest.raises(RuntimeError, match="stage=prompt_timeout prompt_id=prompt-b"):
        backend.infer_batch(
            artifact=artifact,
            job=job,
            prompts=(
                ("prompt-a", "What is 17 + 28? Reply using only digits."),
                ("prompt-b", "Which planet is known as the Red Planet? Reply with one word."),
            ),
            max_tokens=64,
        )


def test_service_indexes_exact_revision_profile_after_runner_sdk_success(tmp_path: Path) -> None:
    process_runner = ContractProcessRunner()
    service = LocalOnboardingService(
        db_path=tmp_path / "state.sqlite3",
        workspace_base=tmp_path / "w",
        model_cache_dir=tmp_path / "cache",
        process_runner=process_runner,
        preflight_inspector=PinnedPreflight(),  # type: ignore[arg-type]
        build_stage_runner=ProductionBuildStageRunner(
            process_runner,
            model_acquisition=PinnedSnapshot(),  # type: ignore[arg-type]
        ),
    )
    try:
        job, _ = service.create_build(
            BuildSubmission(
                model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
                task=PHASE0_CANDIDATES["smollm2-1.7b-instruct"].modality,
                task_profile="llm-cpu-int4",
            ),
            idempotency_key="production-index",
        )
        deadline = time.time() + 5
        while time.time() < deadline and service.get_build(job.job_id).state not in {
            JobState.SUCCEEDED,
            JobState.FAILED,
        }:
            time.sleep(0.02)
        assert service.get_build(job.job_id).state == JobState.SUCCEEDED
        tested = service.health()["compatibility_index"]
        while time.time() < deadline and not tested:
            time.sleep(0.02)
            tested = service.health()["compatibility_index"]
        assert tested == [
            {
                "model_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                "task": "llm",
                "artifact_id": job.result_artifact_id,
                "verified_utc": tested[0]["verified_utc"],  # type: ignore[index]
                "evidence": "successful_fl_inference",
                "revision": SMOLLM2_REVISION,
                "task_profile": "llm-cpu-int4",
                "display_name": "SmolLM2-1.7B-Instruct",
                "tested_status": "tested",
            }
        ]
    finally:
        service.close()


def test_production_runner_rejects_unverified_profile_without_running_tools(tmp_path: Path) -> None:
    request = replace(_request(tmp_path), task_profile="llm-cpu-f16")
    request.workspace_root.mkdir(parents=True)
    request.model_cache_dir.mkdir(parents=True)
    job = BuildJob(job_id="wrong-profile", request=request)
    transition(job, JobState.PREFLIGHT, "Preflight passed.")
    process_runner = ContractProcessRunner()

    ProductionBuildStageRunner(
        process_runner,
        model_acquisition=PinnedSnapshot(),  # type: ignore[arg-type]
    ).run(job, persist=lambda: None, cancellation_event=Event())

    assert job.state == JobState.FAILED
    assert job.failure is not None
    assert "llm-cpu-int4" in job.failure.message
    assert process_runner.specs == []


def test_production_runner_rejects_non_verified_recipe_without_running_tools(tmp_path: Path) -> None:
    request = BuildRequest(
        candidate=ModelCandidate(
            key="distil-whisper-cpu-fp16",
            huggingface_model_id=DISTIL_WHISPER_MODEL_ID,
            modality=CandidateModality.ASR,
            recommended_mobius_dtype="f32",
            recommended_olive_precision="fp32",
            notes="blocked recipe",
        ),
        workspace_root=tmp_path / "w",
        model_cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "w" / "output",
        task_profile="asr-cpu-fp16",
        hf_revision="6e61418885eaf4d5cc9f64e508e80ac5b4c052b7",
    )
    request.workspace_root.mkdir(parents=True)
    request.model_cache_dir.mkdir(parents=True)
    job = BuildJob(job_id="distil-blocked", request=request)
    transition(job, JobState.PREFLIGHT, "Preflight passed.")
    process_runner = ContractProcessRunner()

    ProductionBuildStageRunner(
        process_runner,
        model_acquisition=PinnedSnapshot(),  # type: ignore[arg-type]
    ).run(job, persist=lambda: None, cancellation_event=Event())

    assert job.state == JobState.FAILED
    assert job.failure is not None
    assert "verified" in job.failure.message
    assert process_runner.specs == []


def test_production_runner_executes_generated_attempt_from_persisted_record(tmp_path: Path) -> None:
    generated_candidate = _compile_generated_candidate(
        "owner/unregistered-model",
        "1234567890abcdef1234567890abcdef12345678",
    )
    generated_record = _generated_record_for(generated_candidate)
    attempt_id = "11111111-1111-1111-1111-111111111111"
    generated_attempt = _attempt_for_generated(
        attempt_id=attempt_id,
        record=generated_record,
        state=AttemptState.RUNNING,
    )
    request = _generated_request(tmp_path, generated_candidate, attempt_id=attempt_id)
    request.workspace_root.mkdir(parents=True)
    request.model_cache_dir.mkdir(parents=True)
    job = BuildJob(job_id="generated-success", request=request)
    transition(job, JobState.PREFLIGHT, "Preflight passed.")
    process_runner = ContractProcessRunner()
    store = InMemoryAttemptStore(
        attempt=generated_attempt,
        generated=generated_record,
    )

    ProductionBuildStageRunner(
        process_runner,
        model_acquisition=GenericSnapshot(),  # type: ignore[arg-type]
        recipe_attempt_store=store,  # type: ignore[arg-type]
    ).run(job, persist=lambda: None, cancellation_event=Event())

    assert job.state == JobState.SUCCEEDED
    assert process_runner.specs
    mobius = process_runner.specs[0]
    assert mobius.argv[:3] == ("mobius", "build", "--config")
    assert "--model" not in mobius.argv
    assert generated_candidate.recipe.huggingface_model_id not in mobius.argv
    assert Path(mobius.argv[3]).parent == request.model_cache_dir
    assert "--runtime" in mobius.argv
    assert mobius.argv[mobius.argv.index("--runtime") + 1] == "ort-genai"


def test_production_runner_executes_trusted_block64_candidate_with_real_olive_argv(tmp_path: Path) -> None:
    """Slice 3A1 end-to-end: a trusted candidate-1 recipe compiled by
    `compile_trusted_candidate_recipe` (block_size=64) executes through the
    existing generated-attempt path and actually threads `--block_size 64`
    into the real Olive `optimize` command line, with per-job Mobius/Olive
    invocation evidence recorded."""
    default_candidate = _compile_generated_candidate(
        "owner/fallback-block64-model",
        "2234567890abcdef1234567890abcdef12345678",
    )
    policy = DEFAULT_CPU_INT4_RECIPE_SELECTION_POLICY
    fallback_recipe = compile_trusted_candidate_recipe(
        default_candidate,
        policy=policy,
        candidate=policy.candidates[1],
    )
    assert fallback_recipe.recipe.olive is not None
    assert fallback_recipe.recipe.olive.block_size == 64

    generated_record = _generated_record_for(fallback_recipe)
    attempt_id = "22222222-2222-2222-2222-222222222222"
    generated_attempt = _attempt_for_generated(
        attempt_id=attempt_id,
        record=generated_record,
        state=AttemptState.RUNNING,
    )
    request = _generated_request(tmp_path, fallback_recipe, attempt_id=attempt_id)
    request.workspace_root.mkdir(parents=True)
    request.model_cache_dir.mkdir(parents=True)
    job = BuildJob(job_id="generated-block64", request=request)
    transition(job, JobState.PREFLIGHT, "Preflight passed.")
    process_runner = ContractProcessRunner()
    store = InMemoryAttemptStore(
        attempt=generated_attempt,
        generated=generated_record,
    )

    ProductionBuildStageRunner(
        process_runner,
        model_acquisition=GenericSnapshot(),  # type: ignore[arg-type]
        recipe_attempt_store=store,  # type: ignore[arg-type]
    ).run(job, persist=lambda: None, cancellation_event=Event())

    assert job.state == JobState.SUCCEEDED
    olive = next(spec for spec in process_runner.specs if spec.argv[:2] == ("olive", "optimize"))
    assert "--block_size" in olive.argv
    assert olive.argv[olive.argv.index("--block_size") + 1] == "64"

    evidence = job.production_invocation_evidence
    assert evidence is not None
    assert evidence.mobius.invocation_count == 1
    assert evidence.olive.invocation_count == 1
    counters = production_runner_module.production_invocation_evidence_to_candidate_counters(evidence)
    assert counters.mobius_build_invocation_count == 1
    assert counters.olive_optimize_invocation_count == 1
    assert counters.total_invocation_count == 2


@pytest.mark.parametrize(("model_id", "revision_sha"), _FROZEN_FIVE_MODELS)
def test_generated_execution_plan_uses_local_snapshot_config_for_frozen_models(
    tmp_path: Path,
    model_id: str,
    revision_sha: str,
) -> None:
    generated_candidate = _compile_generated_candidate(model_id, revision_sha)
    generated_record = _generated_record_for(generated_candidate)
    attempt_id = "77777777-7777-7777-7777-777777777777"
    generated_attempt = _attempt_for_generated(
        attempt_id=attempt_id,
        record=generated_record,
        state=AttemptState.RUNNING,
    )
    request = _generated_request(tmp_path, generated_candidate, attempt_id=attempt_id)
    request.workspace_root.mkdir(parents=True)
    request.model_cache_dir.mkdir(parents=True)
    job = BuildJob(job_id=f"generated-plan-{hash(model_id) & 0xffff:x}", request=request)
    transition(job, JobState.PREFLIGHT, "Preflight passed.")
    process_runner = ContractProcessRunner()
    store = InMemoryAttemptStore(
        attempt=generated_attempt,
        generated=generated_record,
    )

    ProductionBuildStageRunner(
        process_runner,
        model_acquisition=GenericSnapshot(),  # type: ignore[arg-type]
        recipe_attempt_store=store,  # type: ignore[arg-type]
    ).run(job, persist=lambda: None, cancellation_event=Event())

    assert job.state == JobState.SUCCEEDED
    mobius = process_runner.specs[0]
    assert mobius.argv[:3] == ("mobius", "build", "--config")
    assert "--model" not in mobius.argv
    assert model_id not in mobius.argv
    assert "\\" in mobius.argv[3]
    assert Path(mobius.argv[3]).parent == request.model_cache_dir


def test_generated_execution_passes_pinned_revision_and_snapshot_config_source(tmp_path: Path) -> None:
    generated_candidate = _compile_generated_candidate(
        "owner/unregistered-model",
        "1234567890abcdef1234567890abcdef12345678",
    )
    generated_record = _generated_record_for(generated_candidate)
    attempt_id = "88888888-8888-8888-8888-888888888888"
    generated_attempt = _attempt_for_generated(
        attempt_id=attempt_id,
        record=generated_record,
        state=AttemptState.RUNNING,
    )
    request = _generated_request(tmp_path, generated_candidate, attempt_id=attempt_id)
    request.workspace_root.mkdir(parents=True)
    request.model_cache_dir.mkdir(parents=True)
    job = BuildJob(job_id="generated-pinned-source", request=request)
    transition(job, JobState.PREFLIGHT, "Preflight passed.")
    process_runner = ContractProcessRunner()
    snapshot = CapturingSnapshot()
    store = InMemoryAttemptStore(
        attempt=generated_attempt,
        generated=generated_record,
    )

    ProductionBuildStageRunner(
        process_runner,
        model_acquisition=snapshot,  # type: ignore[arg-type]
        recipe_attempt_store=store,  # type: ignore[arg-type]
    ).run(job, persist=lambda: None, cancellation_event=Event())

    assert job.state == JobState.SUCCEEDED
    assert len(snapshot.calls) == 1
    model_id, snapshot_dir, revision = snapshot.calls[0]
    assert model_id == generated_candidate.recipe.huggingface_model_id
    assert revision == generated_candidate.pinned_revision
    assert revision is not None
    assert len(revision) == 40
    assert snapshot_dir.parent == request.model_cache_dir
    assert (snapshot_dir / "tokenizer.json").is_file()
    mobius = process_runner.specs[0]
    assert mobius.argv[:3] == ("mobius", "build", "--config")
    assert Path(mobius.argv[3]) == snapshot_dir
    assert "--model" not in mobius.argv


def test_production_runner_rejects_generated_attempt_revision_mismatch_before_tools(tmp_path: Path) -> None:
    generated_candidate = _compile_generated_candidate(
        "owner/unregistered-model",
        "1234567890abcdef1234567890abcdef12345678",
    )
    generated_record = _generated_record_for(generated_candidate)
    attempt_id = "22222222-2222-2222-2222-222222222222"
    generated_attempt = _attempt_for_generated(
        attempt_id=attempt_id,
        record=generated_record,
        state=AttemptState.RUNNING,
    )
    request = replace(
        _generated_request(tmp_path, generated_candidate, attempt_id=attempt_id),
        hf_revision="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )
    request.workspace_root.mkdir(parents=True)
    request.model_cache_dir.mkdir(parents=True)
    job = BuildJob(job_id="generated-revision-mismatch", request=request)
    transition(job, JobState.PREFLIGHT, "Preflight passed.")
    process_runner = ContractProcessRunner()
    store = InMemoryAttemptStore(
        attempt=generated_attempt,
        generated=generated_record,
    )

    ProductionBuildStageRunner(
        process_runner,
        model_acquisition=GenericSnapshot(),  # type: ignore[arg-type]
        recipe_attempt_store=store,  # type: ignore[arg-type]
    ).run(job, persist=lambda: None, cancellation_event=Event())

    assert job.state == JobState.FAILED
    assert job.failure is not None
    assert job.failure.classification == FailureClassification.INVALID_REQUEST
    assert process_runner.specs == []


def test_production_runner_rejects_missing_generated_attempt_before_tools(tmp_path: Path) -> None:
    generated_candidate = _compile_generated_candidate(
        "owner/unregistered-model",
        "1234567890abcdef1234567890abcdef12345678",
    )
    generated_record = _generated_record_for(generated_candidate)
    request = _generated_request(
        tmp_path,
        generated_candidate,
        attempt_id="missing-attempt-id",
    )
    request.workspace_root.mkdir(parents=True)
    request.model_cache_dir.mkdir(parents=True)
    job = BuildJob(job_id="generated-missing-attempt", request=request)
    transition(job, JobState.PREFLIGHT, "Preflight passed.")
    process_runner = ContractProcessRunner()
    store = InMemoryAttemptStore(
        attempt=None,
        generated=generated_record,
    )

    ProductionBuildStageRunner(
        process_runner,
        model_acquisition=GenericSnapshot(),  # type: ignore[arg-type]
        recipe_attempt_store=store,  # type: ignore[arg-type]
    ).run(job, persist=lambda: None, cancellation_event=Event())

    assert job.state == JobState.FAILED
    assert job.failure is not None
    assert job.failure.classification == FailureClassification.INVALID_REQUEST
    assert process_runner.specs == []


def test_production_runner_rejects_generated_fingerprint_mismatch_before_tools(tmp_path: Path) -> None:
    generated_candidate = _compile_generated_candidate(
        "owner/unregistered-model",
        "1234567890abcdef1234567890abcdef12345678",
    )
    generated_record = _generated_record_for(generated_candidate)
    attempt_id = "66666666-6666-6666-6666-666666666666"
    generated_attempt = _attempt_for_generated(
        attempt_id=attempt_id,
        record=generated_record,
        state=AttemptState.RUNNING,
    )
    request = replace(
        _generated_request(tmp_path, generated_candidate, attempt_id=attempt_id),
        generated_recipe_attempt=GeneratedRecipeAttemptBinding(
            attempt_id=attempt_id,
            recipe_fingerprint="0" * 64,
            confirmed=True,
            confirmation_provenance="api.confirm_automatic_recipe_attempt",
        ),
    )
    request.workspace_root.mkdir(parents=True)
    request.model_cache_dir.mkdir(parents=True)
    job = BuildJob(job_id="generated-fingerprint-mismatch", request=request)
    transition(job, JobState.PREFLIGHT, "Preflight passed.")
    process_runner = ContractProcessRunner()
    store = InMemoryAttemptStore(
        attempt=generated_attempt,
        generated=generated_record,
    )

    ProductionBuildStageRunner(
        process_runner,
        model_acquisition=GenericSnapshot(),  # type: ignore[arg-type]
        recipe_attempt_store=store,  # type: ignore[arg-type]
    ).run(job, persist=lambda: None, cancellation_event=Event())

    assert job.state == JobState.FAILED
    assert job.failure is not None
    assert job.failure.classification == FailureClassification.INVALID_REQUEST
    assert process_runner.specs == []


def test_production_runner_rejects_generated_attempt_client_injected_optimization_before_tools(
    tmp_path: Path,
) -> None:
    generated_candidate = _compile_generated_candidate(
        "owner/unregistered-model",
        "1234567890abcdef1234567890abcdef12345678",
    )
    generated_record = _generated_record_for(generated_candidate)
    attempt_id = "33333333-3333-3333-3333-333333333333"
    generated_attempt = _attempt_for_generated(
        attempt_id=attempt_id,
        record=generated_record,
        state=AttemptState.RUNNING,
    )
    request = replace(
        _generated_request(tmp_path, generated_candidate, attempt_id=attempt_id),
        optimization_precision="fp32",
    )
    request.workspace_root.mkdir(parents=True)
    request.model_cache_dir.mkdir(parents=True)
    job = BuildJob(job_id="generated-client-override", request=request)
    transition(job, JobState.PREFLIGHT, "Preflight passed.")
    process_runner = ContractProcessRunner()
    store = InMemoryAttemptStore(
        attempt=generated_attempt,
        generated=generated_record,
    )

    ProductionBuildStageRunner(
        process_runner,
        model_acquisition=GenericSnapshot(),  # type: ignore[arg-type]
        recipe_attempt_store=store,  # type: ignore[arg-type]
    ).run(job, persist=lambda: None, cancellation_event=Event())

    assert job.state == JobState.FAILED
    assert job.failure is not None
    assert job.failure.classification == FailureClassification.INVALID_REQUEST
    assert process_runner.specs == []


def test_production_runner_rejects_cancelled_generated_attempt_before_tools(tmp_path: Path) -> None:
    generated_candidate = _compile_generated_candidate(
        "owner/unregistered-model",
        "1234567890abcdef1234567890abcdef12345678",
    )
    generated_record = _generated_record_for(generated_candidate)
    attempt_id = "44444444-4444-4444-4444-444444444444"
    generated_attempt = _attempt_for_generated(
        attempt_id=attempt_id,
        record=generated_record,
        state=AttemptState.CANCELLED,
    )
    request = _generated_request(tmp_path, generated_candidate, attempt_id=attempt_id)
    request.workspace_root.mkdir(parents=True)
    request.model_cache_dir.mkdir(parents=True)
    job = BuildJob(job_id="generated-cancelled", request=request)
    transition(job, JobState.PREFLIGHT, "Preflight passed.")
    process_runner = ContractProcessRunner()
    store = InMemoryAttemptStore(
        attempt=generated_attempt,
        generated=generated_record,
    )

    ProductionBuildStageRunner(
        process_runner,
        model_acquisition=GenericSnapshot(),  # type: ignore[arg-type]
        recipe_attempt_store=store,  # type: ignore[arg-type]
    ).run(job, persist=lambda: None, cancellation_event=Event())

    assert job.state == JobState.FAILED
    assert job.failure is not None
    assert job.failure.classification == FailureClassification.INVALID_REQUEST
    assert process_runner.specs == []


def test_production_runner_rejects_experimental_static_recipe_without_running_tools(tmp_path: Path) -> None:
    experimental = PHASE0_CANDIDATES["smollm2-1.7b-instruct"]
    request = BuildRequest(
        candidate=experimental,
        workspace_root=tmp_path / "w-static-experimental",
        model_cache_dir=tmp_path / "cache-static-experimental",
        output_dir=tmp_path / "w-static-experimental" / "output",
        task_profile="llm-cpu-int4",
        hf_revision=SMOLLM2_REVISION,
        recipe_id="smollm2-1.7b-cpu-int4",
        recipe_version="1.0.0",
        recipe_status=RecipeStatus.EXPERIMENTAL.value,
        allow_experimental=True,
    )
    request.workspace_root.mkdir(parents=True)
    request.model_cache_dir.mkdir(parents=True)
    job = BuildJob(job_id="static-experimental", request=request)
    transition(job, JobState.PREFLIGHT, "Preflight passed.")
    process_runner = ContractProcessRunner()
    base_recipe = next(recipe for recipe in DEFAULT_MODEL_RECIPES if recipe.id == "smollm2-1.7b-cpu-int4")
    experimental_recipe = replace(
        base_recipe,
        status=RecipeStatus.EXPERIMENTAL,
        status_reason="Recipe remains experimental.",
    )
    recipe_registry = RecipeRegistry(
        tuple(
            experimental_recipe if recipe.id == experimental_recipe.id else recipe
            for recipe in DEFAULT_MODEL_RECIPES
        )
    )

    ProductionBuildStageRunner(
        process_runner,
        recipe_registry=recipe_registry,
        model_acquisition=PinnedSnapshot(),  # type: ignore[arg-type]
    ).run(job, persist=lambda: None, cancellation_event=Event())

    assert job.state == JobState.FAILED
    assert job.failure is not None
    assert job.failure.classification == FailureClassification.NOT_VERIFIED
    assert "verified" in job.failure.message
    assert process_runner.specs == []


def test_production_package_paths_prefer_resolved_generated_recipe_prefix(tmp_path: Path) -> None:
    generated_candidate = _compile_generated_candidate(
        "owner/unregistered-model",
        "1234567890abcdef1234567890abcdef12345678",
    )
    request = replace(
        _generated_request(
            tmp_path,
            generated_candidate,
            attempt_id="55555555-5555-5555-5555-555555555555",
        ),
        recipe_artifact_cache_prefix="tampered-prefix",
    )
    job = BuildJob(job_id="generated-path-prefix", request=request)
    staging, package = production_package_paths(
        job,
        recipe_registry=RecipeRegistry(()),
        resolved_recipe=generated_candidate.recipe,
    )

    assert "tampered-prefix" not in staging.name
    assert "tampered-prefix" not in package.name
    assert generated_candidate.recipe.artifact_cache_prefix in staging.name
    assert generated_candidate.recipe.artifact_cache_prefix in package.name


def test_package_collision_does_not_delete_existing_immutable_artifact(tmp_path: Path) -> None:
    request = _request(tmp_path)
    request.workspace_root.mkdir(parents=True)
    request.model_cache_dir.mkdir(parents=True)
    job = BuildJob(job_id="collision", request=request)
    transition(job, JobState.PREFLIGHT, "Preflight passed.")
    production = ProductionBuildStageRunner(
        ContractProcessRunner(),
        model_acquisition=PinnedSnapshot(),  # type: ignore[arg-type]
    )
    _, package_dir = production_package_paths(job)
    package_dir.mkdir()
    marker = package_dir / "existing.txt"
    marker.write_text("keep", encoding="utf-8")

    production.run(job, persist=lambda: None, cancellation_event=Event())

    assert job.state == JobState.FAILED
    assert marker.read_text(encoding="utf-8") == "keep"


def test_decoder_output_reconciliation_remaps_unique_quantized_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir(parents=True, exist_ok=False)
    (staging / "model.onnx").write_bytes(b"onnx-placeholder")
    (staging / "genai_config.json").write_text(
        json.dumps(
            {
                "model": {
                    "decoder": {
                        "outputs": {
                            "logits": "logits",
                            "present_key_names": "present.0.key",
                        }
                    }
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        production_runner_module,
        "_load_onnx_graph_output_names",
        lambda _: ("logits_Q4", "present.0.key"),
    )

    result = production_runner_module._reconcile_decoder_outputs_in_staging_package(staging_dir=staging)

    assert result["status"] == "applied"
    assert result["remapped_outputs"] == {"logits": "logits_Q4"}
    updated = json.loads((staging / "genai_config.json").read_text(encoding="utf-8"))
    outputs = updated["model"]["decoder"]["outputs"]
    assert outputs["logits"] == "logits_Q4"
    assert outputs["present_key_names"] == "present.0.key"


def test_decoder_output_reconciliation_fails_closed_on_ambiguous_suffix_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir(parents=True, exist_ok=False)
    (staging / "model.onnx").write_bytes(b"onnx-placeholder")
    original = {
        "model": {
            "decoder": {
                "outputs": {
                    "logits": "logits",
                }
            }
        }
    }
    (staging / "genai_config.json").write_text(json.dumps(original, indent=2), encoding="utf-8")
    monkeypatch.setattr(
        production_runner_module,
        "_load_onnx_graph_output_names",
        lambda _: ("logits_Q4", "logits_Q8"),
    )

    with pytest.raises(RuntimeError, match="unresolved mappings"):
        production_runner_module._reconcile_decoder_outputs_in_staging_package(staging_dir=staging)
    current = json.loads((staging / "genai_config.json").read_text(encoding="utf-8"))
    assert current == original


def test_decoder_output_reconciliation_fails_closed_on_missing_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir(parents=True, exist_ok=False)
    (staging / "model.onnx").write_bytes(b"onnx-placeholder")
    original = {
        "model": {
            "decoder": {
                "outputs": {
                    "logits": "logits",
                }
            }
        }
    }
    (staging / "genai_config.json").write_text(json.dumps(original, indent=2), encoding="utf-8")
    monkeypatch.setattr(
        production_runner_module,
        "_load_onnx_graph_output_names",
        lambda _: ("other_output",),
    )

    with pytest.raises(RuntimeError, match="unresolved mappings"):
        production_runner_module._reconcile_decoder_outputs_in_staging_package(staging_dir=staging)
    current = json.loads((staging / "genai_config.json").read_text(encoding="utf-8"))
    assert current == original


def test_decoder_output_reconciliation_leaves_already_correct_mapping_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir(parents=True, exist_ok=False)
    (staging / "model.onnx").write_bytes(b"onnx-placeholder")
    original = {
        "model": {
            "decoder": {
                "outputs": {
                    "logits": "logits_Q4",
                }
            }
        }
    }
    (staging / "genai_config.json").write_text(json.dumps(original, indent=2), encoding="utf-8")
    monkeypatch.setattr(
        production_runner_module,
        "_load_onnx_graph_output_names",
        lambda _: ("logits_Q4", "present.0.key"),
    )

    result = production_runner_module._reconcile_decoder_outputs_in_staging_package(staging_dir=staging)

    assert result["status"] == "verified"
    current = json.loads((staging / "genai_config.json").read_text(encoding="utf-8"))
    assert current == original


def test_decoder_output_reconciliation_accepts_indexed_decoder_output_templates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir(parents=True, exist_ok=False)
    (staging / "model.onnx").write_bytes(b"onnx-placeholder")
    original = {
        "model": {
            "decoder": {
                "outputs": {
                    "logits": "logits_Q4",
                    "present_key_names": "present.%d.key",
                    "present_value_names": "present.%d.value",
                }
            }
        }
    }
    (staging / "genai_config.json").write_text(json.dumps(original, indent=2), encoding="utf-8")
    monkeypatch.setattr(
        production_runner_module,
        "_load_onnx_graph_output_names",
        lambda _: (
            "logits_Q4",
            "present.0.key",
            "present.1.key",
            "present.0.value",
        ),
    )

    result = production_runner_module._reconcile_decoder_outputs_in_staging_package(staging_dir=staging)

    assert result["status"] == "verified"
    assert result["remapped_outputs"] == {}
    current = json.loads((staging / "genai_config.json").read_text(encoding="utf-8"))
    assert current == original


def test_decoder_output_reconciliation_rejects_unsafe_paths_and_invalid_config_shape(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir(parents=True, exist_ok=False)
    (staging / "model.onnx").write_bytes(b"onnx-placeholder")
    (staging / "genai_config.json").write_text(
        json.dumps({"model": {"decoder": {"outputs": {"logits": "logits"}}}}, indent=2),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="unsafe path components"):
        production_runner_module._reconcile_decoder_outputs_in_staging_package(
            staging_dir=staging,
            model_relative_path="../model.onnx",
        )
    (staging / "genai_config.json").write_text(
        json.dumps({"model": {"decoder": {"outputs": ["not", "a", "mapping"]}}}, indent=2),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="must be an object"):
        production_runner_module._reconcile_decoder_outputs_in_staging_package(staging_dir=staging)


class TinyOutputMismatchRunner(ContractProcessRunner):
    def run(self, spec: CommandSpec, cancel_event=None) -> CommandResult:  # noqa: ANN001
        self.specs.append(spec)
        self.cancel_events.append(cancel_event)
        argv = spec.argv
        stdout = ""
        if argv[:2] == ("mobius", "build"):
            output = Path(argv[-1])
            (output / "model.onnx").write_bytes(b"mobius")
            (output / "genai_config.json").write_text(
                json.dumps({"model": {"decoder": {"outputs": {"logits": "logits"}}}}, indent=2),
                encoding="utf-8",
            )
            (output / "tokenizer.json").write_text("{}", encoding="utf-8")
        elif argv[:2] == ("olive", "optimize"):
            output = Path(argv[argv.index("--output_path") + 1])
            (output / "model.onnx").write_bytes(b"olive")
            (output / "genai_config.json").write_text(
                json.dumps({"model": {"decoder": {"outputs": {"logits": "logits"}}}}, indent=2),
                encoding="utf-8",
            )
            (output / "tokenizer.json").write_text("{}", encoding="utf-8")
        elif "validate-runtime" in argv:
            stdout = json.dumps(
                {
                    "ok": True,
                    "checks": [
                        "onnx_checker=1",
                        "ort_cpu_load=passed",
                        "oga_generation=passed",
                    ],
                }
            )
        elif "foundry-infer" in argv:
            stdout = json.dumps({"ok": True, "output": "OK"})
        return CommandResult(spec=spec, exit_code=0, stdout=stdout, stderr="")


def test_production_runner_reconciles_staging_outputs_before_runtime_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    request.workspace_root.mkdir(parents=True)
    request.model_cache_dir.mkdir(parents=True)
    job = BuildJob(job_id="tiny-reconcile", request=request)
    transition(job, JobState.PREFLIGHT, "Preflight passed.")
    runner = TinyOutputMismatchRunner()
    monkeypatch.setattr(
        production_runner_module,
        "_load_onnx_graph_output_names",
        lambda _: ("logits_Q4",),
    )

    ProductionBuildStageRunner(runner, model_acquisition=PinnedSnapshot()).run(
        job,
        persist=lambda: None,
        cancellation_event=Event(),
    )

    assert job.state == JobState.SUCCEEDED
    assert job.artifacts
    package_config = json.loads((job.artifacts[0].path / "genai_config.json").read_text(encoding="utf-8"))
    assert package_config["model"]["decoder"]["outputs"]["logits"] == "logits_Q4"
    assert any(
        "decoder output reconciliation applied" in event.message.lower()
        for event in job.events
    )
