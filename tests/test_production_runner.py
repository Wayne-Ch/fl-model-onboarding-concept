from __future__ import annotations

import json
import time

from dataclasses import replace
from pathlib import Path
from threading import Event

from fl_model_onboarding.adapters.interfaces import CommandResult, CommandSpec
from fl_model_onboarding.candidates import PHASE0_CANDIDATES
from fl_model_onboarding.contracts import (
    BuildJob,
    BuildRequest,
    JobState,
    PreflightResult,
    ToolAvailability,
    ValidationStatus,
)
from fl_model_onboarding.local_service import BuildSubmission, LocalOnboardingService
from fl_model_onboarding.production_runner import (
    ProductionBuildStageRunner,
    SMOLLM2_REVISION,
    production_package_paths,
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
    assert mobius.argv[:3] == ("mobius", "build", "--model")
    assert Path(mobius.argv[3]).name == "snapshot"
    assert Path(mobius.argv[-1]).name == "mobius"
    olive = runner.specs[1]
    assert olive.argv[:2] == ("olive", "optimize")
    assert "text-generation-with-past" in olive.argv
    assert all(spec.timeout_seconds <= 7200 for spec in runner.specs)
    assert runner.cancel_events[-1] is not None


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
