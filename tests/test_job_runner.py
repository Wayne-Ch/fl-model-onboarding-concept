from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from fl_model_onboarding.candidates import PHASE0_CANDIDATES
from fl_model_onboarding.contracts import (
    BuildRequest,
    FailureClassification,
    FailureInfo,
    JobState,
    PreflightResult,
    ToolAvailability,
)
from fl_model_onboarding.job_runner import LocalJobRunner


class FakeInspector:
    def __init__(self, preflight: PreflightResult) -> None:
        self._preflight = preflight

    def inspect(self, request: BuildRequest) -> PreflightResult:
        return self._preflight


def _request(tmp_path: Path) -> BuildRequest:
    return BuildRequest(
        candidate=PHASE0_CANDIDATES["smollm2-1.7b-instruct"],
        workspace_root=tmp_path,
        model_cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
        dry_run=True,
    )


def _preflight_ok(request: BuildRequest) -> PreflightResult:
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
            ToolAvailability("onnxruntime", "python-package", True, "1.26.0"),
            ToolAvailability("onnxruntime-genai", "python-package", True, "0.14.0"),
            ToolAvailability("foundry-local-sdk", "python-package", True, "1.2.0"),
            ToolAvailability("huggingface_hub", "python-package", True, "1.22.0"),
        ),
        foundry_catalog_matches=(),
        huggingface_revision="abc",
        huggingface_sha="abc",
        huggingface_private=False,
        huggingface_gated=False,
        blockers=(),
        warnings=(),
    )


def test_dry_run_reaches_succeeded(tmp_path: Path) -> None:
    request = _request(tmp_path)
    runner = LocalJobRunner(FakeInspector(_preflight_ok(request)))  # type: ignore[arg-type]
    plan = runner.run_dry(request)
    assert plan.job.state == JobState.SUCCEEDED
    stage_values = [event.state for event in plan.job.events]
    assert JobState.PREFLIGHT in stage_values
    assert JobState.INFERENCING in stage_values


def test_dry_run_stops_on_preflight_blocker(tmp_path: Path) -> None:
    request = _request(tmp_path)
    blocking = _preflight_ok(request)
    blocking = replace(
        blocking,
        blockers=(
            FailureInfo(
                stage=JobState.MOBIUS_BUILDING,
                classification=FailureClassification.MISSING_DEPENDENCY,
                message="mobius missing",
            ),
        ),
    )
    runner = LocalJobRunner(FakeInspector(blocking))  # type: ignore[arg-type]
    plan = runner.run_dry(request)
    assert plan.job.state == JobState.FAILED
    assert plan.job.failure is not None
    assert plan.job.failure.classification == FailureClassification.MISSING_DEPENDENCY
