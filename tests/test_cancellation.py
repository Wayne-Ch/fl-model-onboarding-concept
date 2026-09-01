from __future__ import annotations

from pathlib import Path

from fl_model_onboarding.candidates import PHASE0_CANDIDATES
from fl_model_onboarding.cancellation import ProcessOwnershipRegistry
from fl_model_onboarding.contracts import BuildJob, BuildRequest, JobState
from fl_model_onboarding.state_machine import transition


def test_cancel_quarantines_partial_output(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    output = workspace / "output"
    output.mkdir(parents=True)
    (output / "partial.bin").write_text("x", encoding="utf-8")
    request = BuildRequest(
        candidate=PHASE0_CANDIDATES["smollm2-1.7b-instruct"],
        workspace_root=workspace,
        model_cache_dir=tmp_path / "cache",
        output_dir=output,
        dry_run=True,
    )
    job = BuildJob(job_id="job-1", request=request)
    transition(job, JobState.PREFLIGHT, "preflight")
    transition(job, JobState.DOWNLOADING, "downloading")

    registry = ProcessOwnershipRegistry()
    quarantine = registry.cancel(job, reason="cancel")
    assert quarantine is not None
    assert quarantine.exists()
    assert (quarantine / "partial.bin").exists()
    assert job.state == JobState.CANCELLED
