from __future__ import annotations

from pathlib import Path

import pytest

from fl_model_onboarding.candidates import PHASE0_CANDIDATES
from fl_model_onboarding.contracts import BuildJob, BuildRequest, FailureClassification, FailureInfo, JobState
from fl_model_onboarding.state_machine import StateTransitionError, cancel_job, fail_job, transition


def _request(tmp_path: Path) -> BuildRequest:
    return BuildRequest(
        candidate=PHASE0_CANDIDATES["smollm2-1.7b-instruct"],
        workspace_root=tmp_path,
        model_cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
        dry_run=True,
    )


def test_valid_state_sequence(tmp_path: Path) -> None:
    job = BuildJob(job_id="job-1", request=_request(tmp_path))
    transition(job, JobState.PREFLIGHT, "preflight")
    transition(job, JobState.DOWNLOADING, "downloading")
    transition(job, JobState.MOBIUS_BUILDING, "mobius")
    transition(job, JobState.MOBIUS_VALIDATING, "validate mobius")
    transition(job, JobState.OLIVE_OPTIMIZING, "olive")
    transition(job, JobState.PACKAGING, "packaging")
    transition(job, JobState.RUNTIME_VALIDATING, "runtime")
    transition(job, JobState.FL_LOADING, "load")
    transition(job, JobState.INFERENCING, "infer")
    transition(job, JobState.SUCCEEDED, "done")
    assert job.state == JobState.SUCCEEDED
    assert len(job.events) == 10


def test_invalid_transition_raises(tmp_path: Path) -> None:
    job = BuildJob(job_id="job-1", request=_request(tmp_path))
    with pytest.raises(StateTransitionError):
        transition(job, JobState.MOBIUS_BUILDING, "skip preflight")


def test_fail_job_records_failure(tmp_path: Path) -> None:
    job = BuildJob(job_id="job-1", request=_request(tmp_path))
    transition(job, JobState.PREFLIGHT, "preflight")
    failure = FailureInfo(
        stage=JobState.PREFLIGHT,
        classification=FailureClassification.MISSING_DEPENDENCY,
        message="mobius missing",
    )
    fail_job(job, failure)
    assert job.state == JobState.FAILED
    assert job.failure == failure


def test_cancel_job_sets_terminal_state(tmp_path: Path) -> None:
    job = BuildJob(job_id="job-1", request=_request(tmp_path))
    transition(job, JobState.PREFLIGHT, "preflight")
    cancel_job(job, "cancelled by user")
    assert job.state == JobState.CANCELLED
    assert job.failure is not None
    assert job.failure.classification == FailureClassification.CANCELLED
