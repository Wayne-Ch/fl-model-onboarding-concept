from __future__ import annotations

from .contracts import BuildJob, FailureClassification, FailureInfo, JobState


class StateTransitionError(ValueError):
    pass


ALLOWED_TRANSITIONS: dict[JobState, set[JobState]] = {
    JobState.QUEUED: {JobState.PREFLIGHT, JobState.CANCELLED},
    JobState.PREFLIGHT: {JobState.DOWNLOADING, JobState.FAILED, JobState.CANCELLED},
    JobState.DOWNLOADING: {JobState.MOBIUS_BUILDING, JobState.FAILED, JobState.CANCELLED},
    JobState.MOBIUS_BUILDING: {JobState.MOBIUS_VALIDATING, JobState.FAILED, JobState.CANCELLED},
    JobState.MOBIUS_VALIDATING: {JobState.OLIVE_OPTIMIZING, JobState.FAILED, JobState.CANCELLED},
    JobState.OLIVE_OPTIMIZING: {JobState.PACKAGING, JobState.FAILED, JobState.CANCELLED},
    JobState.PACKAGING: {JobState.RUNTIME_VALIDATING, JobState.FAILED, JobState.CANCELLED},
    JobState.RUNTIME_VALIDATING: {JobState.FL_LOADING, JobState.FAILED, JobState.CANCELLED},
    JobState.FL_LOADING: {JobState.INFERENCING, JobState.FAILED, JobState.CANCELLED},
    JobState.INFERENCING: {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED},
    JobState.SUCCEEDED: set(),
    JobState.FAILED: set(),
    JobState.CANCELLED: set(),
}

EXECUTION_ORDER: tuple[JobState, ...] = (
    JobState.PREFLIGHT,
    JobState.DOWNLOADING,
    JobState.MOBIUS_BUILDING,
    JobState.MOBIUS_VALIDATING,
    JobState.OLIVE_OPTIMIZING,
    JobState.PACKAGING,
    JobState.RUNTIME_VALIDATING,
    JobState.FL_LOADING,
    JobState.INFERENCING,
)


def ensure_transition(current: JobState, target: JobState) -> None:
    if target not in ALLOWED_TRANSITIONS[current]:
        raise StateTransitionError(f"Invalid state transition: {current.value} -> {target.value}")


def transition(job: BuildJob, target: JobState, message: str) -> None:
    ensure_transition(job.state, target)
    job.state = target
    job.add_event(message)


def fail_job(job: BuildJob, failure: FailureInfo) -> None:
    ensure_transition(job.state, JobState.FAILED)
    job.failure = failure
    job.state = JobState.FAILED
    job.add_event(f"Failed ({failure.classification.value}): {failure.message}")


def cancel_job(job: BuildJob, reason: str) -> None:
    ensure_transition(job.state, JobState.CANCELLED)
    job.failure = FailureInfo(
        stage=job.state,
        classification=FailureClassification.CANCELLED,
        message=reason,
    )
    job.state = JobState.CANCELLED
    job.add_event(reason)
