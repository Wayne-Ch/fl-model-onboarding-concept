from __future__ import annotations

from .contracts import BuildJob, FailureClassification, FailureInfo, JobState
from .state_contract import (
    cancellable_states_from_contract,
    execution_order_from_contract,
    load_state_contract,
    transition_map_from_contract,
)


class StateTransitionError(ValueError):
    pass


_STATE_CONTRACT = load_state_contract()
ALLOWED_TRANSITIONS: dict[JobState, set[JobState]] = transition_map_from_contract(_STATE_CONTRACT)
EXECUTION_ORDER: tuple[JobState, ...] = execution_order_from_contract(_STATE_CONTRACT)
CANCELLABLE_STATES = cancellable_states_from_contract(_STATE_CONTRACT)


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
    if job.state not in CANCELLABLE_STATES:
        raise StateTransitionError(f"State '{job.state.value}' is not cancellable.")
    ensure_transition(job.state, JobState.CANCELLED)
    job.failure = FailureInfo(
        stage=job.state,
        classification=FailureClassification.CANCELLED,
        message=reason,
    )
    job.state = JobState.CANCELLED
    job.add_event(reason)
