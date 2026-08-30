from __future__ import annotations

from fl_model_onboarding.contracts import JobState
from fl_model_onboarding.state_contract import load_state_contract
from fl_model_onboarding.state_machine import ALLOWED_TRANSITIONS, CANCELLABLE_STATES, EXECUTION_ORDER


def test_state_contract_file_covers_all_job_states() -> None:
    contract = load_state_contract()
    rows = contract["states"]
    names = {row["name"] for row in rows}
    assert names == {state.value for state in JobState}


def test_execution_order_matches_contract_sequence() -> None:
    assert EXECUTION_ORDER[0] == JobState.PREFLIGHT
    assert EXECUTION_ORDER[-1] == JobState.INFERENCING


def test_cancellable_states_include_active_pipeline() -> None:
    assert JobState.MOBIUS_BUILDING in CANCELLABLE_STATES
    assert JobState.SUCCEEDED not in CANCELLABLE_STATES
    assert JobState.CANCELLED not in CANCELLABLE_STATES


def test_transitions_loaded_from_contract() -> None:
    assert JobState.PREFLIGHT in ALLOWED_TRANSITIONS[JobState.QUEUED]
    assert JobState.SUCCEEDED in ALLOWED_TRANSITIONS[JobState.INFERENCING]
