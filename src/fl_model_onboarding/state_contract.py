from __future__ import annotations

import json

from pathlib import Path

from .contracts import JobState


def contract_file_path() -> Path:
    return Path(__file__).resolve().parents[2] / "contracts" / "job-state-machine.json"


def load_state_contract(path: Path | None = None) -> dict[str, object]:
    contract_path = path or contract_file_path()
    return json.loads(contract_path.read_text(encoding="utf-8"))


def transition_map_from_contract(contract: dict[str, object]) -> dict[JobState, set[JobState]]:
    transitions_raw = contract.get("transitions")
    if not isinstance(transitions_raw, list):
        raise ValueError("job-state-machine.json is missing 'transitions' list.")

    mapping: dict[JobState, set[JobState]] = {state: set() for state in JobState}
    for row in transitions_raw:
        if not isinstance(row, dict):
            continue
        src = JobState(str(row["from"]))
        dst = JobState(str(row["to"]))
        mapping[src].add(dst)
    return mapping


def cancellable_states_from_contract(contract: dict[str, object]) -> frozenset[JobState]:
    states_raw = contract.get("states")
    if not isinstance(states_raw, list):
        raise ValueError("job-state-machine.json is missing 'states' list.")
    cancellable: set[JobState] = set()
    for row in states_raw:
        if not isinstance(row, dict):
            continue
        if bool(row.get("cancellable")):
            cancellable.add(JobState(str(row["name"])))
    return frozenset(cancellable)


def execution_order_from_contract(contract: dict[str, object]) -> tuple[JobState, ...]:
    states_raw = contract.get("states")
    if not isinstance(states_raw, list):
        raise ValueError("job-state-machine.json is missing 'states' list.")
    ordered: list[JobState] = []
    for row in states_raw:
        if not isinstance(row, dict):
            continue
        state = JobState(str(row["name"]))
        if state in {JobState.QUEUED, JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}:
            continue
        ordered.append(state)
    return tuple(ordered)
