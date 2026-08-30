from __future__ import annotations

import json

from pathlib import Path

import yaml


def test_openapi_contains_required_paths() -> None:
    path = Path("contracts") / "openapi.yaml"
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    paths = spec["paths"]
    assert "/api/builds" in paths
    assert "/api/builds/{job_id}/events" in paths
    assert "/api/artifacts/{artifact_id}/infer/text" in paths
    assert "/api/artifacts/{artifact_id}/infer/asr" in paths
    parameters = paths["/api/builds"]["post"]["parameters"]
    idempotency = next(p for p in parameters if p["name"] == "Idempotency-Key")
    assert idempotency["required"] is True


def test_state_machine_contract_contains_cancellable_flags() -> None:
    path = Path("contracts") / "job-state-machine.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    states = {row["name"]: row for row in contract["states"]}
    assert states["mobius_building"]["cancellable"] is True
    assert states["succeeded"]["cancellable"] is False
