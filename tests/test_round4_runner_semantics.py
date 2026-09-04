from __future__ import annotations

import importlib.util
import json
import sys

from pathlib import Path
from typing import Any

import pytest


def _load_round_runner_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "evaluation"
        / "recipe-agent-v1"
        / "round-4"
        / "run_round4.py"
    )
    spec = importlib.util.spec_from_file_location("round4_runner", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_required_environment_blockers_detect_missing_project_package_and_runtime_module() -> None:
    module = _load_round_runner_module()
    blockers = module._required_environment_blockers(
        {
            "probes": [
                {
                    "kind": "python-package",
                    "name": "fl-model-onboarding",
                    "required": True,
                    "available": False,
                },
                {
                    "kind": "python-module",
                    "name": "fl_model_onboarding.runtime_worker",
                    "required": True,
                    "available": False,
                },
                {
                    "kind": "command",
                    "name": "mobius",
                    "required": True,
                    "available": True,
                },
            ]
        }
    )
    assert "python-package:fl-model-onboarding" in blockers
    assert "python-module:fl_model_onboarding.runtime_worker" in blockers


def test_required_environment_blockers_clear_when_project_package_and_module_are_available() -> None:
    module = _load_round_runner_module()
    blockers = module._required_environment_blockers(
        {
            "probes": [
                {
                    "kind": "python-package",
                    "name": "fl-model-onboarding",
                    "required": True,
                    "available": True,
                },
                {
                    "kind": "python-module",
                    "name": "fl_model_onboarding.runtime_worker",
                    "required": True,
                    "available": True,
                },
            ]
        }
    )
    assert blockers == []


def test_safe_delete_tree_within_root_rejects_outside_path(tmp_path: Path) -> None:
    module = _load_round_runner_module()
    runtime_root = tmp_path / "runtime"
    outside = tmp_path / "outside"
    runtime_root.mkdir()
    outside.mkdir()
    (outside / "x.txt").write_text("x", encoding="utf-8")

    with pytest.raises(RuntimeError, match="outside runtime root"):
        module._safe_delete_tree_within_root(root=runtime_root, target=outside)


def test_cleanup_failed_workspaces_deletes_only_failed_jobs_within_root(tmp_path: Path) -> None:
    module = _load_round_runner_module()
    runtime_root = tmp_path / "runtime"
    state_root = runtime_root / "state"
    workspace_root = runtime_root / "workspace"
    cache_root = runtime_root / "cache"
    for path in (runtime_root, state_root, workspace_root, cache_root):
        path.mkdir(parents=True, exist_ok=True)
    failed_workspace = workspace_root / "job-failed"
    succeeded_workspace = workspace_root / "job-succeeded"
    failed_workspace.mkdir()
    succeeded_workspace.mkdir()
    (failed_workspace / "artifact.bin").write_bytes(b"a" * 256)
    (succeeded_workspace / "artifact.bin").write_bytes(b"b" * 256)

    paths = module.RunPaths(
        repo_root=tmp_path,
        output_root=tmp_path,
        manifest_path=tmp_path / "models.json",
        scratch_root=tmp_path,
        runtime_root=runtime_root,
        state_root=state_root,
        workspace_base=workspace_root,
        model_cache_dir=cache_root,
        service_db_path=state_root / "service.sqlite3",
        recipe_attempt_db_path=state_root / "recipe-attempts.sqlite3",
        run_id="r4-test",
    )
    model_results = [
        {
            "model_id": "failed-model",
            "job": {"job_id": "job-failed"},
            "failure_summary": {"attempt_state": "failed"},
        },
        {
            "model_id": "succeeded-model",
            "job": {"job_id": "job-succeeded"},
            "failure_summary": {"attempt_state": "succeeded"},
        },
    ]

    cleanup = module._cleanup_failed_workspaces_for_current_run(
        paths=paths,
        model_results=model_results,
        retain_failed_workspaces=False,
    )

    assert cleanup["cleaned_count"] == 1
    assert cleanup["freed_bytes"] >= 256
    assert not failed_workspace.exists()
    assert succeeded_workspace.exists()
    first_retention = model_results[0].get("workspace_retention")
    assert isinstance(first_retention, dict)
    assert first_retention.get("action") == "failed-workspace-deleted"


def test_cleanup_failed_workspaces_from_summary_preserves_succeeded_job(tmp_path: Path) -> None:
    module = _load_round_runner_module()
    runtime_root = tmp_path / "obsolete"
    workspace_root = runtime_root / "workspace"
    failed_workspace = workspace_root / "job-failed"
    succeeded_workspace = workspace_root / "job-succeeded"
    failed_workspace.mkdir(parents=True)
    succeeded_workspace.mkdir(parents=True)
    (failed_workspace / "x.log").write_text("failed", encoding="utf-8")
    (succeeded_workspace / "y.log").write_text("succeeded", encoding="utf-8")

    summary_path = tmp_path / "round-summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "run_id": "r-obsolete",
                "results": [
                    {"model_id": "m1", "attempt_state": "failed", "job_id": "job-failed"},
                    {"model_id": "m2", "attempt_state": "succeeded", "job_id": "job-succeeded"},
                ],
            }
        ),
        encoding="utf-8",
    )

    cleanup = module._cleanup_failed_workspaces_from_summary(
        summary_path=summary_path,
        runtime_root=runtime_root,
    )

    assert cleanup["status"] == "completed"
    assert cleanup["deleted_count"] == 1
    assert cleanup["freed_bytes"] > 0
    assert not failed_workspace.exists()
    assert succeeded_workspace.exists()


def test_sanitize_text_redacts_tokens_and_truncates() -> None:
    module = _load_round_runner_module()
    raw = (
        "token hf_abcdefghijklmnopqrstuvwxyz123456 "
        "Bearer SUPERSECRET012345 api_key=ABCDEF012345 "
        "path C:\\temp\\secret.txt \u0007"
    )
    sanitized = module._sanitize_text(raw, (), max_chars=80)
    assert "hf_abcdefghijklmnopqrstuvwxyz123456" not in sanitized
    assert "SUPERSECRET012345" not in sanitized
    assert "ABCDEF012345" not in sanitized
    assert "C:\\temp\\secret.txt" not in sanitized
    assert "\u0007" not in sanitized
    assert len(sanitized) <= 80


def test_load_quality_validation_evidence_from_metrics_ref(tmp_path: Path) -> None:
    module = _load_round_runner_module()
    runtime_root = tmp_path / "runtime"
    state_root = runtime_root / "state"
    workspace_root = runtime_root / "workspace"
    cache_root = runtime_root / "cache"
    for path in (runtime_root, state_root, workspace_root, cache_root):
        path.mkdir(parents=True, exist_ok=True)

    job_id = "11111111-1111-1111-1111-111111111111"
    evidence_path = workspace_root / job_id / "quality-validation-evidence.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "profile_id": "textgen-basic-quality-v1",
                "profile_version": 1,
                "baseline_comparison": {"passed": False, "regressions": ["regressed"]},
                "per_prompt": [
                    {
                        "prompt_id": "prompt-a",
                        "category": "correctness",
                        "prompt": "p" * 400,
                        "baseline": {
                            "output_text": "b" * 900,
                            "checks": {"passed": True, "failures": []},
                        },
                        "optimized": {
                            "output_text": "o" * 900,
                            "checks": {"passed": False, "failures": ["missing_keyword"]},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    paths = module.RunPaths(
        repo_root=tmp_path,
        output_root=tmp_path,
        manifest_path=tmp_path / "models.json",
        scratch_root=tmp_path,
        runtime_root=runtime_root,
        state_root=state_root,
        workspace_base=workspace_root,
        model_cache_dir=cache_root,
        service_db_path=state_root / "service.sqlite3",
        recipe_attempt_db_path=state_root / "recipe-attempts.sqlite3",
        run_id="r4-test",
    )
    attempt_payload = {
        "gates": [
            {
                "gate": "quality_validation",
                "metrics_ref": f"quality-metrics://{job_id}/quality-validation-evidence.json",
            }
        ]
    }
    loaded = module._load_quality_validation_evidence(
        paths=paths,
        attempt_payload=attempt_payload,
        job_id=job_id,
    )

    assert isinstance(loaded, dict)
    assert loaded["status"] == "loaded"
    assert loaded["metrics_ref"] == f"quality-metrics://{job_id}/quality-validation-evidence.json"
    prompts = loaded["per_prompt"]
    assert isinstance(prompts, list) and len(prompts) == 1
    prompt = prompts[0]
    baseline = prompt["baseline"]
    optimized = prompt["optimized"]
    assert isinstance(baseline, dict) and isinstance(optimized, dict)
    assert isinstance(prompt["prompt"], str) and len(prompt["prompt"]) <= module.QUALITY_EVIDENCE_MAX_PROMPT_CHARS
    assert isinstance(baseline["output_text"], str)
    assert len(baseline["output_text"]) <= module.QUALITY_EVIDENCE_MAX_OUTPUT_CHARS
    assert isinstance(optimized["output_text"], str)
    assert len(optimized["output_text"]) <= module.QUALITY_EVIDENCE_MAX_OUTPUT_CHARS
    excerpt = loaded["failure_excerpt"]
    assert isinstance(excerpt, list) and len(excerpt) == 1
    assert excerpt[0]["prompt_id"] == "prompt-a"
    assert excerpt[0]["optimized_passed"] is False


def test_load_quality_validation_evidence_detects_job_mismatch(tmp_path: Path) -> None:
    module = _load_round_runner_module()
    runtime_root = tmp_path / "runtime"
    state_root = runtime_root / "state"
    workspace_root = runtime_root / "workspace"
    cache_root = runtime_root / "cache"
    for path in (runtime_root, state_root, workspace_root, cache_root):
        path.mkdir(parents=True, exist_ok=True)
    paths = module.RunPaths(
        repo_root=tmp_path,
        output_root=tmp_path,
        manifest_path=tmp_path / "models.json",
        scratch_root=tmp_path,
        runtime_root=runtime_root,
        state_root=state_root,
        workspace_base=workspace_root,
        model_cache_dir=cache_root,
        service_db_path=state_root / "service.sqlite3",
        recipe_attempt_db_path=state_root / "recipe-attempts.sqlite3",
        run_id="r4-test",
    )
    loaded = module._load_quality_validation_evidence(
        paths=paths,
        attempt_payload={
            "gates": [
                {
                    "gate": "quality_validation",
                    "metrics_ref": "quality-metrics://aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa/quality-validation-evidence.json",
                }
            ]
        },
        job_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
    )
    assert isinstance(loaded, dict)
    assert loaded["status"] == "metrics-ref-job-mismatch"


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self) -> dict[str, Any]:
        return self._payload


class _SequencedClient:
    def __init__(self, responses: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]]) -> None:
        self._responses = responses
        self._offsets: dict[tuple[str, str], int] = {}
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def request(self, method: str, path: str, **kwargs: Any) -> _FakeResponse:
        key = (method, path)
        self.calls.append((method, path, kwargs))
        if key not in self._responses:
            raise AssertionError(f"Unexpected request: {method} {path}")
        rows = self._responses[key]
        if not rows:
            raise AssertionError(f"No responses configured for {method} {path}")
        index = self._offsets.get(key, 0)
        if index >= len(rows):
            status, payload = rows[-1]
        else:
            status, payload = rows[index]
            self._offsets[key] = index + 1
        return _FakeResponse(status, payload)


def _attempt_payload(
    *,
    workflow_outcome: str,
    lineage_selection_state: str,
    fallback_attempt_state: str,
    fallback_selection_status: str,
) -> dict[str, Any]:
    fallback_candidate = {
        "candidate_attempt_id": "cand-fallback",
        "attempt_id": "attempt-fallback",
        "candidate_index": 1,
        "candidate_id": "int4-block-size-64",
        "role": "quality_retry",
        "attempt_state": fallback_attempt_state,
        "selection_status": fallback_selection_status,
        "invocation_counters": {
            "mobius_build_invocation_count": None,
            "olive_optimize_invocation_count": None,
            "total_invocation_count": None,
        },
        "validated_scope": {},
    }
    selected_candidate = fallback_candidate if fallback_selection_status == "selected" else None
    return {
        "attempt_id": "attempt-parent",
        "state": "failed",
        "workflow_outcome": workflow_outcome,
        "build_job_id": "job-parent",
        "candidate_selection": {
            "lineage_selection_state": lineage_selection_state,
            "selected_candidate": selected_candidate,
            "candidates": [
                {
                    "candidate_attempt_id": "cand-default",
                    "attempt_id": "attempt-parent",
                    "candidate_index": 0,
                    "candidate_id": "default-int4",
                    "role": "default",
                    "attempt_state": "failed",
                    "selection_status": "not_selected",
                    "invocation_counters": {
                        "mobius_build_invocation_count": 1,
                        "olive_optimize_invocation_count": 1,
                        "total_invocation_count": 2,
                    },
                    "validated_scope": {},
                },
                fallback_candidate,
            ],
        },
    }


@pytest.mark.parametrize(
    ("terminal_workflow_outcome", "terminal_lineage_state", "terminal_fallback_state", "terminal_selection"),
    [
        ("selected", "selected", "succeeded", "selected"),
        ("exhausted", "exhausted", "failed", "not_selected"),
    ],
)
def test_poll_terminal_attempt_workflow_waits_until_lineage_terminal(
    monkeypatch: pytest.MonkeyPatch,
    terminal_workflow_outcome: str,
    terminal_lineage_state: str,
    terminal_fallback_state: str,
    terminal_selection: str,
) -> None:
    module = _load_round_runner_module()
    parent_path = "/api/recipes/generated/attempts/attempt-parent"
    pending_payload = _attempt_payload(
        workflow_outcome="pending",
        lineage_selection_state="pending",
        fallback_attempt_state="running",
        fallback_selection_status="not_selected",
    )
    terminal_payload = _attempt_payload(
        workflow_outcome=terminal_workflow_outcome,
        lineage_selection_state=terminal_lineage_state,
        fallback_attempt_state=terminal_fallback_state,
        fallback_selection_status=terminal_selection,
    )
    client = _SequencedClient(
        {
            ("GET", parent_path): [
                (200, pending_payload),
                (200, pending_payload),
                (200, terminal_payload),
            ]
        }
    )
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    payload, timed_out, timeout_detail = module._poll_terminal_attempt_workflow(
        client,
        attempt_id="attempt-parent",
        timeout_seconds=30,
        poll_seconds=1,
    )
    assert timed_out is False
    assert timeout_detail is None
    assert payload["workflow_outcome"] == terminal_workflow_outcome
    assert payload["candidate_selection"]["lineage_selection_state"] == terminal_lineage_state
    parent_get_calls = [call for call in client.calls if call[0] == "GET" and call[1] == parent_path]
    assert len(parent_get_calls) == 3
    assert not [call for call in client.calls if call[0] == "POST" and "/cancel" in call[1]]


def test_poll_terminal_attempt_workflow_timeout_cancels_inflight_child_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_round_runner_module()
    parent_path = "/api/recipes/generated/attempts/attempt-parent"
    child_path = "/api/recipes/generated/attempts/attempt-fallback"
    cancel_path = "/api/builds/job-fallback/cancel"
    pending_payload = _attempt_payload(
        workflow_outcome="pending",
        lineage_selection_state="pending",
        fallback_attempt_state="running",
        fallback_selection_status="not_selected",
    )
    exhausted_payload = _attempt_payload(
        workflow_outcome="exhausted",
        lineage_selection_state="exhausted",
        fallback_attempt_state="cancelled",
        fallback_selection_status="not_selected",
    )
    client = _SequencedClient(
        {
            ("GET", parent_path): [
                (200, pending_payload),
                (200, exhausted_payload),
            ],
            ("GET", child_path): [
                (
                    200,
                    {
                        "attempt_id": "attempt-fallback",
                        "state": "running",
                        "build_job_id": "job-fallback",
                    },
                )
            ],
            ("POST", cancel_path): [(200, {"job_id": "job-fallback", "state": "cancelled"})],
        }
    )

    clock = {"value": 0.0}

    def _fake_time() -> float:
        clock["value"] += 1.0
        return clock["value"]

    monkeypatch.setattr(module.time, "time", _fake_time)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    payload, timed_out, timeout_detail = module._poll_terminal_attempt_workflow(
        client,
        attempt_id="attempt-parent",
        timeout_seconds=2,
        poll_seconds=1,
    )
    assert timed_out is True
    assert payload["workflow_outcome"] == "exhausted"
    assert isinstance(timeout_detail, dict)
    cancelled_children = timeout_detail.get("cancelled_children")
    assert isinstance(cancelled_children, list) and len(cancelled_children) == 1
    row = cancelled_children[0]
    assert row["attempt_id"] == "attempt-fallback"
    assert row["build_job_id"] == "job-fallback"
    assert row["cancel"]["status_code"] == 200
