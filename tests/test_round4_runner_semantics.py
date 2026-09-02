from __future__ import annotations

import importlib.util
import json
import sys

from pathlib import Path

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

