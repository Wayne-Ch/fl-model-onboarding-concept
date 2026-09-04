from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from fl_model_onboarding.local_api import create_app
from fl_model_onboarding.local_service import LocalOnboardingService

EXPECTED_MODELS: tuple[str, ...] = (
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "Qwen/Qwen2-1.5B-Instruct",
    "Qwen/Qwen2-0.5B-Instruct",
    "ibm-granite/granite-3.2-2b-instruct",
)
EXPECTED_SMOL_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
DEFAULT_SCRATCH_ROOT = Path(r"C:\fmo-r7")
DEFAULT_VENV_PYTHON = Path(r"C:\fl-recipe-v1-venv\Scripts\python.exe")
REUSE_JOB_REF_RE = re.compile(r"^job://(?P<job_id>[0-9a-fA-F-]+)/artifact/[0-9a-fA-F]+$")


@dataclass(frozen=True)
class RuntimeContext:
    script_root: Path
    repo_root: Path
    output_root: Path
    manifest_path: Path
    round_name: str
    run_id: str
    scratch_root: Path
    runtime_root: Path
    state_root: Path
    workspace_root: Path
    cache_root: Path
    service_db_path: Path
    recipe_attempt_db_path: Path


def _load_round4_runner_module():
    round4_script = Path(__file__).resolve().parents[1] / "round-4" / "run_round4.py"
    spec = importlib.util.spec_from_file_location("round4_runner", round4_script)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load round-4 runner module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _has_arg(argv: list[str], arg_name: str) -> bool:
    return any(part == arg_name or part.startswith(f"{arg_name}=") for part in argv)


def _arg_value(argv: list[str], arg_name: str) -> str | None:
    for index, part in enumerate(argv):
        if part == arg_name and index + 1 < len(argv):
            return argv[index + 1]
        if part.startswith(f"{arg_name}="):
            return part.split("=", 1)[1]
    return None


def _ensure_arg(argv: list[str], arg_name: str, value: str) -> None:
    if _has_arg(argv, arg_name):
        return
    argv.extend([arg_name, value])


def _discover_round6_runtime_root(script_root: Path) -> Path | None:
    summary_path = script_root.parent / "round-6" / "round-6-summary.json"
    if not summary_path.is_file():
        return None
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    run_id = str(payload.get("run_id") or "").strip()
    if not run_id:
        return None
    candidate = Path(r"C:\fmo-r6") / run_id
    return candidate if candidate.exists() else None


def _resolve_default_python() -> Path:
    return DEFAULT_VENV_PYTHON if DEFAULT_VENV_PYTHON.is_file() else Path(sys.executable).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object at {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _request(client: TestClient, method: str, path: str, **kwargs: Any) -> tuple[int, Any]:
    response = client.request(method, path, **kwargs)
    status_code = int(response.status_code)
    try:
        return status_code, response.json()
    except Exception:
        return status_code, {"raw_text": response.text}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        return value
    return None


def _candidate_role(index: int | None) -> str | None:
    if index is None:
        return None
    return "default" if index == 0 else "quality_retry"


def _find_selected_candidate_entry(candidate_selection: dict[str, Any]) -> dict[str, Any] | None:
    selected = _as_dict(candidate_selection.get("selected_candidate"))
    candidate_attempt_id = str(selected.get("candidate_attempt_id") or "").strip()
    if not candidate_attempt_id:
        return None
    for row in _as_list(candidate_selection.get("candidates")):
        if not isinstance(row, dict):
            continue
        if str(row.get("candidate_attempt_id") or "").strip() == candidate_attempt_id:
            return row
    return None


def _parse_winner_job_id(selected_candidate_entry: dict[str, Any] | None) -> str | None:
    if not isinstance(selected_candidate_entry, dict):
        return None
    artifact_ref = str(selected_candidate_entry.get("artifact_ref") or "").strip()
    if not artifact_ref:
        return None
    match = REUSE_JOB_REF_RE.fullmatch(artifact_ref)
    if match is None:
        return None
    return match.group("job_id")


def _build_runtime_context(
    *,
    script_root: Path,
    round_name: str,
    scratch_root: Path,
    run_id: str,
) -> RuntimeContext:
    repo_root = script_root.parents[2]
    output_root = script_root
    manifest_path = script_root.parent / "models.json"
    runtime_root = (scratch_root / run_id).resolve()
    state_root = (runtime_root / "state").resolve()
    workspace_root = (runtime_root / "workspace").resolve()
    cache_root = (runtime_root / "cache").resolve()
    service_db_path = (state_root / "service.sqlite3").resolve()
    recipe_attempt_db_path = (state_root / "recipe-attempts.sqlite3").resolve()
    return RuntimeContext(
        script_root=script_root,
        repo_root=repo_root,
        output_root=output_root,
        manifest_path=manifest_path,
        round_name=round_name,
        run_id=run_id,
        scratch_root=scratch_root.resolve(),
        runtime_root=runtime_root,
        state_root=state_root,
        workspace_root=workspace_root,
        cache_root=cache_root,
        service_db_path=service_db_path,
        recipe_attempt_db_path=recipe_attempt_db_path,
    )


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        try:
            total += item.stat().st_size
        except OSError:
            continue
    return total


def _safe_delete_tree_within_root(*, root: Path, target: Path) -> int:
    resolved_root = root.resolve()
    resolved_target = target.resolve()
    if resolved_target == resolved_root:
        raise RuntimeError(f"Refusing to delete runtime root: {resolved_target}")
    try:
        resolved_target.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError(
            f"Refusing to delete path outside runtime root: {resolved_target} not under {resolved_root}"
        ) from exc
    if not target.exists():
        return 0
    freed_bytes = _dir_size_bytes(target)
    shutil.rmtree(target)
    return freed_bytes


def _snapshot_paths(runtime: RuntimeContext) -> dict[str, Any]:
    usage = shutil.disk_usage(runtime.scratch_root)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "disk_free_gb": round(usage.free / (1024**3), 3),
        "disk_used_gb": round(usage.used / (1024**3), 3),
        "runtime_bytes": _dir_size_bytes(runtime.runtime_root),
        "cache_bytes": _dir_size_bytes(runtime.cache_root),
        "workspace_bytes": _dir_size_bytes(runtime.workspace_root),
        "state_bytes": _dir_size_bytes(runtime.state_root),
    }


def _load_reuse_dispatch_evidence_row(db_path: Path, reused_attempt_id: str) -> dict[str, Any] | None:
    if not db_path.is_file():
        return None
    with sqlite3.connect(str(db_path)) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT
                reused_attempt_id,
                source_attempt_id,
                source_candidate_attempt_id,
                parent_attempt_id,
                policy_id,
                policy_version,
                policy_fingerprint,
                quality_profile_fingerprint,
                reused_without_build,
                runner_dispatch_count,
                mobius_invocation_count,
                olive_invocation_count,
                recorded_utc
            FROM candidate_reuse_dispatch_evidence
            WHERE reused_attempt_id = ?
            """,
            (reused_attempt_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "reused_attempt_id": row["reused_attempt_id"],
        "source_attempt_id": row["source_attempt_id"],
        "source_candidate_attempt_id": row["source_candidate_attempt_id"],
        "source_parent_attempt_id": row["parent_attempt_id"],
        "policy_id": row["policy_id"],
        "policy_version": row["policy_version"],
        "policy_fingerprint": row["policy_fingerprint"],
        "quality_profile_fingerprint": row["quality_profile_fingerprint"],
        "reused_without_build": bool(row["reused_without_build"]),
        "runner_dispatch_count": int(row["runner_dispatch_count"]),
        "mobius_invocation_count": int(row["mobius_invocation_count"]),
        "olive_invocation_count": int(row["olive_invocation_count"]),
        "recorded_utc": row["recorded_utc"],
    }


def _collect_branch_source_identity(python_exe: Path, repo_root: Path, expected_commit: str) -> dict[str, Any]:
    code = (
        "import importlib, importlib.metadata, json, pathlib; "
        "m = importlib.import_module('fl_model_onboarding'); "
        "dist = importlib.metadata.distribution('fl-model-onboarding'); "
        "direct = dist.read_text('direct_url.json'); "
        "payload = {"
        "'module_file': str(pathlib.Path(m.__file__).resolve()), "
        "'distribution_root': str(pathlib.Path(dist.locate_file('')).resolve()), "
        "'version': dist.version, "
        "'direct_url': json.loads(direct) if direct else None"
        "}; "
        "print(json.dumps(payload))"
    )
    completed = subprocess.run(
        [str(python_exe), "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    payload: dict[str, Any] | None = None
    if completed.returncode == 0:
        try:
            parsed = json.loads(completed.stdout.strip())
            if isinstance(parsed, dict):
                payload = parsed
        except json.JSONDecodeError:
            payload = None
    module_file = str(payload.get("module_file")) if isinstance(payload, dict) else ""
    distribution_root = str(payload.get("distribution_root")) if isinstance(payload, dict) else ""
    module_under_repo = False
    distribution_under_repo = False
    if module_file:
        try:
            Path(module_file).resolve().relative_to(repo_root.resolve())
            module_under_repo = True
        except ValueError:
            module_under_repo = False
    if distribution_root:
        try:
            Path(distribution_root).resolve().relative_to(repo_root.resolve())
            distribution_under_repo = True
        except ValueError:
            distribution_under_repo = False
    return {
        "python_executable": str(python_exe.resolve()),
        "module_file": module_file or None,
        "distribution_root": distribution_root or None,
        "distribution_version": payload.get("version") if isinstance(payload, dict) else None,
        "distribution_direct_url": payload.get("direct_url") if isinstance(payload, dict) else None,
        "module_under_repo_root": module_under_repo,
        "distribution_under_repo_root": distribution_under_repo,
        "expected_repo_commit": expected_commit or None,
    }


def _cleanup_non_winner_workspaces(
    *,
    runtime_root: Path,
    workspace_root: Path,
    winner_job_ids: set[str],
) -> dict[str, Any]:
    before_bytes = _dir_size_bytes(workspace_root)
    cleaned_count = 0
    blocked_count = 0
    missing_count = 0
    retained_count = 0
    freed_bytes = 0
    details: list[dict[str, Any]] = []
    for child in sorted(workspace_root.iterdir(), key=lambda item: item.name) if workspace_root.exists() else []:
        if not child.is_dir():
            continue
        job_id = child.name
        if job_id in winner_job_ids:
            retained_count += 1
            details.append(
                {
                    "job_id": job_id,
                    "action": "retained-selected-winner-workspace",
                }
            )
            continue
        detail = {"job_id": job_id, "action": "unknown"}
        try:
            reclaimed = _safe_delete_tree_within_root(root=runtime_root, target=child)
        except RuntimeError as exc:
            blocked_count += 1
            detail["action"] = "delete-blocked"
            detail["error"] = str(exc)
        else:
            if reclaimed > 0:
                cleaned_count += 1
                freed_bytes += reclaimed
                detail["action"] = "deleted-nonwinner-workspace"
                detail["freed_bytes"] = reclaimed
            elif child.exists():
                detail["action"] = "delete-no-bytes"
            else:
                missing_count += 1
                detail["action"] = "missing-before-delete"
        details.append(detail)
    after_bytes = _dir_size_bytes(workspace_root)
    return {
        "retention_policy": "retain-selected-winner-workspaces-only",
        "retained_count": retained_count,
        "cleaned_count": cleaned_count,
        "missing_count": missing_count,
        "blocked_count": blocked_count,
        "freed_bytes": freed_bytes,
        "workspace_bytes_before": before_bytes,
        "workspace_bytes_after": after_bytes,
        "details": details,
    }


def _expected_first_request_behavior(model_id: str) -> dict[str, Any]:
    if model_id == EXPECTED_SMOL_MODEL:
        return {
            "selected_candidate_index": 1,
            "selected_candidate_role": "quality_retry",
            "default_candidate_expected": "failed-and-preserved",
            "fallback_candidate_expected": "verified-and-selected",
            "aggregate_mobius_build_invocation_count": 1,
            "aggregate_olive_optimize_invocation_count": 2,
        }
    return {
        "selected_candidate_index": 0,
        "selected_candidate_role": "default",
        "default_candidate_expected": "verified-and-selected",
        "fallback_candidate_expected": "not-registered",
        "aggregate_mobius_build_invocation_count": 1,
        "aggregate_olive_optimize_invocation_count": 1,
    }


def _first_request_behavior_observation(
    *,
    model_id: str,
    workflow_outcome: str | None,
    selected_candidate_index: int | None,
    selected_candidate_role: str | None,
    candidate_count: int,
    aggregate_mobius: int | None,
    aggregate_olive: int | None,
) -> dict[str, Any]:
    expected = _expected_first_request_behavior(model_id)
    mismatches: list[str] = []
    if selected_candidate_index != expected["selected_candidate_index"]:
        mismatches.append("selected_candidate_index")
    if selected_candidate_role != expected["selected_candidate_role"]:
        mismatches.append("selected_candidate_role")
    if workflow_outcome != "selected":
        mismatches.append("workflow_outcome")
    if aggregate_mobius != expected["aggregate_mobius_build_invocation_count"]:
        mismatches.append("aggregate_mobius_build_invocation_count")
    if aggregate_olive != expected["aggregate_olive_optimize_invocation_count"]:
        mismatches.append("aggregate_olive_optimize_invocation_count")
    if model_id == EXPECTED_SMOL_MODEL:
        if candidate_count < 2:
            mismatches.append("candidate_count_for_retry")
    else:
        if candidate_count != 1:
            mismatches.append("candidate_count_without_retry")
    return {
        "expected": expected,
        "actual": {
            "workflow_outcome": workflow_outcome,
            "selected_candidate_index": selected_candidate_index,
            "selected_candidate_role": selected_candidate_role,
            "candidate_count": candidate_count,
            "aggregate_mobius_build_invocation_count": aggregate_mobius,
            "aggregate_olive_optimize_invocation_count": aggregate_olive,
        },
        "matches_expected": len(mismatches) == 0,
        "mismatch_fields": mismatches,
    }


def _run_second_request_reuse_check(
    *,
    client: TestClient,
    service: LocalOnboardingService,
    runtime: RuntimeContext,
    model_id: str,
    model_index: int,
    run_id: str,
    expected_parent_attempt_id: str,
    expected_source_attempt_id: str,
    expected_source_candidate_attempt_id: str,
) -> dict[str, Any]:
    preview_status, preview_payload = _request(
        client,
        "GET",
        "/api/recipes/generated/preview",
        params={"id": model_id, "task": "llm"},
    )
    generated_preview = _as_dict(_as_dict(preview_payload).get("generated_recipe"))
    recipe_fingerprint = str(generated_preview.get("fingerprint") or "").strip().lower()
    preview_reuse_resolution = _as_dict(generated_preview.get("candidate_selection_reuse"))
    jobs_before = len(getattr(service, "_jobs", {}))
    mapping_before = len(getattr(service, "_attempt_to_build_job", {}))
    idempotency_key = f"{run_id}-reuse-{model_index:02d}"
    create_status = 0
    create_payload: dict[str, Any] = {}
    if recipe_fingerprint:
        create_status, raw_create_payload = _request(
            client,
            "POST",
            "/api/recipes/generated/attempts",
            headers={"Idempotency-Key": idempotency_key},
            json={
                "model_id": model_id,
                "recipe_fingerprint": recipe_fingerprint,
                "confirm_automatic_recipe_attempt": True,
            },
        )
        create_payload = _as_dict(raw_create_payload)
    jobs_after = len(getattr(service, "_jobs", {}))
    mapping_after = len(getattr(service, "_attempt_to_build_job", {}))
    created_attempt = _as_dict(create_payload.get("attempt"))
    create_reuse_resolution = _as_dict(created_attempt.get("candidate_selection_reuse"))
    reuse_attempt_id = str(created_attempt.get("attempt_id") or "").strip()
    poll_status = 0
    poll_payload: dict[str, Any] = {}
    if reuse_attempt_id:
        poll_status, raw_poll_payload = _request(
            client,
            "GET",
            f"/api/recipes/generated/attempts/{reuse_attempt_id}",
        )
        poll_payload = _as_dict(raw_poll_payload)
    final_attempt = poll_payload if poll_payload else created_attempt
    final_candidate_selection = _as_dict(final_attempt.get("candidate_selection"))
    final_reuse_payload = _as_dict(final_candidate_selection.get("reuse"))
    reuse_evidence_row = (
        _load_reuse_dispatch_evidence_row(runtime.recipe_attempt_db_path, reuse_attempt_id)
        if reuse_attempt_id
        else None
    )
    build_job_alias = _as_dict(create_payload.get("job")).get("job_id")
    winner_job_id = (
        create_reuse_resolution.get("winner_job_id")
        or preview_reuse_resolution.get("winner_job_id")
        or None
    )
    runner_dispatch_count = _as_int(final_reuse_payload.get("runner_dispatch_count"))
    mobius_invocation_count = _as_int(final_reuse_payload.get("mobius_invocation_count"))
    olive_invocation_count = _as_int(final_reuse_payload.get("olive_invocation_count"))
    reused_without_build = final_reuse_payload.get("reused_without_build") is True
    source_parent_attempt_id = str(final_reuse_payload.get("source_parent_attempt_id") or "").strip()
    source_attempt_id = str(final_reuse_payload.get("source_attempt_id") or "").strip()
    source_candidate_attempt_id = str(final_reuse_payload.get("source_candidate_attempt_id") or "").strip()
    workflow_outcome = str(final_attempt.get("workflow_outcome") or "").strip().lower()
    mapping_present_for_reuse_attempt = reuse_attempt_id in getattr(service, "_attempt_to_build_job", {})
    reuse_zero_build_verified = (
        reused_without_build
        and runner_dispatch_count == 0
        and mobius_invocation_count == 0
        and olive_invocation_count == 0
        and workflow_outcome == "reused"
        and mapping_present_for_reuse_attempt is False
    )
    source_ids_match = (
        source_parent_attempt_id == expected_parent_attempt_id
        and source_attempt_id == expected_source_attempt_id
        and source_candidate_attempt_id == expected_source_candidate_attempt_id
    )
    db_row_matches = (
        isinstance(reuse_evidence_row, dict)
        and reuse_evidence_row.get("source_parent_attempt_id") == source_parent_attempt_id
        and reuse_evidence_row.get("source_attempt_id") == source_attempt_id
        and reuse_evidence_row.get("source_candidate_attempt_id") == source_candidate_attempt_id
        and reuse_evidence_row.get("reused_without_build") is True
        and reuse_evidence_row.get("runner_dispatch_count") == 0
        and reuse_evidence_row.get("mobius_invocation_count") == 0
        and reuse_evidence_row.get("olive_invocation_count") == 0
    )
    return {
        "model_id": model_id,
        "model_index": model_index,
        "second_request_submitted": bool(recipe_fingerprint),
        "preview_http_status": preview_status,
        "create_http_status": create_status,
        "poll_http_status": poll_status,
        "idempotency_key": idempotency_key,
        "recipe_fingerprint": recipe_fingerprint or None,
        "reuse_attempt_id": reuse_attempt_id or None,
        "reuse_attempt_state": final_attempt.get("state"),
        "reuse_workflow_outcome": final_attempt.get("workflow_outcome"),
        "winner_resolution": create_reuse_resolution or preview_reuse_resolution or None,
        "build_job_alias_id": build_job_alias,
        "winner_job_id": winner_job_id,
        "build_job_alias_matches_winner_job": bool(build_job_alias and winner_job_id and build_job_alias == winner_job_id),
        "attempt_build_job_id": final_attempt.get("build_job_id"),
        "runner_dispatch_mapping_present_for_reuse_attempt": mapping_present_for_reuse_attempt,
        "service_dispatch_observation": {
            "jobs_before": jobs_before,
            "jobs_after": jobs_after,
            "attempt_mapping_count_before": mapping_before,
            "attempt_mapping_count_after": mapping_after,
            "new_job_created_for_reuse_attempt": jobs_after > jobs_before,
        },
        "reuse_evidence": final_reuse_payload or None,
        "reuse_evidence_row": reuse_evidence_row,
        "source_ids_match_expected_winner": source_ids_match,
        "reuse_evidence_row_matches_response": db_row_matches,
        "reuse_zero_build_verified": reuse_zero_build_verified,
    }


def _build_round7_report(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Recipe Agent v1 Round 7 Report")
    lines.append("")
    lines.append(f"- **Run ID:** `{summary.get('run_id')}`")
    lines.append(f"- **Branch:** `{summary.get('branch')}`")
    lines.append(f"- **Commit:** `{summary.get('commit')}`")
    lines.append(f"- **Window (UTC):** `{summary.get('started_utc')}` -> `{summary.get('finished_utc')}`")
    lines.append(f"- **valid_baseline:** `{summary.get('valid_baseline')}`")
    lines.append(f"- **Recipe Verification (winner-selected):** `{summary.get('recipe_verified_rate')}`")
    lines.append(f"- **Model Capability (all checks passed):** `{summary.get('model_capability_all_pass_rate')}`")
    lines.append(f"- **Selected-candidate reuse zero-build evidence:** `{summary.get('selected_candidate_reuse_zero_build_rate')}`")
    lines.append("")
    lines.append("## Recipe Verification (winner-selected)")
    lines.append("")
    lines.append("| Model | Workflow outcome | Winner candidate | Winner recipe status | First request Mobius/Olive | Reuse 0-build |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    reuse_by_model = {
        str(row.get("model_id")): row for row in _as_list(summary.get("reuse_checks")) if isinstance(row, dict)
    }
    for row in _as_list(summary.get("results")):
        if not isinstance(row, dict):
            continue
        model_id = str(row.get("model_id"))
        winner = _as_dict(row.get("winner_candidate"))
        winner_index = _as_int(winner.get("candidate_index"))
        winner_id = winner.get("candidate_id")
        winner_text = "-" if winner_index is None else f"{winner_index}:{winner_id}"
        verification = _as_dict(row.get("winner_recipe_verification"))
        aggregate = _as_dict(row.get("first_request_aggregate_invocation_counters"))
        mobius = aggregate.get("mobius_build_invocation_count")
        olive = aggregate.get("olive_optimize_invocation_count")
        reuse_row = reuse_by_model.get(model_id, {})
        reuse_ok = _as_dict(reuse_row).get("reuse_zero_build_verified")
        lines.append(
            "| "
            + f"{model_id} | {row.get('workflow_outcome')} | {winner_text} | {verification.get('status')} "
            + f"| {mobius}/{olive} | {reuse_ok} |"
        )
    lines.append("")
    lines.append("## Model Capability (non-blocking advisory)")
    lines.append("")
    lines.append("| Model | Checks passed | Confidence | Warnings |")
    lines.append("| --- | --- | --- | --- |")
    for row in _as_list(summary.get("results")):
        if not isinstance(row, dict):
            continue
        capability = _as_dict(row.get("winner_model_capability"))
        checks_passed = capability.get("checks_passed")
        total_checks = capability.get("total_checks")
        checks = (
            f"{checks_passed}/{total_checks}"
            if isinstance(checks_passed, int) and isinstance(total_checks, int)
            else "-"
        )
        confidence = _as_dict(capability.get("confidence")).get("level")
        warnings = capability.get("warnings")
        warning_text = ", ".join(str(item) for item in warnings) if isinstance(warnings, list) and warnings else "-"
        lines.append(
            "| "
            + f"{row.get('model_id')} | {checks} | {confidence or '-'} | {warning_text} |"
        )
    lines.append("")
    lines.append("## Retry and dispatch evidence")
    lines.append("")
    lines.append(
        f"- Retry expected only for `{EXPECTED_SMOL_MODEL}`: expected `{summary.get('retry_count_expected')}`, actual `{summary.get('retry_count_actual')}`."
    )
    lines.append(
        "- First-request aggregate invocation totals: "
        + f"Mobius={_as_dict(summary.get('first_request_aggregate_dispatch_totals')).get('mobius_build_invocation_count')}, "
        + f"Olive={_as_dict(summary.get('first_request_aggregate_dispatch_totals')).get('olive_optimize_invocation_count')}."
    )
    lines.append(
        f"- Selected-candidate reuse with measured zero dispatch (5-model denominator): `{summary.get('selected_candidate_reuse_zero_build_rate')}`."
    )
    lines.append("")
    lines.append("## Cleanup and process evidence")
    lines.append("")
    cleanup = _as_dict(summary.get("post_reuse_workspace_cleanup"))
    lines.append(
        "- Workspace cleanup after durable evidence extraction: "
        + f"retained={cleanup.get('retained_count')}, deleted={cleanup.get('cleaned_count')}, "
        + f"blocked={cleanup.get('blocked_count')}, freed_bytes={cleanup.get('freed_bytes')}."
    )
    lines.append(
        f"- Final lingering process count under runtime root: `{summary.get('final_lingering_process_count')}`."
    )
    source_identity = _as_dict(summary.get("branch_source_identity"))
    lines.append(
        "- Branch source identity: "
        + f"module_under_repo_root={source_identity.get('module_under_repo_root')}, "
        + f"distribution_under_repo_root={source_identity.get('distribution_under_repo_root')}, "
        + f"expected_commit={source_identity.get('expected_repo_commit')}."
    )
    lines.append("")
    lines.append("## Delta from Round 6")
    lines.append("")
    delta = _as_dict(summary.get("round6_delta"))
    lines.append(
        f"- Round 6 Recipe Verification: `{delta.get('round6_recipe_verified_rate')}` -> Round 7: `{delta.get('round7_recipe_verified_rate')}`."
    )
    explanation = delta.get("explanation")
    if isinstance(explanation, str) and explanation.strip():
        lines.append(f"- Explanation: {explanation}")
    remaining = _as_list(summary.get("remaining_path_to_5of5_recipe_verification"))
    if remaining:
        lines.append("- Remaining path to 5/5:")
        for row in remaining:
            if not isinstance(row, dict):
                continue
            lines.append(
                f"  - `{row.get('model_id')}` -> status `{row.get('winner_recipe_status')}`; next action: {row.get('next_action')}"
            )
    else:
        lines.append("- Remaining path to 5/5: none.")
    lines.append("")
    return "\n".join(lines)


def _enrich_round7_artifacts(
    *,
    round4_module: Any,
    script_root: Path,
    round_name: str,
    scratch_root: Path,
    python_exe: Path,
) -> None:
    summary_path = script_root / "round-7-summary.json"
    manifest_path = script_root / "round-manifest.json"
    if not summary_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("Round 7 base artifacts were not created by the round runner.")
    summary = _read_json(summary_path)
    manifest = _read_json(manifest_path)
    run_id = str(summary.get("run_id") or "").strip()
    if not run_id:
        raise RuntimeError("Round 7 summary is missing run_id.")
    runtime = _build_runtime_context(
        script_root=script_root,
        round_name=round_name,
        scratch_root=scratch_root,
        run_id=run_id,
    )
    if not runtime.service_db_path.is_file():
        raise RuntimeError(
            f"Round 7 runtime state database was not found at expected path: {runtime.service_db_path}"
        )
    python_runtime = round4_module._resolve_python_runtime(python_exe)
    run_paths = round4_module.RunPaths(
        repo_root=runtime.repo_root,
        output_root=runtime.output_root,
        manifest_path=runtime.manifest_path,
        scratch_root=runtime.scratch_root,
        runtime_root=runtime.runtime_root,
        state_root=runtime.state_root,
        workspace_base=runtime.workspace_root,
        model_cache_dir=runtime.cache_root,
        service_db_path=runtime.service_db_path,
        recipe_attempt_db_path=runtime.recipe_attempt_db_path,
        run_id=runtime.run_id,
    )
    replacements = round4_module._replacement_pairs(
        run_paths,
        round_name=round_name,
        python_runtime=python_runtime,
    )

    def sanitize(value: Any) -> Any:
        return round4_module._sanitize_obj(value, replacements)

    results = [row for row in _as_list(summary.get("results")) if isinstance(row, dict)]
    model_results_dir = script_root / "model-results"
    recipe_status_counts = {"verified": 0, "blocked": 0, "inconclusive": 0, "unknown": 0}
    model_capability_all_pass_count = 0
    retry_models: list[str] = []
    first_request_mobius_total = 0
    first_request_olive_total = 0
    reuse_checks: list[dict[str, Any]] = []
    winner_job_ids: set[str] = set()

    source_identity = _collect_branch_source_identity(
        python_exe=python_exe,
        repo_root=runtime.repo_root,
        expected_commit=str(summary.get("commit") or ""),
    )

    service = LocalOnboardingService(
        db_path=runtime.service_db_path,
        workspace_base=runtime.workspace_root,
        model_cache_dir=runtime.cache_root,
        enable_production_runner=True,
        runtime_python_executable=python_exe,
    )
    try:
        with TestClient(create_app(service=service)) as client:
            for row in results:
                model_index = int(row.get("model_index", 0))
                model_id = str(row.get("model_id") or "")
                parent_attempt_id = str(row.get("attempt_id") or "")
                first_attempt_status = 0
                first_attempt_payload: dict[str, Any] = {}
                if parent_attempt_id:
                    first_attempt_status, attempt_payload = _request(
                        client,
                        "GET",
                        f"/api/recipes/generated/attempts/{parent_attempt_id}",
                    )
                    first_attempt_payload = _as_dict(attempt_payload)
                candidate_selection = _as_dict(first_attempt_payload.get("candidate_selection"))
                selected_candidate = _as_dict(candidate_selection.get("selected_candidate"))
                selected_candidate_entry = _find_selected_candidate_entry(candidate_selection)
                aggregate_counters = _as_dict(candidate_selection.get("aggregate_invocation_counters"))
                selected_candidate_index = _as_int(selected_candidate.get("candidate_index"))
                selected_candidate_role = _candidate_role(selected_candidate_index)
                candidate_count = len(_as_list(candidate_selection.get("candidates")))
                aggregate_mobius = _as_int(aggregate_counters.get("mobius_build_invocation_count"))
                aggregate_olive = _as_int(aggregate_counters.get("olive_optimize_invocation_count"))
                if aggregate_mobius is not None:
                    first_request_mobius_total += aggregate_mobius
                if aggregate_olive is not None:
                    first_request_olive_total += aggregate_olive
                if selected_candidate_index is not None and selected_candidate_index > 0:
                    retry_models.append(model_id)
                winner_attempt_id = str(selected_candidate.get("attempt_id") or parent_attempt_id).strip()
                winner_attempt_payload = first_attempt_payload
                winner_attempt_status = first_attempt_status
                if winner_attempt_id and winner_attempt_id != parent_attempt_id:
                    winner_attempt_status, winner_payload = _request(
                        client,
                        "GET",
                        f"/api/recipes/generated/attempts/{winner_attempt_id}",
                    )
                    winner_attempt_payload = _as_dict(winner_payload)
                winner_quality = _as_dict(winner_attempt_payload.get("quality_validation"))
                winner_recipe_verification = _as_dict(winner_quality.get("recipe_integrity"))
                winner_model_capability = _as_dict(winner_quality.get("model_capability"))
                winner_recipe_status = str(winner_recipe_verification.get("status") or "").strip().lower()
                if winner_recipe_status not in recipe_status_counts:
                    winner_recipe_status = "unknown"
                recipe_status_counts[winner_recipe_status] += 1
                checks_passed = _as_int(winner_model_capability.get("checks_passed"))
                total_checks = _as_int(winner_model_capability.get("total_checks"))
                if checks_passed is not None and total_checks is not None and total_checks > 0 and checks_passed == total_checks:
                    model_capability_all_pass_count += 1
                winner_job_id = _parse_winner_job_id(selected_candidate_entry)
                if winner_job_id:
                    winner_job_ids.add(winner_job_id)

                first_request_behavior = _first_request_behavior_observation(
                    model_id=model_id,
                    workflow_outcome=str(first_attempt_payload.get("workflow_outcome") or None),
                    selected_candidate_index=selected_candidate_index,
                    selected_candidate_role=selected_candidate_role,
                    candidate_count=candidate_count,
                    aggregate_mobius=aggregate_mobius,
                    aggregate_olive=aggregate_olive,
                )

                row["workflow_outcome"] = first_attempt_payload.get("workflow_outcome")
                row["candidate_selection"] = candidate_selection or None
                row["winner_candidate"] = selected_candidate or None
                row["winner_candidate_timeline_entry"] = selected_candidate_entry
                row["winner_attempt_id"] = winner_attempt_id or None
                row["winner_attempt_state"] = winner_attempt_payload.get("state")
                row["winner_recipe_verification"] = winner_recipe_verification or None
                row["winner_model_capability"] = winner_model_capability or None
                row["winner_recipe_status"] = winner_recipe_status
                row["first_request_aggregate_invocation_counters"] = aggregate_counters or None
                row["first_request_behavior_observation"] = first_request_behavior
                row["first_request_attempt_http_status"] = first_attempt_status
                row["winner_attempt_http_status"] = winner_attempt_status
                row["winner_job_id"] = winner_job_id

                if (
                    winner_recipe_status == "verified"
                    and isinstance(selected_candidate, dict)
                    and selected_candidate.get("candidate_attempt_id")
                    and winner_attempt_id
                ):
                    reuse_check = _run_second_request_reuse_check(
                        client=client,
                        service=service,
                        runtime=runtime,
                        model_id=model_id,
                        model_index=model_index,
                        run_id=runtime.run_id,
                        expected_parent_attempt_id=parent_attempt_id,
                        expected_source_attempt_id=winner_attempt_id,
                        expected_source_candidate_attempt_id=str(
                            selected_candidate.get("candidate_attempt_id")
                        ),
                    )
                else:
                    reuse_check = {
                        "model_id": model_id,
                        "model_index": model_index,
                        "second_request_submitted": False,
                        "reason": "winner_not_verified_or_missing_selection",
                        "reuse_zero_build_verified": False,
                    }
                reuse_checks.append(reuse_check)
                row["second_request_reuse"] = reuse_check

                model_filename = f"{model_index:02d}-{round4_module._slugify(model_id)}.json"
                model_path = model_results_dir / model_filename
                model_payload = _read_json(model_path) if model_path.is_file() else {}
                model_payload["round7_first_request"] = {
                    "workflow_outcome": row.get("workflow_outcome"),
                    "candidate_selection": row.get("candidate_selection"),
                    "winner_candidate": row.get("winner_candidate"),
                    "winner_candidate_timeline_entry": row.get("winner_candidate_timeline_entry"),
                    "winner_attempt_id": row.get("winner_attempt_id"),
                    "winner_attempt_state": row.get("winner_attempt_state"),
                    "winner_recipe_verification": row.get("winner_recipe_verification"),
                    "winner_model_capability": row.get("winner_model_capability"),
                    "first_request_aggregate_invocation_counters": row.get(
                        "first_request_aggregate_invocation_counters"
                    ),
                    "first_request_behavior_observation": row.get("first_request_behavior_observation"),
                    "winner_job_id": row.get("winner_job_id"),
                }
                model_payload["round7_second_request_reuse"] = reuse_check
                _write_json(model_path, sanitize(model_payload))
    finally:
        try:
            service.close()
        except Exception:
            pass

    post_reuse_cleanup = _cleanup_non_winner_workspaces(
        runtime_root=runtime.runtime_root,
        workspace_root=runtime.workspace_root,
        winner_job_ids=winner_job_ids,
    )
    lingering = round4_module._lingering_processes_for_runtime(runtime.runtime_root)
    final_snapshot = _snapshot_paths(runtime)
    models_total = len(results)
    zero_reuse_count = sum(
        1
        for row in reuse_checks
        if isinstance(row, dict) and row.get("reuse_zero_build_verified") is True
    )
    parent_recipe_verified_rate = summary.get("recipe_verified_rate")
    parent_recipe_verified_count = summary.get("recipe_verified_count")
    summary["results"] = results
    summary["valid_baseline"] = bool(summary.get("baseline_valid"))
    summary["parent_attempt_recipe_verified_count"] = parent_recipe_verified_count
    summary["parent_attempt_recipe_verified_rate"] = parent_recipe_verified_rate
    summary["recipe_verified_count"] = recipe_status_counts["verified"]
    summary["recipe_blocked_count"] = recipe_status_counts["blocked"]
    summary["recipe_inconclusive_count"] = recipe_status_counts["inconclusive"]
    summary["recipe_status_unknown_count"] = recipe_status_counts["unknown"]
    summary["recipe_verified_rate"] = f"{recipe_status_counts['verified']}/{models_total}"
    summary["model_capability_all_pass_count"] = model_capability_all_pass_count
    summary["model_capability_all_pass_rate"] = f"{model_capability_all_pass_count}/{models_total}"
    summary["recipe_verification_status_counts"] = {
        "VERIFIED": recipe_status_counts["verified"],
        "BLOCKED": recipe_status_counts["blocked"],
        "INCONCLUSIVE": recipe_status_counts["inconclusive"],
        "UNKNOWN": recipe_status_counts["unknown"],
    }
    summary["retry_count_expected"] = 1
    summary["retry_count_actual"] = len(retry_models)
    summary["retry_models_expected"] = [EXPECTED_SMOL_MODEL]
    summary["retry_models_actual"] = sorted(set(retry_models))
    summary["first_request_aggregate_dispatch_totals"] = {
        "mobius_build_invocation_count": first_request_mobius_total,
        "olive_optimize_invocation_count": first_request_olive_total,
    }
    summary["selected_candidate_reuse_zero_build_count"] = zero_reuse_count
    summary["selected_candidate_reuse_zero_build_rate"] = f"{zero_reuse_count}/{models_total}"
    summary["reuse_checks"] = reuse_checks
    summary["post_reuse_workspace_cleanup"] = post_reuse_cleanup
    summary["final_snapshot_after_reuse_cleanup"] = final_snapshot
    summary["final_lingering_processes"] = lingering
    summary["final_lingering_process_count"] = (
        len(lingering) if isinstance(lingering, list) else None
    )
    summary["branch_source_identity"] = source_identity
    remaining = []
    for row in results:
        if not isinstance(row, dict):
            continue
        if str(row.get("winner_recipe_status")) == "verified":
            continue
        remaining.append(
            {
                "model_id": row.get("model_id"),
                "winner_recipe_status": row.get("winner_recipe_status"),
                "next_action": row.get("next_action"),
            }
        )
    summary["remaining_path_to_5of5_recipe_verification"] = remaining

    round6_summary_path = script_root.parent / "round-6" / "round-6-summary.json"
    round6_summary = _read_json(round6_summary_path) if round6_summary_path.is_file() else {}
    round6_recipe_verified_rate = round6_summary.get("recipe_verified_rate")
    summary["round6_delta"] = {
        "round6_recipe_verified_rate": round6_recipe_verified_rate,
        "round7_recipe_verified_rate": summary.get("recipe_verified_rate"),
        "round6_recipe_verified_count": round6_summary.get("recipe_verified_count"),
        "round7_recipe_verified_count": summary.get("recipe_verified_count"),
        "explanation": (
            "Round 7 counts recipe verification by selected winner candidate (including quality-retry fallback), "
            "while preserving parent-attempt-only rate in parent_attempt_recipe_verified_rate."
        ),
    }

    manifest["reuse_checks"] = reuse_checks
    manifest["round7_enrichment"] = {
        "recipe_verification_status_counts": summary.get("recipe_verification_status_counts"),
        "recipe_verified_rate": summary.get("recipe_verified_rate"),
        "model_capability_all_pass_rate": summary.get("model_capability_all_pass_rate"),
        "retry_count_expected": summary.get("retry_count_expected"),
        "retry_count_actual": summary.get("retry_count_actual"),
        "retry_models_expected": summary.get("retry_models_expected"),
        "retry_models_actual": summary.get("retry_models_actual"),
        "first_request_aggregate_dispatch_totals": summary.get("first_request_aggregate_dispatch_totals"),
        "selected_candidate_reuse_zero_build_rate": summary.get("selected_candidate_reuse_zero_build_rate"),
        "branch_source_identity": source_identity,
        "post_reuse_workspace_cleanup": post_reuse_cleanup,
        "final_snapshot_after_reuse_cleanup": final_snapshot,
        "final_lingering_process_count": summary.get("final_lingering_process_count"),
    }

    sanitized_summary = sanitize(summary)
    sanitized_manifest = sanitize(manifest)
    _write_json(summary_path, sanitized_summary)
    _write_json(manifest_path, sanitized_manifest)
    report = _build_round7_report(sanitized_summary)
    (script_root / "round-7-report.md").write_text(report, encoding="utf-8")


def main() -> int:
    argv = list(sys.argv[1:])
    script_root = Path(__file__).resolve().parent
    round4_module = _load_round4_runner_module()
    _ensure_arg(argv, "--output-dir", str(script_root))
    _ensure_arg(argv, "--round-name", script_root.name)
    _ensure_arg(argv, "--scratch-root", str(DEFAULT_SCRATCH_ROOT))
    _ensure_arg(argv, "--python-exe", str(_resolve_default_python()))
    if not _has_arg(argv, "--seed-runtime-root"):
        round6_runtime = _discover_round6_runtime_root(script_root)
        if round6_runtime is not None:
            argv.extend(["--seed-runtime-root", str(round6_runtime)])
    _ensure_arg(argv, "--seed-mode", "junction")

    round_name = str(_arg_value(argv, "--round-name") or script_root.name)
    scratch_root = Path(_arg_value(argv, "--scratch-root") or str(DEFAULT_SCRATCH_ROOT)).resolve()
    python_exe = Path(_arg_value(argv, "--python-exe") or str(_resolve_default_python())).resolve()
    if not python_exe.is_file():
        raise SystemExit(f"Python executable not found for Round 7 runtime: {python_exe}")

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *argv]
        exit_code = int(round4_module.main())
    finally:
        sys.argv = original_argv
    if exit_code != 0:
        return exit_code
    _enrich_round7_artifacts(
        round4_module=round4_module,
        script_root=script_root,
        round_name=round_name,
        scratch_root=scratch_root,
        python_exe=python_exe,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
