from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from fl_model_onboarding.local_api import create_app
from fl_model_onboarding.local_service import LocalOnboardingService

EXPECTED_FROZEN_MODELS: tuple[tuple[str, str], ...] = (
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "fe8a4ea1ffedaf415f4da2f062534de366a451e6"),
    ("HuggingFaceTB/SmolLM2-360M-Instruct", "a10cc1512eabd3dde888204e902eca88bddb4951"),
    ("Qwen/Qwen2-1.5B-Instruct", "ba1cf1846d7df0a0591d6c00649f57e798519da8"),
    ("Qwen/Qwen2-0.5B-Instruct", "c540970f9e29518b1d8f06ab8b24cba66ad77b6d"),
    ("ibm-granite/granite-3.2-2b-instruct", "641593c3b25bec0b1efe9f0f7d7a67f7243f86a3"),
)
TERMINAL_STATES = {"succeeded", "failed", "cancelled"}
WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"[A-Za-z]:(?:\\|/(?!/))[^\s\"']+")
REQUIRED_COMMANDS: tuple[str, ...] = ("foundry", "mobius", "olive")
REQUIRED_DISTRIBUTIONS: tuple[tuple[str, str], ...] = (
    ("onnx", "onnx"),
    ("onnxruntime", "onnxruntime"),
    ("onnxruntime-genai", "onnxruntime_genai"),
    ("foundry-local-sdk", "foundry_local_sdk"),
    ("mobius-onnx", "mobius"),
    ("olive-ai", "olive"),
)


@dataclass(frozen=True)
class RunPaths:
    repo_root: Path
    output_root: Path
    manifest_path: Path
    scratch_root: Path
    runtime_root: Path
    state_root: Path
    workspace_base: Path
    model_cache_dir: Path
    service_db_path: Path
    recipe_attempt_db_path: Path
    run_id: str


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    return slug.strip("-") or "model"


def _parse_json_maybe(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidate = stripped[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None
    return None


def _run_cli_json(
    python_exe: Path,
    cli_args: list[str],
    *,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    cmd = [str(python_exe), "-m", "fl_model_onboarding.cli", *cli_args]
    completed = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    payload = _parse_json_maybe(completed.stdout) or _parse_json_maybe(completed.stderr)
    return {
        "command": cmd,
        "exit_code": completed.returncode,
        "json": payload,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _normalize_path_for_compare(value: str) -> str:
    return value.replace("/", "\\").rstrip("\\").lower()


def _path_contains_dir(path_value: str, directory: Path) -> bool:
    expected = _normalize_path_for_compare(str(directory))
    if not expected:
        return False
    for segment in path_value.split(os.pathsep):
        current = segment.strip()
        if not current:
            continue
        if _normalize_path_for_compare(current) == expected:
            return True
    return False


def _with_scripts_on_path(base_env: dict[str, str], scripts_dir: Path) -> tuple[dict[str, str], bool]:
    env = dict(base_env)
    current = env.get("PATH", "")
    if _path_contains_dir(current, scripts_dir):
        return env, False
    env["PATH"] = str(scripts_dir) if not current else f"{scripts_dir}{os.pathsep}{current}"
    return env, True


def _resolve_console_script_path(scripts_dir: Path, name: str) -> Path | None:
    for suffix in ("", ".exe", ".cmd", ".bat", ".ps1"):
        candidate = scripts_dir / f"{name}{suffix}"
        if candidate.exists():
            return candidate.resolve()
    return None


def _resolve_python_runtime(python_exe: Path) -> dict[str, Any]:
    code = (
        "import json,sys,sysconfig;"
        "print(json.dumps({"
        "'executable': sys.executable,"
        "'version': sys.version.split()[0],"
        "'scripts_dir': sysconfig.get_path('scripts')"
        "}))"
    )
    completed = subprocess.run(
        [str(python_exe), "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    payload = _parse_json_maybe(completed.stdout)
    if completed.returncode != 0 or not isinstance(payload, dict):
        raise RuntimeError(
            "Unable to resolve Python runtime identity from the selected interpreter: "
            + (completed.stderr.strip() or completed.stdout.strip() or f"exit_code={completed.returncode}")
        )
    scripts_dir_raw = payload.get("scripts_dir")
    if not isinstance(scripts_dir_raw, str) or not scripts_dir_raw.strip():
        raise RuntimeError("Resolved Python runtime is missing a scripts directory.")
    base_env = dict(os.environ)
    scripts_dir = Path(scripts_dir_raw).resolve()
    child_env, path_prefixed = _with_scripts_on_path(base_env, scripts_dir)
    return {
        "python_exe": Path(str(payload.get("executable") or python_exe)).resolve(),
        "python_version": str(payload.get("version") or "unknown"),
        "scripts_dir": scripts_dir,
        "scripts_dir_name": scripts_dir.name,
        "venv_name": scripts_dir.parent.name,
        "parent_path_has_scripts": _path_contains_dir(base_env.get("PATH", ""), scripts_dir),
        "child_path_prefixed": path_prefixed,
        "child_env": child_env,
        "command_paths": {
            name: _resolve_console_script_path(scripts_dir, name) for name in REQUIRED_COMMANDS
        },
    }


def _python_runtime_identity(runtime: dict[str, Any]) -> dict[str, Any]:
    command_paths = runtime.get("command_paths")
    command_resolution: dict[str, str] = {}
    if isinstance(command_paths, dict):
        for name in REQUIRED_COMMANDS:
            command_resolution[name] = (
                "venv-scripts-absolute"
                if isinstance(command_paths.get(name), Path)
                else "path-lookup"
            )
    return {
        "python_version": runtime.get("python_version"),
        "python_executable_name": runtime.get("python_exe").name
        if isinstance(runtime.get("python_exe"), Path)
        else "python",
        "venv_name": runtime.get("venv_name"),
        "scripts_dir_name": runtime.get("scripts_dir_name"),
        "scripts_dir_resolution": "sysconfig",
        "parent_path_has_scripts": bool(runtime.get("parent_path_has_scripts")),
        "child_path_prefixed_scripts": bool(runtime.get("child_path_prefixed")),
        "command_resolution": command_resolution,
    }


def _extract_semver(value: str) -> str | None:
    match = re.search(r"\d+\.\d+\.\d+(?:\.\d+)?", value)
    return match.group(0) if match else None


def _probe_command(
    name: str,
    *,
    executable: Path | None = None,
    env: dict[str, str] | None = None,
    availability_args: tuple[str, ...] = ("--version",),
    version_args: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    invocation = str(executable) if executable is not None else name
    resolution = "venv-scripts-absolute" if executable is not None else "path-lookup"
    try:
        availability = subprocess.run(
            [invocation, *availability_args],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
    except FileNotFoundError:
        return {
            "name": name,
            "kind": "command",
            "required": True,
            "available": False,
            "version": None,
            "probe": "command-not-found",
            "resolution": resolution,
        }
    output = (availability.stdout or availability.stderr or "").strip()
    first_line = output.splitlines()[0] if output else ""
    version = _extract_semver(first_line) if first_line else None
    probe = first_line or (
        "probe-ok" if availability.returncode == 0 else f"exit-code-{availability.returncode}"
    )
    available = availability.returncode == 0
    if version_args is not None:
        try:
            version_probe = subprocess.run(
                [invocation, *version_args],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            version_output = (version_probe.stdout or version_probe.stderr or "").strip()
            version_first_line = version_output.splitlines()[0] if version_output else ""
            if version_probe.returncode == 0:
                parsed = _extract_semver(version_first_line)
                if parsed:
                    version = parsed
                if not probe:
                    probe = version_first_line or "probe-ok"
            elif not available and version_first_line:
                probe = version_first_line
        except FileNotFoundError:
            available = False
            probe = "command-not-found"
    return {
        "name": name,
        "kind": "command",
        "required": True,
        "available": available,
        "version": version,
        "probe": probe or "probe-empty",
        "resolution": resolution,
    }


def _probe_command_version(
    name: str,
    *,
    executable: Path | None = None,
    env: dict[str, str] | None = None,
    args: tuple[str, ...] = ("--version",),
) -> dict[str, Any]:
    return _probe_command(
        name,
        executable=executable,
        env=env,
        availability_args=args,
    )


def _probe_python_distribution(
    python_exe: Path,
    distribution: str,
    import_name: str,
    *,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    code = (
        "import importlib, importlib.metadata, json; "
        f"importlib.import_module('{import_name}'); "
        f"print(json.dumps({{'version': importlib.metadata.version('{distribution}')}}))"
    )
    completed = subprocess.run(
        [str(python_exe), "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    payload = _parse_json_maybe(completed.stdout)
    version = payload.get("version") if isinstance(payload, dict) else None
    detail = (completed.stderr or completed.stdout or "").strip().splitlines()
    probe = detail[-1] if detail else ("probe-ok" if completed.returncode == 0 else f"exit-code-{completed.returncode}")
    return {
        "name": distribution,
        "kind": "python-package",
        "required": True,
        "import_name": import_name,
        "available": completed.returncode == 0 and isinstance(version, str),
        "version": version if isinstance(version, str) else None,
        "probe": probe,
        "resolution": "python-interpreter",
    }


def _probe_toolchain(
    python_exe: Path,
    *,
    runtime: dict[str, Any],
    env: dict[str, str],
) -> dict[str, Any]:
    command_paths = runtime.get("command_paths")
    resolved_paths = command_paths if isinstance(command_paths, dict) else {}
    probes = [
        _probe_command(
            "foundry",
            executable=resolved_paths.get("foundry"),
            env=env,
            availability_args=("--version",),
        ),
        _probe_command(
            "mobius",
            executable=resolved_paths.get("mobius"),
            env=env,
            availability_args=("--help",),
            version_args=("--version",),
        ),
        _probe_command(
            "olive",
            executable=resolved_paths.get("olive"),
            env=env,
            availability_args=("--help",),
            version_args=("--version",),
        ),
    ]
    probes.extend(
        _probe_python_distribution(
            python_exe,
            distribution,
            import_name,
            env=env,
        )
        for distribution, import_name in REQUIRED_DISTRIBUTIONS
    )
    missing_required = [
        f"{row.get('kind')}:{row.get('name')}"
        for row in probes
        if row.get("required") is True and row.get("available") is False
    ]
    return {
        "python_runtime": _python_runtime_identity(runtime),
        "probes": probes,
        "missing_required": missing_required,
        "ready_for_round": len(missing_required) == 0,
    }


def _git_value(repo_root: Path, args: list[str]) -> str:
    completed = subprocess.run(
        ["git", "--no-pager", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return f"git-error:{completed.stderr.strip()}"
    return completed.stdout.strip()


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for entry in path.rglob("*"):
        if not entry.is_file():
            continue
        try:
            total += entry.stat().st_size
        except OSError:
            continue
    return total


def _snapshot_paths(paths: RunPaths) -> dict[str, Any]:
    usage = shutil.disk_usage(paths.scratch_root)
    return {
        "timestamp_utc": _now_utc_iso(),
        "disk_free_gb": round(usage.free / (1024**3), 3),
        "disk_used_gb": round(usage.used / (1024**3), 3),
        "runtime_bytes": _dir_size_bytes(paths.runtime_root),
        "cache_bytes": _dir_size_bytes(paths.model_cache_dir),
        "workspace_bytes": _dir_size_bytes(paths.workspace_base),
        "state_bytes": _dir_size_bytes(paths.state_root),
    }


def _decode_response(response: Any) -> tuple[int, Any]:
    status_code = int(getattr(response, "status_code", 500))
    try:
        return status_code, response.json()
    except Exception:
        return status_code, {"raw_text": getattr(response, "text", "")}


def _request(client: TestClient, method: str, path: str, **kwargs: Any) -> tuple[int, Any]:
    response = client.request(method, path, **kwargs)
    return _decode_response(response)


def _poll_terminal_job(
    client: TestClient,
    *,
    job_id: str,
    timeout_seconds: int,
    poll_seconds: int,
) -> tuple[dict[str, Any], bool, dict[str, Any] | None]:
    deadline = time.time() + timeout_seconds
    last_payload: dict[str, Any] | None = None
    while time.time() < deadline:
        status, payload = _request(client, "GET", f"/api/builds/{job_id}")
        if status != 200:
            return (
                {
                    "job_id": job_id,
                    "state": "failed",
                    "failure": {
                        "stage": "polling",
                        "classification": "api_error",
                        "message": f"Polling /api/builds/{job_id} returned HTTP {status}.",
                        "detail": payload,
                    },
                },
                False,
                None,
            )
        assert isinstance(payload, dict)
        last_payload = payload
        state = str(payload.get("state", "")).lower()
        if state in TERMINAL_STATES:
            return payload, False, None
        time.sleep(max(1, poll_seconds))
    cancel_status, cancel_payload = _request(
        client,
        "POST",
        f"/api/builds/{job_id}/cancel",
        json={"reason": "Round 1 timeout guard reached before terminal state."},
    )
    final_status, final_payload = _request(client, "GET", f"/api/builds/{job_id}")
    if final_status == 200 and isinstance(final_payload, dict):
        return final_payload, True, {"status_code": cancel_status, "payload": cancel_payload}
    if last_payload is not None:
        return last_payload, True, {"status_code": cancel_status, "payload": cancel_payload}
    return (
        {
            "job_id": job_id,
            "state": "failed",
            "failure": {
                "stage": "polling",
                "classification": "timeout",
                "message": "Timed out and failed to retrieve terminal build state.",
            },
        },
        True,
        {"status_code": cancel_status, "payload": cancel_payload},
    )


def _compact_model_detail(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"raw": payload}
    matches = payload.get("foundry_catalog_matches")
    return {
        "model_id": payload.get("model_id"),
        "sha": payload.get("sha"),
        "task": payload.get("task"),
        "buildable": payload.get("buildable"),
        "build_blockers": payload.get("build_blockers"),
        "recipe_status": payload.get("recipe_status"),
        "recipe_reason": payload.get("recipe_reason"),
        "foundry_catalog_status": payload.get("foundry_catalog_status"),
        "foundry_catalog_matches_count": len(matches) if isinstance(matches, list) else None,
        "warnings": payload.get("warnings"),
    }


def _compact_preflight(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"raw": payload}
    result = payload.get("result")
    tools: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    if isinstance(result, dict):
        result_tools = result.get("tools")
        if isinstance(result_tools, list):
            for row in result_tools:
                if isinstance(row, dict):
                    tools.append(
                        {
                            "name": row.get("name"),
                            "available": row.get("available"),
                            "version": row.get("version"),
                            "probe": row.get("detail") or row.get("version") or "probe-empty",
                        }
                    )
        result_blockers = result.get("blockers")
        if isinstance(result_blockers, list):
            for row in result_blockers:
                if isinstance(row, dict):
                    blockers.append(
                        {
                            "stage": row.get("stage"),
                            "classification": row.get("classification"),
                            "message": row.get("message"),
                        }
                    )
    generated_recipe = payload.get("generated_recipe")
    generated_summary: dict[str, Any] = {}
    if isinstance(generated_recipe, dict):
        generated_summary = {
            "eligible_for_automatic_recipe_attempt": generated_recipe.get("eligible_for_automatic_recipe_attempt"),
            "compile_error": generated_recipe.get("compile_error"),
            "fingerprint": generated_recipe.get("fingerprint"),
            "verified_reuse": generated_recipe.get("verified_reuse"),
        }
    matches = payload.get("foundry_catalog_matches")
    return {
        "ok": payload.get("ok"),
        "cached": payload.get("cached"),
        "cache_key": payload.get("cache_key"),
        "recipe_status": payload.get("recipe_status"),
        "recipe_reason": payload.get("recipe_reason"),
        "foundry_catalog_status": payload.get("foundry_catalog_status"),
        "foundry_catalog_matches_count": len(matches) if isinstance(matches, list) else None,
        "warnings": payload.get("warnings"),
        "tools": tools,
        "blockers": blockers,
        "generated_recipe": generated_summary,
    }


def _compact_job(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"raw": payload}
    events_compact: list[dict[str, Any]] = []
    events = payload.get("events")
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, dict):
                continue
            events_compact.append(
                {
                    "sequence": event.get("sequence"),
                    "state": event.get("state"),
                    "timestamp_utc": event.get("timestamp_utc"),
                    "message": event.get("message"),
                }
            )
    validations_compact: list[dict[str, Any]] = []
    validations = payload.get("validations")
    if isinstance(validations, list):
        for row in validations:
            if not isinstance(row, dict):
                continue
            failure = row.get("failure")
            compact_failure = (
                {
                    "stage": failure.get("stage"),
                    "classification": failure.get("classification"),
                    "message": failure.get("message"),
                }
                if isinstance(failure, dict)
                else None
            )
            validations_compact.append(
                {
                    "stage": row.get("stage"),
                    "status": row.get("status"),
                    "checks": row.get("checks"),
                    "failure": compact_failure,
                }
            )
    artifacts_compact: list[dict[str, Any]] = []
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, list):
        for row in artifacts:
            if not isinstance(row, dict):
                continue
            artifacts_compact.append(
                {
                    "artifact_id": row.get("artifact_id"),
                    "kind": row.get("kind"),
                    "description": row.get("description"),
                }
            )
    failure = payload.get("failure")
    compact_failure = (
        {
            "stage": failure.get("stage"),
            "classification": failure.get("classification"),
            "message": failure.get("message"),
        }
        if isinstance(failure, dict)
        else None
    )
    return {
        "job_id": payload.get("job_id"),
        "state": payload.get("state"),
        "started_utc": payload.get("started_utc"),
        "finished_utc": payload.get("finished_utc"),
        "result_artifact_id": payload.get("result_artifact_id"),
        "failure": compact_failure,
        "events": events_compact,
        "validations": validations_compact,
        "artifacts": artifacts_compact,
    }


def _compact_attempt(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"raw": payload}
    gates: list[dict[str, Any]] = []
    for gate in payload.get("gates", []) if isinstance(payload.get("gates"), list) else []:
        if not isinstance(gate, dict):
            continue
        gates.append(
            {
                "sequence": gate.get("sequence"),
                "gate": gate.get("gate"),
                "status": gate.get("status"),
                "evidence_ref": gate.get("evidence_ref"),
                "metrics_ref": gate.get("metrics_ref"),
                "started_utc": gate.get("started_utc"),
                "finished_utc": gate.get("finished_utc"),
            }
        )
    failure = payload.get("failure")
    compact_failure = (
        {
            "classification": failure.get("classification"),
            "stage": failure.get("stage"),
            "message": failure.get("message"),
            "evidence_refs": failure.get("evidence_refs"),
            "source_owner": failure.get("source_owner"),
            "next_action": failure.get("next_action"),
        }
        if isinstance(failure, dict)
        else None
    )
    return {
        "attempt_id": payload.get("attempt_id"),
        "state": payload.get("state"),
        "build_job_id": payload.get("build_job_id"),
        "idempotency_key": payload.get("idempotency_key"),
        "request_fingerprint": payload.get("request_fingerprint"),
        "recipe_fingerprint": payload.get("recipe_fingerprint"),
        "model_id": payload.get("model_id"),
        "revision_sha": payload.get("revision_sha"),
        "requested_device": payload.get("requested_device"),
        "requested_precision": payload.get("requested_precision"),
        "compiler_version": payload.get("compiler_version"),
        "capability_fingerprint": payload.get("capability_fingerprint"),
        "toolchain_fingerprint": payload.get("toolchain_fingerprint"),
        "profile_fingerprint": payload.get("profile_fingerprint"),
        "created_utc": payload.get("created_utc"),
        "finished_utc": payload.get("finished_utc"),
        "gates": gates,
        "failure": compact_failure,
    }


def _load_generated_record(db_path: Path, recipe_fingerprint: str) -> dict[str, Any] | None:
    if not db_path.exists():
        return None
    with sqlite3.connect(str(db_path)) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT
                recipe_fingerprint,
                schema_version,
                recipe_status,
                model_id,
                revision_sha,
                requested_device,
                requested_precision,
                compiler_version,
                capability_fingerprint,
                toolchain_fingerprint,
                profile_fingerprint,
                canonical_json,
                created_utc
            FROM generated_recipes
            WHERE recipe_fingerprint = ?
            """,
            (recipe_fingerprint,),
        ).fetchone()
    if row is None:
        return None
    canonical_json = str(row["canonical_json"])
    canonical_payload = json.loads(canonical_json)
    return {
        "recipe_fingerprint": row["recipe_fingerprint"],
        "schema_version": row["schema_version"],
        "recipe_status": row["recipe_status"],
        "model_id": row["model_id"],
        "revision_sha": row["revision_sha"],
        "requested_device": row["requested_device"],
        "requested_precision": row["requested_precision"],
        "compiler_version": row["compiler_version"],
        "capability_fingerprint": row["capability_fingerprint"],
        "toolchain_fingerprint": row["toolchain_fingerprint"],
        "profile_fingerprint": row["profile_fingerprint"],
        "created_utc": row["created_utc"],
        "canonical_payload_sha256": hashlib.sha256(canonical_json.encode("utf-8")).hexdigest(),
        "canonical_payload": canonical_payload,
    }


def _replacement_pairs(
    paths: RunPaths,
    *,
    python_runtime: dict[str, Any] | None = None,
) -> tuple[tuple[str, str], ...]:
    root_placeholder = f"scratch://round-1/{paths.run_id}"
    replacements: list[tuple[str, str]] = [
        (str(paths.runtime_root), root_placeholder),
        (str(paths.state_root), f"{root_placeholder}/state"),
        (str(paths.workspace_base), f"{root_placeholder}/workspace"),
        (str(paths.model_cache_dir), f"{root_placeholder}/cache"),
        (str(paths.service_db_path), f"{root_placeholder}/state/service.sqlite3"),
        (str(paths.recipe_attempt_db_path), f"{root_placeholder}/state/recipe-attempts.sqlite3"),
        (str(paths.repo_root), "<repo-root>"),
    ]
    if isinstance(python_runtime, dict):
        python_exe = python_runtime.get("python_exe")
        scripts_dir = python_runtime.get("scripts_dir")
        command_paths = python_runtime.get("command_paths")
        if isinstance(python_exe, Path):
            replacements.append((str(python_exe), "<python-exe>"))
        if isinstance(scripts_dir, Path):
            replacements.append((str(scripts_dir), "<python-scripts-dir>"))
        if isinstance(command_paths, dict):
            for tool_name, tool_path in command_paths.items():
                if isinstance(tool_path, Path):
                    replacements.append((str(tool_path), f"<{tool_name}-exe>"))
    return tuple(replacements)


def _sanitize_text(value: str, replacements: tuple[tuple[str, str], ...]) -> str:
    out = value
    for src, dst in replacements:
        src_backslash = src.replace("/", "\\")
        src_forward = src.replace("\\", "/")
        out = out.replace(src_backslash, dst).replace(src_forward, dst).replace(src, dst)
    out = WINDOWS_ABSOLUTE_PATH_RE.sub("<redacted-absolute-path>", out)
    return out


def _sanitize_obj(value: Any, replacements: tuple[tuple[str, str], ...]) -> Any:
    if isinstance(value, str):
        return _sanitize_text(value, replacements)
    if isinstance(value, list):
        return [_sanitize_obj(item, replacements) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_obj(item, replacements) for item in value]
    if isinstance(value, dict):
        return {str(key): _sanitize_obj(item, replacements) for key, item in value.items()}
    return value


@contextmanager
def _temporary_path(path_value: str):
    original = os.environ.get("PATH")
    os.environ["PATH"] = path_value
    try:
        yield
    finally:
        if original is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = original


def _lingering_processes_for_runtime(runtime_root: Path) -> list[dict[str, Any]]:
    command = (
        "$root = "
        + json.dumps(str(runtime_root))
        + ";"
        + "$rows = Get-CimInstance Win32_Process | Where-Object { "
        + "$_.ProcessId -ne $PID -and $_.Name -ne 'powershell.exe' -and $_.Name -ne 'pwsh.exe' -and "
        + "$_.CommandLine -like ('*' + $root + '*') "
        + "} | Select-Object ProcessId, Name, CommandLine;"
        + "if ($rows) { $rows | ConvertTo-Json -Compress } else { '[]' }"
    )
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return [
            {
                "error": "process-scan-failed",
                "exit_code": completed.returncode,
                "stderr": completed.stderr.strip(),
            }
        ]
    payload = completed.stdout.strip() or "[]"
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return [{"error": "process-scan-json-parse-failed", "raw": payload}]
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return [{"error": "process-scan-unexpected-shape", "raw": parsed}]
    out: list[dict[str, Any]] = []
    for row in parsed:
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "process_id": row.get("ProcessId"),
                "name": row.get("Name"),
                "command_line": row.get("CommandLine"),
            }
        )
    return out


def _first_non_pass_gate(gates: list[dict[str, Any]]) -> dict[str, Any] | None:
    for gate in gates:
        status = str(gate.get("status", "")).lower()
        if status != "passed":
            return gate
    return None


def _prior_successful_gates(gates: list[dict[str, Any]]) -> list[str]:
    successful: list[str] = []
    for gate in gates:
        status = str(gate.get("status", "")).lower()
        name = str(gate.get("gate", ""))
        if status == "passed":
            successful.append(name)
            continue
        break
    return successful


def _extract_failure_summary(model_result: dict[str, Any]) -> dict[str, Any]:
    attempt = model_result.get("attempt")
    job = model_result.get("job")
    attempt_create = model_result.get("attempt_create")
    preflight = model_result.get("preflight")
    preflight_recipe_status = (
        preflight.get("recipe_status") if isinstance(preflight, dict) else None
    )
    preflight_unregistered_expected = bool(
        model_result.get("preflight_recipe_unregistered_expected")
    )

    attempt_state = str(attempt.get("state")) if isinstance(attempt, dict) else "not_started"
    if attempt_state == "succeeded":
        return {
            "attempt_state": attempt_state,
            "first_failed_stage": None,
            "first_failed_classification": None,
            "error_signature": None,
            "prior_successful_gates": _prior_successful_gates(attempt.get("gates", [])),
            "source_owner": None,
            "next_action": None,
            "preflight_recipe_status": preflight_recipe_status,
            "preflight_recipe_unregistered_expected": preflight_unregistered_expected,
        }

    gates = attempt.get("gates", []) if isinstance(attempt, dict) else []
    first_failed_gate = _first_non_pass_gate(gates if isinstance(gates, list) else [])
    attempt_failure = attempt.get("failure") if isinstance(attempt, dict) else None
    if isinstance(attempt_failure, dict):
        return {
            "attempt_state": attempt_state,
            "first_failed_stage": attempt_failure.get("stage"),
            "first_failed_classification": attempt_failure.get("classification"),
            "error_signature": attempt_failure.get("message"),
            "prior_successful_gates": _prior_successful_gates(gates if isinstance(gates, list) else []),
            "first_failed_gate": first_failed_gate,
            "source_owner": attempt_failure.get("source_owner"),
            "next_action": attempt_failure.get("next_action"),
            "evidence_refs": attempt_failure.get("evidence_refs"),
            "preflight_recipe_status": preflight_recipe_status,
            "preflight_recipe_unregistered_expected": preflight_unregistered_expected,
        }

    job_failure = job.get("failure") if isinstance(job, dict) else None
    if isinstance(job_failure, dict):
        return {
            "attempt_state": attempt_state,
            "first_failed_stage": job_failure.get("stage"),
            "first_failed_classification": job_failure.get("classification"),
            "error_signature": job_failure.get("message"),
            "prior_successful_gates": _prior_successful_gates(gates if isinstance(gates, list) else []),
            "first_failed_gate": first_failed_gate,
            "source_owner": "fl-onboarding",
            "next_action": "Inspect the failed build stage and rerun with a fresh generated-attempt idempotency key.",
            "evidence_refs": [f"job://{job.get('job_id')}"] if isinstance(job, dict) and job.get("job_id") else [],
            "preflight_recipe_status": preflight_recipe_status,
            "preflight_recipe_unregistered_expected": preflight_unregistered_expected,
        }

    if isinstance(attempt_create, dict):
        payload = attempt_create.get("payload")
        if isinstance(payload, dict):
            return {
                "attempt_state": attempt_state,
                "first_failed_stage": "attempt_create",
                "first_failed_classification": payload.get("code", f"http_{attempt_create.get('status_code')}"),
                "error_signature": payload.get("message") or payload,
                "prior_successful_gates": [],
                "source_owner": "fl-onboarding",
                "next_action": "Resolve API creation failure and retry explicit-confirm generated attempt.",
                "preflight_recipe_status": preflight_recipe_status,
                "preflight_recipe_unregistered_expected": preflight_unregistered_expected,
            }
    return {
        "attempt_state": attempt_state,
        "first_failed_stage": "unknown",
        "first_failed_classification": "unknown",
        "error_signature": "No terminal attempt or job failure payload available.",
        "prior_successful_gates": [],
        "source_owner": "fl-onboarding",
        "next_action": "Collect additional service diagnostics and retry.",
        "preflight_recipe_status": preflight_recipe_status,
        "preflight_recipe_unregistered_expected": preflight_unregistered_expected,
    }


def _load_quality_profile(repo_root: Path) -> dict[str, Any]:
    profile_path = repo_root / "config" / "quality-validation-profiles.json"
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    profiles = payload.get("profiles")
    if not isinstance(profiles, list):
        return {"profile_id": None}
    profile = next((row for row in profiles if isinstance(row, dict) and row.get("profile_id") == "textgen-basic-quality-v1"), None)
    if not isinstance(profile, dict):
        return {"profile_id": None}
    deterministic = profile.get("deterministic_inference")
    prompt_count = len(profile.get("prompts", [])) if isinstance(profile.get("prompts"), list) else None
    return {
        "profile_id": profile.get("profile_id"),
        "version": profile.get("version"),
        "task": profile.get("task"),
        "deterministic_inference": deterministic if isinstance(deterministic, dict) else None,
        "unsupported_determinism_fields_reported_by_runtime": ["temperature", "seed"],
        "prompt_count": prompt_count,
    }


def _run_one_model(
    *,
    paths: RunPaths,
    model_entry: dict[str, Any],
    model_index: int,
    model_timeout_seconds: int,
    poll_seconds: int,
) -> dict[str, Any]:
    model_id = str(model_entry["model_id"])
    result: dict[str, Any] = {
        "model_index": model_index,
        "model_id": model_id,
        "manifest_sha": model_entry.get("sha"),
        "started_utc": _now_utc_iso(),
        "resource_before": _snapshot_paths(paths),
        "manifest_catalog_match": model_entry.get("catalog_match"),
        "manifest_recipe_exists": model_entry.get("recipe_exists"),
    }

    service: LocalOnboardingService | None = None
    try:
        service = LocalOnboardingService(
            db_path=paths.service_db_path,
            workspace_base=paths.workspace_base,
            model_cache_dir=paths.model_cache_dir,
            enable_production_runner=True,
        )
        with TestClient(create_app(service=service)) as client:
            detail_status, detail_payload = _request(client, "GET", "/api/models/detail", params={"id": model_id})
            result["model_detail_http_status"] = detail_status
            result["model_detail"] = _compact_model_detail(detail_payload)

            preflight_status, preflight_payload = _request(
                client,
                "POST",
                "/api/models/preflight",
                json={"model_id": model_id, "task": "llm", "task_profile": "default"},
            )
            result["preflight_http_status"] = preflight_status
            result["preflight"] = _compact_preflight(preflight_payload)
            result["preflight_recipe_unregistered_expected"] = _expected_unregistered_recipe_preflight(
                preflight_payload
            )

            preview_status, preview_payload = _request(
                client,
                "GET",
                "/api/recipes/generated/preview",
                params={"id": model_id, "task": "llm"},
            )
            result["preview_http_status"] = preview_status
            if isinstance(preview_payload, dict):
                generated_recipe = preview_payload.get("generated_recipe")
            else:
                generated_recipe = None
            result["generated_preview"] = {
                "model_id": preview_payload.get("model_id") if isinstance(preview_payload, dict) else model_id,
                "sha": preview_payload.get("sha") if isinstance(preview_payload, dict) else None,
                "foundry_catalog_status": (
                    preview_payload.get("foundry_catalog_status") if isinstance(preview_payload, dict) else None
                ),
                "foundry_catalog_matches_count": (
                    len(preview_payload.get("foundry_catalog_matches", []))
                    if isinstance(preview_payload, dict) and isinstance(preview_payload.get("foundry_catalog_matches"), list)
                    else None
                ),
                "generated_recipe": generated_recipe,
            }

            fingerprint = (
                str(generated_recipe.get("fingerprint"))
                if isinstance(generated_recipe, dict) and isinstance(generated_recipe.get("fingerprint"), str)
                else None
            )
            if fingerprint:
                result["generated_recipe_record"] = _load_generated_record(paths.recipe_attempt_db_path, fingerprint)
            else:
                result["generated_recipe_record"] = None

            if not fingerprint:
                result["attempt_create"] = {
                    "status_code": 0,
                    "payload": {
                        "code": "GENERATED_RECIPE_FINGERPRINT_MISSING",
                        "message": "Generated preview did not produce a recipe fingerprint.",
                    },
                }
            else:
                idem_key = f"{paths.run_id}-{model_index:02d}"
                create_status, create_payload = _request(
                    client,
                    "POST",
                    "/api/recipes/generated/attempts",
                    headers={"Idempotency-Key": idem_key},
                    json={
                        "model_id": model_id,
                        "recipe_fingerprint": fingerprint,
                        "confirm_automatic_recipe_attempt": True,
                    },
                )
                result["attempt_create"] = {
                    "status_code": create_status,
                    "payload": create_payload,
                }
                if create_status == 200 and isinstance(create_payload, dict):
                    job_info = create_payload.get("job")
                    attempt_info = create_payload.get("attempt")
                    if isinstance(job_info, dict):
                        job_id = str(job_info.get("job_id"))
                    else:
                        job_id = ""
                    if isinstance(attempt_info, dict):
                        attempt_id = str(attempt_info.get("attempt_id"))
                    else:
                        attempt_id = ""
                    if job_id:
                        job_payload, timed_out, cancel_result = _poll_terminal_job(
                            client,
                            job_id=job_id,
                            timeout_seconds=model_timeout_seconds,
                            poll_seconds=poll_seconds,
                        )
                        result["job_poll_timed_out"] = timed_out
                        if cancel_result is not None:
                            result["timeout_cancel"] = cancel_result
                        result["job"] = _compact_job(job_payload)

                        events_status, events_payload = _request(
                            client,
                            "GET",
                            f"/api/builds/{job_id}/events",
                            params={"after": 0},
                        )
                        result["events_http_status"] = events_status
                        if isinstance(events_payload, dict):
                            events_rows = events_payload.get("events")
                            result["event_count"] = len(events_rows) if isinstance(events_rows, list) else None
                        else:
                            result["event_count"] = None
                    else:
                        result["job"] = {
                            "state": "failed",
                            "failure": {
                                "stage": "attempt_create",
                                "classification": "invalid-response",
                                "message": "Attempt create response did not include a job id.",
                            },
                        }

                    if attempt_id:
                        attempt_status, attempt_payload = _request(
                            client,
                            "GET",
                            f"/api/recipes/generated/attempts/{attempt_id}",
                        )
                        result["attempt_http_status"] = attempt_status
                        result["attempt"] = _compact_attempt(attempt_payload)
                    else:
                        result["attempt"] = {
                            "state": "failed",
                            "failure": {
                                "classification": "invalid-response",
                                "stage": "attempt_create",
                                "message": "Attempt create response did not include attempt id.",
                            },
                            "gates": [],
                        }
                else:
                    result["job"] = None
                    result["attempt"] = None
    finally:
        if service is not None:
            try:
                service.close()
            except Exception as exc:
                result["service_close_error"] = str(exc)

    result["lingering_processes_after_close"] = _lingering_processes_for_runtime(paths.runtime_root)
    result["resource_after"] = _snapshot_paths(paths)
    result["finished_utc"] = _now_utc_iso()
    failure_summary = _extract_failure_summary(result)
    result["failure_summary"] = failure_summary
    return result


def _run_reuse_checks(paths: RunPaths, successful_model_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not successful_model_results:
        return []
    checks: list[dict[str, Any]] = []
    service = LocalOnboardingService(
        db_path=paths.service_db_path,
        workspace_base=paths.workspace_base,
        model_cache_dir=paths.model_cache_dir,
        enable_production_runner=True,
    )
    try:
        with TestClient(create_app(service=service)) as client:
            for row in successful_model_results:
                model_id = str(row.get("model_id"))
                original_attempt = row.get("attempt") if isinstance(row.get("attempt"), dict) else {}
                original_attempt_id = original_attempt.get("attempt_id")
                original_fingerprint = original_attempt.get("recipe_fingerprint")
                status, payload = _request(
                    client,
                    "GET",
                    "/api/recipes/generated/preview",
                    params={"id": model_id, "task": "llm"},
                )
                preview = payload.get("generated_recipe") if isinstance(payload, dict) else None
                verified_reuse = preview.get("verified_reuse") if isinstance(preview, dict) else None
                checks.append(
                    {
                        "model_id": model_id,
                        "http_status": status,
                        "eligible_for_automatic_recipe_attempt": (
                            preview.get("eligible_for_automatic_recipe_attempt")
                            if isinstance(preview, dict)
                            else None
                        ),
                        "verified_reuse": verified_reuse,
                        "expected_attempt_id": original_attempt_id,
                        "expected_fingerprint": original_fingerprint,
                        "reuse_identity_match": (
                            isinstance(verified_reuse, dict)
                            and verified_reuse.get("available") is True
                            and verified_reuse.get("attempt_id") == original_attempt_id
                            and verified_reuse.get("source_recipe_fingerprint") == original_fingerprint
                        ),
                    }
                )
    finally:
        try:
            service.close()
        except Exception:
            pass
    return checks


def _manifest_invariants(manifest: dict[str, Any]) -> dict[str, Any]:
    models = manifest.get("models")
    errors: list[str] = []
    if not isinstance(models, list):
        return {
            "ok": False,
            "errors": ["manifest.models must be an array."],
            "count": 0,
            "order_matches": False,
            "sha_matches": False,
            "full_sha_lengths": False,
        }
    expected_order = [row[0] for row in EXPECTED_FROZEN_MODELS]
    expected_sha = {row[0]: row[1] for row in EXPECTED_FROZEN_MODELS}
    actual_order = []
    order_matches = True
    sha_matches = True
    full_sha_lengths = True
    for index, row in enumerate(models, start=1):
        if not isinstance(row, dict):
            errors.append(f"models[{index}] must be an object.")
            continue
        model_id = str(row.get("model_id", ""))
        sha = str(row.get("sha", ""))
        actual_order.append(model_id)
        if len(sha) != 40 or re.fullmatch(r"[0-9a-f]{40}", sha.lower()) is None:
            full_sha_lengths = False
            errors.append(f"{model_id or f'model#{index}'} sha is not full 40-char lowercase hex.")
        expected_model_id = expected_order[index - 1] if index - 1 < len(expected_order) else None
        if expected_model_id != model_id:
            order_matches = False
            errors.append(
                f"models[{index}] expected {expected_model_id!r}, found {model_id!r}."
            )
        expected_model_sha = expected_sha.get(model_id)
        if expected_model_sha is None or expected_model_sha != sha:
            sha_matches = False
            errors.append(
                f"{model_id}: expected sha {expected_model_sha!r}, found {sha!r}."
            )
    if len(models) != 5:
        errors.append(f"models must contain exactly 5 entries; found {len(models)}.")
    return {
        "ok": not errors,
        "errors": errors,
        "count": len(models),
        "actual_order": actual_order,
        "expected_order": expected_order,
        "order_matches": order_matches,
        "sha_matches": sha_matches,
        "full_sha_lengths": full_sha_lengths,
    }


def _required_environment_blockers(toolchain_probe: dict[str, Any]) -> list[str]:
    missing = toolchain_probe.get("missing_required")
    if isinstance(missing, list):
        out = [str(item) for item in missing if str(item).strip()]
        if out:
            return out
    probes = toolchain_probe.get("probes")
    if not isinstance(probes, list):
        return []
    blockers: list[str] = []
    for row in probes:
        if not isinstance(row, dict):
            continue
        if row.get("required") is True and row.get("available") is False:
            blockers.append(f"{row.get('kind')}:{row.get('name')}")
    return blockers


def _expected_unregistered_recipe_preflight(preflight_payload: Any) -> bool:
    if not isinstance(preflight_payload, dict):
        return False
    recipe_status = str(preflight_payload.get("recipe_status", "")).lower()
    generated = preflight_payload.get("generated_recipe")
    generated_eligible = (
        isinstance(generated, dict)
        and generated.get("eligible_for_automatic_recipe_attempt") is True
    )
    return recipe_status == "unregistered" and generated_eligible


def _classify_round_outcome(
    *,
    manifest_invariants: dict[str, Any],
    environment_blockers: list[str],
    model_results: list[dict[str, Any]],
    environment_fail_fast_triggered: bool,
) -> dict[str, Any]:
    total_count = len(model_results)
    success_count = sum(
        1
        for row in model_results
        if row.get("failure_summary", {}).get("attempt_state") == "succeeded"
    )
    attempt_success_rate = f"{success_count}/{total_count}"
    if not bool(manifest_invariants.get("ok")):
        return {
            "round_classification": "invalid_manifest",
            "baseline_valid": False,
            "model_success_rate_applicable": False,
            "success_rate": "not_applicable",
            "attempt_success_rate": attempt_success_rate,
            "environment_blockers": [],
            "environment_fail_fast_triggered": False,
        }
    if environment_blockers:
        return {
            "round_classification": "invalid_environment",
            "baseline_valid": False,
            "model_success_rate_applicable": False,
            "success_rate": "not_applicable",
            "attempt_success_rate": attempt_success_rate,
            "environment_blockers": environment_blockers,
            "environment_fail_fast_triggered": environment_fail_fast_triggered,
        }
    return {
        "round_classification": "valid_baseline",
        "baseline_valid": True,
        "model_success_rate_applicable": True,
        "success_rate": attempt_success_rate,
        "attempt_success_rate": attempt_success_rate,
        "environment_blockers": [],
        "environment_fail_fast_triggered": False,
    }


def _make_environment_failfast_result(
    *,
    paths: RunPaths,
    model_entry: dict[str, Any],
    model_index: int,
    environment_blockers: list[str],
) -> dict[str, Any]:
    model_id = str(model_entry.get("model_id", f"invalid-model-entry-{model_index}"))
    now = _now_utc_iso()
    message = (
        "Round fail-fast before attempts because required shared tools are unavailable: "
        + ", ".join(environment_blockers)
    )
    return {
        "model_index": model_index,
        "model_id": model_id,
        "manifest_sha": model_entry.get("sha"),
        "started_utc": now,
        "finished_utc": now,
        "resource_before": _snapshot_paths(paths),
        "resource_after": _snapshot_paths(paths),
        "manifest_catalog_match": model_entry.get("catalog_match"),
        "manifest_recipe_exists": model_entry.get("recipe_exists"),
        "preflight_recipe_unregistered_expected": None,
        "failure_summary": {
            "attempt_state": "not_attempted",
            "first_failed_stage": "round_environment_preflight",
            "first_failed_classification": "invalid_environment",
            "error_signature": message,
            "prior_successful_gates": [],
            "source_owner": "fl-onboarding",
            "next_action": "Install/resolve required toolchain commands in the selected Python environment and rerun.",
            "environment_blockers": list(environment_blockers),
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _format_bytes(value: int) -> str:
    gib = value / (1024**3)
    return f"{gib:.3f} GiB"


def _build_report(
    *,
    run_paths: RunPaths,
    git_head: str,
    git_branch: str,
    started_utc: str,
    finished_utc: str,
    manifest_invariants: dict[str, Any],
    frozen_validate: dict[str, Any],
    frozen_dry_run: dict[str, Any],
    quality_profile: dict[str, Any],
    round_outcome: dict[str, Any],
    model_results: list[dict[str, Any]],
    reuse_checks: list[dict[str, Any]],
) -> str:
    model_success_rate = str(round_outcome.get("success_rate", "not_applicable"))
    attempt_success_rate = str(round_outcome.get("attempt_success_rate", "0/0"))
    lines: list[str] = []
    lines.append("# Recipe Agent v1 Round 1 Report")
    lines.append("")
    lines.append(f"- **Run ID:** `{run_paths.run_id}`")
    lines.append(f"- **Branch:** `{git_branch}`")
    lines.append(f"- **Commit:** `{git_head}`")
    lines.append(f"- **Window (UTC):** `{started_utc}` -> `{finished_utc}`")
    lines.append(f"- **Round classification:** `{round_outcome.get('round_classification')}`")
    lines.append(f"- **Baseline valid evidence:** `{round_outcome.get('baseline_valid')}`")
    lines.append(
        "- **Model success rate:** "
        + (f"**{model_success_rate}**" if round_outcome.get("model_success_rate_applicable") else "**not_applicable**")
    )
    if not round_outcome.get("model_success_rate_applicable"):
        lines.append(f"- **Raw attempt outcome (diagnostic only):** `{attempt_success_rate}`")
    lines.append(f"- **Retained external evidence root:** `scratch://round-1/{run_paths.run_id}`")
    lines.append("")
    lines.append("## Frozen manifest and deterministic checks")
    lines.append("")
    lines.append(f"- Manifest invariants pass: **{manifest_invariants.get('ok')}**")
    lines.append(f"- `recipe-agent frozen-validate` exit code: **{frozen_validate.get('exit_code')}**")
    lines.append(f"- `recipe-agent frozen-dry-run` exit code: **{frozen_dry_run.get('exit_code')}**")
    environment_blockers = round_outcome.get("environment_blockers") or []
    if environment_blockers:
        lines.append("- Round-level environment blockers: `" + ", ".join(str(item) for item in environment_blockers) + "`")
        lines.append(
            f"- Round fail-fast before attempts triggered: `{round_outcome.get('environment_fail_fast_triggered')}`"
        )
    lines.append("")
    lines.append("## Deterministic quality profile snapshot")
    lines.append("")
    lines.append(
        f"- Profile: `{quality_profile.get('profile_id')}` v`{quality_profile.get('version')}` for task `{quality_profile.get('task')}`"
    )
    lines.append(f"- Deterministic inference config: `{json.dumps(quality_profile.get('deterministic_inference'))}`")
    lines.append(
        "- Runtime-reported unsupported deterministic fields: "
        f"`{', '.join(quality_profile.get('unsupported_determinism_fields_reported_by_runtime', []))}`"
    )
    lines.append("")
    lines.append("## Per-model outcomes")
    lines.append("")
    lines.append("| Model | Attempt state | First failed stage | Classification | Prior passed gates |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in model_results:
        summary = row.get("failure_summary", {})
        prior = summary.get("prior_successful_gates") or []
        prior_text = ", ".join(str(item) for item in prior) if prior else "-"
        lines.append(
            "| "
            + f"{row.get('model_id')} | {summary.get('attempt_state')} | {summary.get('first_failed_stage') or '-'} "
            + f"| {summary.get('first_failed_classification') or '-'} | {prior_text} |"
        )
    lines.append("")
    failed_rows = [row for row in model_results if row.get("failure_summary", {}).get("attempt_state") != "succeeded"]
    if failed_rows:
        lines.append("## Failure analysis")
        lines.append("")
        for index, row in enumerate(failed_rows, start=1):
            summary = row.get("failure_summary", {})
            attempt = row.get("attempt")
            gates = attempt.get("gates", []) if isinstance(attempt, dict) else []
            first_non_pass = _first_non_pass_gate(gates if isinstance(gates, list) else [])
            lines.append(f"{index}. **{row.get('model_id')}**")
            lines.append(f"   - First failed stage/classification: `{summary.get('first_failed_stage')}` / `{summary.get('first_failed_classification')}`")
            lines.append(f"   - Error signature: `{summary.get('error_signature')}`")
            if summary.get("preflight_recipe_unregistered_expected"):
                lines.append(
                    "   - Preflight recipe status: `unregistered` (expected in generated-recipe preflight; not an environment blocker)."
                )
            if first_non_pass is not None:
                lines.append(
                    f"   - First failed gate: `{first_non_pass.get('gate')}` with status `{first_non_pass.get('status')}`"
                )
            lines.append(f"   - Source owner: `{summary.get('source_owner')}`")
            lines.append(f"   - Next action: {summary.get('next_action')}")
            evidence_refs = summary.get("evidence_refs") or []
            if evidence_refs:
                lines.append(f"   - Evidence refs: `{', '.join(str(item) for item in evidence_refs)}`")
            record = row.get("generated_recipe_record")
            if isinstance(record, dict):
                lines.append(
                    "   - Generated recipe provenance: "
                    + f"fingerprint `{record.get('recipe_fingerprint')}`, capability `{record.get('capability_fingerprint')}`, "
                    + f"toolchain `{record.get('toolchain_fingerprint')}`, profile `{record.get('profile_fingerprint')}`"
                )
        lines.append("")
    lines.append("## Verified recipe reuse re-check (no rebuild)")
    lines.append("")
    if not reuse_checks:
        lines.append("- No successful models to re-check for verified reuse.")
    else:
        for row in reuse_checks:
            lines.append(
                "- "
                + f"{row.get('model_id')}: reuse identity match = **{row.get('reuse_identity_match')}** "
                + f"(attempt `{row.get('expected_attempt_id')}`)"
            )
    lines.append("")
    lines.append("## Scratch retention")
    lines.append("")
    final_runtime_bytes = _dir_size_bytes(run_paths.runtime_root)
    final_cache_bytes = _dir_size_bytes(run_paths.model_cache_dir)
    final_workspace_bytes = _dir_size_bytes(run_paths.workspace_base)
    final_state_bytes = _dir_size_bytes(run_paths.state_root)
    lines.append(f"- Runtime root retained size: **{_format_bytes(final_runtime_bytes)}**")
    lines.append(f"- Cache retained size: **{_format_bytes(final_cache_bytes)}**")
    lines.append(f"- Workspace retained size: **{_format_bytes(final_workspace_bytes)}**")
    lines.append(f"- State retained size: **{_format_bytes(final_state_bytes)}**")
    lines.append("- Paths represented in committed artifacts by `scratch://round-1/<run_id>/...` placeholders.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Recipe Agent v1 Round 1 baseline over frozen models.")
    parser.add_argument(
        "--scratch-root",
        default=r"C:\fmo-recipe-v1\round-1",
        help="Short external scratch root for runtime/cache/state data.",
    )
    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python executable used for CLI invariant checks.",
    )
    parser.add_argument(
        "--model-timeout-seconds",
        type=int,
        default=14400,
        help="Per-model terminal poll timeout in seconds.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=10,
        help="Polling interval in seconds while waiting for build terminal states.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional explicit run id. Defaults to UTC timestamp run id.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory. Defaults to evaluation/recipe-agent-v1/round-1.",
    )
    args = parser.parse_args()

    script_root = Path(__file__).resolve().parent
    output_root = Path(args.output_dir).resolve() if args.output_dir.strip() else script_root
    repo_root = script_root.parents[2]
    manifest_path = script_root.parent / "models.json"
    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("r1-%Y%m%dT%H%M%SZ")

    scratch_root = Path(args.scratch_root).resolve()
    runtime_root = (scratch_root / run_id).resolve()
    state_root = (runtime_root / "state").resolve()
    workspace_base = (runtime_root / "workspace").resolve()
    model_cache_dir = (runtime_root / "cache").resolve()
    service_db_path = (state_root / "service.sqlite3").resolve()
    recipe_attempt_db_path = (state_root / "recipe-attempts.sqlite3").resolve()

    for directory in (scratch_root, runtime_root, state_root, workspace_base, model_cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    paths = RunPaths(
        repo_root=repo_root,
        output_root=output_root,
        manifest_path=manifest_path,
        scratch_root=scratch_root,
        runtime_root=runtime_root,
        state_root=state_root,
        workspace_base=workspace_base,
        model_cache_dir=model_cache_dir,
        service_db_path=service_db_path,
        recipe_attempt_db_path=recipe_attempt_db_path,
        run_id=run_id,
    )
    output_root.mkdir(parents=True, exist_ok=True)

    selected_python = Path(args.python_exe).resolve()
    python_runtime = _resolve_python_runtime(selected_python)
    replacements = _replacement_pairs(paths, python_runtime=python_runtime)

    started_utc = _now_utc_iso()
    git_head = _git_value(paths.repo_root, ["rev-parse", "HEAD"])
    git_branch = _git_value(paths.repo_root, ["rev-parse", "--abbrev-ref", "HEAD"])

    manifest_raw = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_raw, dict):
        raise SystemExit("Frozen manifest must be a JSON object.")
    manifest_invariants = _manifest_invariants(manifest_raw)
    models = manifest_raw.get("models")
    if not isinstance(models, list):
        raise SystemExit("Frozen manifest models must be a list.")

    per_model_results: list[dict[str, Any]] = []
    child_env = python_runtime.get("child_env")
    if not isinstance(child_env, dict):
        raise RuntimeError("Resolved Python runtime did not provide a child environment.")
    child_path = str(child_env.get("PATH", ""))

    with _temporary_path(child_path):
        toolchain_probe = _probe_toolchain(
            selected_python,
            runtime=python_runtime,
            env=child_env,
        )
        frozen_list = _run_cli_json(
            selected_python,
            ["recipe-agent", "frozen-list", "--path", str(paths.manifest_path)],
            env=child_env,
        )
        frozen_validate = _run_cli_json(
            selected_python,
            ["recipe-agent", "frozen-validate", "--path", str(paths.manifest_path)],
            env=child_env,
        )
        frozen_dry_run = _run_cli_json(
            selected_python,
            ["recipe-agent", "frozen-dry-run", "--path", str(paths.manifest_path)],
            env=child_env,
        )

        environment_blockers = _required_environment_blockers(toolchain_probe)
        environment_fail_fast_triggered = False
        if environment_blockers:
            environment_fail_fast_triggered = True
            for index, model_entry in enumerate(models, start=1):
                if not isinstance(model_entry, dict):
                    per_model_results.append(
                        {
                            "model_index": index,
                            "model_id": f"invalid-model-entry-{index}",
                            "failure_summary": {
                                "attempt_state": "failed",
                                "first_failed_stage": "manifest-parse",
                                "first_failed_classification": "invalid-manifest-entry",
                                "error_signature": "Manifest entry was not a JSON object.",
                                "prior_successful_gates": [],
                                "source_owner": "fl-onboarding",
                                "next_action": "Fix frozen manifest row shape and rerun.",
                            },
                        }
                    )
                    continue
                per_model_results.append(
                    _make_environment_failfast_result(
                        paths=paths,
                        model_entry=model_entry,
                        model_index=index,
                        environment_blockers=environment_blockers,
                    )
                )
        else:
            for index, model_entry in enumerate(models, start=1):
                if not isinstance(model_entry, dict):
                    per_model_results.append(
                        {
                            "model_index": index,
                            "model_id": f"invalid-model-entry-{index}",
                            "failure_summary": {
                                "attempt_state": "failed",
                                "first_failed_stage": "manifest-parse",
                                "first_failed_classification": "invalid-manifest-entry",
                                "error_signature": "Manifest entry was not a JSON object.",
                                "prior_successful_gates": [],
                                "source_owner": "fl-onboarding",
                                "next_action": "Fix frozen manifest row shape and rerun.",
                            },
                        }
                    )
                    continue
                per_model_results.append(
                    _run_one_model(
                        paths=paths,
                        model_entry=model_entry,
                        model_index=index,
                        model_timeout_seconds=max(60, int(args.model_timeout_seconds)),
                        poll_seconds=max(1, int(args.poll_seconds)),
                    )
                )

        successful_rows = [
            row
            for row in per_model_results
            if row.get("failure_summary", {}).get("attempt_state") == "succeeded"
        ]
        reuse_checks = _run_reuse_checks(paths, successful_rows)

    round_outcome = _classify_round_outcome(
        manifest_invariants=manifest_invariants,
        environment_blockers=_required_environment_blockers(toolchain_probe),
        model_results=per_model_results,
        environment_fail_fast_triggered=environment_fail_fast_triggered,
    )
    finished_utc = _now_utc_iso()

    final_snapshot = _snapshot_paths(paths)
    quality_profile = _load_quality_profile(paths.repo_root)

    model_results_slim: list[dict[str, Any]] = []
    for row in per_model_results:
        failure_summary = row.get("failure_summary", {})
        generated_record = row.get("generated_recipe_record")
        generated_preview = row.get("generated_preview", {})
        preflight = row.get("preflight", {})
        model_results_slim.append(
            {
                "model_index": row.get("model_index"),
                "model_id": row.get("model_id"),
                "manifest_sha": row.get("manifest_sha"),
                "attempt_state": failure_summary.get("attempt_state"),
                "first_failed_stage": failure_summary.get("first_failed_stage"),
                "first_failed_classification": failure_summary.get("first_failed_classification"),
                "error_signature": failure_summary.get("error_signature"),
                "prior_successful_gates": failure_summary.get("prior_successful_gates"),
                "source_owner": failure_summary.get("source_owner"),
                "next_action": failure_summary.get("next_action"),
                "preflight_recipe_status": failure_summary.get("preflight_recipe_status"),
                "preflight_recipe_unregistered_expected": failure_summary.get(
                    "preflight_recipe_unregistered_expected"
                ),
                "recipe_fingerprint": (
                    generated_record.get("recipe_fingerprint") if isinstance(generated_record, dict) else None
                ),
                "capability_fingerprint": (
                    generated_record.get("capability_fingerprint") if isinstance(generated_record, dict) else None
                ),
                "toolchain_fingerprint": (
                    generated_record.get("toolchain_fingerprint") if isinstance(generated_record, dict) else None
                ),
                "profile_fingerprint": (
                    generated_record.get("profile_fingerprint") if isinstance(generated_record, dict) else None
                ),
                "catalog_status": generated_preview.get("foundry_catalog_status"),
                "catalog_matches_count": generated_preview.get("foundry_catalog_matches_count"),
                "recipe_registry_status": row.get("model_detail", {}).get("recipe_status"),
                "preflight_ok": preflight.get("ok"),
                "preflight_recipe_interpretation": (
                    "expected_unregistered_generated_recipe"
                    if failure_summary.get("preflight_recipe_unregistered_expected")
                    else None
                ),
                "job_state": row.get("job", {}).get("state") if isinstance(row.get("job"), dict) else None,
                "job_id": row.get("job", {}).get("job_id") if isinstance(row.get("job"), dict) else None,
                "attempt_id": row.get("attempt", {}).get("attempt_id") if isinstance(row.get("attempt"), dict) else None,
                "event_count": row.get("event_count"),
                "resource_before": row.get("resource_before"),
                "resource_after": row.get("resource_after"),
                "lingering_process_count": (
                    len(row.get("lingering_processes_after_close", []))
                    if isinstance(row.get("lingering_processes_after_close"), list)
                    else None
                ),
            }
        )

    success_count = sum(1 for row in per_model_results if row.get("failure_summary", {}).get("attempt_state") == "succeeded")
    summary_payload: dict[str, Any] = {
        "run_id": paths.run_id,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "branch": git_branch,
        "commit": git_head,
        "models_total": len(per_model_results),
        "models_succeeded": success_count,
        "models_failed": len(per_model_results) - success_count,
        "round_classification": round_outcome.get("round_classification"),
        "baseline_valid": round_outcome.get("baseline_valid"),
        "model_success_rate_applicable": round_outcome.get("model_success_rate_applicable"),
        "success_rate": round_outcome.get("success_rate"),
        "attempt_success_rate": round_outcome.get("attempt_success_rate"),
        "environment_blockers": round_outcome.get("environment_blockers"),
        "environment_fail_fast_triggered": round_outcome.get("environment_fail_fast_triggered"),
        "results": model_results_slim,
        "reuse_checks": reuse_checks,
    }

    manifest_payload: dict[str, Any] = {
        "run_id": paths.run_id,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "branch": git_branch,
        "commit": git_head,
        "manifest_path": "evaluation/recipe-agent-v1/models.json",
        "manifest_sha256": hashlib.sha256(paths.manifest_path.read_bytes()).hexdigest(),
        "manifest_invariants": manifest_invariants,
        "round_outcome": round_outcome,
        "frozen_cli": {
            "frozen_list": frozen_list,
            "frozen_validate": frozen_validate,
            "frozen_dry_run": frozen_dry_run,
        },
        "quality_profile": quality_profile,
        "python_environment": toolchain_probe.get("python_runtime"),
        "toolchain_probe": toolchain_probe,
        "scratch_locations": {
            "root": f"scratch://round-1/{paths.run_id}",
            "state": f"scratch://round-1/{paths.run_id}/state",
            "workspace": f"scratch://round-1/{paths.run_id}/workspace",
            "cache": f"scratch://round-1/{paths.run_id}/cache",
            "service_db": f"scratch://round-1/{paths.run_id}/state/service.sqlite3",
            "recipe_attempt_db": f"scratch://round-1/{paths.run_id}/state/recipe-attempts.sqlite3",
        },
        "final_snapshot": final_snapshot,
        "model_registry_and_catalog_snapshot": [
            {
                "model_id": row.get("model_id"),
                "manifest_catalog_match": row.get("manifest_catalog_match"),
                "live_catalog_status": row.get("generated_preview", {}).get("foundry_catalog_status")
                if isinstance(row.get("generated_preview"), dict)
                else None,
                "live_catalog_matches_count": row.get("generated_preview", {}).get("foundry_catalog_matches_count")
                if isinstance(row.get("generated_preview"), dict)
                else None,
                "recipe_registry_status": row.get("model_detail", {}).get("recipe_status")
                if isinstance(row.get("model_detail"), dict)
                else None,
                "recipe_registry_reason": row.get("model_detail", {}).get("recipe_reason")
                if isinstance(row.get("model_detail"), dict)
                else None,
            }
            for row in per_model_results
        ],
        "reuse_checks": reuse_checks,
    }

    model_results_dir = paths.output_root / "model-results"
    model_results_dir.mkdir(parents=True, exist_ok=True)
    for row in per_model_results:
        model_id = str(row.get("model_id"))
        model_index = int(row.get("model_index", 0))
        filename = f"{model_index:02d}-{_slugify(model_id)}.json"
        _write_json(
            model_results_dir / filename,
            _sanitize_obj(row, replacements),
        )

    _write_json(paths.output_root / "round-manifest.json", _sanitize_obj(manifest_payload, replacements))
    _write_json(paths.output_root / "round-1-summary.json", _sanitize_obj(summary_payload, replacements))

    report = _build_report(
        run_paths=paths,
        git_head=git_head,
        git_branch=git_branch,
        started_utc=started_utc,
        finished_utc=finished_utc,
        manifest_invariants=manifest_invariants,
        frozen_validate=frozen_validate,
        frozen_dry_run=frozen_dry_run,
        quality_profile=quality_profile,
        round_outcome=_sanitize_obj(round_outcome, replacements),
        model_results=_sanitize_obj(per_model_results, replacements),
        reuse_checks=_sanitize_obj(reuse_checks, replacements),
    )
    (paths.output_root / "round-1-report.md").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
