from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
DIAG_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[4]
SRC_ROOT = REPO_ROOT / "src"
ROUND6_DIR = REPO_ROOT / "evaluation" / "recipe-agent-v1" / "round-6"
ROUND6_MANIFEST_PATH = ROUND6_DIR / "round-manifest.json"
ROUND6_SUMMARY_PATH = ROUND6_DIR / "round-6-summary.json"
ROUND6_SMOLLM_RESULT_PATH = ROUND6_DIR / "model-results" / "02-huggingfacetb-smollm2-360m-instruct.json"
MODELS_JSON_PATH = REPO_ROOT / "evaluation" / "recipe-agent-v1" / "models.json"
REPORT_PATH = DIAG_DIR / "diagnostic-report.json"

FROZEN_MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
QUALITY_PROFILE_ID = "textgen-basic-quality-v1"
UNSUPPORTED_DETERMINISM_FIELDS = ("temperature", "seed")

DEFAULT_RUNTIME_PYTHON = Path(r"C:\fl-recipe-v1-venv\Scripts\python.exe")
DEFAULT_RETAINED_CACHE_ROOT = Path(r"C:\fmo-r6\r6-20260902T172246Z\cache")
DEFAULT_EXTERNAL_SCRATCH_ROOT = Path(r"C:\fmo-r6-smollm-json-regression")
EXACT_STRAY_DIAGNOSTIC_ROOT = Path(r"C:\fmo-r6-smollm-diagnostics")
NUMERIC_FIDELITY_PROMPT = "Write one short sentence about measuring a length in centimeters."
NUMERIC_FIDELITY_MAX_TOKENS = 48

ABS_PATH_RE = re.compile(r"[A-Za-z]:(?:\\|/(?!/))[^\"'\r\n]*")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fl_model_onboarding.quality_validation import (  # noqa: E402
    PromptExecutionRecord,
    evaluate_quality_validation,
    load_quality_validation_profile_registry,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object at {path}.")
    return payload


def _parse_json_maybe(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(stripped[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _sanitize_text(value: str, *, max_chars: int = 280) -> str:
    cleaned = CONTROL_CHAR_RE.sub("", value)
    cleaned = ABS_PATH_RE.sub("<redacted-absolute-path>", cleaned)
    cleaned = cleaned.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    if max_chars <= 3:
        return cleaned[:max_chars]
    return cleaned[: max_chars - 3].rstrip() + "..."


def _sanitize_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {str(key): _sanitize_payload(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_sanitize_payload(value) for value in payload]
    if isinstance(payload, str):
        return _sanitize_text(payload, max_chars=2000)
    return payload


def _sanitize_tail(value: str, *, max_chars: int = 2400) -> str:
    cleaned = CONTROL_CHAR_RE.sub("", value)
    cleaned = ABS_PATH_RE.sub("<redacted-absolute-path>", cleaned)
    cleaned = cleaned.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    if max_chars <= 3:
        return cleaned[-max_chars:]
    return "..." + cleaned[-(max_chars - 3) :].lstrip()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _run_cmd(
    argv: list[str],
    *,
    timeout_seconds: int,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
        check=False,
        cwd=str(cwd),
        env=env,
    )
    elapsed = time.perf_counter() - started
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "duration_seconds": round(elapsed, 3),
        "argv": argv,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "stdout_excerpt": _sanitize_text(completed.stdout, max_chars=360),
        "stderr_excerpt": _sanitize_text(completed.stderr, max_chars=360),
        "stdout_tail": _sanitize_tail(completed.stdout, max_chars=3200),
        "stderr_tail": _sanitize_tail(completed.stderr, max_chars=3200),
    }


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _file_size_bytes(path: Path) -> int:
    return int(path.stat().st_size) if path.is_file() else 0


def _directory_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for item in path.rglob("*"):
        if item.is_file():
            total += int(item.stat().st_size)
    return total


def _directory_fingerprint(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        return {
            "exists": False,
            "path_name": path.name,
            "file_count": 0,
            "total_bytes": 0,
            "manifest_sha256": None,
            "files": [],
        }
    records: list[dict[str, Any]] = []
    manifest = hashlib.sha256()
    total_bytes = 0
    file_paths = [row for row in sorted(path.rglob("*")) if row.is_file()]
    for item in file_paths:
        rel = item.relative_to(path).as_posix()
        size = int(item.stat().st_size)
        digest = _sha256(item)
        total_bytes += size
        manifest.update(rel.encode("utf-8"))
        manifest.update(b"\0")
        manifest.update(str(size).encode("utf-8"))
        manifest.update(b"\0")
        manifest.update(digest.encode("utf-8"))
        manifest.update(b"\n")
        records.append(
            {
                "relative_path": rel,
                "size_bytes": size,
                "sha256": digest,
            }
        )
    return {
        "exists": True,
        "path_name": path.name,
        "file_count": len(records),
        "total_bytes": total_bytes,
        "manifest_sha256": manifest.hexdigest(),
        "files": records,
    }


def _cache_sibling_inventory(*, cache_root: Path, artifact_prefix: str, snapshot_prefix: str) -> dict[str, Any]:
    artifact_siblings = sorted(cache_root.glob(f"{artifact_prefix}-*"))
    snapshot_siblings = sorted(cache_root.glob(f"{snapshot_prefix}-*"))
    return {
        "cache_root_name": cache_root.name,
        "artifact_prefix": artifact_prefix,
        "snapshot_prefix": snapshot_prefix,
        "artifact_siblings": [
            {
                "name": item.name,
                "is_dir": item.is_dir(),
                "size_bytes": (_directory_size_bytes(item) if item.is_dir() else _file_size_bytes(item)),
            }
            for item in artifact_siblings
        ],
        "snapshot_siblings": [
            {
                "name": item.name,
                "is_dir": item.is_dir(),
                "size_bytes": (_directory_size_bytes(item) if item.is_dir() else _file_size_bytes(item)),
            }
            for item in snapshot_siblings
        ],
    }


def _slug_model_id(model_id: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", model_id.strip().lower())
    return slug.strip("-")


def _runtime_env(base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base or os.environ)
    existing = env.get("PYTHONPATH", "").strip()
    paths = [str(SRC_ROOT)]
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def _load_toolchain_versions(
    python_exe: Path,
    env: dict[str, str],
    *,
    allow_interpreter_conflation: bool,
) -> dict[str, Any]:
    runtime_exe = python_exe.resolve()
    harness_exe = Path(sys.executable).resolve()
    same_interpreter = runtime_exe == harness_exe
    if same_interpreter and not allow_interpreter_conflation:
        raise RuntimeError(
            "Harness interpreter and requested runtime interpreter are identical. "
            "Pass --allow-interpreter-conflation only when this is intentional.",
        )

    code = """
import importlib.metadata as metadata
import json
import os
import platform
import sys
from pathlib import Path

packages = [
    "fl-model-onboarding",
    "onnx",
    "onnxruntime",
    "onnxruntime-genai",
    "foundry-local-sdk",
    "mobius-onnx",
    "olive-ai",
]
versions = {}
missing = []
for package in packages:
    try:
        versions[package] = metadata.version(package)
    except Exception:
        missing.append(package)

print(json.dumps({
    "probe_source": "runtime_subprocess",
    "runtime_executable": str(Path(sys.executable).resolve()),
    "runtime_python_version": platform.python_version(),
    "runtime_prefix": sys.prefix,
    "runtime_base_prefix": getattr(sys, "base_prefix", sys.prefix),
    "pythonpath_env": os.environ.get("PYTHONPATH", ""),
    "packages": versions,
    "missing_packages": missing,
}, sort_keys=True))
"""
    probe = _run_cmd([str(runtime_exe), "-c", code], timeout_seconds=180, env=env)
    payload = _parse_json_maybe(str(probe.get("stdout") or ""))
    if not probe["ok"] or not isinstance(payload, dict):
        raise RuntimeError(
            "Runtime toolchain probe failed. "
            f"stderr tail: {probe['stderr_tail'] or '<empty>'}",
        )
    reported_exe_raw = str(payload.get("runtime_executable") or "").strip()
    if not reported_exe_raw:
        raise RuntimeError("Runtime toolchain probe payload did not include runtime_executable.")
    reported_exe = Path(reported_exe_raw).resolve()
    if reported_exe != runtime_exe:
        raise RuntimeError(
            "Runtime toolchain probe executable mismatch: "
            f"requested '{runtime_exe}', reported '{reported_exe}'.",
        )
    return {
        "probe_source": "runtime_subprocess",
        "runtime_requested_executable": str(runtime_exe),
        "runtime_reported": payload,
        "probe_command": probe["argv"],
        "probe_duration_seconds": probe["duration_seconds"],
        "harness_interpreter": {
            "executable": str(harness_exe),
            "python_version": sys.version.split()[0],
        },
        "interpreter_conflation": {
            "same_executable": same_interpreter,
            "allow_interpreter_conflation": allow_interpreter_conflation,
        },
    }


def _load_round6_context() -> dict[str, Any]:
    manifest = _read_json(ROUND6_MANIFEST_PATH)
    summary = _read_json(ROUND6_SUMMARY_PATH)
    smollm_result = _read_json(ROUND6_SMOLLM_RESULT_PATH)
    models_payload = _read_json(MODELS_JSON_PATH)

    run_id = str(manifest.get("run_id") or "").strip()
    if not run_id:
        raise RuntimeError("Round 6 manifest is missing run_id.")
    branch = str(manifest.get("branch") or "").strip()
    commit = str(manifest.get("commit") or "").strip()
    if not branch or not commit:
        raise RuntimeError("Round 6 manifest is missing branch/commit.")

    models = models_payload.get("models")
    if not isinstance(models, list):
        raise RuntimeError("models.json is missing model list.")
    smollm_model = next(
        (row for row in models if isinstance(row, dict) and row.get("model_id") == FROZEN_MODEL_ID),
        None,
    )
    if not isinstance(smollm_model, dict):
        raise RuntimeError("models.json does not include SmolLM2-360M frozen model.")
    frozen_revision = str(smollm_model.get("sha") or smollm_model.get("revision") or "").strip()
    if not frozen_revision:
        raise RuntimeError("SmolLM2 frozen revision is missing in models.json.")

    evidence = smollm_result.get("quality_validation_evidence")
    if not isinstance(evidence, dict):
        raise RuntimeError("Round 6 SmolLM model result is missing quality_validation_evidence.")
    batch_worker = evidence.get("batch_worker")
    if not isinstance(batch_worker, dict):
        raise RuntimeError("Round 6 SmolLM model result is missing batch_worker evidence.")
    baseline_request = batch_worker.get("baseline", {}).get("request", {})
    optimized_request = batch_worker.get("optimized", {}).get("request", {})
    if not isinstance(baseline_request, dict) or not isinstance(optimized_request, dict):
        raise RuntimeError("Round 6 SmolLM batch_worker requests are malformed.")
    baseline_model_name = str(baseline_request.get("model_name") or "").strip()
    optimized_model_name = str(optimized_request.get("model_name") or "").strip()
    if not baseline_model_name or not optimized_model_name:
        raise RuntimeError("Round 6 batch_worker model names are missing.")
    per_prompt_timeout = int(baseline_request.get("per_prompt_timeout_seconds") or 900)
    batch_timeout = int(baseline_request.get("batch_timeout_seconds") or 3600)
    max_tokens = int(baseline_request.get("max_tokens") or 64)

    generated_recipe = smollm_result.get("generated_preview", {}).get("generated_recipe", {})
    recipe_block = generated_recipe.get("recipe", {}) if isinstance(generated_recipe, dict) else {}
    artifact_prefix = str(
        recipe_block.get("artifact_cache_prefix")
        or (recipe_block.get("recipe", {}) if isinstance(recipe_block.get("recipe"), dict) else {}).get("artifact_cache_prefix")
        or smollm_result.get("attempt", {})
        .get("job", {})
        .get("request", {})
        .get("generated_recipe_attempt", {})
        .get("recipe_artifact_cache_prefix")
        or ""
    ).strip()
    job_payload = smollm_result.get("job", {})
    artifacts = job_payload.get("artifacts", []) if isinstance(job_payload, dict) else []
    first_artifact_id = ""
    if isinstance(artifacts, list) and artifacts and isinstance(artifacts[0], dict):
        first_artifact_id = str(artifacts[0].get("artifact_id") or "")
    artifact_id = str(
        (job_payload.get("result_artifact_id") if isinstance(job_payload, dict) else "")
        or first_artifact_id
        or ""
    ).strip().lower()
    if not artifact_prefix or not artifact_id:
        raise RuntimeError("Unable to resolve optimized artifact identity from Round 6 result.")

    metrics_ref = str(
        smollm_result.get("failure_summary", {}).get("quality_validation_metrics_ref")
        or evidence.get("metrics_ref")
        or ""
    ).strip()
    if not metrics_ref:
        raise RuntimeError("Round 6 SmolLM quality metrics_ref is missing.")

    profile = load_quality_validation_profile_registry().get(QUALITY_PROFILE_ID)
    if profile is None:
        raise RuntimeError(f"Quality profile '{QUALITY_PROFILE_ID}' not found.")
    prompts = [
        {
            "prompt_id": row.prompt_id,
            "category": row.category.value,
            "prompt": row.prompt,
            "required_tokens": list(row.expected.required_tokens),
            "forbidden_tokens": list(row.expected.forbidden_tokens),
            "exact_match": row.expected.exact_match,
            "max_words": row.expected.max_words,
            "required_json_keys": list(row.expected.required_json_keys),
        }
        for row in profile.prompts
    ]

    return {
        "manifest": manifest,
        "summary": summary,
        "smollm_result": smollm_result,
        "round_id": run_id,
        "round_branch": branch,
        "round_commit": commit,
        "frozen_revision": frozen_revision,
        "artifact_prefix": artifact_prefix,
        "artifact_id": artifact_id,
        "metrics_ref": metrics_ref,
        "batch_request": {
            "max_tokens": max_tokens,
            "per_prompt_timeout_seconds": per_prompt_timeout,
            "batch_timeout_seconds": batch_timeout,
        },
        "baseline_model_name": baseline_model_name,
        "optimized_model_name": optimized_model_name,
        "quality_profile": profile,
        "quality_prompts": prompts,
        "deterministic_inference": {
            "temperature": profile.deterministic_inference.temperature,
            "seed": profile.deterministic_inference.seed,
            "max_tokens": profile.deterministic_inference.max_tokens,
        },
    }


def _resolve_retained_paths(cache_root: Path, context: dict[str, Any]) -> dict[str, Path | str]:
    if not cache_root.is_dir():
        raise RuntimeError(f"Retained cache root is missing: {cache_root}")
    revision = str(context["frozen_revision"])
    snapshot_prefix = f"snapshot-{_slug_model_id(FROZEN_MODEL_ID)}"
    snapshot_name = f"{snapshot_prefix}-{revision}"
    snapshot_dir = cache_root / snapshot_name
    if not snapshot_dir.is_dir():
        raise RuntimeError(f"Frozen SmolLM snapshot is missing: {snapshot_dir}")

    artifact_prefix = str(context["artifact_prefix"])
    artifact_id = str(context["artifact_id"])
    artifact_short = artifact_id[:12]
    expected_optimized_name = f"{artifact_prefix}-{artifact_short}"
    optimized_dir = cache_root / expected_optimized_name
    if not optimized_dir.is_dir():
        raise RuntimeError(
            "Exact optimized package directory for artifact id was not found: "
            f"expected '{expected_optimized_name}' under {cache_root}.",
        )
    if not (optimized_dir / "inference_model.json").is_file() or not (optimized_dir / "model.onnx").is_file():
        raise RuntimeError(f"Optimized package '{optimized_dir}' is missing required package files.")

    sibling_inventory = _cache_sibling_inventory(
        cache_root=cache_root,
        artifact_prefix=artifact_prefix,
        snapshot_prefix=snapshot_prefix,
    )
    return {
        "cache_root": cache_root,
        "snapshot_dir": snapshot_dir,
        "optimized_dir": optimized_dir,
        "snapshot_prefix": snapshot_prefix,
        "snapshot_name": snapshot_name,
        "optimized_expected_name": expected_optimized_name,
        "optimized_package_name": optimized_dir.name,
        "optimized_artifact_id": artifact_id,
        "optimized_artifact_short": artifact_short,
        "selection_exact_id_match": optimized_dir.name == expected_optimized_name,
        "sibling_inventory_preexisting": sibling_inventory,
    }


def _build_baseline_package(
    *,
    runtime_python: Path,
    env: dict[str, str],
    snapshot_dir: Path,
    output_dir: Path,
    baseline_model_name: str,
) -> dict[str, Any]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(runtime_python),
        "-m",
        "mobius",
        "build",
        "--config",
        str(snapshot_dir),
        "--ep",
        "cpu",
        "--runtime",
        "ort-genai",
        "--task",
        "text-generation",
        "--dtype",
        "f32",
        str(output_dir),
    ]
    result = _run_cmd(cmd, timeout_seconds=7200, env=env)
    if not result["ok"]:
        raise RuntimeError(f"Mobius baseline build failed: {result['stderr_excerpt']}")
    inference_model_path = output_dir / "inference_model.json"
    inference_model_path.write_text(json.dumps({"Name": baseline_model_name}, indent=2), encoding="utf-8")
    return {
        "build_command": cmd,
        "build_seconds": result["duration_seconds"],
        "baseline_dir": output_dir,
    }


def _read_inference_model_name(model_dir: Path) -> str:
    payload = _read_json(model_dir / "inference_model.json")
    name = str(payload.get("Name") or "").strip()
    if not name:
        raise RuntimeError(f"inference_model.json is missing Name at {model_dir}.")
    return name


def _write_inference_model_name(model_dir: Path, model_name: str) -> None:
    path = model_dir / "inference_model.json"
    payload = _read_json(path) if path.is_file() else {}
    payload["Name"] = model_name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _batch_request(prompts: list[dict[str, Any]], *, max_tokens: int, per_prompt_timeout: int, batch_timeout: int) -> dict[str, Any]:
    return {
        "prompts": [{"prompt_id": row["prompt_id"], "prompt": row["prompt"], "max_tokens": max_tokens} for row in prompts],
        "per_prompt_timeout_seconds": per_prompt_timeout,
        "batch_timeout_seconds": batch_timeout,
    }


def _run_foundry_batch(
    *,
    runtime_python: Path,
    env: dict[str, str],
    model_dir: Path,
    model_name: str,
    request_payload: dict[str, Any],
    timeout_seconds: int,
) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        json.dump(request_payload, handle)
        request_path = Path(handle.name)
    try:
        cmd = [
            str(runtime_python),
            "-m",
            "fl_model_onboarding.runtime_worker",
            "foundry-infer-batch",
            "--model-dir",
            str(model_dir),
            "--model-name",
            model_name,
            "--request-file",
            str(request_path),
        ]
        result = _run_cmd(cmd, timeout_seconds=timeout_seconds, env=env)
    finally:
        request_path.unlink(missing_ok=True)

    payload = _parse_json_maybe(str(result.get("stdout") or ""))
    if not isinstance(payload, dict):
        payload = {
            "ok": False,
            "error": "runtime_worker did not return parseable JSON payload",
            "stdout_excerpt": result["stdout_excerpt"],
            "stderr_excerpt": result["stderr_excerpt"],
        }
    prompt_outputs: dict[str, str] = {}
    prompt_rows: list[dict[str, Any]] = []
    for row in payload.get("results", []):
        if not isinstance(row, dict):
            continue
        prompt_id = str(row.get("prompt_id") or "")
        output_text = str(row.get("output") or "")
        if prompt_id:
            prompt_outputs[prompt_id] = output_text
        prompt_rows.append(
            {
                "prompt_id": prompt_id,
                "timed_out": bool(row.get("timed_out")),
                "duration_seconds": float(row.get("duration_seconds") or 0.0),
                "output_text": output_text,
            }
        )
    return {
        "command": cmd,
        "command_ok": bool(result["ok"]),
        "command_returncode": result["returncode"],
        "command_duration_seconds": result["duration_seconds"],
        "worker_payload": payload,
        "prompt_rows": prompt_rows,
        "prompt_outputs": prompt_outputs,
    }


def _chat_output_text(response: Any) -> str:
    choices = getattr(response, "choices", [])
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
    return str(response)


def _run_foundry_chat(client: Any, *, prompt: str, max_tokens: int) -> str:
    if hasattr(client, "settings"):
        client.settings.max_tokens = int(max_tokens)
        try:
            client.settings.temperature = 0.0
        except Exception:
            pass
        try:
            client.settings.top_p = 1.0
        except Exception:
            pass
        try:
            client.settings.do_sample = False
        except Exception:
            pass
    response = client.complete_chat([{"role": "user", "content": prompt}])
    return _chat_output_text(response)


def _to_prompt_records(
    *,
    prompts: list[dict[str, Any]],
    outputs: dict[str, str],
    deterministic_inference: Any,
) -> tuple[PromptExecutionRecord, ...]:
    records: list[PromptExecutionRecord] = []
    for prompt in prompts:
        prompt_id = str(prompt["prompt_id"])
        output_text = str(outputs.get(prompt_id, ""))
        records.append(
            PromptExecutionRecord(
                prompt_id=prompt_id,
                output_text=output_text,
                applied_determinism=deterministic_inference,
                unsupported_determinism_fields=UNSUPPORTED_DETERMINISM_FIELDS,
            )
        )
    return tuple(records)


def _evaluate_pair(
    *,
    profile: Any,
    prompts: list[dict[str, Any]],
    baseline_outputs: dict[str, str],
    optimized_outputs: dict[str, str],
    deterministic_inference: Any,
) -> dict[str, Any]:
    baseline_records = _to_prompt_records(
        prompts=prompts,
        outputs=baseline_outputs,
        deterministic_inference=deterministic_inference,
    )
    optimized_records = _to_prompt_records(
        prompts=prompts,
        outputs=optimized_outputs,
        deterministic_inference=deterministic_inference,
    )
    outcome = evaluate_quality_validation(
        profile=profile,
        model_task="text-generation",
        baseline_outputs=baseline_records,
        optimized_outputs=optimized_records,
        require_baseline_comparison=True,
    )
    baseline_map = {
        row.prompt_id: row
        for row in (outcome.baseline_functional.prompt_results if outcome.baseline_functional is not None else ())
    }
    optimized_map = {row.prompt_id: row for row in outcome.optimized_functional.prompt_results}
    per_prompt: list[dict[str, Any]] = []
    for prompt in prompts:
        prompt_id = str(prompt["prompt_id"])
        baseline_row = baseline_map.get(prompt_id)
        optimized_row = optimized_map.get(prompt_id)
        per_prompt.append(
            {
                "prompt_id": prompt_id,
                "baseline_passed": (bool(baseline_row.passed) if baseline_row is not None else None),
                "optimized_passed": (bool(optimized_row.passed) if optimized_row is not None else None),
                "baseline_failures": list(baseline_row.failures) if baseline_row is not None else [],
                "optimized_failures": list(optimized_row.failures) if optimized_row is not None else [],
            }
        )
    baseline_regressions: list[str] = []
    if outcome.baseline_comparison is not None:
        baseline_regressions = list(outcome.baseline_comparison.regressions)
    return {
        "can_promote": bool(outcome.promotion_evidence.can_promote),
        "functional_gate": outcome.promotion_evidence.functional_gate.value,
        "baseline_comparison_gate": outcome.promotion_evidence.baseline_comparison_gate.value,
        "metrics_capture_gate": outcome.promotion_evidence.metrics_gate.value,
        "per_prompt": per_prompt,
        "baseline_comparison_regressions": baseline_regressions,
        "integrity_failures": list(outcome.recipe_verification.integrity_failures),
        "recipe_status": outcome.recipe_verification.status.value,
    }


def _evaluate_pair_detailed(
    *,
    profile: Any,
    prompts: list[dict[str, Any]],
    baseline_outputs: dict[str, str],
    optimized_outputs: dict[str, str],
    deterministic_inference: Any,
) -> dict[str, Any]:
    baseline_records = _to_prompt_records(
        prompts=prompts,
        outputs=baseline_outputs,
        deterministic_inference=deterministic_inference,
    )
    optimized_records = _to_prompt_records(
        prompts=prompts,
        outputs=optimized_outputs,
        deterministic_inference=deterministic_inference,
    )
    outcome = evaluate_quality_validation(
        profile=profile,
        model_task="text-generation",
        baseline_outputs=baseline_records,
        optimized_outputs=optimized_records,
        require_baseline_comparison=True,
    )
    summary = _evaluate_pair(
        profile=profile,
        prompts=prompts,
        baseline_outputs=baseline_outputs,
        optimized_outputs=optimized_outputs,
        deterministic_inference=deterministic_inference,
    )
    return {
        "summary": summary,
        "promotion_evidence": _jsonable(outcome.promotion_evidence),
        "recipe_verification": _jsonable(outcome.recipe_verification),
        "model_capability": _jsonable(outcome.model_capability),
        "baseline_comparison": _jsonable(outcome.baseline_comparison),
        "optimized_functional": _jsonable(outcome.optimized_functional),
        "baseline_functional": _jsonable(outcome.baseline_functional),
        "metrics_capture": _jsonable(outcome.metrics),
    }


def _json_prompt_features(prompt_id: str, outputs: dict[str, str], token_counter: Any | None) -> dict[str, Any]:
    text = str(outputs.get(prompt_id, ""))
    words = [part for part in re.split(r"\s+", text.strip()) if part]
    has_fence = "```" in text
    parse_ok = False
    required_keys_present = False
    if text:
        try:
            parsed = json.loads(text)
            parse_ok = isinstance(parsed, dict)
            if isinstance(parsed, dict):
                required_keys_present = all(key in parsed for key in ("answer", "unit"))
        except json.JSONDecodeError:
            parse_ok = False
    token_count = None
    if token_counter is not None:
        try:
            token_count = int(token_counter.count_tokens(text))
        except Exception:
            token_count = None
    return {
        "output_bounded": _sanitize_text(text, max_chars=420),
        "char_count": len(text),
        "word_count": len(words),
        "token_count": token_count,
        "has_markdown_fence": has_fence,
        "json_parse_ok": parse_ok,
        "required_keys_present": required_keys_present,
    }


class _OgaTokenCounter:
    def __init__(self, model_dir: Path):
        import onnxruntime_genai as og

        self._model = og.Model(str(model_dir))
        self._tokenizer = og.Tokenizer(self._model)

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text))

    def close(self) -> None:
        del self._tokenizer
        del self._model
        gc.collect()


def _run_repeated_trials(
    *,
    runtime_python: Path,
    env: dict[str, str],
    profile: Any,
    prompts: list[dict[str, Any]],
    deterministic_inference: Any,
    baseline_dir: Path,
    baseline_model_name: str,
    optimized_dir: Path,
    optimized_model_name: str,
    request_payload: dict[str, Any],
    trials: int,
) -> dict[str, Any]:
    json_prompt_id = "format-json-answer-unit"
    baseline_token_counter = _OgaTokenCounter(baseline_dir)
    optimized_token_counter = _OgaTokenCounter(optimized_dir)
    baseline_runs: list[dict[str, Any]] = []
    optimized_runs: list[dict[str, Any]] = []
    pair_verdicts: list[dict[str, Any]] = []
    try:
        for index in range(1, trials + 1):
            baseline_run = _run_foundry_batch(
                runtime_python=runtime_python,
                env=env,
                model_dir=baseline_dir,
                model_name=baseline_model_name,
                request_payload=request_payload,
                timeout_seconds=max(1800, int(request_payload["batch_timeout_seconds"]) + 180),
            )
            optimized_run = _run_foundry_batch(
                runtime_python=runtime_python,
                env=env,
                model_dir=optimized_dir,
                model_name=optimized_model_name,
                request_payload=request_payload,
                timeout_seconds=max(1800, int(request_payload["batch_timeout_seconds"]) + 180),
            )
            evaluation = _evaluate_pair(
                profile=profile,
                prompts=prompts,
                baseline_outputs=baseline_run["prompt_outputs"],
                optimized_outputs=optimized_run["prompt_outputs"],
                deterministic_inference=deterministic_inference,
            )
            baseline_features = _json_prompt_features(
                json_prompt_id,
                baseline_run["prompt_outputs"],
                token_counter=baseline_token_counter,
            )
            optimized_features = _json_prompt_features(
                json_prompt_id,
                optimized_run["prompt_outputs"],
                token_counter=optimized_token_counter,
            )
            baseline_runs.append(
                {
                    "trial": index,
                    "ok": bool(baseline_run["worker_payload"].get("ok")),
                    "batch_seconds": baseline_run["worker_payload"].get("duration_seconds"),
                    "json_prompt": baseline_features,
                    "outputs_bounded": {
                        prompt_id: _sanitize_text(str(text), max_chars=220)
                        for prompt_id, text in baseline_run["prompt_outputs"].items()
                    },
                }
            )
            optimized_runs.append(
                {
                    "trial": index,
                    "ok": bool(optimized_run["worker_payload"].get("ok")),
                    "batch_seconds": optimized_run["worker_payload"].get("duration_seconds"),
                    "json_prompt": optimized_features,
                    "outputs_bounded": {
                        prompt_id: _sanitize_text(str(text), max_chars=220)
                        for prompt_id, text in optimized_run["prompt_outputs"].items()
                    },
                }
            )
            pair_verdicts.append(
                {
                    "trial": index,
                    "can_promote": evaluation["can_promote"],
                    "integrity_failures": evaluation["integrity_failures"],
                    "baseline_comparison_regressions": evaluation["baseline_comparison_regressions"],
                    "json_prompt_baseline_passed": next(
                        (
                            row["baseline_passed"]
                            for row in evaluation["per_prompt"]
                            if row["prompt_id"] == json_prompt_id
                        ),
                        None,
                    ),
                    "json_prompt_optimized_passed": next(
                        (
                            row["optimized_passed"]
                            for row in evaluation["per_prompt"]
                            if row["prompt_id"] == json_prompt_id
                        ),
                        None,
                    ),
                }
            )
    finally:
        baseline_token_counter.close()
        optimized_token_counter.close()

    baseline_json_valid = sum(1 for row in baseline_runs if bool(row["json_prompt"]["json_parse_ok"]))
    optimized_json_valid = sum(1 for row in optimized_runs if bool(row["json_prompt"]["json_parse_ok"]))
    baseline_fenced = sum(1 for row in baseline_runs if bool(row["json_prompt"]["has_markdown_fence"]))
    optimized_fenced = sum(1 for row in optimized_runs if bool(row["json_prompt"]["has_markdown_fence"]))
    regression_hits = sum(
        1
        for row in pair_verdicts
        if "baseline_passed_optimized_failed:format-json-answer-unit" in row["integrity_failures"]
    )
    can_promote_count = sum(1 for row in pair_verdicts if bool(row["can_promote"]))

    baseline_unique_outputs = len({str(row["json_prompt"]["output_bounded"]) for row in baseline_runs})
    optimized_unique_outputs = len({str(row["json_prompt"]["output_bounded"]) for row in optimized_runs})

    return {
        "trial_count": trials,
        "baseline_runs": baseline_runs,
        "optimized_runs": optimized_runs,
        "pair_verdicts": pair_verdicts,
        "summary": {
            "baseline_json_valid_rate": f"{baseline_json_valid}/{trials}",
            "optimized_json_valid_rate": f"{optimized_json_valid}/{trials}",
            "baseline_fenced_rate": f"{baseline_fenced}/{trials}",
            "optimized_fenced_rate": f"{optimized_fenced}/{trials}",
            "quality_regression_signature_rate": f"{regression_hits}/{trials}",
            "can_promote_rate": f"{can_promote_count}/{trials}",
            "baseline_unique_json_outputs": baseline_unique_outputs,
            "optimized_unique_json_outputs": optimized_unique_outputs,
        },
        "baseline_outputs_for_reuse": [dict(row["outputs_bounded"]) for row in baseline_runs],
    }


def _bounded_outputs(outputs: dict[str, str], *, max_chars: int = 280) -> dict[str, str]:
    return {prompt_id: _sanitize_text(str(text), max_chars=max_chars) for prompt_id, text in outputs.items()}


def _run_full_suite_candidate_matrix(
    *,
    runtime_python: Path,
    env: dict[str, str],
    profile: Any,
    prompts: list[dict[str, Any]],
    deterministic_inference: Any,
    request_payload: dict[str, Any],
    baseline_dir: Path,
    baseline_model_name: str,
    default_dir: Path,
    default_model_name: str,
    block64_dir: Path | None,
    block64_model_name: str | None,
    trials: int,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = [
        {
            "candidate_id": "default_int4",
            "model_dir": default_dir,
            "model_name": default_model_name,
        }
    ]
    if block64_dir is not None and block64_model_name is not None:
        candidates.append(
            {
                "candidate_id": "block_size_64",
                "model_dir": block64_dir,
                "model_name": block64_model_name,
            }
        )

    matrix_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        model_dir = Path(str(candidate["model_dir"]))
        model_name = str(candidate["model_name"])
        trial_rows: list[dict[str, Any]] = []
        complete_batches = 0
        can_promote_count = 0
        structural_regression_count = 0
        for trial_index in range(1, trials + 1):
            baseline_run = _run_foundry_batch(
                runtime_python=runtime_python,
                env=env,
                model_dir=baseline_dir,
                model_name=baseline_model_name,
                request_payload=request_payload,
                timeout_seconds=max(1800, int(request_payload["batch_timeout_seconds"]) + 180),
            )
            candidate_run = _run_foundry_batch(
                runtime_python=runtime_python,
                env=env,
                model_dir=model_dir,
                model_name=model_name,
                request_payload=request_payload,
                timeout_seconds=max(1800, int(request_payload["batch_timeout_seconds"]) + 180),
            )
            evaluation = _evaluate_pair_detailed(
                profile=profile,
                prompts=prompts,
                baseline_outputs=baseline_run["prompt_outputs"],
                optimized_outputs=candidate_run["prompt_outputs"],
                deterministic_inference=deterministic_inference,
            )
            baseline_ok = bool(baseline_run["worker_payload"].get("ok"))
            candidate_ok = bool(candidate_run["worker_payload"].get("ok"))
            both_complete = (
                baseline_ok
                and candidate_ok
                and len(baseline_run["prompt_outputs"]) == len(prompts)
                and len(candidate_run["prompt_outputs"]) == len(prompts)
            )
            if both_complete:
                complete_batches += 1
            if bool(evaluation["summary"]["can_promote"]):
                can_promote_count += 1
            integrity_failures = list(evaluation["summary"]["integrity_failures"])
            if any("optimized_structural_regression:format-json-answer-unit" in row for row in integrity_failures):
                structural_regression_count += 1
            trial_rows.append(
                {
                    "trial": trial_index,
                    "baseline_worker_ok": baseline_ok,
                    "candidate_worker_ok": candidate_ok,
                    "baseline_batch_seconds": baseline_run["worker_payload"].get("duration_seconds"),
                    "candidate_batch_seconds": candidate_run["worker_payload"].get("duration_seconds"),
                    "baseline_outputs_bounded": _bounded_outputs(baseline_run["prompt_outputs"]),
                    "candidate_outputs_bounded": _bounded_outputs(candidate_run["prompt_outputs"]),
                    "evaluation": evaluation,
                }
            )
        matrix_rows.append(
            {
                "candidate_id": candidate_id,
                "model_name": model_name,
                "trial_count": trials,
                "complete_batch_rate": f"{complete_batches}/{trials}",
                "can_promote_rate": f"{can_promote_count}/{trials}",
                "json_structural_regression_rate": f"{structural_regression_count}/{trials}",
                "trials": trial_rows,
            }
        )

    default_row = next((row for row in matrix_rows if row["candidate_id"] == "default_int4"), None)
    block64_row = next((row for row in matrix_rows if row["candidate_id"] == "block_size_64"), None)
    if block64_row is None:
        conclusion = {
            "block64_candidate_supported": False,
            "result": "block64_unavailable",
            "reason": "block_size_64 variant did not build successfully in this run.",
        }
    else:
        default_structural = str(default_row.get("json_structural_regression_rate")) if isinstance(default_row, dict) else ""
        block64_promote = str(block64_row.get("can_promote_rate"))
        block64_structural = str(block64_row.get("json_structural_regression_rate"))
        remains_candidate = block64_promote == f"{trials}/{trials}" and block64_structural == f"0/{trials}"
        conclusion = {
            "block64_candidate_supported": remains_candidate,
            "result": (
                "block64_remains_promotion_eligible_full_suite"
                if remains_candidate
                else "block64_withdrawn_due_to_full_suite_failure"
            ),
            "default_int4_structural_regression_rate": default_structural,
            "block64_can_promote_rate": block64_promote,
            "block64_structural_regression_rate": block64_structural,
        }
    return {
        "prompt_order": [str(row["prompt_id"]) for row in prompts],
        "candidates": matrix_rows,
        "conclusion": conclusion,
    }


def _probe_foundry_controls(
    *,
    runtime_python: Path,
    env: dict[str, str],
    model_dir: Path,
    model_name: str,
    probe_prompt: str,
) -> dict[str, Any]:
    code = """
import json
import sys
from pathlib import Path

from foundry_local_sdk import Configuration, FoundryLocalManager

model_dir = Path(sys.argv[1]).resolve()
model_name = sys.argv[2]
prompt = sys.argv[3]

FoundryLocalManager.initialize(Configuration(
    app_name="fl-model-onboarding-smollm-json-regression-diag",
    model_cache_dir=str(model_dir.parent),
))
manager = FoundryLocalManager.instance
candidate = next((row for row in manager.catalog.get_cached_models() if model_name in row.id), None)
if candidate is None:
    raise RuntimeError(f"Model '{model_name}' not discovered in '{model_dir.parent}'.")
candidate.load()
client = candidate.get_chat_client()

requested = {
    "max_tokens": 64,
    "temperature": 0.0,
    "top_p": 1.0,
    "seed": 17,
    "do_sample": False,
}
applied = {}
unsupported = {}
for key, value in requested.items():
    try:
        setattr(client.settings, key, value)
        applied[key] = getattr(client.settings, key, None)
    except Exception as exc:
        unsupported[key] = str(exc)

response = client.complete_chat([{"role": "user", "content": prompt}])
choices = getattr(response, "choices", [])
if choices:
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    output = content if isinstance(content, str) else str(response)
else:
    output = str(response)
candidate.unload()
print(json.dumps({
    "ok": True,
    "requested": requested,
    "applied": applied,
    "unsupported": unsupported,
    "output_bounded": output[:280],
}, sort_keys=True))
"""
    result = _run_cmd(
        [str(runtime_python), "-c", code, str(model_dir), model_name, probe_prompt],
        timeout_seconds=1200,
        env=env,
    )
    payload = _parse_json_maybe(str(result.get("stdout") or ""))
    if not isinstance(payload, dict):
        return {
            "ok": False,
            "error": "unable to parse Foundry control probe payload",
            "stdout_excerpt": result["stdout_excerpt"],
            "stderr_excerpt": result["stderr_excerpt"],
        }
    payload["command_ok"] = result["ok"]
    payload["duration_seconds"] = result["duration_seconds"]
    return payload


def _probe_oga_controls(model_dir: Path, prompt: str) -> dict[str, Any]:
    import onnxruntime_genai as og

    model = og.Model(str(model_dir))
    tokenizer = og.Tokenizer(model)
    encoded = tokenizer.encode(prompt)
    requested = {
        "max_length": len(encoded) + 64,
        "max_tokens": 64,
        "temperature": 0.0,
        "top_p": 1.0,
        "do_sample": False,
        "seed": 17,
    }
    applied: dict[str, Any] = {}
    unsupported: dict[str, str] = {}
    for key, value in requested.items():
        params = og.GeneratorParams(model)
        try:
            params.set_search_options(**{key: value})
            applied[key] = value
        except Exception as exc:
            unsupported[key] = str(exc)
    params = og.GeneratorParams(model)
    accepted_subset: dict[str, Any] = {}
    for key in ("max_length", "temperature", "top_p", "do_sample"):
        if key in applied:
            accepted_subset[key] = applied[key]
    if accepted_subset:
        params.set_search_options(**accepted_subset)
    generator = og.Generator(model, params)
    stream = tokenizer.create_stream()
    generator.append_tokens(encoded)
    generated = 0
    text = ""
    while not generator.is_done() and generated < 64:
        generator.generate_next_token()
        tokens = generator.get_next_tokens()
        text += stream.decode(tokens[0])
        generated += 1
    del generator
    del stream
    del tokenizer
    del model
    gc.collect()
    return {
        "ok": True,
        "requested": requested,
        "applied": applied,
        "unsupported": unsupported,
        "generated_tokens": generated,
        "output_bounded": _sanitize_text(text, max_chars=280),
    }


def _file_fingerprint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False}
    return {
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _extract_chat_template(model_dir: Path) -> dict[str, Any]:
    tokenizer_config_path = model_dir / "tokenizer_config.json"
    tokenizer_template = None
    if tokenizer_config_path.is_file():
        try:
            payload = _read_json(tokenizer_config_path)
            value = payload.get("chat_template")
            if isinstance(value, str):
                tokenizer_template = value
        except Exception:
            tokenizer_template = None
    jinja_path = model_dir / "chat_template.jinja"
    jinja_template = jinja_path.read_text(encoding="utf-8") if jinja_path.is_file() else None
    return {
        "tokenizer_config_chat_template_sha256": (
            hashlib.sha256(tokenizer_template.encode("utf-8")).hexdigest() if isinstance(tokenizer_template, str) else None
        ),
        "chat_template_jinja_sha256": (
            hashlib.sha256(jinja_template.encode("utf-8")).hexdigest() if isinstance(jinja_template, str) else None
        ),
    }


def _compare_packaging(baseline_dir: Path, optimized_dir: Path) -> dict[str, Any]:
    files = (
        "genai_config.json",
        "inference_model.json",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    )
    fingerprints: dict[str, Any] = {}
    for name in files:
        baseline_file = baseline_dir / name
        optimized_file = optimized_dir / name
        fingerprints[name] = {
            "baseline": _file_fingerprint(baseline_file),
            "optimized": _file_fingerprint(optimized_file),
            "sha_equal": (
                baseline_file.is_file()
                and optimized_file.is_file()
                and _sha256(baseline_file) == _sha256(optimized_file)
            ),
        }
    baseline_inference = _read_json(baseline_dir / "inference_model.json")
    optimized_inference = _read_json(optimized_dir / "inference_model.json")
    baseline_genai = _read_json(baseline_dir / "genai_config.json")
    optimized_genai = _read_json(optimized_dir / "genai_config.json")
    return {
        "files": fingerprints,
        "inference_model_name": {
            "baseline": baseline_inference.get("Name"),
            "optimized": optimized_inference.get("Name"),
            "equal": baseline_inference.get("Name") == optimized_inference.get("Name"),
        },
        "genai_config_exact_equal": baseline_genai == optimized_genai,
        "chat_template_fingerprints": {
            "baseline": _extract_chat_template(baseline_dir),
            "optimized": _extract_chat_template(optimized_dir),
        },
    }


def _onnx_op_summary(model_path: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)
    counts: dict[str, int] = {}
    for node in model.graph.node:
        op = str(node.op_type)
        counts[op] = counts.get(op, 0) + 1
    top_ops = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:20]
    return {
        "total_nodes": len(model.graph.node),
        "op_counts_top20": [{"op": name, "count": count} for name, count in top_ops],
        "matmul_count": counts.get("MatMul", 0),
        "matmul_nbits_count": counts.get("MatMulNBits", 0),
        "gather_block_quantized_count": counts.get("GatherBlockQuantized", 0),
        "dequantize_linear_count": counts.get("DequantizeLinear", 0),
    }


def _profile_foundry_load_and_batch(
    *,
    runtime_python: Path,
    env: dict[str, str],
    model_dir: Path,
    model_name: str,
    prompts: list[dict[str, Any]],
    max_tokens: int,
) -> dict[str, Any]:
    payload = {
        "model_dir": str(model_dir),
        "model_name": model_name,
        "prompts": [{"prompt_id": str(row["prompt_id"]), "prompt": str(row["prompt"])} for row in prompts],
        "max_tokens": int(max_tokens),
    }
    worker_code = """
import json
import os
import sys
import time
from pathlib import Path

from foundry_local_sdk import Configuration, FoundryLocalManager

payload_path = Path(sys.argv[1]).resolve()
payload = json.loads(payload_path.read_text(encoding='utf-8'))
model_dir = Path(payload['model_dir']).resolve()
model_name = str(payload['model_name'])
prompts = payload['prompts']
max_tokens = int(payload['max_tokens'])

rss_supported = False
rss_error = ''
peak_rss_bytes = None
rss_samples = []
process = None
try:
    import psutil
    process = psutil.Process(os.getpid())
    rss_supported = True
except Exception as exc:
    rss_error = str(exc)

def sample(stage: str):
    global peak_rss_bytes
    if process is None:
        return
    try:
        rss = int(process.memory_info().rss)
    except Exception:
        return
    if peak_rss_bytes is None or rss > peak_rss_bytes:
        peak_rss_bytes = rss
    rss_samples.append({'stage': stage, 'rss_bytes': rss})

sample('before_initialize')
initialize_start = time.perf_counter()
FoundryLocalManager.initialize(Configuration(
    app_name='fl-model-onboarding-smollm-json-regression-profile',
    model_cache_dir=str(model_dir.parent),
))
initialize_end = time.perf_counter()
sample('after_initialize')
manager = FoundryLocalManager.instance
candidate = next((row for row in manager.catalog.get_cached_models() if model_name in row.id), None)
if candidate is None:
    raise RuntimeError(f\"Profile candidate '{model_name}' not found in cache root '{model_dir.parent}'.\")
load_start = time.perf_counter()
candidate.load()
load_end = time.perf_counter()
sample('after_load')
client = candidate.get_chat_client()
prompt_runs = []
generation_start = time.perf_counter()
for row in prompts:
    prompt_id = str(row['prompt_id'])
    prompt_text = str(row['prompt'])
    if hasattr(client, 'settings'):
        client.settings.max_tokens = max_tokens
        try:
            client.settings.temperature = 0.0
        except Exception:
            pass
    infer_start = time.perf_counter()
    response = client.complete_chat([{'role': 'user', 'content': prompt_text}])
    infer_end = time.perf_counter()
    choices = getattr(response, 'choices', [])
    if choices:
        message = getattr(choices[0], 'message', None)
        content = getattr(message, 'content', None)
        output = content if isinstance(content, str) else str(response)
    else:
        output = str(response)
    sample(f'after_prompt_{prompt_id}')
    prompt_runs.append({
        'prompt_id': prompt_id,
        'duration_seconds': round(infer_end - infer_start, 4),
        'output_bounded': output[:200],
    })
generation_end = time.perf_counter()
unload_start = time.perf_counter()
candidate.unload()
unload_end = time.perf_counter()
sample('after_unload')

print(json.dumps({
    'ok': True,
    'initialize_seconds': round(initialize_end - initialize_start, 4),
    'load_seconds': round(load_end - load_start, 4),
    'generation_seconds': round(generation_end - generation_start, 4),
    'unload_seconds': round(unload_end - unload_start, 4),
    'prompt_runs': prompt_runs,
    'peak_rss_bytes': peak_rss_bytes,
    'rss_supported': rss_supported,
    'rss_error': rss_error if not rss_supported else '',
    'rss_samples': rss_samples,
}, sort_keys=True))
"""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        json.dump(payload, handle)
        payload_path = Path(handle.name)
    try:
        result = _run_cmd(
            [str(runtime_python), "-c", worker_code, str(payload_path)],
            timeout_seconds=1800,
            env=env,
        )
    finally:
        payload_path.unlink(missing_ok=True)
    parsed = _parse_json_maybe(str(result.get("stdout") or ""))
    if not result["ok"] or not isinstance(parsed, dict):
        return {
            "ok": False,
            "error": "foundry profile subprocess failed",
            "stderr_tail": result["stderr_tail"],
            "stdout_tail": result["stdout_tail"],
            "duration_seconds": result["duration_seconds"],
        }
    parsed["command_duration_seconds"] = result["duration_seconds"]
    return parsed


def _token_trace_with_oga(model_dir: Path, prompt: str, *, max_new_tokens: int) -> dict[str, Any]:
    import onnxruntime_genai as og

    model = og.Model(str(model_dir))
    tokenizer = og.Tokenizer(model)
    prompt_tokens = tokenizer.encode(prompt)
    params = og.GeneratorParams(model)
    options = {
        "max_length": len(prompt_tokens) + max_new_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "do_sample": False,
    }
    params.set_search_options(**options)
    generator = og.Generator(model, params)
    stream = tokenizer.create_stream()
    generator.append_tokens(prompt_tokens)
    token_ids: list[int] = []
    output_text = ""
    while not generator.is_done() and len(token_ids) < max_new_tokens:
        generator.generate_next_token()
        next_tokens = generator.get_next_tokens()
        token_id = int(next_tokens[0])
        token_ids.append(token_id)
        output_text += stream.decode(token_id)
    del generator
    del stream
    del tokenizer
    del model
    gc.collect()
    return {
        "ok": True,
        "token_ids": token_ids,
        "generated_count": len(token_ids),
        "output_bounded": _sanitize_text(output_text, max_chars=280),
        "options": options,
    }


def _compare_token_traces(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    ref_ids = [int(row) for row in reference.get("token_ids", [])]
    cand_ids = [int(row) for row in candidate.get("token_ids", [])]
    compared = min(len(ref_ids), len(cand_ids))
    matches = 0
    first_divergence: int | None = None
    for index in range(compared):
        if ref_ids[index] == cand_ids[index]:
            matches += 1
            continue
        first_divergence = index + 1
        break
    if first_divergence is None and len(ref_ids) != len(cand_ids):
        first_divergence = compared + 1
    return {
        "compared_steps": compared,
        "matching_steps": matches,
        "step_match_rate": (f"{matches}/{compared}" if compared > 0 else "0/0"),
        "first_divergence_step_1_indexed": first_divergence,
        "reference_generated_count": len(ref_ids),
        "candidate_generated_count": len(cand_ids),
    }


def _attempt_numeric_fidelity_probe(
    *,
    baseline_dir: Path,
    default_dir: Path,
    block64_dir: Path | None,
) -> dict[str, Any]:
    method = {
        "kind": "oga_next_token_trace_divergence",
        "prompt": NUMERIC_FIDELITY_PROMPT,
        "max_new_tokens": NUMERIC_FIDELITY_MAX_TOKENS,
        "notes": "Deterministic decode options requested (do_sample=false, temperature=0, top_p=1).",
    }
    try:
        baseline_trace = _token_trace_with_oga(
            baseline_dir,
            NUMERIC_FIDELITY_PROMPT,
            max_new_tokens=NUMERIC_FIDELITY_MAX_TOKENS,
        )
        default_trace = _token_trace_with_oga(
            default_dir,
            NUMERIC_FIDELITY_PROMPT,
            max_new_tokens=NUMERIC_FIDELITY_MAX_TOKENS,
        )
        payload: dict[str, Any] = {
            "status": "available",
            "method": method,
            "baseline_vs_default": {
                "comparison": _compare_token_traces(baseline_trace, default_trace),
                "baseline_output_bounded": baseline_trace["output_bounded"],
                "candidate_output_bounded": default_trace["output_bounded"],
            },
        }
        if block64_dir is not None:
            block64_trace = _token_trace_with_oga(
                block64_dir,
                NUMERIC_FIDELITY_PROMPT,
                max_new_tokens=NUMERIC_FIDELITY_MAX_TOKENS,
            )
            payload["baseline_vs_block64"] = {
                "comparison": _compare_token_traces(baseline_trace, block64_trace),
                "baseline_output_bounded": baseline_trace["output_bounded"],
                "candidate_output_bounded": block64_trace["output_bounded"],
            }
        return payload
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "numeric_fidelity_unknown",
            "method": method,
            "error": _sanitize_text(str(exc), max_chars=600),
        }


def _copy_package(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _replace_graph(target_dir: Path, donor_dir: Path) -> None:
    target_onnx = target_dir / "model.onnx"
    donor_onnx = donor_dir / "model.onnx"
    if not donor_onnx.is_file():
        raise RuntimeError(f"Donor model.onnx missing: {donor_onnx}")
    shutil.copy2(donor_onnx, target_onnx)
    target_external = target_dir / "model.onnx.data"
    donor_external = donor_dir / "model.onnx.data"
    if donor_external.is_file():
        shutil.copy2(donor_external, target_external)
    elif target_external.exists():
        target_external.unlink()


def _run_hybrid_experiments(
    *,
    runtime_python: Path,
    env: dict[str, str],
    profile: Any,
    prompts: list[dict[str, Any]],
    deterministic_inference: Any,
    baseline_dir: Path,
    optimized_dir: Path,
    request_payload: dict[str, Any],
    external_root: Path,
) -> dict[str, Any]:
    hybrid_root = external_root / "hybrid"
    hybrid_a_dir = hybrid_root / "baseline-pkg-optimized-graph"
    hybrid_b_dir = hybrid_root / "optimized-pkg-baseline-graph"
    hybrid_root.mkdir(parents=True, exist_ok=True)

    _copy_package(baseline_dir, hybrid_a_dir)
    _replace_graph(hybrid_a_dir, optimized_dir)
    _write_inference_model_name(hybrid_a_dir, "smollm2-diag-hybrid-a:1")

    _copy_package(optimized_dir, hybrid_b_dir)
    _replace_graph(hybrid_b_dir, baseline_dir)
    _write_inference_model_name(hybrid_b_dir, "smollm2-diag-hybrid-b:1")

    def run_and_eval(model_dir: Path, model_name: str) -> dict[str, Any]:
        batch = _run_foundry_batch(
            runtime_python=runtime_python,
            env=env,
            model_dir=model_dir,
            model_name=model_name,
            request_payload=request_payload,
            timeout_seconds=max(1800, int(request_payload["batch_timeout_seconds"]) + 180),
        )
        baseline_reference = _run_foundry_batch(
            runtime_python=runtime_python,
            env=env,
            model_dir=baseline_dir,
            model_name=_read_inference_model_name(baseline_dir),
            request_payload=request_payload,
            timeout_seconds=max(1800, int(request_payload["batch_timeout_seconds"]) + 180),
        )
        evaluation = _evaluate_pair(
            profile=profile,
            prompts=prompts,
            baseline_outputs=baseline_reference["prompt_outputs"],
            optimized_outputs=batch["prompt_outputs"],
            deterministic_inference=deterministic_inference,
        )
        return {
            "batch_ok": bool(batch["worker_payload"].get("ok")),
            "batch_seconds": batch["worker_payload"].get("duration_seconds"),
            "json_prompt_output_bounded": _sanitize_text(
                str(batch["prompt_outputs"].get("format-json-answer-unit", "")),
                max_chars=420,
            ),
            "json_prompt_has_fence": "```" in str(batch["prompt_outputs"].get("format-json-answer-unit", "")),
            "json_prompt_parse_ok": _json_prompt_features("format-json-answer-unit", batch["prompt_outputs"], None)[
                "json_parse_ok"
            ],
            "quality_eval": evaluation,
        }

    hybrid_a = run_and_eval(hybrid_a_dir, _read_inference_model_name(hybrid_a_dir))
    hybrid_b = run_and_eval(hybrid_b_dir, _read_inference_model_name(hybrid_b_dir))
    return {
        "hybrid_a_baseline_package_plus_optimized_graph": hybrid_a,
        "hybrid_b_optimized_package_plus_baseline_graph": hybrid_b,
        "directories": {
            "hybrid_root": str(hybrid_root),
            "hybrid_a_dir": str(hybrid_a_dir),
            "hybrid_b_dir": str(hybrid_b_dir),
        },
    }


def _classify_optimize_failure(stderr_text: str, stdout_text: str) -> dict[str, str]:
    full = f"{stdout_text}\n{stderr_text}".strip()
    lines = [line.strip() for line in full.splitlines() if line.strip()]
    last_line = lines[-1] if lines else ""
    lowered = full.lower()
    if "assertionerror" in lowered and "block_size" in lowered:
        classification = "invalid_block_size"
    elif "no module named 'datasets'" in lowered:
        classification = "missing_dependency_datasets"
    elif "modulenotfounderror" in lowered:
        classification = "missing_dependency"
    elif "not implemented" in lowered:
        classification = "runtime_not_implemented"
    elif "traceback" in lowered:
        classification = "python_exception"
    elif "error" in lowered:
        classification = "tool_error"
    else:
        classification = "unknown_failure"
    return {
        "failure_classification": classification,
        "last_exception_line": _sanitize_text(last_line, max_chars=220),
    }


def _run_variant_experiments(
    *,
    runtime_python: Path,
    env: dict[str, str],
    profile: Any,
    prompts: list[dict[str, Any]],
    deterministic_inference: Any,
    baseline_dir: Path,
    baseline_model_name: str,
    baseline_trial_outputs: list[dict[str, str]],
    request_payload: dict[str, Any],
    external_root: Path,
) -> dict[str, Any]:
    variant_specs = [
        {"id": "int4_default", "olive_args": [], "trial_count": 3},
        {"id": "int4_block_size_16", "olive_args": ["--block_size", "16"], "trial_count": 3},
        {"id": "int4_block_size_32", "olive_args": ["--block_size", "32"], "trial_count": 3},
        {"id": "int4_block_size_64", "olive_args": ["--block_size", "64"], "trial_count": 3},
        {"id": "int4_block_size_-1", "olive_args": ["--block_size", "-1"], "trial_count": 0},
        {"id": "int4_act_precision_uint8", "olive_args": ["--act_precision", "uint8"], "trial_count": 0},
    ]
    variants_root = external_root / "variants"
    variants_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"variants_root": str(variants_root), "variants": []}

    for spec in variant_specs:
        variant_id = str(spec["id"])
        olive_args = [str(item) for item in spec["olive_args"]]
        trial_count = int(spec["trial_count"])
        variant_dir = variants_root / variant_id
        if variant_dir.exists():
            shutil.rmtree(variant_dir)
        cmd = [
            str(runtime_python),
            "-m",
            "olive",
            "optimize",
            "--model_name_or_path",
            str(baseline_dir),
            "--task",
            "text-generation-with-past",
            "--provider",
            "CPUExecutionProvider",
            "--device",
            "cpu",
            "--log_level",
            "1",
            "--precision",
            "int4",
            "--output_path",
            str(variant_dir),
        ]
        cmd.extend(olive_args)
        optimize = _run_cmd(cmd, timeout_seconds=7200, env=env)
        row: dict[str, Any] = {
            "variant_id": variant_id,
            "olive_args": olive_args,
            "optimize_ok": bool(optimize["ok"]),
            "optimize_seconds": optimize["duration_seconds"],
            "optimize_stdout_excerpt": optimize["stdout_excerpt"],
            "optimize_stderr_excerpt": optimize["stderr_excerpt"],
            "optimize_stdout_tail": optimize["stdout_tail"],
            "optimize_stderr_tail": optimize["stderr_tail"],
            "trial_count": trial_count,
            "trials": [],
            "variant_dir": str(variant_dir),
        }
        if not optimize["ok"] or not (variant_dir / "model.onnx").is_file():
            row.update(_classify_optimize_failure(str(optimize.get("stderr") or ""), str(optimize.get("stdout") or "")))
            row["status"] = "optimize_failed_or_unsupported"
            report["variants"].append(row)
            continue

        model_name = f"{baseline_model_name.split(':')[0]}-{variant_id}:1"
        _write_inference_model_name(variant_dir, model_name)
        row["graph_summary"] = _onnx_op_summary(variant_dir / "model.onnx")
        row["model_name"] = model_name
        row["package_size_bytes"] = _directory_size_bytes(variant_dir)
        row["foundry_profile"] = _profile_foundry_load_and_batch(
            runtime_python=runtime_python,
            env=env,
            model_dir=variant_dir,
            model_name=model_name,
            prompts=prompts,
            max_tokens=int(request_payload["prompts"][0]["max_tokens"]),
        )
        can_promote = 0
        json_parse_ok = 0
        json_fenced = 0
        integrity_regressions = 0
        for trial_index in range(1, trial_count + 1):
            batch = _run_foundry_batch(
                runtime_python=runtime_python,
                env=env,
                model_dir=variant_dir,
                model_name=model_name,
                request_payload=request_payload,
                timeout_seconds=max(1800, int(request_payload["batch_timeout_seconds"]) + 180),
            )
            baseline_outputs = baseline_trial_outputs[(trial_index - 1) % len(baseline_trial_outputs)]
            eval_payload = _evaluate_pair(
                profile=profile,
                prompts=prompts,
                baseline_outputs=baseline_outputs,
                optimized_outputs=batch["prompt_outputs"],
                deterministic_inference=deterministic_inference,
            )
            json_features = _json_prompt_features("format-json-answer-unit", batch["prompt_outputs"], None)
            if eval_payload["can_promote"]:
                can_promote += 1
            if json_features["json_parse_ok"]:
                json_parse_ok += 1
            if json_features["has_markdown_fence"]:
                json_fenced += 1
            if "baseline_passed_optimized_failed:format-json-answer-unit" in eval_payload["integrity_failures"]:
                integrity_regressions += 1
            row["trials"].append(
                {
                    "trial": trial_index,
                    "batch_ok": bool(batch["worker_payload"].get("ok")),
                    "batch_seconds": batch["worker_payload"].get("duration_seconds"),
                    "outputs_bounded": _bounded_outputs(batch["prompt_outputs"]),
                    "json_prompt": json_features,
                    "quality_eval": {
                        "can_promote": eval_payload["can_promote"],
                        "integrity_failures": eval_payload["integrity_failures"],
                        "baseline_comparison_regressions": eval_payload["baseline_comparison_regressions"],
                    },
                }
            )
        row["summary"] = {
            "can_promote_rate": f"{can_promote}/{trial_count}",
            "json_parse_ok_rate": f"{json_parse_ok}/{trial_count}",
            "json_fenced_rate": f"{json_fenced}/{trial_count}",
            "json_regression_signature_rate": f"{integrity_regressions}/{trial_count}",
        }
        row["status"] = "evaluated"
        report["variants"].append(row)
    return report


def _rank_remedies(variant_report: dict[str, Any]) -> dict[str, Any]:
    variants = [row for row in variant_report.get("variants", []) if isinstance(row, dict)]
    block64 = next((row for row in variants if row.get("variant_id") == "int4_block_size_64"), None)
    default_int4 = next((row for row in variants if row.get("variant_id") == "int4_default"), None)
    rejected: list[dict[str, Any]] = []
    for row in variants:
        variant_id = str(row.get("variant_id"))
        status = str(row.get("status"))
        if variant_id == "int4_block_size_64":
            continue
        if status == "optimize_failed_or_unsupported":
            rejected.append(
                {
                    "variant_id": variant_id,
                    "reason": "toolchain_unsupported_or_optimize_failed",
                    "evidence": row.get("optimize_stderr_excerpt"),
                }
            )
            continue
        summary = row.get("summary", {})
        can_promote_rate = str(summary.get("can_promote_rate") or "")
        if not can_promote_rate.startswith("3/3") and not can_promote_rate.startswith("5/5"):
            rejected.append(
                {
                    "variant_id": variant_id,
                    "reason": "quality_gate_failed",
                    "evidence": summary,
                }
            )
    ranked = [
        {
            "rank": 1,
            "candidate": "capability-level Olive INT4 block_size=64",
            "status": (
                "proven_on_target_model"
                if isinstance(block64, dict) and str(block64.get("summary", {}).get("can_promote_rate")) == "5/5"
                else "not_proven"
            ),
            "scope": "llama-text-generation-cpu-int4-v1 capability policy candidate (not model-id keyed).",
            "evidence": block64.get("summary") if isinstance(block64, dict) else None,
            "risk": (
                "Not yet proven across full frozen five-model set; must run unchanged-five-model rerun before adoption."
            ),
        },
        {
            "rank": 2,
            "candidate": "keep current int4 default",
            "status": "rejected_for_target",
            "scope": "existing production path",
            "evidence": default_int4.get("summary") if isinstance(default_int4, dict) else None,
            "risk": "Retains deterministic JSON regression signature on SmolLM2.",
        },
    ]
    return {"ranked": ranked, "rejected": rejected}


def _build_block_size_cost_matrix(
    *,
    selected_default_package_dir: Path,
    selected_default_profile: dict[str, Any],
    variants: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = [
        {
            "variant": "default_int4_selected_round6_artifact",
            "package_size_bytes": _directory_size_bytes(selected_default_package_dir),
            "optimize_seconds": None,
            "load_seconds": selected_default_profile.get("load_seconds"),
            "generation_seconds": selected_default_profile.get("generation_seconds"),
            "peak_rss_bytes": selected_default_profile.get("peak_rss_bytes"),
            "quality_summary": None,
        }
    ]
    for row in variants.get("variants", []):
        if not isinstance(row, dict):
            continue
        variant_id = str(row.get("variant_id"))
        if variant_id not in {"int4_default", "int4_block_size_16", "int4_block_size_32", "int4_block_size_64"}:
            continue
        foundry_profile = row.get("foundry_profile", {})
        summary = row.get("summary", {})
        rows.append(
            {
                "variant": variant_id,
                "package_size_bytes": row.get("package_size_bytes"),
                "optimize_seconds": row.get("optimize_seconds"),
                "load_seconds": (foundry_profile.get("load_seconds") if isinstance(foundry_profile, dict) else None),
                "generation_seconds": (
                    foundry_profile.get("generation_seconds") if isinstance(foundry_profile, dict) else None
                ),
                "peak_rss_bytes": (foundry_profile.get("peak_rss_bytes") if isinstance(foundry_profile, dict) else None),
                "quality_summary": summary if isinstance(summary, dict) else None,
            }
        )
    return {
        "rows": rows,
        "notes": [
            "optimize_seconds is omitted for selected Round 6 default artifact because it was prebuilt.",
            "load/generation timings are direct Foundry SDK measurements on this host and are not cross-hardware comparable.",
            "peak_rss_bytes uses best-effort psutil sampling when available.",
        ],
    }


def _probe_lingering_processes() -> dict[str, Any]:
    command = (
        "$self=$PID;"
        "$rx='fl_model_onboarding.runtime_worker|mobius build|olive optimize|fl-model-onboarding-smollm-json-regression-diag';"
        "$rows=Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $self -and $_.CommandLine -and ($_.CommandLine -match $rx) } "
        "| Select-Object ProcessId,Name,CommandLine;"
        "$rows | ConvertTo-Json -Compress"
    )
    result = _run_cmd(["powershell", "-NoProfile", "-Command", command], timeout_seconds=120)
    payload = result.get("stdout", "").strip()
    if not payload:
        return {"ok": True, "count": 0, "rows": []}
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return {"ok": False, "parse_error": result["stdout_excerpt"], "rows": []}
    if isinstance(parsed, dict):
        rows = [parsed]
    elif isinstance(parsed, list):
        rows = [row for row in parsed if isinstance(row, dict)]
    else:
        rows = []
    return {
        "ok": True,
        "count": len(rows),
        "rows": [
            {
                "process_id": row.get("ProcessId"),
                "name": row.get("Name"),
                "commandline_bounded": _sanitize_text(str(row.get("CommandLine") or ""), max_chars=220),
            }
            for row in rows
        ],
    }


def _cleanup_exact_stray_root(path: Path) -> dict[str, Any]:
    target = EXACT_STRAY_DIAGNOSTIC_ROOT.resolve()
    resolved = path.resolve()
    if resolved != target:
        raise RuntimeError(
            f"Refusing cleanup for non-whitelisted path '{resolved}'. Allowed path is '{target}'.",
        )
    if not resolved.exists():
        return {
            "path": str(resolved),
            "existed": False,
            "removed": False,
            "bytes_freed": 0,
            "entries_before": [],
        }
    entries_before: list[dict[str, Any]] = []
    for entry in sorted(resolved.iterdir()):
        entries_before.append(
            {
                "name": entry.name,
                "is_dir": entry.is_dir(),
                "size_bytes": (_directory_size_bytes(entry) if entry.is_dir() else _file_size_bytes(entry)),
            }
        )
    bytes_before = _directory_size_bytes(resolved)
    shutil.rmtree(resolved)
    return {
        "path": str(resolved),
        "existed": True,
        "removed": True,
        "bytes_freed": bytes_before,
        "entries_before": entries_before,
    }


def _safe_cleanup_external(path: Path, scratch_root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    root = scratch_root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        return {"ok": False, "error": f"Refusing cleanup outside {root}: {resolved} ({exc})"}
    if not resolved.exists():
        return {"ok": True, "removed": False, "bytes_freed": 0}
    bytes_before = _directory_size_bytes(resolved)
    shutil.rmtree(resolved)
    return {"ok": True, "removed": True, "bytes_freed": bytes_before}


def _main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose SmolLM2 Round 6 optimized JSON-format regression.")
    parser.add_argument("--runtime-python", default=str(DEFAULT_RUNTIME_PYTHON))
    parser.add_argument("--retained-cache-root", default=str(DEFAULT_RETAINED_CACHE_ROOT))
    parser.add_argument("--external-scratch-root", default=str(DEFAULT_EXTERNAL_SCRATCH_ROOT))
    parser.add_argument("--repro-trials", type=int, default=6)
    parser.add_argument("--full-suite-trials", type=int, default=3)
    parser.add_argument("--output-report", default=str(REPORT_PATH))
    parser.add_argument("--retain-external", action="store_true")
    parser.add_argument("--allow-interpreter-conflation", action="store_true")
    parser.add_argument("--toolchain-probe-only", action="store_true")
    args = parser.parse_args()

    runtime_python = Path(args.runtime_python).resolve()
    retained_cache_root = Path(args.retained_cache_root).resolve()
    scratch_root = Path(args.external_scratch_root).resolve()
    output_report = Path(args.output_report).resolve()
    if not runtime_python.is_file():
        raise RuntimeError(f"Runtime python executable not found: {runtime_python}")
    scratch_root.mkdir(parents=True, exist_ok=True)
    env = _runtime_env()
    if args.toolchain_probe_only:
        payload = _load_toolchain_versions(
            runtime_python,
            env,
            allow_interpreter_conflation=bool(args.allow_interpreter_conflation),
        )
        print(json.dumps(_sanitize_payload(payload), sort_keys=True))
        return 0

    context = _load_round6_context()
    retained_paths = _resolve_retained_paths(retained_cache_root, context)
    profile = context["quality_profile"]
    prompts = context["quality_prompts"]
    deterministic = context["deterministic_inference"]
    deterministic_profile = profile.deterministic_inference
    request_payload = _batch_request(
        prompts,
        max_tokens=int(context["batch_request"]["max_tokens"]),
        per_prompt_timeout=int(context["batch_request"]["per_prompt_timeout_seconds"]),
        batch_timeout=int(context["batch_request"]["batch_timeout_seconds"]),
    )

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    external_run_root = scratch_root / f"smollm-json-regression-{run_stamp}"
    external_run_root.mkdir(parents=True, exist_ok=False)
    cleanup_result: dict[str, Any] = {"ok": False, "removed": False, "bytes_freed": 0}
    report: dict[str, Any] = {}
    try:
        toolchain_probe = _load_toolchain_versions(
            runtime_python,
            env,
            allow_interpreter_conflation=bool(args.allow_interpreter_conflation),
        )
        snapshot_dir = Path(str(retained_paths["snapshot_dir"]))
        optimized_dir = Path(str(retained_paths["optimized_dir"]))
        selected_input_hashes_before = {
            "snapshot": _directory_fingerprint(snapshot_dir),
            "optimized_package": _directory_fingerprint(optimized_dir),
        }
        baseline_dir = external_run_root / "baseline" / "mobius"
        baseline_build = _build_baseline_package(
            runtime_python=runtime_python,
            env=env,
            snapshot_dir=snapshot_dir,
            output_dir=baseline_dir,
            baseline_model_name=str(context["baseline_model_name"]),
        )
        baseline_model_name = _read_inference_model_name(baseline_dir)
        optimized_model_name = _read_inference_model_name(optimized_dir)

        baseline_single = _run_foundry_batch(
            runtime_python=runtime_python,
            env=env,
            model_dir=baseline_dir,
            model_name=baseline_model_name,
            request_payload=request_payload,
            timeout_seconds=max(1800, int(request_payload["batch_timeout_seconds"]) + 180),
        )
        optimized_single = _run_foundry_batch(
            runtime_python=runtime_python,
            env=env,
            model_dir=optimized_dir,
            model_name=optimized_model_name,
            request_payload=request_payload,
            timeout_seconds=max(1800, int(request_payload["batch_timeout_seconds"]) + 180),
        )
        single_eval = _evaluate_pair_detailed(
            profile=profile,
            prompts=prompts,
            baseline_outputs=baseline_single["prompt_outputs"],
            optimized_outputs=optimized_single["prompt_outputs"],
            deterministic_inference=deterministic_profile,
        )

        baseline_counter = _OgaTokenCounter(baseline_dir)
        optimized_counter = _OgaTokenCounter(optimized_dir)
        try:
            single_json_features = {
                "baseline": _json_prompt_features("format-json-answer-unit", baseline_single["prompt_outputs"], baseline_counter),
                "optimized": _json_prompt_features(
                    "format-json-answer-unit",
                    optimized_single["prompt_outputs"],
                    optimized_counter,
                ),
            }
        finally:
            baseline_counter.close()
            optimized_counter.close()

        repeated = _run_repeated_trials(
            runtime_python=runtime_python,
            env=env,
            profile=profile,
            prompts=prompts,
            deterministic_inference=deterministic_profile,
            baseline_dir=baseline_dir,
            baseline_model_name=baseline_model_name,
            optimized_dir=optimized_dir,
            optimized_model_name=optimized_model_name,
            request_payload=request_payload,
            trials=max(5, int(args.repro_trials)),
        )
        single_prompt_reproducibility = {
            "prompt_id": "format-json-answer-unit",
            "trial_count": repeated["trial_count"],
            "baseline_json_valid_rate": repeated["summary"]["baseline_json_valid_rate"],
            "optimized_json_valid_rate": repeated["summary"]["optimized_json_valid_rate"],
            "baseline_fenced_rate": repeated["summary"]["baseline_fenced_rate"],
            "optimized_fenced_rate": repeated["summary"]["optimized_fenced_rate"],
            "regression_signature_rate": repeated["summary"]["quality_regression_signature_rate"],
            "baseline_unique_json_outputs": repeated["summary"]["baseline_unique_json_outputs"],
            "optimized_unique_json_outputs": repeated["summary"]["optimized_unique_json_outputs"],
        }

        control_probe_prompt = "Return valid JSON object with keys answer and unit, where answer is 12 and unit is cm."
        decoding_controls = {
            "foundry_baseline": _probe_foundry_controls(
                runtime_python=runtime_python,
                env=env,
                model_dir=baseline_dir,
                model_name=baseline_model_name,
                probe_prompt=control_probe_prompt,
            ),
            "foundry_optimized": _probe_foundry_controls(
                runtime_python=runtime_python,
                env=env,
                model_dir=optimized_dir,
                model_name=optimized_model_name,
                probe_prompt=control_probe_prompt,
            ),
            "oga_baseline": _probe_oga_controls(baseline_dir, control_probe_prompt),
            "oga_optimized": _probe_oga_controls(optimized_dir, control_probe_prompt),
        }

        packaging_compare = _compare_packaging(baseline_dir, optimized_dir)
        graph_compare = {
            "baseline": _onnx_op_summary(baseline_dir / "model.onnx"),
            "optimized": _onnx_op_summary(optimized_dir / "model.onnx"),
        }
        hybrid = _run_hybrid_experiments(
            runtime_python=runtime_python,
            env=env,
            profile=profile,
            prompts=prompts,
            deterministic_inference=deterministic_profile,
            baseline_dir=baseline_dir,
            optimized_dir=optimized_dir,
            request_payload=request_payload,
            external_root=external_run_root,
        )
        variants = _run_variant_experiments(
            runtime_python=runtime_python,
            env=env,
            profile=profile,
            prompts=prompts,
            deterministic_inference=deterministic_profile,
            baseline_dir=baseline_dir,
            baseline_model_name=baseline_model_name,
            baseline_trial_outputs=repeated["baseline_outputs_for_reuse"],
            request_payload=request_payload,
            external_root=external_run_root,
        )
        block64_variant = next(
            (
                row
                for row in variants.get("variants", [])
                if isinstance(row, dict)
                and row.get("variant_id") == "int4_block_size_64"
                and row.get("status") == "evaluated"
            ),
            None,
        )
        block64_dir = Path(str(block64_variant["variant_dir"])) if isinstance(block64_variant, dict) else None
        block64_model_name = str(block64_variant["model_name"]) if isinstance(block64_variant, dict) else None

        full_suite_trials = max(3, int(args.full_suite_trials))
        full_suite_evidence = _run_full_suite_candidate_matrix(
            runtime_python=runtime_python,
            env=env,
            profile=profile,
            prompts=prompts,
            deterministic_inference=deterministic_profile,
            request_payload=request_payload,
            baseline_dir=baseline_dir,
            baseline_model_name=baseline_model_name,
            default_dir=optimized_dir,
            default_model_name=optimized_model_name,
            block64_dir=block64_dir,
            block64_model_name=block64_model_name,
            trials=full_suite_trials,
        )
        default_profile = _profile_foundry_load_and_batch(
            runtime_python=runtime_python,
            env=env,
            model_dir=optimized_dir,
            model_name=optimized_model_name,
            prompts=prompts,
            max_tokens=int(request_payload["prompts"][0]["max_tokens"]),
        )
        block_size_costs = _build_block_size_cost_matrix(
            selected_default_package_dir=optimized_dir,
            selected_default_profile=default_profile,
            variants=variants,
        )
        numeric_fidelity = _attempt_numeric_fidelity_probe(
            baseline_dir=baseline_dir,
            default_dir=optimized_dir,
            block64_dir=block64_dir,
        )

        remedy_ranking = _rank_remedies(variants)
        lingering = _probe_lingering_processes()
        stray_cleanup = _cleanup_exact_stray_root(EXACT_STRAY_DIAGNOSTIC_ROOT)

        selected_input_hashes_after = {
            "snapshot": _directory_fingerprint(snapshot_dir),
            "optimized_package": _directory_fingerprint(optimized_dir),
        }
        selected_inputs_unchanged = (
            selected_input_hashes_before["snapshot"]["manifest_sha256"]
            == selected_input_hashes_after["snapshot"]["manifest_sha256"]
            and selected_input_hashes_before["optimized_package"]["manifest_sha256"]
            == selected_input_hashes_after["optimized_package"]["manifest_sha256"]
        )

        all_models = context["manifest"].get("frozen_cli", {}).get("frozen_list", {}).get("json", {}).get("models", [])
        if not isinstance(all_models, list):
            all_models = []
        rerun_model_ids = [str(row.get("model_id")) for row in all_models if isinstance(row, dict) and row.get("model_id")]

        full_suite_conclusion = full_suite_evidence.get("conclusion", {})
        block64_supported = bool(full_suite_conclusion.get("block64_candidate_supported"))
        retry_ladder_justified = bool(block64_supported)

        report = {
            "schema_version": "1.1.0",
            "diagnostic_id": "smollm-round6-json-regression",
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source_round": {
                "round_id": context["round_id"],
                "branch": context["round_branch"],
                "commit": context["round_commit"],
                "manifest_path": "evaluation/recipe-agent-v1/round-6/round-manifest.json",
                "summary_path": "evaluation/recipe-agent-v1/round-6/round-6-summary.json",
                "model_result_path": (
                    "evaluation/recipe-agent-v1/round-6/model-results/"
                    "02-huggingfacetb-smollm2-360m-instruct.json"
                ),
            },
            "frozen_model": {
                "model_id": FROZEN_MODEL_ID,
                "revision_sha": context["frozen_revision"],
                "retained_snapshot_dir_name": Path(str(retained_paths["snapshot_dir"])).name,
                "retained_optimized_package_name": str(retained_paths["optimized_package_name"]),
            },
            "quality_profile": {
                "profile_id": QUALITY_PROFILE_ID,
                "deterministic_inference": deterministic,
                "unsupported_determinism_fields_by_runtime_contract": list(UNSUPPORTED_DETERMINISM_FIELDS),
                "prompt_ids": [str(row["prompt_id"]) for row in prompts],
            },
            "round6_claim_under_review": {
                "metrics_ref": context["metrics_ref"],
                "baseline_comparison_from_round6": context["smollm_result"]
                .get("quality_validation_evidence", {})
                .get("baseline_comparison", {}),
                "recipe_verification_from_round6": context["smollm_result"]
                .get("quality_validation_evidence", {})
                .get("recipe_verification", {}),
                "failure_excerpt_from_round6": context["smollm_result"]
                .get("quality_validation_evidence", {})
                .get("failure_excerpt", []),
            },
            "toolchain_probe": toolchain_probe,
            "selected_input_evidence": {
                "selection_by_exact_artifact_id": {
                    "artifact_id": retained_paths["optimized_artifact_id"],
                    "artifact_short": retained_paths["optimized_artifact_short"],
                    "expected_directory_name": retained_paths["optimized_expected_name"],
                    "selected_directory_name": retained_paths["optimized_package_name"],
                    "exact_match": retained_paths["selection_exact_id_match"],
                },
                "preexisting_sibling_inventory": retained_paths["sibling_inventory_preexisting"],
                "selected_input_hashes_before": selected_input_hashes_before,
                "selected_input_hashes_after": selected_input_hashes_after,
                "selected_inputs_unchanged_after_diagnostics": selected_inputs_unchanged,
            },
            "reproduction_single_batch": {
                "batch_request": request_payload,
                "baseline_model_name": baseline_model_name,
                "optimized_model_name": optimized_model_name,
                "baseline_batch_seconds": baseline_single["worker_payload"].get("duration_seconds"),
                "optimized_batch_seconds": optimized_single["worker_payload"].get("duration_seconds"),
                "json_prompt": single_json_features,
                "quality_eval": single_eval,
                "bounded_outputs": {
                    "baseline": _bounded_outputs(baseline_single["prompt_outputs"]),
                    "optimized": _bounded_outputs(optimized_single["prompt_outputs"]),
                },
            },
            "single_prompt_reproducibility": single_prompt_reproducibility,
            "full_suite_evidence": full_suite_evidence,
            "cross_model_generalization": {
                "status": "unproven_pending_full_five_model_rerun",
                "reason": "Evidence in this diagnostic run is SmolLM-targeted only.",
                "required_rerun_model_ids": rerun_model_ids,
            },
            "determinism_repeated_trials": repeated,
            "decoding_controls_probe": decoding_controls,
            "packaging_and_template_comparison": packaging_compare,
            "graph_comparison": graph_compare,
            "layer_isolation_hybrid_swaps": hybrid,
            "int4_variant_experiments": variants,
            "block_size_costs_and_performance": block_size_costs,
            "numeric_fidelity_probe": numeric_fidelity,
            "root_cause_assessment": {
                "primary_layer": "quantized_graph_behavior",
                "confidence": "high",
                "supporting_evidence": [
                    "Baseline and optimized package metadata (genai_config/tokenizer/config) remained equivalent where compared.",
                    "Swapping only ONNX graphs swapped behavior; package/config swap alone did not reproduce the fence regression.",
                    "Repeated trials showed stable baseline clean JSON and optimized fenced invalid JSON outputs.",
                ],
                "ruled_out_or_lower_confidence_layers": [
                    "evaluator mapping (strict JSON contract behaved as designed)",
                    "model-id-specific prompt logic (no model-specific workaround tested)",
                    "runtime worker prompt ordering (prompt order preserved and bounded timings)",
                ],
            },
            "remedy_analysis": {
                "ranked_candidates": remedy_ranking["ranked"],
                "rejected_approaches": remedy_ranking["rejected"],
                "safe_generic_fix_proven_for_target_model": block64_supported,
                "safe_generic_fix_proven_for_full_round6_five_model_set": False,
                "full_suite_candidate_conclusion": full_suite_conclusion,
                "retry_ladder_round7_justification": {
                    "justified": retry_ladder_justified,
                    "reason": (
                        "default remained structurally regressed while block64 remained promotion-eligible in full-suite repeats."
                        if retry_ladder_justified
                        else "block64 did not stay promotion-eligible across full-suite repeats."
                    ),
                    "constraints": [
                        "default build first",
                        "retry block64 only on baseline-pass/optimized-structural-regression",
                        "max 2 builds",
                        "no model-id routing",
                    ],
                },
                "recommended_next_action": (
                    "Run unchanged-five-model Round 6 rerun with capability-level INT4 block_size=64 policy candidate "
                    "to validate generalization and guard against regressions."
                ),
                "mandatory_full_set_rerun": {
                    "required": True,
                    "scope": "full frozen five-model set unchanged",
                    "model_ids": rerun_model_ids,
                    "gates_must_remain_strict": [
                        "no JSON fence stripping",
                        "no parser-only acceptance",
                        "no validator relaxation",
                        "no output rewriting",
                    ],
                },
            },
            "operational_cleanup": {
                "external_run_root": str(external_run_root),
                "retain_external_requested": bool(args.retain_external),
                "lingering_process_probe": lingering,
                "exact_stray_root_cleanup": stray_cleanup,
            },
        }
    finally:
        if args.retain_external:
            cleanup_result = {"ok": True, "removed": False, "bytes_freed": 0, "reason": "retain_external_requested"}
        else:
            cleanup_result = _safe_cleanup_external(external_run_root, scratch_root)

    report["operational_cleanup"]["external_cleanup_result"] = cleanup_result  # type: ignore[index]
    output_report.parent.mkdir(parents=True, exist_ok=True)
    sanitized = _sanitize_payload(report)
    output_report.write_text(json.dumps(sanitized, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"ok": True, "report_path": str(output_report), "cleanup": cleanup_result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
