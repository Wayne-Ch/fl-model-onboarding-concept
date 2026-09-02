from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
DIAG_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[4]
ROUND5_DIR = REPO_ROOT / "evaluation" / "recipe-agent-v1" / "round-5"
ROUND5_MANIFEST_PATH = ROUND5_DIR / "round-manifest.json"
ROUND5_TINY_RESULT_PATH = ROUND5_DIR / "model-results" / "01-tinyllama-tinyllama-1-1b-chat-v1-0.json"
QUALITY_PROFILE_PATH = REPO_ROOT / "config" / "quality-validation-profiles.json"
REPORT_PATH = DIAG_DIR / "diagnostic-report.json"

FROZEN_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
FROZEN_MODEL_SHA = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
QUALITY_PROFILE_ID = "textgen-basic-quality-v1"
UNSUPPORTED_DETERMINISM_FIELDS = ("temperature", "seed")
ABS_PATH_RE = re.compile(r"[A-Za-z]:(?:\\|/(?!/))[^\"'\r\n]*")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MODEL_NAME_FLAG_RE = re.compile(r"'--model-name', '([^']+)'")
TIMEOUT_SECONDS_RE = re.compile(r"timed out after (\d+)s", re.IGNORECASE)
DEFAULT_TIMEOUT_SECONDS = 900
EXTERNAL_SCRATCH_ROOT = Path(r"C:\fmo-r5-diag")
RETAINED_ROUND5_RUNTIME_ROOT = Path(r"C:\fmo-r5\r5-0902d")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object at {path}.")
    return payload


def _sanitize_text(value: str, *, max_chars: int = 220) -> str:
    cleaned = CONTROL_CHAR_RE.sub("", value)
    cleaned = ABS_PATH_RE.sub("<redacted-absolute-path>", cleaned)
    cleaned = cleaned.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    if max_chars <= 3:
        return cleaned[:max_chars]
    return cleaned[: max_chars - 3].rstrip() + "..."


def _load_round5_context() -> dict[str, Any]:
    manifest = _read_json(ROUND5_MANIFEST_PATH)
    tiny_result = _read_json(ROUND5_TINY_RESULT_PATH)
    run_id = str(manifest.get("run_id") or "").strip()
    if not run_id:
        raise RuntimeError("Round 5 manifest is missing run_id.")
    models = manifest.get("frozen_cli", {}).get("frozen_list", {}).get("json", {}).get("models", [])
    if not isinstance(models, list):
        raise RuntimeError("Round 5 manifest frozen model list is missing.")
    tiny_manifest = next(
        (row for row in models if isinstance(row, dict) and row.get("model_id") == FROZEN_MODEL_ID),
        None,
    )
    if not isinstance(tiny_manifest, dict):
        raise RuntimeError("Round 5 manifest does not include the TinyLlama frozen model.")
    sha = str(tiny_manifest.get("sha") or "").strip().lower()
    if sha != FROZEN_MODEL_SHA:
        raise RuntimeError(f"Frozen TinyLlama SHA mismatch: expected {FROZEN_MODEL_SHA}, found {sha or '<missing>'}.")

    failure_message = str(
        tiny_result.get("attempt", {}).get("failure", {}).get("message")
        or tiny_result.get("failure_summary", {}).get("error_signature")
        or ""
    )
    baseline_model_name = ""
    match = MODEL_NAME_FLAG_RE.search(failure_message)
    if match is not None:
        baseline_model_name = match.group(1).strip()
    if not baseline_model_name:
        baseline_model_name = "tinyllama-1-1b-chat-v1-0-onboarding-round5-mobius-baseline:1"
    timeout_match = TIMEOUT_SECONDS_RE.search(failure_message)
    recorded_timeout = int(timeout_match.group(1)) if timeout_match is not None else DEFAULT_TIMEOUT_SECONDS

    runtime_probe = manifest.get("toolchain_probe", {})
    expected_versions: dict[str, str | None] = {}
    probes = runtime_probe.get("probes", []) if isinstance(runtime_probe, dict) else []
    if isinstance(probes, list):
        for row in probes:
            if isinstance(row, dict) and isinstance(row.get("name"), str):
                expected_versions[str(row["name"])] = str(row.get("version")) if row.get("version") is not None else None

    return {
        "manifest": manifest,
        "tiny_result": tiny_result,
        "run_id": run_id,
        "baseline_model_name": baseline_model_name,
        "recorded_timeout_seconds": recorded_timeout,
        "failure_message": _sanitize_text(failure_message, max_chars=400),
        "expected_toolchain_versions": expected_versions,
    }


def _run_cmd(
    argv: list[str],
    *,
    timeout_seconds: int,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
        )
        elapsed = time.perf_counter() - started
        return {
            "ok": completed.returncode == 0,
            "timed_out": False,
            "returncode": completed.returncode,
            "duration_seconds": round(elapsed, 3),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "stdout_excerpt": _sanitize_text(completed.stdout, max_chars=320),
            "stderr_excerpt": _sanitize_text(completed.stderr, max_chars=320),
        }
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return {
            "ok": False,
            "timed_out": True,
            "returncode": None,
            "duration_seconds": round(elapsed, 3),
            "stdout": stdout,
            "stderr": stderr,
            "stdout_excerpt": _sanitize_text(stdout, max_chars=320),
            "stderr_excerpt": _sanitize_text(stderr, max_chars=320),
        }


def _run_worker(mode: str, payload: dict[str, Any], *, timeout_seconds: int) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        json.dump(payload, handle)
        payload_path = Path(handle.name)
    try:
        result = _run_cmd(
            [sys.executable, str(SCRIPT_PATH), "--worker", mode, "--payload-file", str(payload_path)],
            timeout_seconds=timeout_seconds,
            cwd=REPO_ROOT,
        )
        result["mode"] = mode
        if result["timed_out"]:
            return result
        output = str(result.get("stdout") or "")
        parsed = _parse_json_maybe(output)
        if isinstance(parsed, dict):
            result["payload"] = parsed
        else:
            result["ok"] = False
            result["parse_error"] = "worker output was not valid JSON"
        return result
    finally:
        payload_path.unlink(missing_ok=True)


def _parse_json_maybe(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        payload = json.loads(stripped)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(stripped[start : end + 1])
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        return None


def _read_toolchain_versions() -> dict[str, Any]:
    code = (
        "import importlib.metadata as m, json, platform, sys;"
        "pkgs=['fl-model-onboarding','onnx','onnxruntime','onnxruntime-genai','foundry-local-sdk','mobius-onnx','olive-ai'];"
        "print(json.dumps({"
        "'python_version': platform.python_version(),"
        "'python_executable_name': Path(sys.executable).name if 'Path' in globals() else sys.executable.split('\\\\\\\\')[-1],"
        "'packages': {k:m.version(k) for k in pkgs}"
        "}, sort_keys=True))"
    )
    # Path isn't available in the one-liner above; use a second snippet that's explicit.
    code = (
        "import importlib.metadata as m, json, platform, sys;"
        "from pathlib import Path;"
        "pkgs=['fl-model-onboarding','onnx','onnxruntime','onnxruntime-genai','foundry-local-sdk','mobius-onnx','olive-ai'];"
        "print(json.dumps({'python_version': platform.python_version(),'python_executable_name': Path(sys.executable).name,"
        "'packages': {k:m.version(k) for k in pkgs}}, sort_keys=True))"
    )
    result = _run_cmd([sys.executable, "-c", code], timeout_seconds=60, cwd=REPO_ROOT)
    payload = _parse_json_maybe(str(result.get("stdout") or ""))
    if not isinstance(payload, dict):
        raise RuntimeError("Unable to read current toolchain package versions.")
    return payload


def _load_quality_profile() -> dict[str, Any]:
    payload = _read_json(QUALITY_PROFILE_PATH)
    profiles = payload.get("profiles")
    if not isinstance(profiles, list):
        raise RuntimeError("Quality profile registry is missing profiles.")
    profile = next(
        (row for row in profiles if isinstance(row, dict) and row.get("profile_id") == QUALITY_PROFILE_ID),
        None,
    )
    if not isinstance(profile, dict):
        raise RuntimeError(f"Quality profile '{QUALITY_PROFILE_ID}' not found.")
    prompts = profile.get("prompts")
    if not isinstance(prompts, list):
        raise RuntimeError("Quality profile prompts are missing.")
    deterministic = profile.get("deterministic_inference")
    if not isinstance(deterministic, dict):
        raise RuntimeError("Quality profile deterministic_inference is missing.")
    return {
        "profile_id": profile.get("profile_id"),
        "version": profile.get("version"),
        "task": profile.get("task"),
        "deterministic_inference": deterministic,
        "prompts": prompts,
    }


def _find_round5_cache_paths(round5_context: dict[str, Any]) -> dict[str, Path]:
    cache_root = RETAINED_ROUND5_RUNTIME_ROOT / "cache"
    if not cache_root.is_dir():
        raise RuntimeError(f"Retained Round 5 cache root is missing: {cache_root}")
    snapshot = cache_root / f"snapshot-tinyllama-tinyllama-1.1b-chat-v1.0-{FROZEN_MODEL_SHA}"
    if not snapshot.is_dir():
        raise RuntimeError(f"Frozen TinyLlama snapshot is missing: {snapshot}")

    tiny_result = round5_context["tiny_result"]
    artifact_id = str(tiny_result.get("job", {}).get("result_artifact_id") or "").strip().lower()
    optimized_prefix = str(tiny_result.get("generated_preview", {}).get("generated_recipe", {}).get("recipe", {}).get("artifact_cache_prefix") or "")
    optimized_short = artifact_id[:12]
    if optimized_prefix and optimized_short:
        candidate = cache_root / f"{optimized_prefix}-{optimized_short}"
        if candidate.is_dir():
            return {"cache_root": cache_root, "snapshot": snapshot, "optimized_package": candidate}

    fallback = sorted(cache_root.glob("tinyllama-1-1b-chat-v1-0-*"))
    fallback_dir = next((row for row in fallback if row.is_dir() and (row / "inference_model.json").is_file()), None)
    if fallback_dir is None:
        raise RuntimeError("Unable to locate retained TinyLlama optimized package in Round 5 cache.")
    return {"cache_root": cache_root, "snapshot": snapshot, "optimized_package": fallback_dir}


def _build_baseline_package(*, external_run_root: Path, snapshot_dir: Path, baseline_model_name: str) -> Path:
    workspace_root = external_run_root / "workspace" / "tinyllama"
    baseline_dir = workspace_root / "mobius"
    workspace_root.mkdir(parents=True, exist_ok=True)
    if baseline_dir.exists():
        shutil.rmtree(baseline_dir)
    cmd = [
        sys.executable,
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
        str(baseline_dir),
    ]
    result = _run_cmd(cmd, timeout_seconds=5400, cwd=REPO_ROOT)
    if not result["ok"]:
        raise RuntimeError(f"Mobius baseline build failed: {result}")
    (baseline_dir / "inference_model.json").write_text(
        json.dumps({"Name": baseline_model_name}, indent=2),
        encoding="utf-8",
    )
    return baseline_dir


def _model_name_from_descriptor(model_dir: Path) -> str:
    payload = json.loads((model_dir / "inference_model.json").read_text(encoding="utf-8"))
    name = str(payload.get("Name") or "").strip()
    if not name:
        raise RuntimeError(f"Model descriptor missing Name at {model_dir}.")
    return name


def _runtime_worker_prompt_sequence(
    *,
    model_dir: Path,
    model_name: str,
    prompts: list[dict[str, Any]],
    max_tokens: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for prompt in prompts:
        prompt_id = str(prompt.get("prompt_id"))
        request_payload = {"prompt": str(prompt.get("prompt", "")), "max_tokens": int(max_tokens)}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
            json.dump(request_payload, handle)
            request_path = Path(handle.name)
        try:
            cmd = [
                sys.executable,
                "-m",
                "fl_model_onboarding.runtime_worker",
                "foundry-infer",
                "--model-dir",
                str(model_dir),
                "--model-name",
                model_name,
                "--request-file",
                str(request_path),
            ]
            result = _run_cmd(cmd, timeout_seconds=timeout_seconds, cwd=REPO_ROOT)
            output_text = ""
            output_payload = _parse_json_maybe(str(result.get("stdout") or ""))
            if isinstance(output_payload, dict):
                output_text = str(output_payload.get("output") or "")
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "timed_out": bool(result["timed_out"]),
                    "ok": bool(result["ok"]),
                    "returncode": result["returncode"],
                    "duration_seconds": result["duration_seconds"],
                    "output_text": output_text,
                    "output_excerpt": _sanitize_text(output_text, max_chars=160) if output_text else "",
                    "stderr_excerpt": result["stderr_excerpt"],
                }
            )
            if result["timed_out"]:
                break
        finally:
            request_path.unlink(missing_ok=True)
    return {
        "prompt_count_attempted": len(rows),
        "timed_out": any(bool(row["timed_out"]) for row in rows),
        "results": rows,
        "total_seconds": round(sum(float(row["duration_seconds"]) for row in rows), 3),
    }


def _quality_eval_from_outputs(
    *,
    profile_id: str,
    prompts: list[dict[str, Any]],
    deterministic_max_tokens: int,
    baseline_rows: list[dict[str, Any]],
    optimized_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from fl_model_onboarding.quality_validation import (  # pylint: disable=import-outside-toplevel
        PromptExecutionRecord,
        evaluate_quality_validation,
        load_quality_validation_profile_registry,
    )

    registry = load_quality_validation_profile_registry()
    profile = registry.get(profile_id)
    prompt_ids = [str(row.get("prompt_id")) for row in prompts]
    baseline_map = {str(row.get("prompt_id")): row for row in baseline_rows if row.get("ok") and not row.get("timed_out")}
    optimized_map = {str(row.get("prompt_id")): row for row in optimized_rows if row.get("ok") and not row.get("timed_out")}
    if any(prompt_id not in baseline_map for prompt_id in prompt_ids):
        return {"evaluated": False, "reason": "baseline-incomplete"}
    if any(prompt_id not in optimized_map for prompt_id in prompt_ids):
        return {"evaluated": False, "reason": "optimized-incomplete"}

    def to_records(data: dict[str, dict[str, Any]]) -> tuple[Any, ...]:
        records: list[Any] = []
        for prompt_id in prompt_ids:
            output = str(data[prompt_id].get("output_text") or data[prompt_id].get("output_excerpt") or "")
            records.append(
                PromptExecutionRecord(
                    prompt_id=prompt_id,
                    output_text=output,
                    applied_determinism=profile.deterministic_inference,
                    unsupported_determinism_fields=UNSUPPORTED_DETERMINISM_FIELDS,
                )
            )
        return tuple(records)

    baseline_records = to_records(baseline_map)
    optimized_records = to_records(optimized_map)
    outcome = evaluate_quality_validation(
        profile=profile,
        model_task="text-generation",
        baseline_outputs=baseline_records,
        optimized_outputs=optimized_records,
        require_baseline_comparison=True,
    )
    failures = []
    for row in outcome.optimized_functional.prompt_results:
        if not row.passed:
            failures.append({"prompt_id": row.prompt_id, "failures": list(row.failures)})
    regressions = []
    if outcome.baseline_comparison is not None:
        regressions = list(outcome.baseline_comparison.regressions)
    return {
        "evaluated": True,
        "can_promote": bool(outcome.promotion_evidence.can_promote),
        "optimized_functional_gate": outcome.promotion_evidence.functional_gate.value,
        "baseline_comparison_gate": outcome.promotion_evidence.baseline_comparison_gate.value,
        "metrics_gate": outcome.promotion_evidence.metrics_gate.value,
        "optimized_failures": failures,
        "baseline_comparison_regressions": regressions,
        "max_tokens_used": deterministic_max_tokens,
    }


def _diagnose(
    *,
    round5_context: dict[str, Any],
    current_design: dict[str, Any],
    single_worker_design: dict[str, Any],
    flsdk_metrics: dict[str, Any],
    oga_metrics: dict[str, Any],
) -> dict[str, Any]:
    timed_out_current = bool(current_design["baseline"]["timed_out"])
    timed_out_single = bool(single_worker_design["baseline"].get("timed_out"))
    baseline_flsdk_ok = bool(flsdk_metrics["baseline"].get("ok"))
    baseline_oga_ok = bool(oga_metrics["baseline"].get("ok"))
    malformed_baseline = not (baseline_flsdk_ok and baseline_oga_ok)

    if malformed_baseline:
        primary = "malformed_baseline_package_or_runtime_breakage"
    elif timed_out_current and not timed_out_single:
        primary = "repeated_subprocess_or_reload_hang_in_current_harness_design"
    elif timed_out_current and timed_out_single:
        primary = "model_load_or_generation_hang_independent_of_harness_design"
    else:
        primary = "round5_timeout_not_reproduced_with_retained_snapshot"

    likely_cause = []
    if primary == "malformed_baseline_package_or_runtime_breakage":
        likely_cause.append("Baseline package/runtime path appears unhealthy outside quality harness.")
    elif primary == "repeated_subprocess_or_reload_hang_in_current_harness_design":
        likely_cause.append("Per-prompt process+load/unload path timed out while single-worker load-once path stayed bounded.")
    elif primary == "model_load_or_generation_hang_independent_of_harness_design":
        likely_cause.append("Both current and single-worker designs exhibited hangs.")
    else:
        likely_cause.append(
            "Retained frozen snapshot baseline and optimized probes completed; prior Round 5 timeout appears intermittent/transient."
        )
    likely_cause.append(
        "Current quality harness is vulnerable because each prompt pays another subprocess boundary and model discovery/load cycle."
    )

    smallest_fix = {
        "proposal": (
            "Use one bounded quality worker per artifact (baseline and optimized) that loads once, executes all fixed prompts, "
            "records per-prompt timings/output, then unloads once."
        ),
        "does_not_change": [
            "prompt set",
            "deterministic max_tokens",
            "functional scoring",
            "baseline-vs-optimized pass criteria",
        ],
    }

    tests = [
        {
            "id": "quality-worker-load-once-call-count",
            "scope": "unit",
            "assertion": "Quality execution invokes FL SDK load/unload once per artifact, not once per prompt.",
        },
        {
            "id": "quality-worker-timeout-attribution",
            "scope": "unit",
            "assertion": "A forced hang in one prompt yields bounded timeout diagnostics with prompt id and phase (load vs infer).",
        },
        {
            "id": "quality-worker-semantic-equivalence",
            "scope": "integration",
            "assertion": (
                "Given identical captured prompt outputs, the promotion decision and regression labels remain identical "
                "between legacy per-prompt executor and load-once executor."
            ),
        },
    ]

    return {
        "primary_diagnosis": primary,
        "round5_timeout_error_signature": round5_context["failure_message"],
        "likely_cause": likely_cause,
        "smallest_generic_fix": smallest_fix,
        "recommended_tests": tests,
    }


def _safe_cleanup_external(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    root = EXTERNAL_SCRATCH_ROOT.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        return {"ok": False, "error": f"Refusing cleanup outside {root}: {resolved} ({exc})"}
    if not resolved.exists():
        return {"ok": True, "removed": False}
    shutil.rmtree(resolved)
    return {"ok": True, "removed": True}


def _sanitize_measurements_for_report(payload: dict[str, Any]) -> dict[str, Any]:
    clone = json.loads(json.dumps(payload))

    def visit(node: Any) -> Any:
        if isinstance(node, dict):
            for key, value in list(node.items()):
                if key in {"output_text", "stdout", "stderr"} and isinstance(value, str):
                    node[key] = _sanitize_text(value, max_chars=220)
                else:
                    node[key] = visit(value)
            return node
        if isinstance(node, list):
            return [visit(item) for item in node]
        return node

    return visit(clone)


def _collect_current_and_single_design_results(
    *,
    profile: dict[str, Any],
    baseline_dir: Path,
    baseline_name: str,
    optimized_dir: Path,
    optimized_name: str,
    timeout_seconds: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    prompts = [row for row in profile["prompts"] if isinstance(row, dict)]
    deterministic = profile["deterministic_inference"]
    deterministic_max_tokens = int(deterministic.get("max_tokens", 64))

    current_baseline = _runtime_worker_prompt_sequence(
        model_dir=baseline_dir,
        model_name=baseline_name,
        prompts=prompts,
        max_tokens=deterministic_max_tokens,
        timeout_seconds=timeout_seconds,
    )
    current_optimized = _runtime_worker_prompt_sequence(
        model_dir=optimized_dir,
        model_name=optimized_name,
        prompts=prompts,
        max_tokens=deterministic_max_tokens,
        timeout_seconds=timeout_seconds,
    )

    single_baseline_worker = _run_worker(
        "flsdk-profile",
        {
            "model_dir": str(baseline_dir),
            "model_name": baseline_name,
            "latency_prompt": "Reply with one word: Mars",
            "max_tokens_list": [1, 8, 64],
            "quality_prompts": prompts,
            "quality_max_tokens": deterministic_max_tokens,
        },
        timeout_seconds=min(3600, timeout_seconds * 4),
    )
    single_optimized_worker = _run_worker(
        "flsdk-profile",
        {
            "model_dir": str(optimized_dir),
            "model_name": optimized_name,
            "latency_prompt": "Reply with one word: Mars",
            "max_tokens_list": [1, 8, 64],
            "quality_prompts": prompts,
            "quality_max_tokens": deterministic_max_tokens,
        },
        timeout_seconds=min(3600, timeout_seconds * 4),
    )

    single_baseline_payload = single_baseline_worker.get("payload", {}) if isinstance(single_baseline_worker, dict) else {}
    single_optimized_payload = single_optimized_worker.get("payload", {}) if isinstance(single_optimized_worker, dict) else {}

    # Normalize output for quality evaluation helper.
    baseline_prompt_rows = []
    optimized_prompt_rows = []
    if isinstance(single_baseline_payload.get("quality_runs"), list):
        for row in single_baseline_payload["quality_runs"]:
            if not isinstance(row, dict):
                continue
            baseline_prompt_rows.append(
                {
                    "prompt_id": row.get("prompt_id"),
                    "ok": bool(row.get("ok")),
                    "timed_out": False,
                    "output_text": str(row.get("output_text") or ""),
                    "output_excerpt": _sanitize_text(str(row.get("output_text") or ""), max_chars=160),
                }
            )
    if isinstance(single_optimized_payload.get("quality_runs"), list):
        for row in single_optimized_payload["quality_runs"]:
            if not isinstance(row, dict):
                continue
            optimized_prompt_rows.append(
                {
                    "prompt_id": row.get("prompt_id"),
                    "ok": bool(row.get("ok")),
                    "timed_out": False,
                    "output_text": str(row.get("output_text") or ""),
                    "output_excerpt": _sanitize_text(str(row.get("output_text") or ""), max_chars=160),
                }
            )

    current_eval = _quality_eval_from_outputs(
        profile_id=str(profile["profile_id"]),
        prompts=prompts,
        deterministic_max_tokens=deterministic_max_tokens,
        baseline_rows=current_baseline["results"],
        optimized_rows=current_optimized["results"],
    )
    single_eval = _quality_eval_from_outputs(
        profile_id=str(profile["profile_id"]),
        prompts=prompts,
        deterministic_max_tokens=deterministic_max_tokens,
        baseline_rows=baseline_prompt_rows,
        optimized_rows=optimized_prompt_rows,
    )
    return (
        {"baseline": current_baseline, "optimized": current_optimized, "quality_eval": current_eval},
        {
            "baseline": single_baseline_payload,
            "optimized": single_optimized_payload,
            "baseline_worker_status": {
                "ok": single_baseline_worker.get("ok"),
                "timed_out": single_baseline_worker.get("timed_out"),
                "duration_seconds": single_baseline_worker.get("duration_seconds"),
                "stderr_excerpt": single_baseline_worker.get("stderr_excerpt"),
            },
            "optimized_worker_status": {
                "ok": single_optimized_worker.get("ok"),
                "timed_out": single_optimized_worker.get("timed_out"),
                "duration_seconds": single_optimized_worker.get("duration_seconds"),
                "stderr_excerpt": single_optimized_worker.get("stderr_excerpt"),
            },
            "quality_eval": single_eval,
        },
        {
            "baseline": single_baseline_payload,
            "optimized": single_optimized_payload,
        },
        {"prompts": prompts, "deterministic_max_tokens": deterministic_max_tokens},
    )


def _main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose TinyLlama Round 5 baseline timeout.")
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--worker", default="")
    parser.add_argument("--payload-file", default="")
    args = parser.parse_args()

    if args.worker:
        payload_path = Path(args.payload_file).resolve()
        payload = _read_json(payload_path)
        if args.worker == "flsdk-profile":
            output = _worker_flsdk_profile(payload)
        elif args.worker == "oga-profile":
            output = _worker_oga_profile(payload)
        else:
            raise SystemExit(f"Unknown worker mode: {args.worker}")
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0

    round5_context = _load_round5_context()
    toolchain_now = _read_toolchain_versions()
    profile = _load_quality_profile()
    cache_paths = _find_round5_cache_paths(round5_context)

    external_run_root = Path(tempfile.mkdtemp(prefix="tinyllama-timeout-", dir=str(EXTERNAL_SCRATCH_ROOT)))
    cleanup_result: dict[str, Any] = {"ok": False, "removed": False}
    try:
        baseline_dir = _build_baseline_package(
            external_run_root=external_run_root,
            snapshot_dir=cache_paths["snapshot"],
            baseline_model_name=round5_context["baseline_model_name"],
        )
        optimized_dir = cache_paths["optimized_package"]
        optimized_name = _model_name_from_descriptor(optimized_dir)

        current_design, single_design, flsdk_payloads, quality_meta = _collect_current_and_single_design_results(
            profile=profile,
            baseline_dir=baseline_dir,
            baseline_name=round5_context["baseline_model_name"],
            optimized_dir=optimized_dir,
            optimized_name=optimized_name,
            timeout_seconds=max(60, int(args.timeout_seconds)),
        )

        oga_baseline = _run_worker(
            "oga-profile",
            {"model_dir": str(baseline_dir), "prompt": "Reply with one word: Mars", "max_tokens_list": [1, 8, 64]},
            timeout_seconds=1800,
        )
        oga_optimized = _run_worker(
            "oga-profile",
            {"model_dir": str(optimized_dir), "prompt": "Reply with one word: Mars", "max_tokens_list": [1, 8, 64]},
            timeout_seconds=1800,
        )
        oga_metrics = {
            "baseline": oga_baseline.get("payload", {}),
            "optimized": oga_optimized.get("payload", {}),
            "baseline_worker_status": {
                "ok": oga_baseline.get("ok"),
                "timed_out": oga_baseline.get("timed_out"),
                "duration_seconds": oga_baseline.get("duration_seconds"),
                "stderr_excerpt": oga_baseline.get("stderr_excerpt"),
            },
            "optimized_worker_status": {
                "ok": oga_optimized.get("ok"),
                "timed_out": oga_optimized.get("timed_out"),
                "duration_seconds": oga_optimized.get("duration_seconds"),
                "stderr_excerpt": oga_optimized.get("stderr_excerpt"),
            },
        }

        diagnosis = _diagnose(
            round5_context=round5_context,
            current_design=current_design,
            single_worker_design=single_design,
            flsdk_metrics=flsdk_payloads,
            oga_metrics=oga_metrics,
        )

        sanitized_measurements = _sanitize_measurements_for_report(
            {
                "current_design_per_prompt_runtime_worker": current_design,
                "single_worker_load_once_design": single_design,
                "flsdk_load_generate_unload": flsdk_payloads,
                "oga_load_first_token_generate": oga_metrics,
            }
        )

        report: dict[str, Any] = {
            "schema_version": "1.0.0",
            "diagnostic_id": "tinyllama-round5-baseline-timeout",
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_round": {
                "round_id": round5_context["run_id"],
                "report_path": "evaluation/recipe-agent-v1/round-5/round-5-report.md",
                "summary_path": "evaluation/recipe-agent-v1/round-5/round-5-summary.json",
                "manifest_path": "evaluation/recipe-agent-v1/round-5/round-manifest.json",
            },
            "frozen_model": {
                "model_id": FROZEN_MODEL_ID,
                "revision_sha": FROZEN_MODEL_SHA,
                "snapshot_cache_status": "present",
                "snapshot_cache_ref": f"scratch://round-5/{round5_context['run_id']}/cache/snapshot-tinyllama-...-{FROZEN_MODEL_SHA}",
            },
            "round5_failure": {
                "recorded_timeout_seconds": round5_context["recorded_timeout_seconds"],
                "error_signature_excerpt": round5_context["failure_message"],
            },
            "toolchain": {
                "expected_from_round5_probe": round5_context["expected_toolchain_versions"],
                "observed_now": toolchain_now,
            },
            "artifacts_used": {
                "retained_round5_runtime_root": "scratch://round-5/r5-0902d",
                "optimized_package_name": cache_paths["optimized_package"].name,
                "optimized_package_ref": f"scratch://round-5/r5-0902d/cache/{cache_paths['optimized_package'].name}",
            },
            "quality_profile": {
                "profile_id": profile["profile_id"],
                "version": profile["version"],
                "task": profile["task"],
                "deterministic_inference": profile["deterministic_inference"],
                "prompt_ids": [str(row.get("prompt_id")) for row in quality_meta["prompts"]],
            },
            "measurements": sanitized_measurements,
            "diagnosis": diagnosis,
        }
    finally:
        cleanup_result = _safe_cleanup_external(external_run_root)

    report["external_cleanup"] = {
        "scratch_root": "<redacted-absolute-path>",
        "preserved_shared_snapshot_cache": "scratch://round-5/r5-0902d/cache",
        "cleanup_result": cleanup_result,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"ok": True, "report_path": str(REPORT_PATH)}))
    return 0


def _worker_flsdk_profile(payload: dict[str, Any]) -> dict[str, Any]:
    from foundry_local_sdk import Configuration, FoundryLocalManager  # pylint: disable=import-outside-toplevel

    model_dir = Path(str(payload["model_dir"])).resolve()
    model_name = str(payload["model_name"])
    latency_prompt = str(payload.get("latency_prompt") or "Reply with one word: Mars")
    max_tokens_list = [int(item) for item in payload.get("max_tokens_list", [1, 8, 64])]
    quality_prompts = payload.get("quality_prompts", [])
    quality_max_tokens = int(payload.get("quality_max_tokens", 64))

    t0 = time.perf_counter()
    FoundryLocalManager.initialize(
        Configuration(
            app_name="fl-model-onboarding-tinyllama-timeout-diag",
            model_cache_dir=str(model_dir.parent),
        )
    )
    t1 = time.perf_counter()
    manager = FoundryLocalManager.instance
    candidates = list(manager.catalog.get_cached_models())
    t2 = time.perf_counter()
    model = next((candidate for candidate in candidates if model_name in candidate.id), None)
    if model is None:
        raise RuntimeError(f"Model '{model_name}' not discovered in cache root '{model_dir.parent}'.")

    load_start = time.perf_counter()
    model.load()
    load_end = time.perf_counter()
    client = model.get_chat_client()

    latency_runs: list[dict[str, Any]] = []
    for max_tokens in max_tokens_list:
        if hasattr(client, "settings"):
            client.settings.max_tokens = max_tokens
            client.settings.temperature = 0.0
        infer_start = time.perf_counter()
        response = client.complete_chat([{"role": "user", "content": latency_prompt}])
        infer_end = time.perf_counter()
        output = _chat_output_text(response)
        latency_runs.append(
            {
                "max_tokens": max_tokens,
                "duration_seconds": round(infer_end - infer_start, 4),
                "output_text": output,
            }
        )

    quality_runs: list[dict[str, Any]] = []
    quality_total_start = time.perf_counter()
    for prompt_row in quality_prompts:
        if not isinstance(prompt_row, dict):
            continue
        prompt_id = str(prompt_row.get("prompt_id") or "")
        prompt_text = str(prompt_row.get("prompt") or "")
        if hasattr(client, "settings"):
            client.settings.max_tokens = quality_max_tokens
            client.settings.temperature = 0.0
        infer_start = time.perf_counter()
        response = client.complete_chat([{"role": "user", "content": prompt_text}])
        infer_end = time.perf_counter()
        quality_runs.append(
            {
                "prompt_id": prompt_id,
                "ok": True,
                "duration_seconds": round(infer_end - infer_start, 4),
                "output_text": _chat_output_text(response),
            }
        )
    quality_total_end = time.perf_counter()

    unload_start = time.perf_counter()
    model.unload()
    unload_end = time.perf_counter()

    return {
        "ok": True,
        "model_cache_dir_name": model_dir.parent.name,
        "candidate_count": len(candidates),
        "selected_candidate_id": model.id,
        "initialize_seconds": round(t1 - t0, 4),
        "catalog_query_seconds": round(t2 - t1, 4),
        "load_seconds": round(load_end - load_start, 4),
        "latency_runs": latency_runs,
        "quality_runs": quality_runs,
        "quality_total_seconds": round(quality_total_end - quality_total_start, 4),
        "unload_seconds": round(unload_end - unload_start, 4),
    }


def _chat_output_text(response: Any) -> str:
    choices = getattr(response, "choices", [])
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
    return str(response)


def _worker_oga_profile(payload: dict[str, Any]) -> dict[str, Any]:
    import onnxruntime_genai as og  # pylint: disable=import-outside-toplevel

    model_dir = Path(str(payload["model_dir"])).resolve()
    prompt = str(payload.get("prompt") or "Reply with one word: Mars")
    max_tokens_list = [int(item) for item in payload.get("max_tokens_list", [1, 8, 64])]

    load_start = time.perf_counter()
    model = og.Model(str(model_dir))
    tokenizer = og.Tokenizer(model)
    load_end = time.perf_counter()
    prompt_tokens = tokenizer.encode(prompt)
    prompt_len = len(prompt_tokens)

    runs: list[dict[str, Any]] = []
    for max_new_tokens in max_tokens_list:
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=prompt_len + max_new_tokens)
        generator = og.Generator(model, params)
        stream = tokenizer.create_stream()
        generator.append_tokens(prompt_tokens)
        generated = 0
        output_text = ""
        started = time.perf_counter()
        first_token_seconds: float | None = None
        while not generator.is_done() and generated < max_new_tokens:
            generator.generate_next_token()
            if first_token_seconds is None:
                first_token_seconds = time.perf_counter() - started
            tokens = generator.get_next_tokens()
            output_text += stream.decode(tokens[0])
            generated += 1
        total_seconds = time.perf_counter() - started
        runs.append(
            {
                "max_new_tokens": max_new_tokens,
                "generated_tokens": generated,
                "first_token_seconds": round(first_token_seconds or 0.0, 6),
                "total_seconds": round(total_seconds, 6),
                "output_text": output_text,
            }
        )
        del generator
        del stream
    unload_start = time.perf_counter()
    del tokenizer
    del model
    gc.collect()
    unload_end = time.perf_counter()

    return {
        "ok": True,
        "load_seconds": round(load_end - load_start, 6),
        "runs": runs,
        "unload_seconds": round(unload_end - unload_start, 6),
    }


if __name__ == "__main__":
    raise SystemExit(_main())
