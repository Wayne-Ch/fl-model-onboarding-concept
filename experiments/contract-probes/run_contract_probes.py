from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HF_TOKEN_PATTERN = re.compile(r"hf_[A-Za-z0-9]{20,}")
AUTH_BEARER_PATTERN = re.compile(r"(Authorization:\s*Bearer\s+)[^\s]+", flags=re.IGNORECASE)
KV_TOKEN_PATTERN = re.compile(r"(token['\"]?\s*[:=]\s*['\"])[^'\"]+(['\"])", flags=re.IGNORECASE)


@dataclass
class Candidate:
    key: str
    model_id: str
    task: str
    olive_precision: str
    byom_name: str


CANDIDATES = [
    Candidate(
        key="llm",
        model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        task="chat",
        olive_precision="int4",
        byom_name="smollm2-contract-probe:1",
    ),
    Candidate(
        key="asr",
        model_id="distil-whisper/distil-medium.en",
        task="speech",
        olive_precision="int8",
        byom_name="distil-whisper-contract-probe:1",
    ),
]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def redact(text: str) -> str:
    redacted = HF_TOKEN_PATTERN.sub("***REDACTED***", text)
    redacted = AUTH_BEARER_PATTERN.sub(r"\1***REDACTED***", redacted)
    redacted = KV_TOKEN_PATTERN.sub(r"\1***REDACTED***\2", redacted)
    return redacted


def truncate(value: str, limit: int = 1000) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}...<truncated {len(value) - limit} chars>"


@dataclass
class CommandResult:
    stage: str
    args: list[str]
    cwd: str | None
    started_at: str
    duration_seconds: float
    returncode: int | None
    timed_out: bool
    stdout: str
    stderr: str
    log_file: str

    @property
    def success(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "args": self.args,
            "cwd": self.cwd,
            "started_at": self.started_at,
            "duration_seconds": round(self.duration_seconds, 3),
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "success": self.success,
            "stdout_preview": truncate(self.stdout, 4000),
            "stderr_preview": truncate(self.stderr, 4000),
            "log_file": self.log_file,
        }


class CommandRunner:
    def __init__(self, scratch_dir: Path):
        self.scratch_dir = scratch_dir
        self.logs_dir = scratch_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.count = 0

    def run(
        self,
        stage: str,
        args: list[str],
        timeout_seconds: int | None = None,
        cwd: Path | None = None,
    ) -> CommandResult:
        self.count += 1
        started = time.monotonic()
        started_at = utc_now()
        timed_out = False
        returncode: int | None
        stdout = ""
        stderr = ""
        try:
            process_env = os.environ.copy()
            process_env.setdefault("PYTHONIOENCODING", "utf-8")
            process_env.setdefault("PYTHONUTF8", "1")
            completed = subprocess.run(
                args,
                cwd=str(cwd) if cwd else None,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                check=False,
                env=process_env,
            )
            returncode = completed.returncode
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
        except FileNotFoundError as exc:
            returncode = 127
            stderr = str(exc)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = None
            stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode(errors="replace")
            stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode(errors="replace")

        duration = time.monotonic() - started
        stdout = redact(stdout)
        stderr = redact(stderr)
        log_file = self.logs_dir / f"{self.count:03d}-{slug(stage)}.json"
        payload = {
            "stage": stage,
            "args": args,
            "cwd": str(cwd) if cwd else None,
            "started_at": started_at,
            "duration_seconds": round(duration, 3),
            "returncode": returncode,
            "timed_out": timed_out,
            "stdout": stdout,
            "stderr": stderr,
        }
        log_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return CommandResult(
            stage=stage,
            args=args,
            cwd=str(cwd) if cwd else None,
            started_at=started_at,
            duration_seconds=duration,
            returncode=returncode,
            timed_out=timed_out,
            stdout=stdout,
            stderr=stderr,
            log_file=str(log_file),
        )


def infer_mobius_style(commands: dict[str, str | None]) -> str:
    if commands.get("mobius"):
        return "subcommand"
    if commands.get("mobiusbuild"):
        return "split"
    return "missing"


def resolve_executable(name: str) -> str | None:
    located = shutil.which(name)
    if located:
        return located
    scripts_dir = Path(sys.executable).parent
    suffix = ".exe" if os.name == "nt" else ""
    candidate = scripts_dir / f"{name}{suffix}"
    if candidate.exists():
        return str(candidate)
    return None


def find_commands() -> dict[str, str | None]:
    names = [
        "python",
        "pip",
        "foundry",
        "olive",
        "mobius",
        "mobiusbuild",
        "mobiusinfo",
        "mobiuslist",
        "mobiuslisteps",
        "mobiuslisttasks",
        "hf",
    ]
    return {name: resolve_executable(name) for name in names}


def package_versions() -> dict[str, str]:
    packages = [
        "foundry-local-sdk",
        "mobius-onnx",
        "olive-ai",
        "onnx",
        "onnxruntime",
        "onnxruntime-genai",
        "huggingface_hub",
        "torch",
    ]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def safe_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed, None
        return None, "json-root-not-object"
    except json.JSONDecodeError as exc:
        return None, str(exc)


def collect_file_inventory(root: Path, max_files: int = 250) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "path": str(path.relative_to(root)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                }
            )
        if len(rows) >= max_files:
            rows.append({"path": "...truncated...", "bytes": 0})
            break
    return rows


def find_onnx_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.onnx") if path.is_file())


def resolve_revision(model_id: str) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model_id, revision="main")
        return {"success": True, "sha": info.sha, "private": bool(getattr(info, "private", False))}
    except Exception as exc:  # noqa: BLE001 - explicit probe capture
        return {"success": False, "error": str(exc)}


def first_failed_stage(events: list[dict[str, Any]]) -> str | None:
    for event in events:
        if not event.get("success", False):
            return str(event.get("stage"))
    return None


def summarize_foundry_catalog(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {"parsed": False}
    top_keys = list(payload.keys())
    first_key = top_keys[0] if top_keys else None
    first_item = payload.get(first_key, [{}])[0] if first_key and payload.get(first_key) else {}
    return {
        "parsed": True,
        "top_keys": top_keys,
        "first_item_keys": sorted(first_item.keys()) if isinstance(first_item, dict) else [],
        "count": len(payload.get(first_key, [])) if first_key and isinstance(payload.get(first_key), list) else 0,
    }


def build_mobius_command(
    commands: dict[str, str | None],
    style: str,
    action: str,
    args: list[str],
) -> list[str] | None:
    if style == "subcommand" and commands.get("mobius"):
        return [commands["mobius"], action, *args]

    if style == "split":
        split_name = {
            "build": "mobiusbuild",
            "info": "mobiusinfo",
            "listtasks": "mobiuslisttasks",
            "listeps": "mobiuslisteps",
        }.get(action)
        if split_name and commands.get(split_name):
            return [commands[split_name], *args]
        if action == "build" and commands.get("mobiusbuild"):
            return [commands["mobiusbuild"], *args]
    return None


def try_onnx_checker(paths: list[Path]) -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    try:
        import onnx
    except Exception as exc:  # noqa: BLE001 - explicit probe capture
        return {"success": False, "error": f"onnx import failed: {exc}"}

    for path in paths:
        row = {"path": str(path), "success": False}
        try:
            onnx.checker.check_model(str(path))
            row["success"] = True
        except Exception as exc:  # noqa: BLE001 - explicit probe capture
            row["error"] = str(exc)
        output.append(row)
    return {"success": any(item["success"] for item in output), "checks": output}


def try_ort_load(path: Path) -> dict[str, Any]:
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return {
            "success": True,
            "inputs": [
                {"name": item.name, "shape": [str(dim) for dim in item.shape], "type": item.type}
                for item in session.get_inputs()
            ],
            "outputs": [item.name for item in session.get_outputs()],
        }
    except Exception as exc:  # noqa: BLE001 - explicit probe capture
        return {"success": False, "error": str(exc)}


def try_oga_text_generation(model_dir: Path) -> dict[str, Any]:
    try:
        import onnxruntime_genai as og

        model = og.Model(str(model_dir))
        tokenizer = og.Tokenizer(model)
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=24)
        generator = og.Generator(model, params)
        generator.append_tokens(tokenizer.encode("Reply with OK."))
        generated: list[int] = []
        for _ in range(12):
            if generator.is_done():
                break
            generator.generate_next_token()
            next_tokens = generator.get_next_tokens()
            if len(next_tokens) > 0:
                generated.append(int(next_tokens[0]))
        return {
            "success": True,
            "generated_token_count": len(generated),
            "generated_text_preview": truncate(tokenizer.decode(generated), 250),
        }
    except Exception as exc:  # noqa: BLE001 - explicit probe capture
        return {"success": False, "error": str(exc)}


def try_oga_speech_transcription(model_dir: Path, scratch_dir: Path) -> dict[str, Any]:
    try:
        import onnxruntime_genai as og

        sample_path = scratch_dir / "samples" / "oga-silence.wav"
        write_silence_wav(sample_path)

        config = og.Config(str(model_dir))
        config.clear_providers()
        config.append_provider("cpu")
        model = og.Model(config)
        processor = model.create_multimodal_processor()
        audios = og.Audios.open(str(sample_path))
        prompts = ["<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"]
        inputs = processor(prompts, audios=audios)

        params = og.GeneratorParams(model)
        params.set_search_options(
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
            max_length=128,
            batch_size=1,
        )
        generator = og.Generator(model, params)
        generator.set_inputs(inputs)
        while not generator.is_done():
            generator.generate_next_token()
        tokens = generator.get_sequence(0)
        text = processor.decode(tokens)
        return {"success": True, "transcription_preview": truncate(str(text), 250)}
    except Exception as exc:  # noqa: BLE001 - explicit probe capture
        return {"success": False, "error": str(exc)}


def write_silence_wav(path: Path, seconds: int = 1, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = b"\x00\x00" * sample_rate * seconds
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(frames)


def _run_foundry_sdk_probe_subprocess(
    candidate: Candidate,
    model_dir: Path,
    scratch_dir: Path,
) -> dict[str, Any]:
    payload = {
        "task": candidate.task,
        "byom_name": candidate.byom_name,
        "model_cache_dir": str(model_dir.parent),
        "audio_path": str(scratch_dir / "samples" / f"{candidate.key}-silence.wav"),
    }
    script = r"""
import json
import wave
import sys
from pathlib import Path

from foundry_local_sdk import Configuration, FoundryLocalManager

cfg = json.loads(sys.argv[1])
result = {"success": False}
try:
    config = Configuration(app_name="contract-probe-sdk-subprocess", model_cache_dir=cfg["model_cache_dir"])
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance
    cached_models = manager.catalog.get_cached_models()
    result["cached_model_ids"] = [m.id for m in cached_models]
    model = next((m for m in cached_models if cfg["byom_name"] in m.id), None)
    if model is None:
        result["stage"] = "fl_sdk_discovery"
        result["error"] = f"Model '{cfg['byom_name']}' not discovered in model_cache_dir"
        print(json.dumps(result))
        raise SystemExit(0)

    result["selected_model_id"] = model.id
    model.load()
    try:
        if cfg["task"] == "chat":
            client = model.get_chat_client()
            response = client.complete_chat([{"role": "user", "content": "Reply with: OK"}])
            preview = str(response)
            if hasattr(response, "choices") and response.choices:
                msg = getattr(response.choices[0], "message", None)
                if msg is not None and hasattr(msg, "content"):
                    preview = str(msg.content)
            result["success"] = True
            result["stage"] = "fl_sdk_inference"
            result["inference"] = {"response_preview": preview[:250]}
        else:
            wav_path = Path(cfg["audio_path"])
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(wav_path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(b"\x00\x00" * 16000)
            client = model.get_audio_client()
            response = client.transcribe(str(wav_path))
            result["success"] = True
            result["stage"] = "fl_sdk_inference"
            result["inference"] = {"response_preview": str(response)[:250]}
    finally:
        model.unload()
except Exception as exc:  # noqa: BLE001
    if "stage" not in result:
        result["stage"] = "fl_sdk_load"
    result["success"] = False
    result["error"] = str(exc)

print(json.dumps(result))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, json.dumps(payload)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    stdout = redact(completed.stdout or "")
    stderr = redact(completed.stderr or "")
    lines = [line for line in stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            payload_result = json.loads(line)
            if isinstance(payload_result, dict):
                payload_result["subprocess_returncode"] = completed.returncode
                if stderr:
                    payload_result["subprocess_stderr_preview"] = truncate(stderr, 800)
                return payload_result
        except json.JSONDecodeError:
            continue

    return {
        "success": False,
        "stage": "fl_sdk_probe_subprocess",
        "error": "Failed to parse SDK probe subprocess JSON output",
        "subprocess_returncode": completed.returncode,
        "subprocess_stdout_preview": truncate(stdout, 1200),
        "subprocess_stderr_preview": truncate(stderr, 1200),
    }


def try_foundry_sdk(model_dir: Path, candidate: Candidate, scratch_dir: Path) -> dict[str, Any]:
    contract_path = model_dir / "inference_model.json"
    created_contract = False
    if not contract_path.exists():
        contract_path.write_text(json.dumps({"Name": candidate.byom_name}, indent=2), encoding="utf-8")
        created_contract = True

    result = _run_foundry_sdk_probe_subprocess(candidate, model_dir, scratch_dir)
    result["inference_model_created"] = created_contract
    return result


def run_candidate_pipeline(
    candidate: Candidate,
    runner: CommandRunner,
    commands: dict[str, str | None],
    mobius_style: str,
    scratch_dir: Path,
    build_timeout_seconds: int,
    olive_timeout_seconds: int,
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "candidate": candidate.model_id,
        "task": candidate.task,
        "events": events,
    }

    revision = resolve_revision(candidate.model_id)
    result["hf_revision"] = revision
    events.append({"stage": f"{candidate.key}.hf_revision", "success": bool(revision.get("success")), "detail": revision})
    if revision.get("success"):
        result["hf_revision_pinning"] = {
            "strategy": "recorded-only",
            "note": "Current mobius build CLI in this environment exposes no --revision flag; revision SHA is recorded in report.",
        }

    info_cmd = build_mobius_command(commands, mobius_style, "info", [candidate.model_id])
    if info_cmd:
        info_result = runner.run(f"{candidate.key}.mobius_info", info_cmd, timeout_seconds=300)
        events.append({"stage": info_result.stage, "success": info_result.success, "command": info_result.as_dict()})
        result["mobius_info"] = info_result.as_dict()
    else:
        events.append(
            {
                "stage": f"{candidate.key}.mobius_info",
                "success": False,
                "detail": "mobius info command not available",
            }
        )

    mobius_output_dir = scratch_dir / "mobius" / candidate.key
    mobius_output_dir.mkdir(parents=True, exist_ok=True)
    if mobius_style == "subcommand":
        build_args = [
            "--model",
            candidate.model_id,
            "--ep",
            "cpu",
            "--runtime",
            "ort-genai",
            "--dtype",
            "f32",
            str(mobius_output_dir),
        ]
    else:
        build_args = [
            "--model",
            candidate.model_id,
            "--output",
            str(mobius_output_dir),
            "--ep",
            "cpu",
            "--runtime",
            "ort-genai",
            "--dtype",
            "f32",
        ]

    build_attempts: list[dict[str, Any]] = []
    for action, label in [("build", "primary"), ("build", "retry")]:
        build_cmd = build_mobius_command(commands, mobius_style, action, build_args)
        if not build_cmd:
            break
        if label == "retry" and mobius_style != "subcommand":
            break
        if label == "retry" and commands.get("mobiusbuild") is None:
            break
        if label == "retry":
            build_cmd = [commands["mobiusbuild"], *build_args]
        build_result = runner.run(
            f"{candidate.key}.mobius_build.{label}",
            build_cmd,
            timeout_seconds=build_timeout_seconds,
        )
        build_attempts.append(build_result.as_dict())
        events.append({"stage": build_result.stage, "success": build_result.success, "command": build_result.as_dict()})
        if build_result.success:
            break

    mobius_success = any(item.get("success") for item in build_attempts)
    mobius_runtime = mobius_output_dir / "runtime_compatibility.json"
    runtime_metadata: dict[str, Any] | None = None
    if mobius_runtime.exists():
        try:
            runtime_metadata = json.loads(mobius_runtime.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            runtime_metadata = {"parse_error": str(exc)}

    result["mobius"] = {
        "success": mobius_success,
        "output_dir": str(mobius_output_dir),
        "build_attempts": build_attempts,
        "files": collect_file_inventory(mobius_output_dir),
        "runtime_compatibility": runtime_metadata,
    }

    if not mobius_success:
        result["status"] = "blocked"
        result["first_failure_stage"] = first_failed_stage(events)
        return result

    onnx_files = find_onnx_files(mobius_output_dir)
    result["mobius"]["onnx_files"] = [str(path) for path in onnx_files]

    olive_success = False
    olive_attempts: list[dict[str, Any]] = []
    olive_output_dir = scratch_dir / "olive" / candidate.key
    olive_output_dir.mkdir(parents=True, exist_ok=True)
    if commands.get("olive"):
        source_candidates = [str(mobius_output_dir)]
        if onnx_files:
            source_candidates.append(str(onnx_files[0]))
        precision_candidates = [candidate.olive_precision]
        if candidate.key == "llm":
            precision_candidates.append("int8")
        elif candidate.task == "speech":
            precision_candidates.append("fp32")

        for model_source in source_candidates:
            for precision in precision_candidates:
                olive_cmd = [
                    commands["olive"],
                    "optimize",
                    "--model_name_or_path",
                    model_source,
                    "--task",
                    "text-generation-with-past" if candidate.task == "chat" else "automatic-speech-recognition",
                    "--output_path",
                    str(olive_output_dir),
                    "--device",
                    "cpu",
                    "--provider",
                    "CPUExecutionProvider",
                    "--precision",
                    precision,
                    "--log_level",
                    "1",
                ]
                olive_result = runner.run(
                    f"{candidate.key}.olive_optimize.{precision}.{slug(model_source)}",
                    olive_cmd,
                    timeout_seconds=olive_timeout_seconds,
                )
                olive_attempts.append(olive_result.as_dict())
                events.append(
                    {
                        "stage": olive_result.stage,
                        "success": olive_result.success,
                        "precision": precision,
                        "source": model_source,
                        "command": olive_result.as_dict(),
                    }
                )
                if olive_result.success:
                    olive_success = True
                    break
            if olive_success:
                break
    else:
        events.append({"stage": f"{candidate.key}.olive_missing", "success": False, "detail": "olive command not found"})

    result["olive"] = {
        "success": olive_success,
        "output_dir": str(olive_output_dir),
        "attempts": olive_attempts,
        "files": collect_file_inventory(olive_output_dir),
    }

    validation_root = olive_output_dir if olive_success and find_onnx_files(olive_output_dir) else mobius_output_dir
    validation_onnx = find_onnx_files(validation_root)
    if not validation_onnx:
        events.append({"stage": f"{candidate.key}.validation.no_onnx", "success": False})
        result["status"] = "blocked"
        result["first_failure_stage"] = first_failed_stage(events)
        return result

    checker = try_onnx_checker(validation_onnx[:3])
    events.append({"stage": f"{candidate.key}.onnx_checker", "success": bool(checker.get("success")), "detail": checker})
    ort = try_ort_load(validation_onnx[0])
    events.append({"stage": f"{candidate.key}.ort_load", "success": bool(ort.get("success")), "detail": ort})

    oga_dir = None
    for root in [validation_root, mobius_output_dir, olive_output_dir]:
        if (root / "genai_config.json").exists():
            oga_dir = root
            break
    if oga_dir and candidate.task == "chat":
        oga = try_oga_text_generation(oga_dir)
    elif oga_dir and candidate.task == "speech":
        oga = try_oga_speech_transcription(oga_dir, scratch_dir)
    else:
        oga = {
            "success": False,
            "error": f"genai_config.json not found for {candidate.task} task",
        }
    events.append({"stage": f"{candidate.key}.oga", "success": bool(oga.get("success")), "detail": oga})

    sdk_root = validation_root
    if not (sdk_root / "genai_config.json").exists() and (mobius_output_dir / "genai_config.json").exists():
        sdk_root = mobius_output_dir
    foundry_sdk = try_foundry_sdk(sdk_root, candidate, scratch_dir)
    events.append({"stage": f"{candidate.key}.fl_sdk", "success": bool(foundry_sdk.get("success")), "detail": foundry_sdk})

    result["validation"] = {
        "validation_root": str(validation_root),
        "onnx_checker": checker,
        "ort": ort,
        "oga": oga,
        "foundry_sdk": foundry_sdk,
    }

    happy_path = bool(
        checker.get("success")
        and ort.get("success")
        and (oga.get("success") if candidate.task == "chat" else True)
        and foundry_sdk.get("success")
    )
    result["status"] = "happy_path" if happy_path else "blocked"
    result["first_failure_stage"] = first_failed_stage(events)
    return result


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    run_stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{run_stamp}-{uuid.uuid4().hex[:8]}"
    scratch_root = Path(args.scratch_root).expanduser().resolve()
    scratch_dir = scratch_root / run_id
    scratch_dir.mkdir(parents=True, exist_ok=True)

    runner = CommandRunner(scratch_dir)
    commands = find_commands()
    mobius_style = infer_mobius_style(commands)

    probe: dict[str, Any] = {
        "run_id": run_id,
        "started_at": utc_now(),
        "scratch_dir": str(scratch_dir),
        "system": {
            "platform": platform.platform(),
            "python": sys.version,
        },
        "commands": commands,
        "mobius_style": mobius_style,
        "package_versions": package_versions(),
    }

    command_probes: list[dict[str, Any]] = []
    probe_commands: list[tuple[str, list[str], int]] = [
        ("tool.foundry.version", ["foundry", "--version"], 120),
        ("tool.foundry.help", ["foundry", "--help"], 120),
        ("tool.foundry.model.help", ["foundry", "model", "--help"], 120),
        ("tool.foundry.model.list.help", ["foundry", "model", "list", "--help"], 120),
        ("tool.foundry.model.load.help", ["foundry", "model", "load", "--help"], 120),
        ("tool.foundry.model.list.json", ["foundry", "model", "list", "-o", "json"], 180),
        ("tool.foundry.model.list.variants.json", ["foundry", "model", "list", "--variants", "-o", "json"], 240),
    ]
    if commands.get("mobius"):
        probe_commands.extend(
            [
                ("tool.mobius.help", [commands["mobius"], "--help"], 120),
                ("tool.mobius.build.help", [commands["mobius"], "build", "--help"], 120),
                ("tool.mobius.list.tasks", [commands["mobius"], "list", "tasks"], 120),
                ("tool.mobius.list.eps", [commands["mobius"], "list", "eps"], 120),
            ]
        )
    elif commands.get("mobiusbuild"):
        probe_commands.append(("tool.mobiusbuild.help", [commands["mobiusbuild"], "--help"], 120))
        if commands.get("mobiuslisttasks"):
            probe_commands.append(("tool.mobiuslisttasks", [commands["mobiuslisttasks"]], 120))
        if commands.get("mobiuslisteps"):
            probe_commands.append(("tool.mobiuslisteps", [commands["mobiuslisteps"]], 120))

    if commands.get("olive"):
        probe_commands.extend(
            [
                ("tool.olive.help", [commands["olive"], "--help"], 120),
                ("tool.olive.optimize.help", [commands["olive"], "optimize", "--help"], 120),
            ]
        )

    for stage, command, timeout in probe_commands:
        result = runner.run(stage, command, timeout_seconds=timeout)
        command_probes.append(result.as_dict())

    probe["command_probes"] = command_probes

    foundry_list_command = next((item for item in command_probes if item["stage"] == "tool.foundry.model.list.json"), None)
    foundry_variants_command = next(
        (item for item in command_probes if item["stage"] == "tool.foundry.model.list.variants.json"),
        None,
    )
    catalog_summary: dict[str, Any] = {}
    if foundry_list_command:
        raw = json.loads(Path(foundry_list_command["log_file"]).read_text(encoding="utf-8"))
        payload, error = safe_json(raw.get("stdout", ""))
        catalog_summary["models"] = summarize_foundry_catalog(payload)
        if error:
            catalog_summary["models_parse_error"] = error
    if foundry_variants_command:
        raw = json.loads(Path(foundry_variants_command["log_file"]).read_text(encoding="utf-8"))
        payload, error = safe_json(raw.get("stdout", ""))
        catalog_summary["variants"] = summarize_foundry_catalog(payload)
        if error:
            catalog_summary["variants_parse_error"] = error
    probe["foundry_catalog_summary"] = catalog_summary

    candidate_results = [
        run_candidate_pipeline(
            candidate=candidate,
            runner=runner,
            commands=commands,
            mobius_style=mobius_style,
            scratch_dir=scratch_dir,
            build_timeout_seconds=args.build_timeout_seconds,
            olive_timeout_seconds=args.olive_timeout_seconds,
        )
        for candidate in CANDIDATES
    ]
    probe["candidate_results"] = candidate_results
    probe["completed_at"] = utc_now()

    summary_path = scratch_dir / "probe-summary.json"
    summary_path.write_text(json.dumps(probe, indent=2), encoding="utf-8")
    probe["summary_path"] = str(summary_path)
    return probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Foundry Local model onboarding contract probes.")
    parser.add_argument(
        "--scratch-root",
        default=str(Path(tempfile.gettempdir()) / "fl-contract-probe-runs"),
        help="Directory for probe outputs and logs (outside the repo by default).",
    )
    parser.add_argument(
        "--build-timeout-seconds",
        type=int,
        default=7200,
        help="Timeout for each Mobius build command.",
    )
    parser.add_argument(
        "--olive-timeout-seconds",
        type=int,
        default=5400,
        help="Timeout for each Olive optimize command.",
    )
    parser.add_argument(
        "--keep-scratch",
        action="store_true",
        help="Keep scratch directory even if probe fails.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        probe = run_probe(args)
    except Exception as exc:  # noqa: BLE001 - explicit probe capture
        print(f"Probe failed: {exc}", file=sys.stderr)
        return 1

    print(f"Probe run completed: {probe['run_id']}")
    print(f"Summary JSON: {probe['summary_path']}")
    for candidate in probe.get("candidate_results", []):
        print(
            f"- {candidate['candidate']}: {candidate.get('status')} "
            f"(first_failure_stage={candidate.get('first_failure_stage')})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
