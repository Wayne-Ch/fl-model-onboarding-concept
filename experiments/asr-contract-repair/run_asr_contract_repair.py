from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import re
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import onnxruntime_genai as og

from asr_contract_adapter import (
    apply_full_adapter,
    apply_minimal_parser_fix,
    collect_file_inventory,
    compare_config_graph_and_reference,
    ensure_audio_processor_config,
    hardlink_or_copy_tree,
    inspect_onnx_graph,
    read_json,
    sha256_file,
    write_json,
)

WHISPER_PROMPT = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    description: str
    adapter_mode: str


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def safe_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def get_foundry_version() -> str:
    result = subprocess.run(
        ["foundry", "--version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode == 0:
        return (result.stdout or "").strip()
    return f"error({result.returncode}): {(result.stderr or result.stdout).strip()}"


def normalize_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def transcript_match(expected: str, actual: str) -> bool:
    expected_norm = normalize_text(expected)
    actual_norm = normalize_text(actual)
    if not expected_norm:
        return False
    return expected_norm in actual_norm


def detach_file_from_hardlink(path: Path) -> None:
    if not path.exists():
        return
    data = path.read_bytes()
    path.unlink()
    path.write_bytes(data)


def create_tts_wav(path: Path, text: str) -> dict[str, Any]:
    escaped_text = text.replace("'", "''")
    command = (
        "Add-Type -AssemblyName System.Speech;"
        "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        f"$s.SetOutputToWaveFile('{str(path)}');"
        f"$s.Speak('{escaped_text}');"
        "$s.Dispose();"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        return {
            "success": False,
            "stage": "audio_sample_generation",
            "error": (result.stderr or result.stdout).strip(),
            "command": command,
        }
    return {
        "success": True,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "expected_transcript": text,
        "generator": "windows-tts-system.speech",
    }


def run_oga_transcription_gate(
    package_dir: Path,
    audio_path: Path,
    expected_text: str,
) -> dict[str, Any]:
    try:
        model = og.Model(str(package_dir))
        processor = model.create_multimodal_processor()
        audios = og.Audios.open(str(audio_path))
        inputs = processor([WHISPER_PROMPT], audios=audios)
        params = og.GeneratorParams(model)
        params.set_search_options(
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
            max_length=96,
            batch_size=1,
        )
        generator = og.Generator(model, params)
        generator.set_inputs(inputs)
        while not generator.is_done():
            generator.generate_next_token()
        sequence = generator.get_sequence(0)
        transcript = str(processor.decode(sequence))
        return {
            "stage": "oga_transcription",
            "success": True,
            "transcript": transcript,
            "expected_transcript": expected_text,
            "expected_match": transcript_match(expected_text, transcript),
        }
    except RuntimeError as exc:
        return {
            "stage": "oga_transcription",
            "success": False,
            "error": str(exc),
        }


def run_foundry_gate_subprocess(
    package_dir: Path,
    byom_name: str,
    audio_path: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    payload = {
        "model_cache_dir": str(package_dir.parent),
        "byom_name": byom_name,
        "audio_path": str(audio_path),
    }
    script = r"""
import json
import sys

from foundry_local_sdk import Configuration, FoundryLocalManager

payload = json.loads(sys.argv[1])
result = {
    "stage": "fl_sdk_discovery",
    "success": False,
}

try:
    config = Configuration(
        app_name="asr-contract-repair-fl-subprocess",
        model_cache_dir=payload["model_cache_dir"],
    )
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance
    models = manager.catalog.get_cached_models()
    model_ids = [m.id for m in models]
    result["cached_model_ids"] = model_ids

    model = next((m for m in models if m.id == payload["byom_name"]), None)
    if model is None:
        model = next((m for m in models if payload["byom_name"] in m.id), None)
    if model is None:
        result["error"] = f"Model '{payload['byom_name']}' not discovered."
        print(json.dumps(result))
        raise SystemExit(0)

    result["stage"] = "fl_sdk_load"
    result["selected_model_id"] = model.id
    model.load()
    try:
        client = model.get_audio_client()
        result["stage"] = "fl_sdk_transcription"
        response = client.transcribe(payload["audio_path"])
        result["success"] = True
        result["transcript"] = str(getattr(response, "text", response))
    finally:
        model.unload()
except Exception as exc:  # noqa: BLE001 - explicit probe capture
    result["success"] = False
    result["error"] = str(exc)

print(json.dumps(result))
"""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script, json.dumps(payload)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode(errors="replace")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode(errors="replace")
        return {
            "stage": "fl_sdk_probe_timeout",
            "success": False,
            "timed_out": True,
            "timeout_seconds": timeout_seconds,
            "subprocess_stdout_preview": stdout[-1000:],
            "subprocess_stderr_preview": stderr[-1000:],
        }

    lines = [line for line in (completed.stdout or "").splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                parsed["subprocess_returncode"] = completed.returncode
                if completed.stderr:
                    parsed["subprocess_stderr_preview"] = completed.stderr[-1000:]
                return parsed
        except json.JSONDecodeError:
            continue

    return {
        "stage": "fl_sdk_probe_subprocess",
        "success": False,
        "error": "Failed to parse Foundry subprocess JSON output.",
        "subprocess_returncode": completed.returncode,
        "subprocess_stdout_preview": (completed.stdout or "")[-1000:],
        "subprocess_stderr_preview": (completed.stderr or "")[-1000:],
    }


def first_failure_stage(gates: list[dict[str, Any]]) -> str | None:
    for gate in gates:
        if not gate.get("success"):
            return str(gate.get("stage"))
    return None


def run_candidate_gates(
    package_dir: Path,
    byom_name: str,
    audio_path: Path,
    expected_text: str,
    foundry_timeout_seconds: int,
) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []

    try:
        _ = read_json(package_dir / "genai_config.json")
        gates.append({"stage": "json_load", "success": True})
    except json.JSONDecodeError as exc:
        gates.append({"stage": "json_load", "success": False, "error": str(exc)})
        return gates

    try:
        _ = og.Config(str(package_dir))
        gates.append({"stage": "oga_parser_load", "success": True})
    except RuntimeError as exc:
        gates.append({"stage": "oga_parser_load", "success": False, "error": str(exc)})
        return gates

    try:
        _ = og.Model(str(package_dir))
        gates.append({"stage": "oga_model_load", "success": True})
    except RuntimeError as exc:
        gates.append({"stage": "oga_model_load", "success": False, "error": str(exc)})
        return gates

    gates.append(run_oga_transcription_gate(package_dir, audio_path, expected_text))
    gates.append(
        run_foundry_gate_subprocess(
            package_dir=package_dir,
            byom_name=byom_name,
            audio_path=audio_path,
            timeout_seconds=foundry_timeout_seconds,
        )
    )
    return gates


def evaluate_outcome(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    for candidate in candidates:
        gates = candidate["gates"]
        if all(gate.get("success") for gate in gates):
            transcription_gate = next(g for g in gates if g["stage"] == "oga_transcription")
            foundry_gate = next(g for g in gates if g["stage"] == "fl_sdk_transcription")
            if transcription_gate.get("expected_match") and foundry_gate.get("transcript"):
                return {
                    "result": "adapter-success",
                    "winning_candidate": candidate["candidate_id"],
                    "summary": "Deterministic package/config adaptation succeeded through OGA and Foundry transcription gates.",
                }

    full_candidate = next((c for c in candidates if c["adapter_mode"] == "full"), None)
    if full_candidate is None:
        return {
            "result": "inconclusive",
            "summary": "No full-adapter candidate executed.",
        }

    oga_gate = next((g for g in full_candidate["gates"] if g["stage"] == "oga_transcription"), None)
    fl_gate = next((g for g in full_candidate["gates"] if g["stage"] == "fl_sdk_transcription"), None)
    parser_gate = next((g for g in full_candidate["gates"] if g["stage"] == "oga_parser_load"), None)

    irreducible = []
    if oga_gate and not oga_gate.get("success") and "Missing Input: position_ids" in str(oga_gate.get("error", "")):
        irreducible.append(
            {
                "mismatch": "Whisper decoder requires position_ids input",
                "owner": "Mobius producer and/or OGA runtime",
                "evidence_stage": "oga_transcription",
                "error_signature": "Missing Input: position_ids",
            }
        )

    if fl_gate and not fl_gate.get("success"):
        irreducible.append(
            {
                "mismatch": "Foundry audio transcription fails after load",
                "owner": "FL SDK task/runtime contract",
                "evidence_stage": str(fl_gate.get("stage")),
                "error_signature": str(fl_gate.get("error", "")),
            }
        )

    if parser_gate and parser_gate.get("success") and irreducible:
        return {
            "result": "source-change-required",
            "summary": "Parser-compatible config adaptation is exhausted; source changes are required for full ASR transcription path.",
            "irreducible_mismatches": irreducible,
        }

    return {
        "result": "inconclusive",
        "summary": "Unable to prove a successful adapter within bounded candidates.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASR runtime-contract adapter probe for retained Mobius Whisper package.")
    parser.add_argument(
        "--source-package",
        required=True,
        help="Path to retained Mobius ASR package root.",
    )
    parser.add_argument(
        "--scratch-root",
        default=str(Path(tempfile.gettempdir()) / "asr-contract-repair-runs"),
        help="Scratch root for candidate package copies and evidence.",
    )
    parser.add_argument(
        "--expected-transcript",
        default="hello world from contract test",
        help="Expected transcript phrase for generated TTS audio.",
    )
    parser.add_argument(
        "--foundry-timeout-seconds",
        type=int,
        default=900,
        help="Timeout per Foundry SDK gate subprocess.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_package = Path(args.source_package).expanduser().resolve()
    scratch_root = Path(args.scratch_root).expanduser().resolve()
    run_id = f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    run_dir = scratch_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    source_config = read_json(source_package / "genai_config.json")
    encoder_graph = inspect_onnx_graph(source_package / "encoder" / "model.onnx")
    decoder_graph = inspect_onnx_graph(source_package / "decoder" / "model.onnx")
    source_comparison = compare_config_graph_and_reference(
        package_dir=source_package,
        config=source_config,
        encoder_graph=encoder_graph,
        decoder_graph=decoder_graph,
    )

    sample_path = run_dir / "known-transcript.wav"
    audio_sample = create_tts_wav(sample_path, args.expected_transcript)

    report: dict[str, Any] = {
        "run_id": run_id,
        "started_at": utc_now(),
        "source_package": str(source_package),
        "scratch_dir": str(run_dir),
        "versions": {
            "onnxruntime_genai": getattr(og, "__version__", "unknown"),
            "onnxruntime": package_version("onnxruntime"),
            "foundry-local-sdk": package_version("foundry-local-sdk"),
            "mobius-onnx": package_version("mobius-onnx"),
            "foundry_cli": get_foundry_version(),
        },
        "source_inventory": collect_file_inventory(source_package),
        "source_contract": {
            "encoder_graph": encoder_graph,
            "decoder_graph": decoder_graph,
            "comparison": source_comparison,
        },
        "audio_sample": audio_sample,
        "candidates": [],
        "reproduction": {
            "command": (
                "python experiments\\asr-contract-repair\\run_asr_contract_repair.py "
                "--source-package <retained_asr_package> --scratch-root <scratch_root>"
            )
        },
    }

    if not audio_sample.get("success"):
        report["completed_at"] = utc_now()
        report["final_assessment"] = {
            "result": "blocked",
            "summary": "Unable to create known-transcript audio sample.",
        }
        write_json(run_dir / "asr-contract-repair-report.json", report)
        print(json.dumps(report["final_assessment"], indent=2))
        print(f"Report: {run_dir / 'asr-contract-repair-report.json'}")
        return 1

    candidate_specs = [
        CandidateSpec(
            candidate_id="minimal-parser-fix",
            description="Schema-preserving key fix only (decoder_input_ids -> input_ids key).",
            adapter_mode="minimal",
        ),
        CandidateSpec(
            candidate_id="full-contract-adapter",
            description="Programmatic OGA-whisper contract adapter using graph-derived mappings.",
            adapter_mode="full",
        ),
    ]

    for spec in candidate_specs:
        candidate_root = run_dir / "candidates" / safe_slug(spec.candidate_id)
        package_dir = candidate_root / "asr"
        hardlink_or_copy_tree(source_package, package_dir)
        detach_file_from_hardlink(package_dir / "genai_config.json")
        detach_file_from_hardlink(package_dir / "inference_model.json")

        original_config = read_json(package_dir / "genai_config.json")
        if spec.adapter_mode == "minimal":
            adapted_config, config_changes = apply_minimal_parser_fix(original_config)
            audio_cfg_created = False
        else:
            adapted_config, config_changes = apply_full_adapter(
                config=original_config,
                encoder_graph=encoder_graph,
                decoder_graph=decoder_graph,
            )
            audio_cfg_created = ensure_audio_processor_config(package_dir, encoder_graph)

        write_json(package_dir / "genai_config.json", adapted_config)
        byom_name = f"distil-whisper-asr-contract-repair-{safe_slug(spec.candidate_id)}:1"
        write_json(package_dir / "inference_model.json", {"Name": byom_name})

        candidate_comparison = compare_config_graph_and_reference(
            package_dir=package_dir,
            config=adapted_config,
            encoder_graph=encoder_graph,
            decoder_graph=decoder_graph,
        )

        gates = run_candidate_gates(
            package_dir=package_dir,
            byom_name=byom_name,
            audio_path=sample_path,
            expected_text=args.expected_transcript,
            foundry_timeout_seconds=args.foundry_timeout_seconds,
        )
        report["candidates"].append(
            {
                "candidate_id": spec.candidate_id,
                "description": spec.description,
                "adapter_mode": spec.adapter_mode,
                "package_dir": str(package_dir),
                "config_changes": config_changes,
                "audio_processor_config_created": audio_cfg_created,
                "comparison": candidate_comparison,
                "gates": gates,
                "first_failure_stage": first_failure_stage(gates),
            }
        )

    report["final_assessment"] = evaluate_outcome(report["candidates"])
    report["completed_at"] = utc_now()
    report_path = run_dir / "asr-contract-repair-report.json"
    write_json(report_path, report)

    print(f"Run: {run_id}")
    print(f"Result: {report['final_assessment']['result']}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
