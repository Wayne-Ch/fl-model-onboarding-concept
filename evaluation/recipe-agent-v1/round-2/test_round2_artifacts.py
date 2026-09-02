from __future__ import annotations

import json
import re

from pathlib import Path


ROUND_DIR = Path(__file__).resolve().parent
EXPECTED_MODELS = (
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "Qwen/Qwen2-1.5B-Instruct",
    "Qwen/Qwen2-0.5B-Instruct",
    "ibm-granite/granite-3.2-2b-instruct",
)
ABSOLUTE_WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:(?:\\|/(?!/))[^\s\"']+")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _walk_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(_walk_strings(item))
        return out
    if isinstance(value, dict):
        out: list[str] = []
        for key, item in value.items():
            out.extend(_walk_strings(str(key)))
            out.extend(_walk_strings(item))
        return out
    return []


def test_round2_summary_covers_frozen_five_and_valid_baseline() -> None:
    summary = _read_json(ROUND_DIR / "round-2-summary.json")
    assert summary["models_total"] == 5
    assert summary["round_classification"] == "valid_baseline"
    assert summary["baseline_valid"] is True
    assert summary["model_success_rate_applicable"] is True
    result_ids = tuple(
        str(row["model_id"])
        for row in summary["results"]  # type: ignore[index]
    )
    assert result_ids == EXPECTED_MODELS
    success_rate = str(summary["success_rate"])
    assert re.fullmatch(r"\d+/5", success_rate) is not None


def test_round2_manifest_records_ready_environment_probe() -> None:
    manifest = _read_json(ROUND_DIR / "round-manifest.json")
    toolchain = manifest["toolchain_probe"]  # type: ignore[index]
    assert isinstance(toolchain, dict)
    assert toolchain["ready_for_round"] is True
    assert toolchain["missing_required"] == []


def test_round2_artifacts_do_not_expose_raw_windows_absolute_paths() -> None:
    manifest = _read_json(ROUND_DIR / "round-manifest.json")
    summary = _read_json(ROUND_DIR / "round-2-summary.json")
    for payload in (manifest, summary):
        for text in _walk_strings(payload):
            assert ABSOLUTE_WINDOWS_PATH_RE.search(text) is None


def test_round2_per_model_artifacts_exist_for_all_five() -> None:
    model_results_dir = ROUND_DIR / "model-results"
    assert model_results_dir.is_dir()
    files = sorted(path for path in model_results_dir.glob("*.json"))
    assert len(files) == 5

