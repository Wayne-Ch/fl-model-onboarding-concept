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


def test_round1_summary_has_five_frozen_models() -> None:
    summary = _read_json(ROUND_DIR / "round-1-summary.json")
    assert summary["models_total"] == 5
    assert summary["results"]
    result_ids = tuple(
        str(row["model_id"])
        for row in summary["results"]  # type: ignore[index]
    )
    assert result_ids == EXPECTED_MODELS


def test_round1_summary_marks_original_run_invalid_environment() -> None:
    summary = _read_json(ROUND_DIR / "round-1-summary.json")
    assert summary["round_classification"] == "invalid_environment"
    assert summary["baseline_valid"] is False
    assert summary["model_success_rate_applicable"] is False
    assert summary["success_rate"] == "not_applicable"
    assert summary["attempt_success_rate"] == "0/5"


def test_round1_manifest_contains_required_snapshots() -> None:
    manifest = _read_json(ROUND_DIR / "round-manifest.json")
    assert manifest["manifest_path"] == "evaluation/recipe-agent-v1/models.json"
    assert manifest["manifest_invariants"]
    assert manifest["frozen_cli"]
    assert manifest["quality_profile"]
    assert manifest["model_registry_and_catalog_snapshot"]
    assert manifest["round_outcome"]["round_classification"] == "invalid_environment"  # type: ignore[index]
    assert manifest["round_outcome"]["baseline_valid"] is False  # type: ignore[index]


def test_round1_manifest_tool_probe_has_canonical_non_empty_probe_field() -> None:
    manifest = _read_json(ROUND_DIR / "round-manifest.json")
    toolchain = manifest["toolchain_probe"]  # type: ignore[index]
    assert isinstance(toolchain, dict)
    probes = toolchain.get("probes")
    assert isinstance(probes, list) and probes
    for probe in probes:
        assert isinstance(probe, dict)
        assert isinstance(probe.get("probe"), str) and probe.get("probe")
        assert "detail" not in probe


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


def test_round1_artifacts_do_not_expose_raw_windows_absolute_paths() -> None:
    manifest = _read_json(ROUND_DIR / "round-manifest.json")
    summary = _read_json(ROUND_DIR / "round-1-summary.json")
    for payload in (manifest, summary):
        for text in _walk_strings(payload):
            assert ABSOLUTE_WINDOWS_PATH_RE.search(text) is None


def test_per_model_artifacts_exist_for_all_five() -> None:
    model_results_dir = ROUND_DIR / "model-results"
    assert model_results_dir.is_dir()
    files = sorted(path for path in model_results_dir.glob("*.json"))
    assert len(files) == 5
