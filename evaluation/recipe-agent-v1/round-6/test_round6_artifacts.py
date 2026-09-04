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


def test_round6_summary_covers_frozen_five_and_split_outcomes() -> None:
    summary = _read_json(ROUND_DIR / "round-6-summary.json")
    assert summary["models_total"] == 5
    result_ids = tuple(
        str(row["model_id"])
        for row in summary["results"]  # type: ignore[index]
    )
    assert result_ids == EXPECTED_MODELS
    assert summary["valid_baseline"] in {True, False}
    assert re.fullmatch(r"\d+/5", str(summary["recipe_verified_rate"])) is not None
    assert re.fullmatch(r"\d+/5", str(summary["model_capability_all_pass_rate"])) is not None
    counts = summary["recipe_verification_status_counts"]  # type: ignore[index]
    assert isinstance(counts, dict)
    assert counts["VERIFIED"] + counts["BLOCKED"] + counts["INCONCLUSIVE"] + counts["UNKNOWN"] == 5


def test_round6_summary_results_include_quality_split_fields() -> None:
    summary = _read_json(ROUND_DIR / "round-6-summary.json")
    results = summary["results"]  # type: ignore[index]
    assert isinstance(results, list)
    for row in results:
        assert "recipe_verification" in row
        assert "model_capability" in row
        assert "batch_worker" in row
        assert "quality_validation_metrics_ref" in row
        assert "quality_validation_failure_excerpt" in row


def test_round6_manifest_records_ready_environment_probe() -> None:
    manifest = _read_json(ROUND_DIR / "round-manifest.json")
    toolchain = manifest["toolchain_probe"]  # type: ignore[index]
    assert isinstance(toolchain, dict)
    assert "ready_for_round" in toolchain
    assert "missing_required" in toolchain


def test_round6_artifacts_do_not_expose_raw_windows_absolute_paths() -> None:
    manifest = _read_json(ROUND_DIR / "round-manifest.json")
    summary = _read_json(ROUND_DIR / "round-6-summary.json")
    for payload in (manifest, summary):
        for text in _walk_strings(payload):
            assert ABSOLUTE_WINDOWS_PATH_RE.search(text) is None


def test_round6_reuse_checks_record_build_delta_when_reused() -> None:
    summary = _read_json(ROUND_DIR / "round-6-summary.json")
    checks = summary["reuse_checks"]  # type: ignore[index]
    assert isinstance(checks, list)
    for row in checks:
        assert isinstance(row, dict)
        if row.get("reuse_identity_match") is True:
            assert row.get("build_invocation_delta") == 0


def test_round6_report_contains_split_sections() -> None:
    report = (ROUND_DIR / "round-6-report.md").read_text(encoding="utf-8")
    assert "Recipe Verification (blocking / promotion)" in report
    assert "Model Capability (non-blocking advisory)" in report
    assert "Reuse verification (post-promotion)" in report


def test_round6_per_model_artifacts_exist_for_all_five() -> None:
    model_results_dir = ROUND_DIR / "model-results"
    assert model_results_dir.is_dir()
    files = sorted(path for path in model_results_dir.glob("*.json"))
    assert len(files) == 5
