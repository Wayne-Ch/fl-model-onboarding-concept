from __future__ import annotations

import json
import re

from pathlib import Path
from typing import Any

ROUND_DIR = Path(__file__).resolve().parent
EXPECTED_MODELS = (
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "Qwen/Qwen2-1.5B-Instruct",
    "Qwen/Qwen2-0.5B-Instruct",
    "ibm-granite/granite-3.2-2b-instruct",
)
ABSOLUTE_WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:(?:\\|/(?!/))[^\s\"']+")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _walk_strings(value: Any) -> list[str]:
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


def test_round7_summary_covers_frozen_five_models_and_round7_fields() -> None:
    summary = _read_json(ROUND_DIR / "round-7-summary.json")
    assert summary["models_total"] == 5
    result_ids = tuple(str(row["model_id"]) for row in summary["results"])  # type: ignore[index]
    assert result_ids == EXPECTED_MODELS
    assert summary["valid_baseline"] in {True, False}
    assert re.fullmatch(r"\d+/5", str(summary["recipe_verified_rate"])) is not None
    assert re.fullmatch(r"\d+/5", str(summary["model_capability_all_pass_rate"])) is not None
    assert re.fullmatch(r"\d+/5", str(summary["selected_candidate_reuse_zero_build_rate"])) is not None
    assert summary["retry_count_expected"] == 1
    assert summary["retry_models_expected"] == ["HuggingFaceTB/SmolLM2-360M-Instruct"]
    assert "parent_attempt_recipe_verified_rate" in summary
    assert "first_request_aggregate_dispatch_totals" in summary
    assert "branch_source_identity" in summary
    assert "round6_delta" in summary


def test_round7_results_include_winner_and_candidate_timeline_evidence() -> None:
    summary = _read_json(ROUND_DIR / "round-7-summary.json")
    results = summary["results"]  # type: ignore[index]
    assert isinstance(results, list)
    assert len(results) == 5
    for row in results:
        assert isinstance(row, dict)
        assert "workflow_outcome" in row
        assert "candidate_selection" in row
        assert "winner_candidate" in row
        assert "winner_recipe_verification" in row
        assert "winner_model_capability" in row
        assert "winner_recipe_status" in row
        assert row["winner_recipe_status"] in {"verified", "blocked", "inconclusive", "unknown"}
        assert "first_request_behavior_observation" in row
        assert "second_request_reuse" in row


def test_round7_reuse_checks_record_measured_zero_dispatch_for_submitted_rows() -> None:
    summary = _read_json(ROUND_DIR / "round-7-summary.json")
    checks = summary["reuse_checks"]  # type: ignore[index]
    assert isinstance(checks, list)
    submitted = [row for row in checks if isinstance(row, dict) and row.get("second_request_submitted") is True]
    assert len(submitted) == int(summary["recipe_verified_count"])
    for row in submitted:
        assert row["reuse_workflow_outcome"] == "reused"
        assert row["reuse_zero_build_verified"] is True
        assert row["runner_dispatch_mapping_present_for_reuse_attempt"] is False
        reuse = row["reuse_evidence"]
        assert isinstance(reuse, dict)
        assert reuse["reused_without_build"] is True
        assert reuse["runner_dispatch_count"] == 0
        assert reuse["mobius_invocation_count"] == 0
        assert reuse["olive_invocation_count"] == 0
        assert row["source_ids_match_expected_winner"] is True
        assert row["reuse_evidence_row_matches_response"] is True
        persisted = row["reuse_evidence_row"]
        assert isinstance(persisted, dict)
        assert persisted["reused_without_build"] is True
        assert persisted["runner_dispatch_count"] == 0
        assert persisted["mobius_invocation_count"] == 0
        assert persisted["olive_invocation_count"] == 0


def test_round7_manifest_and_summary_include_enrichment_block() -> None:
    manifest = _read_json(ROUND_DIR / "round-manifest.json")
    enrichment = manifest["round7_enrichment"]  # type: ignore[index]
    assert isinstance(enrichment, dict)
    assert "recipe_verified_rate" in enrichment
    assert "selected_candidate_reuse_zero_build_rate" in enrichment
    assert "post_reuse_workspace_cleanup" in enrichment


def test_round7_artifacts_do_not_expose_raw_windows_absolute_paths() -> None:
    manifest = _read_json(ROUND_DIR / "round-manifest.json")
    summary = _read_json(ROUND_DIR / "round-7-summary.json")
    model_results = sorted((ROUND_DIR / "model-results").glob("*.json"))
    assert len(model_results) == 5
    payloads: list[dict[str, Any]] = [manifest, summary]
    payloads.extend(_read_json(path) for path in model_results)
    for payload in payloads:
        for text in _walk_strings(payload):
            assert ABSOLUTE_WINDOWS_PATH_RE.search(text) is None


def test_round7_report_contains_required_sections() -> None:
    report = (ROUND_DIR / "round-7-report.md").read_text(encoding="utf-8")
    assert "Recipe Verification (winner-selected)" in report
    assert "Model Capability (non-blocking advisory)" in report
    assert "Retry and dispatch evidence" in report
    assert "Cleanup and process evidence" in report
    assert "Delta from Round 6" in report
