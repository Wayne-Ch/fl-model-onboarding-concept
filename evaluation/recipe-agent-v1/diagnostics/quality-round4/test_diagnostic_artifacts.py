"""Tests for Round 4 quality diagnostic artifact schema and consistency."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

DIAGNOSTIC_DIR = Path(__file__).resolve().parent
DIAGNOSTIC_REPORT_PATH = DIAGNOSTIC_DIR / "diagnostic-report.json"
ROUND4_DIR = DIAGNOSTIC_DIR.parent.parent / "round-4"
DOCS_PATH = DIAGNOSTIC_DIR.parents[3] / "docs" / "recipe-agent-quality-diagnostics.md"

REQUIRED_TOP_KEYS = {
    "schema_version",
    "diagnostic_id",
    "round_id",
    "round_branch",
    "round_commit",
    "quality_profile",
    "evidence_gap",
    "models_diagnosed",
    "cross_cutting_analysis",
    "recommendations",
    "generic_fixes_required",
    "rerun_requirement",
}

REQUIRED_MODEL_KEYS = {
    "model_id",
    "revision_sha",
    "architecture",
    "error_signature",
    "prompt_failures",
    "overall_diagnosis",
}

REQUIRED_PROMPT_FAILURE_KEYS = {
    "prompt_id",
    "category",
    "optimized_failure_codes",
    "classification",
    "reasoning",
}

VALID_CLASSIFICATIONS = {
    "model_capability_limitation",
    "quantization_regression",
    "evaluator_false_negative",
    "indeterminate_needs_rerun",
    "passed",
}

VALID_RECOMMENDATION_ACTIONS = {
    "keep_failure_as_legitimate",
    "fix_evaluator_semantics",
    "fix_generic_prompt",
    "fix_inference_plumbing",
    "classify_quantization_regression",
}


@pytest.fixture(scope="module")
def diagnostic_report() -> dict:
    assert DIAGNOSTIC_REPORT_PATH.is_file(), f"Missing {DIAGNOSTIC_REPORT_PATH}"
    return json.loads(DIAGNOSTIC_REPORT_PATH.read_text(encoding="utf-8"))


def test_report_exists():
    assert DIAGNOSTIC_REPORT_PATH.is_file()


def test_docs_exist():
    assert DOCS_PATH.is_file()


def test_schema_version(diagnostic_report: dict):
    assert diagnostic_report["schema_version"] == "1.0.0"


def test_required_top_level_keys(diagnostic_report: dict):
    missing = REQUIRED_TOP_KEYS - set(diagnostic_report.keys())
    assert not missing, f"Missing top-level keys: {missing}"


def test_round_id_matches_round4(diagnostic_report: dict):
    report_path = ROUND4_DIR / "round-4-report.md"
    if report_path.is_file():
        text = report_path.read_text(encoding="utf-8")
        assert diagnostic_report["round_id"] in text


def test_models_diagnosed_not_empty(diagnostic_report: dict):
    assert len(diagnostic_report["models_diagnosed"]) >= 2


def test_model_entries_have_required_keys(diagnostic_report: dict):
    for model in diagnostic_report["models_diagnosed"]:
        missing = REQUIRED_MODEL_KEYS - set(model.keys())
        assert not missing, f"Model {model.get('model_id')} missing keys: {missing}"


def test_prompt_failures_have_required_keys(diagnostic_report: dict):
    for model in diagnostic_report["models_diagnosed"]:
        for pf in model["prompt_failures"]:
            missing = REQUIRED_PROMPT_FAILURE_KEYS - set(pf.keys())
            assert not missing, (
                f"Model {model['model_id']} prompt {pf.get('prompt_id')} missing keys: {missing}"
            )


def test_classifications_are_valid(diagnostic_report: dict):
    for model in diagnostic_report["models_diagnosed"]:
        for pf in model["prompt_failures"]:
            assert pf["classification"] in VALID_CLASSIFICATIONS, (
                f"Invalid classification '{pf['classification']}' for "
                f"{model['model_id']}:{pf['prompt_id']}"
            )


def test_recommendation_actions_are_valid(diagnostic_report: dict):
    for rec in diagnostic_report["recommendations"]:
        for action in rec.get("actions", []):
            assert action["action"] in VALID_RECOMMENDATION_ACTIONS, (
                f"Invalid action '{action['action']}' for {rec['model_id']}"
            )


def test_smollm2_has_quantization_regression(diagnostic_report: dict):
    smollm2 = next(
        (m for m in diagnostic_report["models_diagnosed"]
         if "SmolLM2" in m["model_id"]),
        None,
    )
    assert smollm2 is not None
    diag = smollm2["overall_diagnosis"]
    assert len(diag["genuine_quantization_regressions"]) >= 1
    assert "format-json-answer-unit" in diag["genuine_quantization_regressions"]


def test_granite_has_no_quantization_regression(diagnostic_report: dict):
    granite = next(
        (m for m in diagnostic_report["models_diagnosed"]
         if "granite" in m["model_id"]),
        None,
    )
    assert granite is not None
    diag = granite["overall_diagnosis"]
    assert len(diag["genuine_quantization_regressions"]) == 0


def test_evidence_gap_documented(diagnostic_report: dict):
    gap = diagnostic_report["evidence_gap"]
    assert gap["actual_outputs_retained"] is False
    assert "scratch" in gap.get("scratch_ref", "").lower() or "scratch" in gap.get("reason", "").lower()


def test_error_signatures_match_round4_artifacts(diagnostic_report: dict):
    """Verify error signatures in diagnostic match the committed round-4 model results."""
    for model in diagnostic_report["models_diagnosed"]:
        model_id = model["model_id"]
        if "SmolLM2" in model_id:
            result_file = ROUND4_DIR / "model-results" / "02-huggingfacetb-smollm2-360m-instruct.json"
        elif "granite" in model_id:
            result_file = ROUND4_DIR / "model-results" / "05-ibm-granite-granite-3-2-2b-instruct.json"
        else:
            continue
        if not result_file.is_file():
            pytest.skip(f"Round 4 result not found: {result_file}")
        result = json.loads(result_file.read_text(encoding="utf-8"))
        round4_sig = result.get("failure_summary", {}).get("error_signature", "")
        assert model["error_signature"] == round4_sig, (
            f"Error signature mismatch for {model_id}"
        )


def test_generic_fixes_have_required_fields(diagnostic_report: dict):
    for fix in diagnostic_report["generic_fixes_required"]:
        assert "fix_id" in fix
        assert "type" in fix
        assert "component" in fix
        assert "description" in fix
