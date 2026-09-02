from __future__ import annotations

import json

from pathlib import Path

DIAG_DIR = Path(__file__).resolve().parent
REPORT_PATH = DIAG_DIR / "diagnostic-report.json"
DOC_PATH = DIAG_DIR.parents[3] / "docs" / "smollm-json-regression.md"


def _read_report() -> dict[str, object]:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def _find_variant(report: dict[str, object], variant_id: str) -> dict[str, object]:
    variants_root = report["int4_variant_experiments"]  # type: ignore[index]
    assert isinstance(variants_root, dict)
    variants = variants_root["variants"]  # type: ignore[index]
    assert isinstance(variants, list)
    for row in variants:
        if isinstance(row, dict) and row.get("variant_id") == variant_id:
            return row
    raise AssertionError(f"Variant '{variant_id}' not found")


def test_report_exists() -> None:
    assert REPORT_PATH.is_file()


def test_doc_exists() -> None:
    assert DOC_PATH.is_file()


def test_frozen_smollm_identity() -> None:
    report = _read_report()
    frozen = report["frozen_model"]  # type: ignore[index]
    assert isinstance(frozen, dict)
    assert frozen["model_id"] == "HuggingFaceTB/SmolLM2-360M-Instruct"
    assert frozen["revision_sha"] == "a10cc1512eabd3dde888204e902eca88bddb4951"


def test_repeated_trials_reproduce_round6_json_regression() -> None:
    report = _read_report()
    repeated = report["determinism_repeated_trials"]  # type: ignore[index]
    assert isinstance(repeated, dict)
    summary = repeated["summary"]  # type: ignore[index]
    assert isinstance(summary, dict)
    assert summary["baseline_json_valid_rate"] == "6/6"
    assert summary["optimized_json_valid_rate"] == "0/6"
    assert summary["optimized_fenced_rate"] == "6/6"
    assert summary["quality_regression_signature_rate"] == "6/6"


def test_packaging_and_template_are_not_the_regression_layer() -> None:
    report = _read_report()
    packaging = report["packaging_and_template_comparison"]  # type: ignore[index]
    assert isinstance(packaging, dict)
    assert packaging["genai_config_exact_equal"] is True
    templates = packaging["chat_template_fingerprints"]  # type: ignore[index]
    assert isinstance(templates, dict)
    baseline = templates["baseline"]  # type: ignore[index]
    optimized = templates["optimized"]  # type: ignore[index]
    assert isinstance(baseline, dict)
    assert isinstance(optimized, dict)
    assert baseline["tokenizer_config_chat_template_sha256"] == optimized["tokenizer_config_chat_template_sha256"]


def test_hybrid_swaps_isolate_graph_as_introducing_layer() -> None:
    report = _read_report()
    hybrid = report["layer_isolation_hybrid_swaps"]  # type: ignore[index]
    assert isinstance(hybrid, dict)
    hybrid_a = hybrid["hybrid_a_baseline_package_plus_optimized_graph"]  # type: ignore[index]
    hybrid_b = hybrid["hybrid_b_optimized_package_plus_baseline_graph"]  # type: ignore[index]
    assert isinstance(hybrid_a, dict)
    assert isinstance(hybrid_b, dict)
    assert hybrid_a["json_prompt_has_fence"] is True
    assert hybrid_a["quality_eval"]["can_promote"] is False  # type: ignore[index]
    assert hybrid_b["json_prompt_has_fence"] is False
    assert hybrid_b["quality_eval"]["can_promote"] is True  # type: ignore[index]


def test_int4_block_size_64_is_top_candidate_on_target_model() -> None:
    report = _read_report()
    default_variant = _find_variant(report, "int4_default")
    block64 = _find_variant(report, "int4_block_size_64")
    assert default_variant["status"] == "evaluated"
    assert default_variant["summary"]["can_promote_rate"] == "0/3"  # type: ignore[index]
    assert default_variant["summary"]["json_fenced_rate"] == "3/3"  # type: ignore[index]
    assert block64["status"] == "evaluated"
    assert block64["summary"]["can_promote_rate"] == "5/5"  # type: ignore[index]
    assert block64["summary"]["json_parse_ok_rate"] == "5/5"  # type: ignore[index]
    assert block64["summary"]["json_fenced_rate"] == "0/5"  # type: ignore[index]


def test_remedy_analysis_requires_full_unchanged_five_model_rerun() -> None:
    report = _read_report()
    remedy = report["remedy_analysis"]  # type: ignore[index]
    assert isinstance(remedy, dict)
    ranked = remedy["ranked_candidates"]  # type: ignore[index]
    assert isinstance(ranked, list)
    assert ranked[0]["candidate"] == "capability-level Olive INT4 block_size=64"  # type: ignore[index]
    assert ranked[0]["status"] == "proven_on_target_model"  # type: ignore[index]
    assert remedy["safe_generic_fix_proven_for_full_round6_five_model_set"] is False
    rerun = remedy["mandatory_full_set_rerun"]  # type: ignore[index]
    assert isinstance(rerun, dict)
    assert rerun["required"] is True
    model_ids = rerun["model_ids"]  # type: ignore[index]
    assert isinstance(model_ids, list)
    assert len(model_ids) == 5


def test_cleanup_and_process_containment_recorded() -> None:
    report = _read_report()
    cleanup = report["operational_cleanup"]  # type: ignore[index]
    assert isinstance(cleanup, dict)
    cleanup_result = cleanup["external_cleanup_result"]  # type: ignore[index]
    lingering = cleanup["lingering_process_probe"]  # type: ignore[index]
    assert isinstance(cleanup_result, dict)
    assert isinstance(lingering, dict)
    assert cleanup_result["ok"] is True
    assert lingering["ok"] is True
    assert lingering["count"] == 0
