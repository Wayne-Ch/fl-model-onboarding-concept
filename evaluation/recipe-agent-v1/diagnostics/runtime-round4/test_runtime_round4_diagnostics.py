from __future__ import annotations

import importlib.util
import json

from pathlib import Path


DIAG_DIR = Path(__file__).resolve().parent
ROUND4_DIR = DIAG_DIR.parents[1] / "round-4"
EVIDENCE_PATH = DIAG_DIR / "runtime_round4_evidence.json"


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_reconcile_module():
    module_path = DIAG_DIR / "reconcile_genai_outputs.py"
    spec = importlib.util.spec_from_file_location("runtime_round4_reconcile_genai_outputs", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_round4_failure_signatures_match_diagnostic_scope() -> None:
    tiny = _read_json(ROUND4_DIR / "model-results" / "01-tinyllama-tinyllama-1-1b-chat-v1-0.json")
    qwen15 = _read_json(ROUND4_DIR / "model-results" / "03-qwen-qwen2-1-5b-instruct.json")
    qwen05 = _read_json(ROUND4_DIR / "model-results" / "04-qwen-qwen2-0-5b-instruct.json")

    tiny_error = str(tiny["failure_summary"]["error_signature"])  # type: ignore[index]
    qwen15_error = str(qwen15["failure_summary"]["error_signature"])  # type: ignore[index]
    qwen05_error = str(qwen05["failure_summary"]["error_signature"])  # type: ignore[index]

    assert "Model output was not found: logits" in tiny_error
    assert "GatherBlockQuantized(1)" in qwen15_error
    assert "GatherBlockQuantized(1)" in qwen05_error


def test_tinyllama_evidence_confirms_generic_output_reconciliation_fix() -> None:
    evidence = _read_json(EVIDENCE_PATH)
    tiny = evidence["tinyllama"]  # type: ignore[index]
    assert isinstance(tiny, dict)

    assert tiny["olive_decoder_logits_mapping_before"] == "logits"
    assert tiny["olive_decoder_logits_mapping_after"] == "logits_Q4"

    runtime = tiny["runtime_validation_results"]  # type: ignore[index]
    assert isinstance(runtime, dict)
    assert runtime["olive_unpatched"]["ok"] is False  # type: ignore[index]
    assert runtime["olive_reconciled_config"]["ok"] is True  # type: ignore[index]

    foundry = tiny["foundry_results"]  # type: ignore[index]
    assert isinstance(foundry, dict)
    assert foundry["unpatched"]["ok"] is False  # type: ignore[index]
    assert foundry["reconciled_config"]["ok"] is True  # type: ignore[index]


def test_qwen_evidence_confirms_runtime_capability_boundary() -> None:
    evidence = _read_json(EVIDENCE_PATH)
    qwen = evidence["qwen2_0_5b"]  # type: ignore[index]
    assert isinstance(qwen, dict)

    olive_default = qwen["olive_default"]  # type: ignore[index]
    assert isinstance(olive_default, dict)
    assert olive_default["op_counts"]["GatherBlockQuantized"] == 3  # type: ignore[index]
    assert olive_default["op_counts"]["MatMulNBits"] > 0  # type: ignore[index]
    assert olive_default["runtime_validation"]["ok"] is False  # type: ignore[index]
    assert "GatherBlockQuantized" in str(olive_default["runtime_validation"]["error_contains"])  # type: ignore[index]

    no_gather = qwen["rtn_nodes_to_exclude_all_gather"]  # type: ignore[index]
    assert isinstance(no_gather, dict)
    assert no_gather["op_counts"]["GatherBlockQuantized"] == 0  # type: ignore[index]
    assert no_gather["op_counts"]["MatMulNBits"] > 0  # type: ignore[index]
    assert no_gather["runtime_validation"]["ok"] is False  # type: ignore[index]
    assert "MatMulNBits" in str(no_gather["runtime_validation"]["error_contains"])  # type: ignore[index]

    qdq = qwen["rtn_nodes_to_exclude_all_gather_and_qdq_rewrite"]  # type: ignore[index]
    assert isinstance(qdq, dict)
    assert qdq["op_counts"]["MatMulNBits"] == 0  # type: ignore[index]
    assert qdq["op_counts"]["DequantizeLinear"] > 0  # type: ignore[index]
    assert qdq["runtime_validation"]["ok"] is False  # type: ignore[index]
    assert "DequantizeLinear(24)" in str(qdq["runtime_validation"]["error_contains"])  # type: ignore[index]

    f32 = qwen["mobius_f32_then_olive_int4"]  # type: ignore[index]
    assert isinstance(f32, dict)
    assert f32["runtime_validation"]["ok"] is True  # type: ignore[index]
    assert f32["observed_scale_dtypes"] == ["FLOAT"]  # type: ignore[index]


def test_reconcile_helper_remaps_missing_output_by_quantized_suffix() -> None:
    module = _load_reconcile_module()
    report = module.build_decoder_output_reconciliation(
        graph_outputs=["logits_Q4", "present.0.key", "present.0.value"],
        decoder_outputs={"logits": "logits"},
    )
    assert report["is_safe_to_apply"] is True
    assert report["proposed_remap"] == {"logits": "logits_Q4"}
    assert report["resolved_decoder_outputs"]["logits"] == "logits_Q4"
