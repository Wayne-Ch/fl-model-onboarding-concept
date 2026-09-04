from __future__ import annotations

import json
import subprocess
import sys

from pathlib import Path

DIAG_DIR = Path(__file__).resolve().parent
REPORT_PATH = DIAG_DIR / "diagnostic-report.json"
DOC_PATH = DIAG_DIR.parents[3] / "docs" / "smollm-json-regression.md"
HARNESS_PATH = DIAG_DIR / "run_smollm_json_regression_diagnostics.py"


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


def _find_full_suite_candidate(report: dict[str, object], candidate_id: str) -> dict[str, object]:
    full_suite = report["full_suite_evidence"]  # type: ignore[index]
    assert isinstance(full_suite, dict)
    candidates = full_suite["candidates"]  # type: ignore[index]
    assert isinstance(candidates, list)
    for row in candidates:
        if isinstance(row, dict) and row.get("candidate_id") == candidate_id:
            return row
    raise AssertionError(f"Full-suite candidate '{candidate_id}' not found")


def _find_cost_row(report: dict[str, object], variant: str) -> dict[str, object]:
    costs = report["block_size_costs_and_performance"]  # type: ignore[index]
    assert isinstance(costs, dict)
    rows = costs["rows"]  # type: ignore[index]
    assert isinstance(rows, list)
    for row in rows:
        if isinstance(row, dict) and row.get("variant") == variant:
            return row
    raise AssertionError(f"Cost row '{variant}' not found")


def test_report_and_doc_exist() -> None:
    assert REPORT_PATH.is_file()
    assert DOC_PATH.is_file()


def test_toolchain_probe_guard_rejects_unacknowledged_conflation() -> None:
    cmd = [
        sys.executable,
        str(HARNESS_PATH),
        "--toolchain-probe-only",
        "--runtime-python",
        sys.executable,
    ]
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert completed.returncode != 0
    combined = (completed.stdout or "") + "\n" + (completed.stderr or "")
    assert "allow-interpreter-conflation" in combined


def test_toolchain_probe_reports_runtime_subprocess_identity() -> None:
    report = _read_report()
    probe = report["toolchain_probe"]  # type: ignore[index]
    assert isinstance(probe, dict)
    assert probe["probe_source"] == "runtime_subprocess"
    runtime_reported = probe["runtime_reported"]  # type: ignore[index]
    assert isinstance(runtime_reported, dict)
    assert runtime_reported["probe_source"] == "runtime_subprocess"
    conflation = probe["interpreter_conflation"]  # type: ignore[index]
    assert isinstance(conflation, dict)
    assert conflation["allow_interpreter_conflation"] is True


def test_selected_input_evidence_records_exact_id_and_hash_stability() -> None:
    report = _read_report()
    selected = report["selected_input_evidence"]  # type: ignore[index]
    assert isinstance(selected, dict)
    selection = selected["selection_by_exact_artifact_id"]  # type: ignore[index]
    assert isinstance(selection, dict)
    assert selection["exact_match"] is True
    before = selected["selected_input_hashes_before"]  # type: ignore[index]
    after = selected["selected_input_hashes_after"]  # type: ignore[index]
    assert isinstance(before, dict)
    assert isinstance(after, dict)
    assert before["snapshot"]["manifest_sha256"] == after["snapshot"]["manifest_sha256"]  # type: ignore[index]
    assert before["optimized_package"]["manifest_sha256"] == after["optimized_package"]["manifest_sha256"]  # type: ignore[index]
    assert selected["selected_inputs_unchanged_after_diagnostics"] is True


def test_single_prompt_reproducibility_rates_match_regression_signature() -> None:
    report = _read_report()
    repro = report["single_prompt_reproducibility"]  # type: ignore[index]
    assert isinstance(repro, dict)
    assert repro["baseline_json_valid_rate"] == "6/6"
    assert repro["optimized_json_valid_rate"] == "0/6"
    assert repro["optimized_fenced_rate"] == "6/6"
    assert repro["regression_signature_rate"] == "6/6"


def test_full_suite_matrix_shows_default_regressed_and_block64_passing() -> None:
    report = _read_report()
    default_row = _find_full_suite_candidate(report, "default_int4")
    block64_row = _find_full_suite_candidate(report, "block_size_64")
    assert default_row["trial_count"] == 3
    assert block64_row["trial_count"] == 3
    assert default_row["complete_batch_rate"] == "3/3"
    assert block64_row["complete_batch_rate"] == "3/3"
    assert default_row["json_structural_regression_rate"] == "3/3"
    assert block64_row["json_structural_regression_rate"] == "0/3"
    assert block64_row["can_promote_rate"] == "3/3"


def test_variant_failures_capture_tail_and_classification() -> None:
    report = _read_report()
    neg1 = _find_variant(report, "int4_block_size_-1")
    uint8 = _find_variant(report, "int4_act_precision_uint8")
    for row in (neg1, uint8):
        assert row["status"] == "optimize_failed_or_unsupported"
        assert isinstance(row.get("optimize_stderr_tail"), str)
        assert len(str(row["optimize_stderr_tail"])) > 0
        assert isinstance(row.get("failure_classification"), str)
        assert len(str(row["failure_classification"])) > 0
        assert isinstance(row.get("last_exception_line"), str)


def test_block_size_cost_matrix_contains_required_rows() -> None:
    report = _read_report()
    for key in (
        "default_int4_selected_round6_artifact",
        "int4_default",
        "int4_block_size_16",
        "int4_block_size_32",
        "int4_block_size_64",
    ):
        row = _find_cost_row(report, key)
        assert row["package_size_bytes"] is not None
        if key != "default_int4_selected_round6_artifact":
            assert row["optimize_seconds"] is not None
        assert "load_seconds" in row
        assert "generation_seconds" in row
        assert "peak_rss_bytes" in row


def test_numeric_fidelity_is_recorded_or_explicitly_unknown() -> None:
    report = _read_report()
    fidelity = report["numeric_fidelity_probe"]  # type: ignore[index]
    assert isinstance(fidelity, dict)
    status = fidelity["status"]
    assert status in {"available", "numeric_fidelity_unknown"}
    if status == "available":
        baseline_vs_block64 = fidelity.get("baseline_vs_block64")
        assert isinstance(baseline_vs_block64, dict)
        comparison = baseline_vs_block64.get("comparison")
        assert isinstance(comparison, dict)
        assert "step_match_rate" in comparison
    else:
        assert isinstance(fidelity.get("error"), str)
        assert len(str(fidelity["error"])) > 0


def test_remedy_and_generalization_status() -> None:
    report = _read_report()
    remedy = report["remedy_analysis"]  # type: ignore[index]
    assert isinstance(remedy, dict)
    retry = remedy["retry_ladder_round7_justification"]  # type: ignore[index]
    assert isinstance(retry, dict)
    assert retry["justified"] is True
    assert report["cross_model_generalization"]["status"] == "unproven_pending_full_five_model_rerun"  # type: ignore[index]
    rerun = remedy["mandatory_full_set_rerun"]  # type: ignore[index]
    assert isinstance(rerun, dict)
    model_ids = rerun["model_ids"]  # type: ignore[index]
    assert isinstance(model_ids, list)
    assert len(model_ids) == 5


def test_cleanup_records_stray_root_and_no_lingering_processes() -> None:
    report = _read_report()
    cleanup = report["operational_cleanup"]  # type: ignore[index]
    assert isinstance(cleanup, dict)
    external = cleanup["external_cleanup_result"]  # type: ignore[index]
    stray = cleanup["exact_stray_root_cleanup"]  # type: ignore[index]
    lingering = cleanup["lingering_process_probe"]  # type: ignore[index]
    assert isinstance(external, dict)
    assert isinstance(stray, dict)
    assert isinstance(lingering, dict)
    assert external["ok"] is True
    assert "bytes_freed" in external
    assert "bytes_freed" in stray
    assert lingering["ok"] is True
    assert lingering["count"] == 0
