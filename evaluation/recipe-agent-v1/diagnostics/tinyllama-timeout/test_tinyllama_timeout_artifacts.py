from __future__ import annotations

import json

from pathlib import Path

DIAG_DIR = Path(__file__).resolve().parent
REPORT_PATH = DIAG_DIR / "diagnostic-report.json"
DOC_PATH = DIAG_DIR.parents[3] / "docs" / "tinyllama-baseline-timeout.md"


def _read_report() -> dict[str, object]:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_report_exists() -> None:
    assert REPORT_PATH.is_file()


def test_docs_exists() -> None:
    assert DOC_PATH.is_file()


def test_frozen_model_identity_matches_round5() -> None:
    report = _read_report()
    frozen = report["frozen_model"]  # type: ignore[index]
    assert isinstance(frozen, dict)
    assert frozen["model_id"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert frozen["revision_sha"] == "fe8a4ea1ffedaf415f4da2f062534de366a451e6"


def test_round5_timeout_signature_captured() -> None:
    report = _read_report()
    failure = report["round5_failure"]  # type: ignore[index]
    assert isinstance(failure, dict)
    assert failure["recorded_timeout_seconds"] == 900
    assert "factual-red-planet" in str(failure["error_signature_excerpt"])


def test_measurements_cover_current_and_single_worker_designs() -> None:
    report = _read_report()
    measurements = report["measurements"]  # type: ignore[index]
    assert isinstance(measurements, dict)
    current = measurements["current_design_per_prompt_runtime_worker"]  # type: ignore[index]
    single = measurements["single_worker_load_once_design"]  # type: ignore[index]
    assert isinstance(current, dict)
    assert isinstance(single, dict)
    for key in ("baseline", "optimized"):
        assert key in current
        assert key in single


def test_smallest_fix_is_generic_and_load_once() -> None:
    report = _read_report()
    diagnosis = report["diagnosis"]  # type: ignore[index]
    assert isinstance(diagnosis, dict)
    smallest_fix = diagnosis["smallest_generic_fix"]  # type: ignore[index]
    assert isinstance(smallest_fix, dict)
    assert "loads once" in str(smallest_fix["proposal"]).lower()


def test_external_cleanup_recorded() -> None:
    report = _read_report()
    cleanup = report["external_cleanup"]  # type: ignore[index]
    assert isinstance(cleanup, dict)
    result = cleanup["cleanup_result"]  # type: ignore[index]
    assert isinstance(result, dict)
    assert result["ok"] is True
