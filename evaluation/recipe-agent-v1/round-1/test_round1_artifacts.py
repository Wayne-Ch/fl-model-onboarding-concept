from __future__ import annotations

import json

from pathlib import Path


ROUND_DIR = Path(__file__).resolve().parent
EXPECTED_MODELS = (
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "Qwen/Qwen2-1.5B-Instruct",
    "Qwen/Qwen2-0.5B-Instruct",
    "ibm-granite/granite-3.2-2b-instruct",
)


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


def test_round1_manifest_contains_required_snapshots() -> None:
    manifest = _read_json(ROUND_DIR / "round-manifest.json")
    assert manifest["manifest_path"] == "evaluation/recipe-agent-v1/models.json"
    assert manifest["manifest_invariants"]
    assert manifest["frozen_cli"]
    assert manifest["quality_profile"]
    assert manifest["model_registry_and_catalog_snapshot"]


def test_per_model_artifacts_exist_for_all_five() -> None:
    model_results_dir = ROUND_DIR / "model-results"
    assert model_results_dir.is_dir()
    files = sorted(path for path in model_results_dir.glob("*.json"))
    assert len(files) == 5
