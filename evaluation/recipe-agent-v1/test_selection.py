"""Tests for the Recipe Agent v1 evaluation model selection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parent.parent / "recipe-agent-v1"
MODELS_JSON = EVAL_DIR / "models.json"


@pytest.fixture
def manifest() -> dict:
    """Load the frozen models manifest."""
    assert MODELS_JSON.exists(), f"models.json not found at {MODELS_JSON}"
    return json.loads(MODELS_JSON.read_text(encoding="utf-8"))


def test_manifest_schema_version(manifest: dict) -> None:
    assert manifest["schema_version"] == "1.0.0"


def test_exactly_five_models(manifest: dict) -> None:
    assert len(manifest["models"]) == 5


def test_at_least_three_architecture_families(manifest: dict) -> None:
    families = {m["model_type"] for m in manifest["models"]}
    assert len(families) >= 3, f"Only {len(families)} families: {families}"


def test_all_models_have_sha(manifest: dict) -> None:
    for m in manifest["models"]:
        assert m["sha"], f"Model {m['model_id']} has no SHA"
        assert len(m["sha"]) >= 7, f"SHA too short for {m['model_id']}"


def test_no_model_in_excluded_recipes(manifest: dict) -> None:
    excluded = set(manifest["selection_rules"]["excluded_recipe_ids"])
    for m in manifest["models"]:
        assert m["model_id"] not in excluded, (
            f"Model {m['model_id']} is in excluded recipes"
        )


def test_all_models_cpu_practical(manifest: dict) -> None:
    max_params = manifest["selection_rules"]["max_params_billion"]
    for m in manifest["models"]:
        if m["params_billion"] is not None:
            assert m["params_billion"] <= max_params, (
                f"Model {m['model_id']} has {m['params_billion']}B params, "
                f"exceeds {max_params}B limit"
            )


def test_all_models_have_evaluation_prompts(manifest: dict) -> None:
    for m in manifest["models"]:
        prompts = m.get("evaluation_prompts", [])
        assert len(prompts) >= 3, (
            f"Model {m['model_id']} has only {len(prompts)} evaluation prompts"
        )
        for p in prompts:
            assert "name" in p
            assert "prompt" in p
            assert "expected_behavior" in p


def test_no_duplicate_model_ids(manifest: dict) -> None:
    ids = [m["model_id"] for m in manifest["models"]]
    assert len(ids) == len(set(ids)), f"Duplicate model IDs found: {ids}"


def test_models_not_marked_as_catalog_match(manifest: dict) -> None:
    for m in manifest["models"]:
        assert m["catalog_match"] is False, (
            f"Model {m['model_id']} is marked as catalog match"
        )


def test_models_not_marked_as_recipe_exists(manifest: dict) -> None:
    for m in manifest["models"]:
        assert m["recipe_exists"] is False, (
            f"Model {m['model_id']} is marked as having a recipe"
        )


def test_all_models_have_license(manifest: dict) -> None:
    for m in manifest["models"]:
        assert m.get("license"), f"Model {m['model_id']} has no license"


def test_recognized_architectures_list(manifest: dict) -> None:
    recognized = manifest["selection_rules"]["recognized_architectures"]
    assert len(recognized) >= 10, "Too few recognized architectures"
    for m in manifest["models"]:
        assert m["model_type"] in recognized, (
            f"Model {m['model_id']} type '{m['model_type']}' not in recognized list"
        )


def test_selection_timestamp_present(manifest: dict) -> None:
    assert manifest.get("selection_timestamp"), "Missing selection timestamp"


def test_alternates_present(manifest: dict) -> None:
    assert len(manifest.get("alternates", [])) > 0, "No alternates recorded"


@pytest.mark.skipif(
    not MODELS_JSON.exists(),
    reason="models.json must exist for live verification",
)
def test_live_verification_models_exist() -> None:
    """Verify selected models still exist on HF Hub (requires network)."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        pytest.skip("huggingface_hub not installed")

    manifest = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
    api = HfApi()

    for m in manifest["models"]:
        try:
            info = api.model_info(m["model_id"])
            assert info is not None, f"Model {m['model_id']} not found on HF Hub"
        except Exception as e:
            pytest.fail(f"Failed to verify {m['model_id']}: {e}")
