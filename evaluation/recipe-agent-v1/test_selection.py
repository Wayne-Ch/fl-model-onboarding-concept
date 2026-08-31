"""Tests for Recipe Agent v1 evaluation model selection.

Covers manifest schema, selection rules, license documentation,
catalog matching logic, and live HF Hub verification.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parent
MODELS_JSON = EVAL_DIR / "models.json"

# Import catalog matching functions for unit tests
sys.path.insert(0, str(EVAL_DIR))
from select_models import (
    EXCLUDED_MODEL_IDS,
    FAMILY_GROUP,
    MOBIUS_RECOGNIZED_ARCHITECTURES,
    _family_group,
    _normalize_for_matching,
    catalog_match_check,
)


@pytest.fixture
def manifest() -> dict:
    assert MODELS_JSON.exists(), f"models.json not found at {MODELS_JSON}"
    return json.loads(MODELS_JSON.read_text(encoding="utf-8"))


# ---- Manifest schema tests ----

def test_manifest_schema_version(manifest: dict) -> None:
    assert manifest["schema_version"] == "1.0.0"


def test_exactly_five_models(manifest: dict) -> None:
    assert len(manifest["models"]) == 5


def test_at_least_three_family_groups(manifest: dict) -> None:
    groups = {_family_group(m["model_type"]) for m in manifest["models"]}
    assert len(groups) >= 3, f"Only {len(groups)} family groups: {groups}"


def test_all_models_have_sha(manifest: dict) -> None:
    for m in manifest["models"]:
        assert m["sha"], f"Model {m['model_id']} has no SHA"
        assert len(m["sha"]) >= 7, f"SHA too short for {m['model_id']}"


def test_no_model_in_excluded_recipes(manifest: dict) -> None:
    excluded = set(manifest["selection_rules"]["excluded_recipe_ids"])
    for m in manifest["models"]:
        assert m["model_id"] not in excluded


def test_all_models_cpu_practical(manifest: dict) -> None:
    max_p = manifest["selection_rules"]["max_params_billion"]
    for m in manifest["models"]:
        if m["params_billion"] is not None:
            assert m["params_billion"] <= max_p


def test_all_models_have_evaluation_prompts(manifest: dict) -> None:
    for m in manifest["models"]:
        prompts = m.get("evaluation_prompts", [])
        assert len(prompts) >= 3
        for p in prompts:
            assert "name" in p and "prompt" in p and "expected_behavior" in p


def test_no_duplicate_model_ids(manifest: dict) -> None:
    ids = [m["model_id"] for m in manifest["models"]]
    assert len(ids) == len(set(ids))


def test_models_not_catalog_matched(manifest: dict) -> None:
    for m in manifest["models"]:
        assert m["catalog_match"]["matched"] is False


def test_models_not_recipe_exists(manifest: dict) -> None:
    for m in manifest["models"]:
        assert m["recipe_exists"] is False


def test_selection_timestamp_present(manifest: dict) -> None:
    assert manifest.get("selection_timestamp")


def test_alternates_present(manifest: dict) -> None:
    assert len(manifest.get("alternates", [])) > 0


# ---- License documentation tests ----

def test_all_models_have_license_info(manifest: dict) -> None:
    for m in manifest["models"]:
        lic = m["license"]
        assert "spdx_id" in lic, f"Missing spdx_id for {m['model_id']}"
        assert "is_clear_oss" in lic, f"Missing is_clear_oss for {m['model_id']}"
        assert "review_required" in lic, f"Missing review_required for {m['model_id']}"
        assert "confidence" in lic, f"Missing confidence for {m['model_id']}"


def test_other_licenses_have_name_and_url(manifest: dict) -> None:
    """Models with license:other must have license_name and/or license_url documented."""
    for m in manifest["models"]:
        lic = m["license"]
        if lic.get("spdx_id") == "other":
            assert lic.get("license_name") or lic.get("license_url"), (
                f"Model {m['model_id']} has license:other but no license_name/license_url"
            )
            assert lic["review_required"] is True


def test_recognized_architectures_list(manifest: dict) -> None:
    recognized = manifest["selection_rules"]["recognized_architectures"]
    assert len(recognized) >= 10
    for m in manifest["models"]:
        assert m["model_type"] in recognized


# ---- Catalog matching unit tests ----

class TestNormalization:
    def test_basic(self) -> None:
        assert _normalize_for_matching("Qwen2.5-3B-Instruct") == "qwen2-5-3b-instruct"

    def test_org_prefix_stripped(self) -> None:
        assert _normalize_for_matching("Qwen/Qwen2.5-3B-Instruct") == "qwen2-5-3b-instruct"

    def test_underscores_to_hyphens(self) -> None:
        assert _normalize_for_matching("some_model_name") == "some-model-name"

    def test_dots_to_hyphens(self) -> None:
        assert _normalize_for_matching("Mistral-7B-Instruct-v0.2") == "mistral-7b-instruct-v0-2"

    def test_multiple_hyphens_collapsed(self) -> None:
        assert _normalize_for_matching("model--name") == "model-name"


class TestCatalogMatching:
    """Tests for catalog matching with real Foundry catalog entry formats."""

    SAMPLE_ENTRIES = [
        {"original": "qwen2.5-0.5b-instruct-vitis-npu:3", "normalized": "qwen2-5-0-5b-instruct-vitis-npu:3", "source_field": "id"},
        {"original": "qwen2.5-0.5b", "normalized": "qwen2-5-0-5b", "source_field": "alias"},
        {"original": "Mistral-7B-Instruct-v0-2-vitis-npu", "normalized": "mistral-7b-instruct-v0-2-vitis-npu", "source_field": "displayName"},
        {"original": "mistral-7b-v0.2", "normalized": "mistral-7b-v0-2", "source_field": "alias"},
        {"original": "Phi-3.5-mini-instruct-generic-gpu:2", "normalized": "phi-3-5-mini-instruct-generic-gpu:2", "source_field": "id"},
        {"original": "phi-3.5-mini", "normalized": "phi-3-5-mini", "source_field": "alias"},
        {"original": "qwen3-0.6b-generic-gpu:2", "normalized": "qwen3-0-6b-generic-gpu:2", "source_field": "id"},
        {"original": "qwen3-0.6b", "normalized": "qwen3-0-6b", "source_field": "alias"},
        {"original": "ministral-3-3b-instruct-2512-generic-gpu:1", "normalized": "ministral-3-3b-instruct-2512-generic-gpu:1", "source_field": "id"},
        {"original": "ministral-3-3b-instruct-2512", "normalized": "ministral-3-3b-instruct-2512", "source_field": "alias"},
        {"original": "olmo-3-7b-instruct-generic-gpu:1", "normalized": "olmo-3-7b-instruct-generic-gpu:1", "source_field": "id"},
        {"original": "olmo-3-7b-instruct", "normalized": "olmo-3-7b-instruct", "source_field": "alias"},
        {"original": "smollm3-3b-generic-gpu:1", "normalized": "smollm3-3b-generic-gpu:1", "source_field": "id"},
        {"original": "smollm3-3b", "normalized": "smollm3-3b", "source_field": "alias"},
    ]

    def test_exact_catalog_model_detected(self) -> None:
        """Qwen2.5-0.5B-Instruct IS in the catalog."""
        r = catalog_match_check("Qwen/Qwen2.5-0.5B-Instruct", self.SAMPLE_ENTRIES)
        assert r["matched"] is True

    def test_dot_vs_hyphen_mistral(self) -> None:
        """Mistral-7B-Instruct-v0.2 matches catalog (dot normalized to hyphen)."""
        r = catalog_match_check("mistralai/Mistral-7B-Instruct-v0.2", self.SAMPLE_ENTRIES)
        assert r["matched"] is True

    def test_absent_model_not_matched(self) -> None:
        """Qwen2.5-3B-Instruct is NOT in catalog (only 0.5B,1.5B,7B,14B)."""
        r = catalog_match_check("Qwen/Qwen2.5-3B-Instruct", self.SAMPLE_ENTRIES)
        assert r["matched"] is False

    def test_absent_math_variant(self) -> None:
        """Qwen2.5-Math-1.5B-Instruct ≠ qwen2.5-1.5b-instruct."""
        r = catalog_match_check("Qwen/Qwen2.5-Math-1.5B-Instruct", self.SAMPLE_ENTRIES)
        assert r["matched"] is False

    def test_qwen3_base_in_catalog(self) -> None:
        """Qwen3-0.6B IS in catalog."""
        r = catalog_match_check("Qwen/Qwen3-0.6B", self.SAMPLE_ENTRIES)
        assert r["matched"] is True

    def test_phi_in_catalog(self) -> None:
        """Phi-3.5-mini-instruct IS in catalog."""
        r = catalog_match_check("microsoft/Phi-3.5-mini-instruct", self.SAMPLE_ENTRIES)
        assert r["matched"] is True

    def test_tinyllama_not_in_catalog(self) -> None:
        """TinyLlama-1.1B-Chat-v1.0 is NOT in catalog."""
        r = catalog_match_check("TinyLlama/TinyLlama-1.1B-Chat-v1.0", self.SAMPLE_ENTRIES)
        assert r["matched"] is False

    def test_olmo2_instruct_not_in_catalog(self) -> None:
        """OLMo-2-0425-1B-Instruct is NOT in catalog (catalog has olmo-3-7b only)."""
        r = catalog_match_check("allenai/OLMo-2-0425-1B-Instruct", self.SAMPLE_ENTRIES)
        assert r["matched"] is False

    def test_gpt2_not_in_catalog(self) -> None:
        """openai-community/gpt2 is NOT in catalog."""
        r = catalog_match_check("openai-community/gpt2", self.SAMPLE_ENTRIES)
        assert r["matched"] is False

    def test_stablelm_not_in_catalog(self) -> None:
        """stabilityai/stablelm-3b-4e1t is NOT in catalog."""
        r = catalog_match_check("stabilityai/stablelm-3b-4e1t", self.SAMPLE_ENTRIES)
        assert r["matched"] is False

    def test_match_returns_confidence_and_reason(self) -> None:
        r = catalog_match_check("Qwen/Qwen2.5-0.5B-Instruct", self.SAMPLE_ENTRIES)
        assert "confidence" in r
        assert "reason" in r
        assert isinstance(r["reason"], str)

    def test_no_match_returns_confidence_none(self) -> None:
        r = catalog_match_check("SomeOrg/NoSuchModel", self.SAMPLE_ENTRIES)
        assert r["matched"] is False
        assert r["confidence"] == "none"

    def test_smollm3_in_catalog(self) -> None:
        """SmolLM3-3B matches smollm3-3b catalog entry."""
        r = catalog_match_check("HuggingFaceTB/SmolLM3-3B", self.SAMPLE_ENTRIES)
        assert r["matched"] is True

    def test_ministral_3b_in_catalog(self) -> None:
        """ministral-3-3b-instruct-2512 IS in catalog."""
        r = catalog_match_check("mistralai/Ministral-3B-Instruct-2412", self.SAMPLE_ENTRIES)
        # This should NOT match because "ministral-3b-instruct-2412" ≠ "ministral-3-3b-instruct-2512"
        # The catalog entry has "3-3b" (version 3, 3B params) vs "3b" in model name
        assert r["matched"] is False


# ---- Family group tests ----

class TestFamilyGroups:
    def test_olmo_group(self) -> None:
        assert _family_group("olmo") == "olmo-family"
        assert _family_group("olmo2") == "olmo-family"

    def test_qwen_group(self) -> None:
        assert _family_group("qwen2") == "qwen-family"
        assert _family_group("qwen3") == "qwen-family"

    def test_ungrouped(self) -> None:
        assert _family_group("gpt2") == "gpt2"
        assert _family_group("llama") == "llama"
        assert _family_group("stablelm") == "stablelm"


# ---- Live verification ----

@pytest.mark.skipif(
    not MODELS_JSON.exists(),
    reason="models.json must exist",
)
def test_live_verification_models_exist() -> None:
    """Verify selected models still exist on HF Hub (requires network)."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        pytest.skip("huggingface_hub not installed")

    manifest = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
    api = HfApi()
    excluded_recipes = set(manifest["selection_rules"]["excluded_recipe_ids"])

    for m in manifest["models"]:
        mid = m["model_id"]
        try:
            info = api.model_info(mid)
            assert info is not None, f"Model {mid} not found"

            # Verify SHA matches
            live_sha = getattr(info, "sha", None)
            assert live_sha == m["sha"], (
                f"SHA mismatch for {mid}: manifest={m['sha']}, live={live_sha}"
            )

            # Verify not gated
            gated = getattr(info, "gated", None)
            assert not gated or gated is False, f"Model {mid} is gated: {gated}"

            # Verify no auto_map (remote code)
            config = getattr(info, "config", None) or {}
            if isinstance(config, dict):
                assert "auto_map" not in config, f"Model {mid} has auto_map"

            # Verify model_type matches
            mt = config.get("model_type") if isinstance(config, dict) else None
            assert mt == m["model_type"], f"model_type mismatch: {mt} vs {m['model_type']}"

            # Verify not in excluded recipes
            assert mid not in excluded_recipes

        except Exception as e:
            pytest.fail(f"Failed to verify {mid}: {e}")
