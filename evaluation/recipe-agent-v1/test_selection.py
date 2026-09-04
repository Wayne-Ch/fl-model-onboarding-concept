"""Tests for Recipe Agent v1 evaluation model selection.

Covers manifest schema, selection rules, license documentation,
catalog matching (including Qwen2, granite, org-prefix normalization),
family groups, and live HF Hub verification.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parent
MODELS_JSON = EVAL_DIR / "models.json"

sys.path.insert(0, str(EVAL_DIR))
from select_models import (
    CURATED_TARGETS,
    EXCLUDED_MODEL_IDS,
    MOBIUS_RECOGNIZED_ARCHITECTURES,
    _family_group,
    _normalize_for_matching,
    catalog_match_check,
)


@pytest.fixture
def manifest() -> dict:
    assert MODELS_JSON.exists(), f"models.json not found at {MODELS_JSON}"
    return json.loads(MODELS_JSON.read_text(encoding="utf-8"))


# ---- Manifest schema ----

def test_schema_version(manifest: dict) -> None:
    assert manifest["schema_version"] == "1.0.0"


def test_exactly_five_models(manifest: dict) -> None:
    assert len(manifest["models"]) == 5


def test_at_least_three_family_groups(manifest: dict) -> None:
    groups = {_family_group(m["model_type"]) for m in manifest["models"]}
    assert len(groups) >= 3, f"Only {len(groups)} groups: {groups}"


def test_all_instruction_models(manifest: dict) -> None:
    """All selected models must be instruction/chat oriented."""
    for m in manifest["models"]:
        name = m["model_id"].lower().split("/")[-1]
        assert any(kw in name for kw in ("instruct", "chat")), (
            f"{m['model_id']} is not an instruction/chat model"
        )


def test_within_family_pairs(manifest: dict) -> None:
    """At least two families must have 2+ models (within-family generalization)."""
    from collections import Counter
    groups = Counter(_family_group(m["model_type"]) for m in manifest["models"])
    pairs = sum(1 for c in groups.values() if c >= 2)
    assert pairs >= 2, f"Only {pairs} family groups have 2+ models: {dict(groups)}"


def test_all_models_have_sha(manifest: dict) -> None:
    for m in manifest["models"]:
        assert m["sha"] and len(m["sha"]) >= 7


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


def test_timestamp_present(manifest: dict) -> None:
    assert manifest.get("selection_timestamp")


def test_alternates_present(manifest: dict) -> None:
    assert len(manifest.get("alternates", [])) > 0


# ---- License ----

def test_all_clear_oss_licenses(manifest: dict) -> None:
    for m in manifest["models"]:
        lic = m["license"]
        assert lic["is_clear_oss"] is True, f"{m['model_id']} license not clear: {lic}"
        assert lic["review_required"] is False
        assert lic["spdx_id"] == "apache-2.0"


def test_license_fields_present(manifest: dict) -> None:
    for m in manifest["models"]:
        lic = m["license"]
        for key in ("spdx_id", "is_clear_oss", "review_required", "confidence"):
            assert key in lic


# ---- Normalization ----

class TestNormalization:
    def test_basic(self) -> None:
        assert _normalize_for_matching("Qwen2.5-3B-Instruct") == "qwen2-5-3b-instruct"

    def test_org_prefix(self) -> None:
        assert _normalize_for_matching("Qwen/Qwen2.5-3B-Instruct") == "qwen2-5-3b-instruct"

    def test_underscores(self) -> None:
        assert _normalize_for_matching("some_model_name") == "some-model-name"

    def test_dots(self) -> None:
        assert _normalize_for_matching("Mistral-7B-Instruct-v0.2") == "mistral-7b-instruct-v0-2"

    def test_multi_hyphens(self) -> None:
        assert _normalize_for_matching("model--name") == "model-name"


# ---- Catalog matching ----

class TestCatalogMatching:
    """Tests with realistic Foundry catalog entries."""

    ENTRIES = [
        {"original": "qwen2.5-0.5b-instruct-vitis-npu:3", "normalized": "qwen2-5-0-5b-instruct-vitis-npu:3", "source_field": "id"},
        {"original": "qwen2.5-0.5b", "normalized": "qwen2-5-0-5b", "source_field": "alias"},
        {"original": "qwen2.5-1.5b-instruct-generic-gpu:4", "normalized": "qwen2-5-1-5b-instruct-generic-gpu:4", "source_field": "id"},
        {"original": "qwen2.5-1.5b", "normalized": "qwen2-5-1-5b", "source_field": "alias"},
        {"original": "Mistral-7B-Instruct-v0-2-vitis-npu", "normalized": "mistral-7b-instruct-v0-2-vitis-npu", "source_field": "displayName"},
        {"original": "mistral-7b-v0.2", "normalized": "mistral-7b-v0-2", "source_field": "alias"},
        {"original": "qwen3-0.6b-generic-gpu:2", "normalized": "qwen3-0-6b-generic-gpu:2", "source_field": "id"},
        {"original": "qwen3-0.6b", "normalized": "qwen3-0-6b", "source_field": "alias"},
        {"original": "smollm3-3b-generic-gpu:1", "normalized": "smollm3-3b-generic-gpu:1", "source_field": "id"},
        {"original": "smollm3-3b", "normalized": "smollm3-3b", "source_field": "alias"},
        {"original": "olmo-3-7b-instruct-generic-gpu:1", "normalized": "olmo-3-7b-instruct-generic-gpu:1", "source_field": "id"},
        {"original": "Phi-3.5-mini-instruct-generic-gpu:2", "normalized": "phi-3-5-mini-instruct-generic-gpu:2", "source_field": "id"},
        {"original": "phi-3.5-mini", "normalized": "phi-3-5-mini", "source_field": "alias"},
    ]

    # --- Selected models must NOT match ---

    def test_tinyllama_absent(self) -> None:
        r = catalog_match_check("TinyLlama/TinyLlama-1.1B-Chat-v1.0", self.ENTRIES)
        assert r["matched"] is False

    def test_smollm2_360m_absent(self) -> None:
        """SmolLM2-360M is distinct from SmolLM3-3B in catalog."""
        r = catalog_match_check("HuggingFaceTB/SmolLM2-360M-Instruct", self.ENTRIES)
        assert r["matched"] is False

    def test_qwen2_1_5b_absent(self) -> None:
        """Qwen2-1.5B ≠ Qwen2.5-1.5B (different model generation)."""
        r = catalog_match_check("Qwen/Qwen2-1.5B-Instruct", self.ENTRIES)
        assert r["matched"] is False

    def test_qwen2_0_5b_absent(self) -> None:
        """Qwen2-0.5B ≠ Qwen2.5-0.5B."""
        r = catalog_match_check("Qwen/Qwen2-0.5B-Instruct", self.ENTRIES)
        assert r["matched"] is False

    def test_granite_3_2_absent(self) -> None:
        r = catalog_match_check("ibm-granite/granite-3.2-2b-instruct", self.ENTRIES)
        assert r["matched"] is False

    # --- Catalog models MUST match (true positives) ---

    def test_qwen25_0_5b_present(self) -> None:
        r = catalog_match_check("Qwen/Qwen2.5-0.5B-Instruct", self.ENTRIES)
        assert r["matched"] is True

    def test_qwen25_1_5b_present(self) -> None:
        r = catalog_match_check("Qwen/Qwen2.5-1.5B-Instruct", self.ENTRIES)
        assert r["matched"] is True

    def test_mistral_v02_present(self) -> None:
        r = catalog_match_check("mistralai/Mistral-7B-Instruct-v0.2", self.ENTRIES)
        assert r["matched"] is True

    def test_qwen3_0_6b_present(self) -> None:
        r = catalog_match_check("Qwen/Qwen3-0.6B", self.ENTRIES)
        assert r["matched"] is True

    def test_phi35_present(self) -> None:
        r = catalog_match_check("microsoft/Phi-3.5-mini-instruct", self.ENTRIES)
        assert r["matched"] is True

    def test_smollm3_present(self) -> None:
        r = catalog_match_check("HuggingFaceTB/SmolLM3-3B", self.ENTRIES)
        assert r["matched"] is True

    # --- Confidence and reason ---

    def test_match_has_confidence(self) -> None:
        r = catalog_match_check("Qwen/Qwen2.5-0.5B-Instruct", self.ENTRIES)
        assert r["confidence"] in ("exact", "high")
        assert isinstance(r["reason"], str) and len(r["reason"]) > 0

    def test_no_match_confidence_none(self) -> None:
        r = catalog_match_check("SomeOrg/NoModel", self.ENTRIES)
        assert r["confidence"] == "none"

    # --- False positive guards ---

    def test_qwen2_not_confused_with_qwen25(self) -> None:
        """Qwen2-X must NOT match Qwen2.5-X catalog entries."""
        r = catalog_match_check("Qwen/Qwen2-7B-Instruct", self.ENTRIES)
        assert r["matched"] is False

    def test_partial_name_no_false_positive(self) -> None:
        """'qwen' alone must not match qwen2.5 or qwen3 entries."""
        r = catalog_match_check("SomeOrg/qwen-tiny-model", self.ENTRIES)
        assert r["matched"] is False


# ---- Family groups ----

class TestFamilyGroups:
    def test_granite_group(self) -> None:
        assert _family_group("granite") == "granite-family"
        assert _family_group("granitemoe") == "granite-family"

    def test_qwen_group(self) -> None:
        assert _family_group("qwen2") == "qwen-family"
        assert _family_group("qwen3") == "qwen-family"

    def test_llama_ungrouped(self) -> None:
        assert _family_group("llama") == "llama"

    def test_granite_in_recognized(self) -> None:
        assert "granite" in MOBIUS_RECOGNIZED_ARCHITECTURES


# ---- Curated targets ----

def test_curated_targets_count() -> None:
    assert len(CURATED_TARGETS) == 5


def test_curated_targets_all_instruction() -> None:
    for mid in CURATED_TARGETS:
        name = mid.lower().split("/")[-1]
        assert any(kw in name for kw in ("instruct", "chat")), f"{mid} not instruction"


# ---- Live verification ----

@pytest.mark.skipif(not MODELS_JSON.exists(), reason="models.json required")
def test_live_verification() -> None:
    """Verify all selected models on HF Hub: SHA, gated, auto_map, model_type."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        pytest.skip("huggingface_hub not installed")

    manifest = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
    api = HfApi()
    excluded = set(manifest["selection_rules"]["excluded_recipe_ids"])

    for m in manifest["models"]:
        mid = m["model_id"]
        try:
            info = api.model_info(mid)
            assert info is not None
            assert getattr(info, "sha", None) == m["sha"], f"SHA mismatch for {mid}"

            gated = getattr(info, "gated", None)
            assert not gated or gated is False, f"{mid} is gated"

            config = getattr(info, "config", None) or {}
            if isinstance(config, dict):
                assert "auto_map" not in config, f"{mid} has auto_map"
                assert config.get("model_type") == m["model_type"]

            assert mid not in excluded
        except Exception as e:
            pytest.fail(f"Failed to verify {mid}: {e}")
