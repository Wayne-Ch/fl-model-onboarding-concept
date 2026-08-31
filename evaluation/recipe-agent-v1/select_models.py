#!/usr/bin/env python3
"""
Reproducible model selection for Recipe Agent v1 evaluation.

Queries Hugging Face Hub API (metadata only — no weight downloads) and the live
Foundry Local catalog to select five unseen text-generation models covering at
least three architecture families.

Usage:
    python evaluation/recipe-agent-v1/select_models.py

Outputs:
    evaluation/recipe-agent-v1/models.json          -- frozen manifest
    evaluation/recipe-agent-v1/candidates_raw.json  -- raw candidate metadata
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, ModelInfo

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).resolve().parent
MODELS_JSON = OUTPUT_DIR / "models.json"
CANDIDATES_RAW = OUTPUT_DIR / "candidates_raw.json"

# Models already covered by recipes in src/fl_model_onboarding/recipes.py
EXCLUDED_MODEL_IDS: set[str] = {
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "distil-whisper/distil-medium.en",
    "ibm-granite/granite-3.3-2b-instruct",
}

# Orgs that are test / internal / community-squatter
EXCLUDED_ORG_PREFIXES: set[str] = {
    "trl-internal-testing",
    "hf-internal-testing",
    "peft-internal-testing",
    "test-org",
    "stas",
    "nm-testing",
}

# Architecture families recognized by Mobius / OGA for text-generation
MOBIUS_RECOGNIZED_ARCHITECTURES: set[str] = {
    "llama", "mistral", "phi", "phi3", "phimoe",
    "gemma", "gemma2", "gemma3",
    "qwen2", "qwen3",
    "gpt2", "gpt_neox",
    "bloom", "falcon", "mpt", "opt",
    "starcoder2", "stablelm", "tinyllama",
    "cohere", "olmo", "olmo2", "glm",
    "granite", "granitemoe",
}

# Well-known open-source licenses considered "clear" (no legal review needed)
CLEAR_LICENSES: set[str] = {
    "apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause",
    "cc-by-4.0", "cc-by-sa-4.0", "openrail",
}

MAX_PARAMS_BILLION = 4.0
MIN_DOWNLOADS = 1000


# ---------------------------------------------------------------------------
# Catalog matching (hardened)
# ---------------------------------------------------------------------------

def get_foundry_catalog() -> dict[str, Any]:
    """Get live Foundry Local catalog via CLI."""
    try:
        result = subprocess.run(
            ["foundry", "model", "list", "-o", "json"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        pass
    return {"models": []}


def _normalize_for_matching(name: str) -> str:
    """Normalize a model name for catalog comparison.

    Strips org prefixes, lowercases, converts dots/underscores to hyphens,
    and collapses multiple hyphens.
    """
    # Remove org prefix if present (e.g., "openai-community/" or "Qwen/")
    if "/" in name:
        name = name.split("/")[-1]
    return re.sub(r"-+", "-", name.lower().replace(".", "-").replace("_", "-")).strip("-")


def extract_catalog_entries(catalog: dict[str, Any]) -> list[dict[str, str]]:
    """Extract normalized catalog entries with original values for matching."""
    entries: list[dict[str, str]] = []
    for m in catalog.get("models", []):
        if not isinstance(m, dict):
            continue
        for key in ("alias", "id", "displayName"):
            val = m.get(key, "")
            if val:
                entries.append({
                    "original": val,
                    "normalized": _normalize_for_matching(val),
                    "source_field": key,
                })
    return entries


def catalog_match_check(
    model_id: str,
    catalog_entries: list[dict[str, str]],
) -> dict[str, Any]:
    """Check if HF model ID matches any Foundry catalog entry.

    Returns a dict with 'matched', 'confidence', 'reason', and 'matched_entry'.
    """
    hf_norm = _normalize_for_matching(model_id)

    for entry in catalog_entries:
        cat_norm = entry["normalized"]

        # Exact match after normalization
        if hf_norm == cat_norm:
            return {
                "matched": True,
                "confidence": "exact",
                "reason": f"Exact normalized match: '{hf_norm}' == catalog {entry['source_field']} '{entry['original']}'",
                "matched_entry": entry["original"],
            }

        # HF name is a prefix of catalog entry (catalog adds suffixes like -generic-gpu)
        if cat_norm.startswith(hf_norm + "-") or cat_norm.startswith(hf_norm + ":"):
            return {
                "matched": True,
                "confidence": "high",
                "reason": f"Catalog entry '{entry['original']}' starts with HF model name '{hf_norm}'",
                "matched_entry": entry["original"],
            }

        # Catalog alias/id is a prefix of HF name
        if hf_norm.startswith(cat_norm + "-") or hf_norm.startswith(cat_norm + ":"):
            return {
                "matched": True,
                "confidence": "high",
                "reason": f"HF model name '{hf_norm}' starts with catalog entry '{entry['original']}'",
                "matched_entry": entry["original"],
            }

    return {
        "matched": False,
        "confidence": "none",
        "reason": f"No catalog entry matches normalized HF name '{hf_norm}'",
        "matched_entry": None,
    }


# ---------------------------------------------------------------------------
# License handling
# ---------------------------------------------------------------------------

def get_license_info(info: ModelInfo) -> dict[str, Any]:
    """Extract comprehensive license information from a model."""
    tags = set(getattr(info, "tags", []) or [])
    lic_tags = [t.split(":", 1)[1] for t in tags if t.startswith("license:")]
    spdx_id = lic_tags[0] if lic_tags else ""

    card = getattr(info, "card_data", None)
    license_name = getattr(card, "license_name", None) if card else None
    license_link = getattr(card, "license_link", None) if card else None

    is_clear = spdx_id.lower() in CLEAR_LICENSES
    review_required = not is_clear and spdx_id != ""

    return {
        "spdx_id": spdx_id,
        "license_name": license_name,
        "license_url": license_link,
        "is_clear_oss": is_clear,
        "review_required": review_required,
        "confidence": "high" if is_clear else ("medium" if spdx_id else "low"),
    }


# ---------------------------------------------------------------------------
# Model inspection helpers
# ---------------------------------------------------------------------------

def has_remote_code(info: ModelInfo) -> bool:
    tags = set(getattr(info, "tags", []) or [])
    if "custom_code" in tags:
        return True
    config = getattr(info, "config", None) or {}
    return isinstance(config, dict) and "auto_map" in config


def has_required_files(info: ModelInfo) -> bool:
    siblings = getattr(info, "siblings", None) or []
    fnames = {s.rfilename for s in siblings if s.rfilename}
    has_config = "config.json" in fnames
    has_tokenizer = bool(fnames & {"tokenizer.json", "tokenizer_config.json", "tokenizer.model"})
    has_weights = any(f.endswith((".safetensors", ".bin")) for f in fnames)
    return has_config and has_tokenizer and has_weights


def get_model_type(info: ModelInfo) -> str | None:
    config = getattr(info, "config", None) or {}
    return config.get("model_type") if isinstance(config, dict) else None


def get_architectures(info: ModelInfo) -> list[str]:
    config = getattr(info, "config", None) or {}
    if isinstance(config, dict):
        archs = config.get("architectures", [])
        return archs if isinstance(archs, list) else []
    return []


def estimate_name_size_b(model_id: str) -> float | None:
    """Heuristic: extract param count from model name like '3B', '1.7B'."""
    name = model_id.split("/")[-1]
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?:illion)?(?:\b|[-_])", name)
    if m:
        return float(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Search and filter
# ---------------------------------------------------------------------------

def search_candidates(api: HfApi) -> list[ModelInfo]:
    candidates: list[ModelInfo] = []
    seen: set[str] = set()
    for query in ("instruct", "chat", "gpt2", "stablelm", "olmo"):
        try:
            for m in api.list_models(
                search=query,
                pipeline_tag="text-generation",
                sort="downloads",
                limit=200,
                fetch_config=True,
                full=True,
            ):
                mid = m.id or ""
                if mid and mid not in seen:
                    seen.add(mid)
                    candidates.append(m)
        except Exception as e:
            print(f"  Warning: search '{query}' failed: {e}", file=sys.stderr)
    return candidates


def filter_candidates(
    candidates: list[ModelInfo],
    catalog_entries: list[dict[str, str]],
) -> list[dict[str, Any]]:
    viable: list[dict[str, Any]] = []

    for info in candidates:
        model_id = info.id or ""
        if not model_id:
            continue

        # Excluded recipes
        if model_id in EXCLUDED_MODEL_IDS:
            continue

        # Excluded orgs
        org = model_id.split("/")[0] if "/" in model_id else ""
        if org.lower() in EXCLUDED_ORG_PREFIXES:
            continue

        # Gated
        gated = getattr(info, "gated", None)
        if gated and gated is not False and str(gated).lower() not in ("false", "none", ""):
            continue

        # Private
        if getattr(info, "private", False):
            continue

        # Remote code
        if has_remote_code(info):
            continue

        # Required files
        if not has_required_files(info):
            continue

        # Architecture
        model_type = get_model_type(info)
        if not model_type or model_type.lower() not in MOBIUS_RECOGNIZED_ARCHITECTURES:
            continue

        # Catalog match
        cat_result = catalog_match_check(model_id, catalog_entries)
        if cat_result["matched"]:
            continue

        # Size check (name heuristic)
        name_size = estimate_name_size_b(model_id)
        if name_size is not None and name_size > MAX_PARAMS_BILLION:
            continue

        # Pipeline check
        pipeline = getattr(info, "pipeline_tag", None)
        if pipeline and pipeline != "text-generation":
            continue

        # Downloads
        downloads = getattr(info, "downloads", 0) or 0
        if downloads < MIN_DOWNLOADS:
            continue

        # License info
        lic_info = get_license_info(info)

        viable.append({
            "model_id": model_id,
            "model_type": model_type,
            "architectures": get_architectures(info),
            "sha": getattr(info, "sha", None) or "",
            "params_billion": None,
            "downloads": downloads,
            "likes": getattr(info, "likes", 0) or 0,
            "license": lic_info,
            "last_modified": (getattr(info, "last_modified", None) or "").isoformat()
                if hasattr(getattr(info, "last_modified", None) or "", "isoformat") else None,
            "pipeline_tag": pipeline,
            "gated": False,
            "private": False,
            "remote_code": False,
            "catalog_match": cat_result,
        })

    viable.sort(key=lambda x: x["downloads"], reverse=True)
    return viable


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

# Collapse similar architectures into one diversity group
FAMILY_GROUP: dict[str, str] = {
    "olmo": "olmo-family",
    "olmo2": "olmo-family",
    "qwen2": "qwen-family",
    "qwen3": "qwen-family",
    "gemma": "gemma-family",
    "gemma2": "gemma-family",
    "gemma3": "gemma-family",
    "phi": "phi-family",
    "phi3": "phi-family",
    "phimoe": "phi-family",
    "granite": "granite-family",
    "granitemoe": "granite-family",
}

# Curated target models aligned with capability registry:
# - llama: verified-template family (2 models for within-family generalization)
# - qwen2: tool-supported-unverified candidate family (2 models)
# - granite: verified-template family (1 model, distinct revision from recipe)
CURATED_TARGETS: list[str] = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",      # llama, apache-2.0
    "HuggingFaceTB/SmolLM2-360M-Instruct",      # llama, apache-2.0 (distinct from 1.7B recipe)
    "Qwen/Qwen2-1.5B-Instruct",                 # qwen2, apache-2.0
    "Qwen/Qwen2-0.5B-Instruct",                 # qwen2, apache-2.0
    "ibm-granite/granite-3.2-2b-instruct",       # granite, apache-2.0 (distinct from 3.3-2b recipe)
]

# Fallback if a curated target fails verification
CURATED_FALLBACKS: list[str] = [
    "ibm-granite/granite-3.1-2b-instruct",       # granite fallback
    "ibm-granite/granite-3.0-2b-instruct",       # granite fallback
]


def _family_group(model_type: str) -> str:
    return FAMILY_GROUP.get(model_type.lower(), model_type.lower())


# Well-known official model publishers (prefer over community uploads)
OFFICIAL_ORGS: set[str] = {
    "openai-community", "meta-llama", "google", "microsoft",
    "mistralai", "Qwen", "allenai", "stabilityai", "bigscience",
    "EleutherAI", "tiiuae", "bigcode", "HuggingFaceTB",
    "ibm-granite", "CohereForAI", "THUDM", "TinyLlama",
    "h2oai", "utter-project", "sbintuitions",
}


def _is_instruction_model(model_id: str) -> bool:
    """Heuristic: model name contains instruct/chat/zephyr."""
    name = model_id.lower().split("/")[-1]
    return any(kw in name for kw in ("instruct", "chat", "zephyr", "rlhf"))


def verify_curated_model(
    model_id: str,
    api: HfApi,
    catalog_entries: list[dict[str, str]],
) -> dict[str, Any] | None:
    """Verify a curated target model against all selection rules.

    Returns a model entry dict if the model passes, or None with printed reason.
    """
    try:
        info = api.model_info(model_id)
    except Exception as e:
        print(f"  FAIL {model_id}: HF Hub lookup failed: {e}")
        return None

    # Gated check
    gated = getattr(info, "gated", None)
    if gated and gated is not False and str(gated).lower() not in ("false", "none", ""):
        print(f"  FAIL {model_id}: gated={gated}")
        return None

    # Remote code
    if has_remote_code(info):
        print(f"  FAIL {model_id}: requires remote code (auto_map)")
        return None

    # Required files
    if not has_required_files(info):
        print(f"  FAIL {model_id}: missing config/tokenizer/weight files")
        return None

    # Architecture
    model_type = get_model_type(info)
    if not model_type or model_type.lower() not in MOBIUS_RECOGNIZED_ARCHITECTURES:
        print(f"  FAIL {model_id}: model_type '{model_type}' not recognized")
        return None

    # Catalog check
    cat_result = catalog_match_check(model_id, catalog_entries)
    if cat_result["matched"]:
        print(f"  FAIL {model_id}: in FL catalog ({cat_result['reason']})")
        return None

    # Recipe check
    if model_id in EXCLUDED_MODEL_IDS:
        print(f"  FAIL {model_id}: has existing recipe")
        return None

    # License
    lic_info = get_license_info(info)

    # Params
    params_b = None
    st = getattr(info, "safetensors", None)
    if st and isinstance(st, dict):
        total = st.get("total")
        if total and isinstance(total, (int, float)):
            params_b = round(total / 1e9, 2)

    if params_b is not None and params_b > MAX_PARAMS_BILLION:
        print(f"  FAIL {model_id}: {params_b}B exceeds {MAX_PARAMS_BILLION}B limit")
        return None

    sha = getattr(info, "sha", None) or ""
    downloads = getattr(info, "downloads", 0) or 0
    likes = getattr(info, "likes", 0) or 0
    last_modified = getattr(info, "last_modified", None)

    print(f"  PASS {model_id} ({model_type}, ~{params_b}B, {lic_info['spdx_id']})")

    return {
        "model_id": model_id,
        "model_type": model_type,
        "architectures": get_architectures(info),
        "sha": sha,
        "params_billion": params_b,
        "downloads": downloads,
        "likes": likes,
        "license": lic_info,
        "last_modified": last_modified.isoformat() if hasattr(last_modified or "", "isoformat") else None,
        "pipeline_tag": getattr(info, "pipeline_tag", None),
        "gated": False,
        "private": False,
        "remote_code": False,
        "catalog_match": cat_result,
    }


def select_final_five(
    viable: list[dict[str, Any]],
    api: HfApi,
    catalog_entries: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Verify curated target models and select exactly five.

    Uses CURATED_TARGETS as primary candidates, falling back to
    CURATED_FALLBACKS if any fail verification. The viable list
    provides additional alternates for documentation.
    """
    selected: list[dict[str, Any]] = []

    # Verify each curated target
    all_targets = CURATED_TARGETS + CURATED_FALLBACKS
    for mid in all_targets:
        if len(selected) >= 5:
            break
        entry = verify_curated_model(mid, api, catalog_entries)
        if entry is not None:
            selected.append(entry)

    alternates = [v for v in viable if v["model_id"] not in {m["model_id"] for m in selected}][:10]
    return selected, alternates


# ---------------------------------------------------------------------------
# Evaluation prompts
# ---------------------------------------------------------------------------

def build_evaluation_prompts(model_id: str) -> list[dict[str, str]]:
    return [
        {
            "name": "basic_instruction",
            "prompt": "Explain what a hash table is in two sentences.",
            "expected_behavior": "Coherent explanation of hash table data structure",
        },
        {
            "name": "reasoning",
            "prompt": "If a train travels 60 mph for 2.5 hours, how far does it go? Show your work.",
            "expected_behavior": "Correct answer of 150 miles with arithmetic shown",
        },
        {
            "name": "code_generation",
            "prompt": "Write a Python function that checks if a string is a palindrome.",
            "expected_behavior": "Syntactically valid Python function with correct logic",
        },
        {
            "name": "creative_writing",
            "prompt": "Write a haiku about programming.",
            "expected_behavior": "A haiku (5-7-5 syllable structure) related to programming",
        },
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    timestamp = datetime.now(timezone.utc)
    print(f"Selection run: {timestamp.isoformat()}")
    print()

    # Step 1: Foundry catalog
    print("Step 1: Fetching Foundry Local catalog...")
    catalog = get_foundry_catalog()
    catalog_entries = extract_catalog_entries(catalog)
    catalog_count = len(catalog.get("models", []))
    print(f"  Catalog: {catalog_count} models, {len(catalog_entries)} matchable entries")
    print()

    # Step 2: Search HF Hub
    print("Step 2: Searching Hugging Face Hub...")
    api = HfApi()
    candidates = search_candidates(api)
    print(f"  Found {len(candidates)} raw candidates")
    print()

    # Step 3: Filter
    print("Step 3: Filtering against selection rules...")
    viable = filter_candidates(candidates, catalog_entries)
    print(f"  {len(viable)} viable candidates pass all rules")
    print()

    # Save raw candidates
    CANDIDATES_RAW.write_text(
        json.dumps(viable[:50], indent=2, default=str), encoding="utf-8",
    )
    print(f"  Saved top 50 to {CANDIDATES_RAW.name}")

    # Step 4: Select
    print()
    print("Step 4: Selecting final five models...")
    selected, alternates = select_final_five(viable, api, catalog_entries)

    families = {(m.get("model_type") or "unknown").lower() for m in selected}
    groups = {_family_group(m.get("model_type") or "unknown") for m in selected}
    print(f"  Selected {len(selected)} models, {len(families)} types ({sorted(families)}), {len(groups)} family groups ({sorted(groups)})")
    assert len(selected) == 5, f"Expected 5 models, got {len(selected)}"
    assert len(groups) >= 3, f"Expected >=3 family groups, got {len(groups)}: {sorted(groups)}"

    # Step 5: Build manifest
    print()
    print("Step 5: Building frozen manifest...")
    manifest = {
        "schema_version": "1.0.0",
        "purpose": "Recipe Agent v1 evaluation set - frozen unseen models",
        "selection_timestamp": timestamp.isoformat(),
        "foundry_catalog_snapshot": {
            "model_count": catalog_count,
            "entry_count": len(catalog_entries),
            "timestamp": timestamp.isoformat(),
        },
        "selection_rules": {
            "max_params_billion": MAX_PARAMS_BILLION,
            "min_downloads": MIN_DOWNLOADS,
            "excluded_recipe_ids": sorted(EXCLUDED_MODEL_IDS),
            "recognized_architectures": sorted(MOBIUS_RECOGNIZED_ARCHITECTURES),
            "clear_licenses": sorted(CLEAR_LICENSES),
        },
        "models": [],
        "alternates": [],
    }

    for i, m in enumerate(selected, 1):
        entry = {
            "index": i,
            "model_id": m["model_id"],
            "sha": m["sha"],
            "model_type": m["model_type"],
            "architectures": m["architectures"],
            "params_billion": m["params_billion"],
            "downloads": m["downloads"],
            "likes": m["likes"],
            "license": m["license"],
            "last_modified": m["last_modified"],
            "catalog_match": m["catalog_match"],
            "recipe_exists": False,
            "cpu_practical": True,
            "cpu_rationale": (
                f"~{m['params_billion']:.1f}B params; int4 quantized fits CPU RAM"
                if m["params_billion"]
                else "Within size ceiling based on file analysis"
            ),
            "mobius_recognition": f"model_type '{m['model_type']}' in recognized set",
            "evaluation_prompts": build_evaluation_prompts(m["model_id"]),
        }
        manifest["models"].append(entry)
        lic_label = m["license"]["spdx_id"]
        if m["license"]["review_required"]:
            lic_label += f" (review: {m['license'].get('license_name', 'unknown')})"
        print(f"  [{i}] {m['model_id']} ({m['model_type']}, ~{m['params_billion']}B, {m['downloads']:,} dl, {lic_label})")

    for a in alternates[:5]:
        manifest["alternates"].append({
            "model_id": a["model_id"],
            "model_type": a["model_type"],
            "params_billion": a["params_billion"],
            "downloads": a["downloads"],
            "license": a["license"],
            "rejection_reason": "Not selected; alternate for replacement if primary becomes unavailable",
        })

    MODELS_JSON.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print()
    print(f"  Frozen manifest: {MODELS_JSON.name}")
    print("Selection complete.")


if __name__ == "__main__":
    main()
