#!/usr/bin/env python3
"""
Reproducible model selection script for Recipe Agent v1 evaluation.

Queries Hugging Face Hub API (metadata only, no weight downloads) and the live
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

# Architecture families recognized by Mobius / OGA for text-generation
# (model_type values from HF config.json)
MOBIUS_RECOGNIZED_ARCHITECTURES: set[str] = {
    "llama",
    "mistral",
    "phi",
    "phi3",
    "phimoe",
    "gemma",
    "gemma2",
    "gemma3",
    "qwen2",
    "qwen3",
    "gpt2",
    "gpt_neox",
    "bloom",
    "falcon",
    "mpt",
    "opt",
    "starcoder2",
    "stablelm",
    "tinyllama",
    "cohere",
    "olmo",
    "olmo2",
    "glm",
}

# Organizations/prefixes that indicate testing or toy models
EXCLUDED_ORG_PREFIXES: set[str] = {
    "trl-internal-testing",
    "hf-internal-testing",
    "test-org",
    "stas",
}

# Upper bound on parameter count (billions) for CPU-practicality
MAX_PARAMS_BILLION = 4.0

# Minimum downloads to filter out toy / test models
MIN_DOWNLOADS = 500


def get_foundry_catalog() -> dict[str, Any]:
    """Get live Foundry Local catalog via CLI."""
    try:
        result = subprocess.run(
            ["foundry", "model", "list", "-o", "json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        pass
    return {"models": []}


def extract_catalog_identifiers(catalog: dict[str, Any]) -> set[str]:
    """Extract all model name fragments from the Foundry catalog for matching."""
    ids: set[str] = set()
    models = catalog.get("models", [])
    if isinstance(models, list):
        for m in models:
            if isinstance(m, dict):
                for key in ("alias", "id", "displayName"):
                    val = m.get(key, "")
                    if val:
                        ids.add(val.lower())
    return ids


def model_in_catalog(model_id: str, catalog_ids: set[str]) -> bool:
    """Check if an HF model ID appears to match any Foundry catalog entry.

    Uses precise matching: normalizes the HF model name to the format used
    in Foundry catalog aliases (lowercase, hyphens, no org prefix) and
    checks if it matches a catalog alias or if the catalog entry name
    starts with the model name (to catch variant suffixes like -generic-gpu).
    """
    # Normalize: "Qwen/Qwen2.5-3B-Instruct" -> "qwen2.5-3b-instruct"
    name = model_id.split("/")[-1].lower().replace("_", "-").replace(".", "-")
    for cid in catalog_ids:
        cid_norm = cid.lower().replace("_", "-").replace(".", "-")
        # Exact match
        if name == cid_norm:
            return True
        # Catalog entry starts with model name (e.g., "qwen2.5-0.5b-instruct-vitis-npu")
        if cid_norm.startswith(name + "-") or cid_norm.startswith(name + ":"):
            return True
        # Model name starts with catalog alias (e.g., alias "qwen2.5-0.5b")
        if name.startswith(cid_norm + "-"):
            return True
    return False


def estimate_params_billion(info: ModelInfo) -> float | None:
    """Estimate parameter count in billions from safetensors metadata."""
    st = getattr(info, "safetensors", None)
    if st and isinstance(st, dict):
        params = st.get("total", None)
        if params and isinstance(params, (int, float)):
            return params / 1e9
    # Fallback: check siblings for model size hints
    siblings = getattr(info, "siblings", None) or []
    total_bytes = sum(
        (s.size or 0) for s in siblings
        if s.rfilename and s.rfilename.endswith((".safetensors", ".bin"))
    )
    if total_bytes > 0:
        # Rough: 2 bytes per param for fp16, 4 bytes for fp32; assume fp16
        return total_bytes / 2 / 1e9
    return None


def has_remote_code(info: ModelInfo) -> bool:
    """Check if the model requires trust_remote_code."""
    tags = set(getattr(info, "tags", []) or [])
    if "custom_code" in tags:
        return True
    # Check config for auto_map
    config = getattr(info, "config", None) or {}
    if isinstance(config, dict) and "auto_map" in config:
        return True
    return False


def has_required_files(info: ModelInfo) -> bool:
    """Check model has config.json, tokenizer files, and weight files."""
    siblings = getattr(info, "siblings", None) or []
    filenames = {s.rfilename for s in siblings if s.rfilename}
    has_config = "config.json" in filenames
    has_tokenizer = bool(
        filenames & {"tokenizer.json", "tokenizer_config.json", "tokenizer.model"}
    )
    has_weights = any(
        f.endswith((".safetensors", ".bin", ".gguf")) for f in filenames
    )
    return has_config and has_tokenizer and has_weights


def get_model_type(info: ModelInfo) -> str | None:
    """Extract model_type from config."""
    config = getattr(info, "config", None) or {}
    if isinstance(config, dict):
        return config.get("model_type", None)
    return None


def search_candidates(api: HfApi) -> list[ModelInfo]:
    """Search HF Hub for text-generation instruction models."""
    candidates: list[ModelInfo] = []

    # Search across multiple architecture-related queries
    search_queries = [
        "instruct",
        "chat",
    ]
    seen_ids: set[str] = set()

    for query in search_queries:
        try:
            results = api.list_models(
                search=query,
                pipeline_tag="text-generation",
                sort="downloads",
                limit=200,
                cardData=True,
                fetch_config=True,
                full=True,
                num_parameters="max:4B",
            )
            for m in results:
                mid = m.id or ""
                if mid and mid not in seen_ids:
                    seen_ids.add(mid)
                    candidates.append(m)
        except Exception as e:
            print(f"  Warning: search '{query}' failed: {e}", file=sys.stderr)

    return candidates


def filter_and_score(
    candidates: list[ModelInfo],
    catalog_ids: set[str],
) -> list[dict[str, Any]]:
    """Apply selection rules and score candidates."""
    viable: list[dict[str, Any]] = []

    for info in candidates:
        model_id = info.id or ""
        if not model_id:
            continue

        # Rule 2: Not in existing recipes
        if model_id in EXCLUDED_MODEL_IDS:
            continue

        # Filter out testing/internal orgs
        org = model_id.split("/")[0] if "/" in model_id else ""
        if org.lower() in EXCLUDED_ORG_PREFIXES:
            continue

        # Rule 5: Not gated, not private, no remote code
        gated = getattr(info, "gated", None)
        if gated and gated is not False and str(gated).lower() not in ("false", "none", ""):
            continue
        if getattr(info, "private", False):
            continue
        if has_remote_code(info):
            continue

        # Rule 6: Has required files
        if not has_required_files(info):
            continue

        # Rule 4: Recognized architecture
        model_type = get_model_type(info)
        if not model_type or model_type.lower() not in MOBIUS_RECOGNIZED_ARCHITECTURES:
            continue

        # Rule 1: Not in Foundry catalog
        if model_in_catalog(model_id, catalog_ids):
            continue

        # Rule 7: CPU-practical size
        params_b = estimate_params_billion(info)
        if params_b is not None and params_b > MAX_PARAMS_BILLION:
            continue
        # Name-based size heuristic if params unavailable
        if params_b is None:
            name_lower = model_id.lower()
            size_match = re.search(r'(\d+)[bB]', name_lower.split("/")[-1])
            if size_match:
                size_val = int(size_match.group(1))
                if size_val > MAX_PARAMS_BILLION:
                    continue

        # Rule 3: Text-generation oriented
        pipeline = getattr(info, "pipeline_tag", None)
        if pipeline and pipeline != "text-generation":
            continue

        # Minimum downloads filter
        downloads = getattr(info, "downloads", 0) or 0
        if downloads < MIN_DOWNLOADS:
            continue

        # License check
        license_val = getattr(info, "card_data", None)
        license_str = ""
        if license_val and hasattr(license_val, "license"):
            license_str = license_val.license or ""
        if not license_str:
            tags = getattr(info, "tags", []) or []
            for t in tags:
                if t.startswith("license:"):
                    license_str = t.split(":", 1)[1]
                    break

        sha = getattr(info, "sha", None) or ""
        last_modified = getattr(info, "last_modified", None)
        likes = getattr(info, "likes", 0) or 0

        viable.append({
            "model_id": model_id,
            "model_type": model_type,
            "architectures": _get_architectures(info),
            "sha": sha,
            "params_billion": round(params_b, 2) if params_b else None,
            "downloads": downloads,
            "likes": likes,
            "license": license_str,
            "last_modified": last_modified.isoformat() if last_modified else None,
            "pipeline_tag": pipeline,
            "gated": False,
            "private": False,
            "remote_code": False,
            "has_config": True,
            "has_tokenizer": True,
            "has_weights": True,
            "catalog_match": False,
            "cpu_practical": True,
            "cpu_rationale": (
                f"~{params_b:.1f}B params, int4 quantized fits CPU RAM"
                if params_b
                else "Size within CPU range based on file analysis"
            ),
        })

    # Sort by downloads descending
    viable.sort(key=lambda x: x["downloads"], reverse=True)
    return viable


def _get_architectures(info: ModelInfo) -> list[str]:
    """Extract architecture list from model config."""
    config = getattr(info, "config", None) or {}
    if isinstance(config, dict):
        archs = config.get("architectures", [])
        if isinstance(archs, list):
            return archs
    return []


def select_final_five(viable: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Select exactly five models covering at least three architecture families.
    Strategy: pick top candidate per unique architecture family first, then
    fill remaining slots with best diversity/quality trade-off.
    """
    # Group by architecture family (model_type)
    by_family: dict[str, list[dict[str, Any]]] = {}
    for v in viable:
        fam = (v.get("model_type") or "unknown").lower()
        by_family.setdefault(fam, []).append(v)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    # Phase 1: Pick best from each family (by downloads), at least 3 families
    families_sorted = sorted(
        by_family.keys(),
        key=lambda f: max(c["downloads"] for c in by_family[f]),
        reverse=True,
    )

    for fam in families_sorted:
        if len(selected) >= 5:
            break
        candidates = [c for c in by_family[fam] if c["model_id"] not in selected_ids]
        if candidates:
            pick = candidates[0]
            selected.append(pick)
            selected_ids.add(pick["model_id"])

    # Phase 2: If < 5, fill from remaining viable by downloads
    if len(selected) < 5:
        remaining = [v for v in viable if v["model_id"] not in selected_ids]
        for r in remaining:
            if len(selected) >= 5:
                break
            selected.append(r)
            selected_ids.add(r["model_id"])

    # Build alternates: next best not selected
    alternates = [v for v in viable if v["model_id"] not in selected_ids][:10]

    return selected, alternates


def build_evaluation_prompts(model_id: str) -> list[dict[str, str]]:
    """Define objective functional prompts for later validation."""
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


def main() -> None:
    timestamp = datetime.now(timezone.utc)
    print(f"Selection run: {timestamp.isoformat()}")
    print()

    # Step 1: Get Foundry catalog
    print("Step 1: Fetching Foundry Local catalog...")
    catalog = get_foundry_catalog()
    catalog_ids = extract_catalog_identifiers(catalog)
    catalog_count = len(catalog.get("models", []))
    print(f"  Catalog contains {catalog_count} models")
    print()

    # Step 2: Search HF Hub
    print("Step 2: Searching Hugging Face Hub for candidates...")
    api = HfApi()
    candidates = search_candidates(api)
    print(f"  Found {len(candidates)} raw candidates")
    print()

    # Step 3: Filter and score
    print("Step 3: Filtering candidates against selection rules...")
    viable = filter_and_score(candidates, catalog_ids)
    print(f"  {len(viable)} viable candidates pass all rules")
    print()

    # Save raw candidates
    CANDIDATES_RAW.write_text(
        json.dumps(viable[:50], indent=2, default=str),
        encoding="utf-8",
    )
    print(f"  Saved top 50 candidates to {CANDIDATES_RAW.name}")

    # Step 4: Select final five
    print()
    print("Step 4: Selecting final five models...")
    selected, alternates = select_final_five(viable)

    # Enrich selected models with full model_info for accurate params
    print("  Enriching selected models with full metadata...")
    for m in selected:
        try:
            full_info = api.model_info(m["model_id"])
            st = getattr(full_info, "safetensors", None)
            if st and isinstance(st, dict):
                total = st.get("total", None)
                if total and isinstance(total, (int, float)):
                    m["params_billion"] = round(total / 1e9, 2)
            m["sha"] = getattr(full_info, "sha", m["sha"]) or m["sha"]
        except Exception:
            pass

    # Post-enrichment: remove models exceeding size limit
    valid_selected = [m for m in selected if (m.get("params_billion") or 0) <= MAX_PARAMS_BILLION]
    if len(valid_selected) < len(selected):
        removed = [m for m in selected if m not in valid_selected]
        for r in removed:
            print(f"  Removed {r['model_id']} ({r.get('params_billion')}B exceeds {MAX_PARAMS_BILLION}B limit)")
        # Fill from alternates
        remaining_alternates = [v for v in viable if v["model_id"] not in {m["model_id"] for m in valid_selected}]
        for alt in remaining_alternates:
            if len(valid_selected) >= 5:
                break
            try:
                full_info = api.model_info(alt["model_id"])
                st = getattr(full_info, "safetensors", None)
                if st and isinstance(st, dict):
                    total = st.get("total", None)
                    if total and isinstance(total, (int, float)):
                        alt["params_billion"] = round(total / 1e9, 2)
                        if alt["params_billion"] > MAX_PARAMS_BILLION:
                            continue
                alt["sha"] = getattr(full_info, "sha", alt["sha"]) or alt["sha"]
            except Exception:
                pass
            valid_selected.append(alt)
        selected = valid_selected

    # Verify at least 3 architecture families
    families = {(m.get("model_type") or "unknown").lower() for m in selected}
    print(f"  Selected {len(selected)} models across {len(families)} families: {sorted(families)}")
    assert len(selected) == 5, f"Expected 5 models, got {len(selected)}"
    assert len(families) >= 3, f"Expected >=3 families, got {len(families)}"

    # Step 5: Build frozen manifest
    print()
    print("Step 5: Building frozen manifest...")
    manifest = {
        "schema_version": "1.0.0",
        "purpose": "Recipe Agent v1 evaluation set - frozen unseen models",
        "selection_timestamp": timestamp.isoformat(),
        "foundry_catalog_snapshot": {
            "model_count": catalog_count,
            "timestamp": timestamp.isoformat(),
        },
        "selection_rules": {
            "max_params_billion": MAX_PARAMS_BILLION,
            "min_downloads": MIN_DOWNLOADS,
            "excluded_recipe_ids": sorted(EXCLUDED_MODEL_IDS),
            "recognized_architectures": sorted(MOBIUS_RECOGNIZED_ARCHITECTURES),
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
            "catalog_match": False,
            "recipe_exists": False,
            "cpu_practical": True,
            "cpu_rationale": m["cpu_rationale"],
            "mobius_recognition": f"model_type '{m['model_type']}' in recognized set",
            "evaluation_prompts": build_evaluation_prompts(m["model_id"]),
        }
        manifest["models"].append(entry)
        print(f"  [{i}] {m['model_id']} ({m['model_type']}, ~{m['params_billion']}B, {m['downloads']:,} downloads)")

    for a in alternates[:5]:
        manifest["alternates"].append({
            "model_id": a["model_id"],
            "model_type": a["model_type"],
            "params_billion": a["params_billion"],
            "downloads": a["downloads"],
            "rejection_reason": "Not selected; alternates for diversity or if primary becomes unavailable",
        })

    MODELS_JSON.write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )
    print()
    print(f"  Frozen manifest written to {MODELS_JSON.name}")
    print()
    print("Selection complete.")


if __name__ == "__main__":
    main()
