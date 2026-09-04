# Recipe Agent v1 — Evaluation Model Selection Report

**Generated:** 2026-08-31T23:47:14Z (revision 3)  
**Foundry Local catalog snapshot:** 50 models, 150 matchable entries  
**Selection script:** `evaluation/recipe-agent-v1/select_models.py`

## Summary

Five **instruction/chat** models frozen for Recipe Agent v1 evaluation.
All from official publishers with Apache-2.0 licenses. Three architecture
families with within-family generalization pairs, aligned with the capability
registry: llama (verified-template), qwen2 (tool-supported-unverified),
granite (verified-template).

## Selected Models

| # | Model ID | Architecture | Family | Params | Downloads | License |
|---|----------|-------------|--------|--------|-----------|---------|
| 1 | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | llama | llama | ~1.10B | 1,918,507 | Apache-2.0 |
| 2 | `HuggingFaceTB/SmolLM2-360M-Instruct` | llama | llama | ~0.36B | 306,534 | Apache-2.0 |
| 3 | `Qwen/Qwen2-1.5B-Instruct` | qwen2 | qwen-family | ~1.54B | 968,749 | Apache-2.0 |
| 4 | `Qwen/Qwen2-0.5B-Instruct` | qwen2 | qwen-family | ~0.49B | 259,479 | Apache-2.0 |
| 5 | `ibm-granite/granite-3.2-2b-instruct` | granite | granite-family | ~2.53B | 7,959 | Apache-2.0 |

## Within-Family Generalization

| Family | Models | Size Range | Purpose |
|--------|--------|-----------|---------|
| llama | TinyLlama-1.1B-Chat, SmolLM2-360M-Instruct | 0.36B–1.10B | Tests pipeline handles different sizes of same architecture |
| qwen-family | Qwen2-1.5B-Instruct, Qwen2-0.5B-Instruct | 0.49B–1.54B | Tests within-family generalization for tool-supported candidate |
| granite-family | granite-3.2-2b-instruct | 2.53B | Tests distinct Granite revision from registered 3.3-2B recipe |

## Capability Registry Alignment

| Family | Registry Status | Evaluation Purpose |
|--------|----------------|-------------------|
| llama | Verified-template | Validate pipeline generalizes across llama-family sizes |
| qwen2 | Tool-supported-unverified | Validate unverified candidate family can be onboarded |
| granite | Verified-template | Validate distinct revision (3.2 vs 3.3) works with existing template |

## License Verification

All five models have Apache-2.0 (SPDX: `apache-2.0`).
No models with `license: other` or custom licenses. No legal review required.

## Selection Criteria Applied

Each model verified against:
1. ✅ **Not in Foundry Local catalog** (50 models checked)
2. ✅ **No exact recipe in recipes.py** (SmolLM2-1.7B excluded, 360M is distinct)
3. ✅ **Instruction/chat oriented** with assessable output
4. ✅ **Architecture recognized** by Mobius (llama, qwen2, granite)
5. ✅ **Not gated**, not private, no remote code (auto_map)
6. ✅ **Apache-2.0 license** with complete metadata
7. ✅ **CPU-practical** (all ≤4B params)
8. ✅ **Official publisher** (TinyLlama, HuggingFaceTB, Qwen, IBM Granite)

## Changes from Revision 2

| Issue | Resolution |
|-------|-----------|
| GPT-2 and StableLM-3b were base models | Replaced with all-instruction set |
| Only 1 model per family (no generalization testing) | 2 llama + 2 qwen pairs |
| Families not aligned with capability registry | llama/qwen2/granite match registry |
| No granite coverage | Added granite-3.2-2b-instruct |

## Excluded Models

### Already in Recipe Registry
| Model ID | Recipe |
|----------|--------|
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | smollm2-1.7b-cpu-int4 (verified) |
| `distil-whisper/distil-medium.en` | distil-whisper-cpu-fp16 (blocked) |
| `ibm-granite/granite-3.3-2b-instruct` | granite-3.3-2b-cpu-int4 (verified) |

### Fallback Candidates (if primary fails)
| Model ID | Architecture | Status |
|----------|-------------|--------|
| `ibm-granite/granite-3.1-2b-instruct` | granite | Available fallback |
| `ibm-granite/granite-3.0-2b-instruct` | granite | Available fallback |

## Evaluation Prompts

Four generic, architecture-independent functional prompts per model:
1. **basic_instruction** — "Explain what a hash table is in two sentences."
2. **reasoning** — "If a train travels 60 mph for 2.5 hours, how far does it go?"
3. **code_generation** — "Write a Python function that checks if a string is a palindrome."
4. **creative_writing** — "Write a haiku about programming."

## Reproducibility

```bash
python evaluation/recipe-agent-v1/select_models.py
```

Verifies curated targets against live HF Hub and Foundry catalog. Falls back
to alternates if any target fails. Broader candidate pool (133 viable) also
searched for documentation purposes.
