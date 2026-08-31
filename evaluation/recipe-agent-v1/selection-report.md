# Recipe Agent v1 — Evaluation Model Selection Report

**Generated:** 2026-08-31T23:02:16Z  
**Foundry Local catalog snapshot:** 50 models  
**Selection script:** `evaluation/recipe-agent-v1/select_models.py`

## Summary

Five text-generation models were selected for the Recipe Agent v1 evaluation set.
The models span **5 architecture families** (qwen2, llama, olmo2, mistral, qwen3),
cover a meaningful range of parameter counts (1.1B–3.5B), downloads (8K–7.5M),
and represent diverse organizations.

## Selected Models

| # | Model ID | Architecture | Params | Downloads | License |
|---|----------|-------------|--------|-----------|---------|
| 1 | `Qwen/Qwen2.5-3B-Instruct` | qwen2 (Qwen2ForCausalLM) | ~3.09B | 7,540,588 | apache-2.0 |
| 2 | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | llama (LlamaForCausalLM) | ~1.10B | 1,918,507 | apache-2.0 |
| 3 | `allenai/OLMo-2-0425-1B-Instruct` | olmo2 (Olmo2ForCausalLM) | ~1.48B | 74,486 | apache-2.0 |
| 4 | `ministral/Ministral-3b-instruct` | mistral (MistralForCausalLM) | ~3.32B | 17,902 | apache-2.0 |
| 5 | `kakaocorp/kanana-2-3b-instruct` | qwen3 (Qwen3ForCausalLM) | ~3.51B | 8,179 | apache-2.0 |

## Selection Rationale

### Architecture Diversity
- **qwen2** — Qwen2.5-3B is a mid-size general-purpose instruct model from Alibaba's Qwen family.
  Most popular text-generation model in the 3B range not already in the FL catalog.
- **llama** — TinyLlama-1.1B is a well-known compact model based on the Llama architecture.
  Provides a lower bound on model size for stress-testing the onboarding pipeline.
- **olmo2** — OLMo-2 is Allen AI's open-source research LLM with a unique architecture variant.
  Tests the pipeline against a less common but Mobius-recognized architecture.
- **mistral** — Ministral-3b-instruct provides Mistral architecture coverage at a CPU-feasible size.
  Complements the catalog's larger Mistral variants (7B+).
- **qwen3** — Kanana-2-3b-instruct uses Qwen3ForCausalLM architecture from Kakao Corp.
  Exercises the newer qwen3 model_type distinct from qwen2.

### Size Spread
- 1.1B (TinyLlama) → 1.48B (OLMo-2) → 3.09B (Qwen2.5) → 3.32B (Ministral) → 3.51B (Kanana)
- All within the 4B parameter ceiling for CPU int4 inference practicality.

### Popularity/Quality Spread
- High: Qwen2.5-3B (7.5M downloads), TinyLlama (1.9M downloads)
- Medium: OLMo-2 (74K downloads)
- Lower: Ministral-3b (18K), Kanana-2 (8K)
- This spread tests whether the pipeline works for both popular and niche models.

## Excluded Models (Recipes Already Exist)

| Model ID | Reason |
|----------|--------|
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | Recipe `smollm2-1.7b-cpu-int4` exists (verified) |
| `distil-whisper/distil-medium.en` | Recipe `distil-whisper-cpu-fp16` exists (blocked) |
| `ibm-granite/granite-3.3-2b-instruct` | Recipe `granite-3.3-2b-cpu-int4` exists (verified) |

## Excluded Models (In Foundry Catalog)

Models filtered out because they already appear in the Foundry Local catalog:
- Qwen2.5-{0.5B,1.5B,7B,14B}-Instruct variants
- Qwen3 variants (0.6B–14B)
- Phi-3/3.5/4 variants
- DeepSeek-R1 variants
- SmolLM3-3B
- Mistral-Nemo-12B, OLMo-3-7B
- All Whisper variants

## Rejected Alternates

Top alternates that passed all rules but were not selected:

1. Additional qwen2/llama variants — redundant with selected models
2. Community fine-tunes — lower confidence in metadata completeness
3. Models with borderline param counts — risk of exceeding CPU feasibility

Full candidate list available in `candidates_raw.json` (top 50 viable candidates).

## Catalog Verification

All five selected models were verified absent from the Foundry Local catalog
(50 models) at selection time. Matching uses normalized name comparison with
dot/hyphen/underscore equivalence.

## Reproducibility

The selection can be reproduced by running:
```bash
python evaluation/recipe-agent-v1/select_models.py
```

This will:
1. Query the live Foundry Local catalog via `foundry model list -o json`
2. Search Hugging Face Hub for text-generation instruct/chat models
3. Apply all selection rules (architecture, size, gating, catalog exclusion, etc.)
4. Select five models across at least three architecture families
5. Freeze exact SHA revisions and write `models.json`

**Note:** Results may differ if the HF Hub or Foundry catalog changes. The frozen
manifest in `models.json` captures the exact selection at the timestamp above.

## Evaluation Prompts

Each model has four objective functional prompts defined for later validation:
1. **basic_instruction** — "Explain what a hash table is in two sentences."
2. **reasoning** — "If a train travels 60 mph for 2.5 hours, how far does it go?"
3. **code_generation** — "Write a Python function that checks if a string is a palindrome."
4. **creative_writing** — "Write a haiku about programming."

These prompts test coherence, arithmetic, code syntax, and creative output
without requiring domain-specific knowledge.

## Uncertainties

1. **Ministral org legitimacy** — `ministral/Ministral-3b-instruct` appears to be a community
   upload (not from `mistralai`). The model has proper config/weights but lower provenance confidence.
2. **Kanana-2 Qwen3 architecture** — Uses Qwen3ForCausalLM architecture despite being from Kakao Corp.
   May have custom training adaptations not reflected in model_type.
3. **Catalog drift** — The Foundry catalog may add these models between selection and evaluation.
   The frozen SHAs in `models.json` remain valid regardless.
