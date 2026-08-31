# Recipe Agent v1 — Evaluation Model Selection Report

**Generated:** 2026-08-31T23:35:18Z (revision 2)  
**Foundry Local catalog snapshot:** 50 models, 150 matchable entries  
**Selection script:** `evaluation/recipe-agent-v1/select_models.py`

## Summary

Five text-generation models selected for the Recipe Agent v1 evaluation set.
All from **official publishers** with **clear open-source licenses** (MIT,
Apache-2.0, CC-BY-SA-4.0). The set spans **5 architecture families** across
5 distinct family groups (gpt2, qwen-family, llama, olmo-family, stablelm).

## Selected Models

| # | Model ID | Architecture | Family Group | Params | Downloads | License | Publisher |
|---|----------|-------------|-------------|--------|-----------|---------|-----------|
| 1 | `openai-community/gpt2` | gpt2 | gpt2 | ~0.14B | 14,317,307 | MIT | OpenAI |
| 2 | `Qwen/Qwen2-1.5B-Instruct` | qwen2 | qwen-family | ~1.54B | 968,749 | Apache-2.0 | Alibaba Qwen |
| 3 | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | llama | llama | ~1.10B | 1,918,507 | Apache-2.0 | TinyLlama |
| 4 | `allenai/OLMo-2-0425-1B-Instruct` | olmo2 | olmo-family | ~1.48B | 74,486 | Apache-2.0 | Allen AI |
| 5 | `stabilityai/stablelm-3b-4e1t` | stablelm | stablelm | ~2.80B | 47,217 | CC-BY-SA-4.0 | Stability AI |

## License Verification

All five models have well-known, clear open-source licenses:

| Model | SPDX ID | License Name | Review Required | Confidence |
|-------|---------|-------------|-----------------|------------|
| gpt2 | `mit` | MIT License | No | High |
| Qwen2-1.5B-Instruct | `apache-2.0` | Apache License 2.0 | No | High |
| TinyLlama-1.1B-Chat | `apache-2.0` | Apache License 2.0 | No | High |
| OLMo-2-0425-1B-Instruct | `apache-2.0` | Apache License 2.0 | No | High |
| stablelm-3b-4e1t | `cc-by-sa-4.0` | CC BY-SA 4.0 | No | High |

**No models with `license: other` or custom/unclear licenses are included.**
This was a deliberate design choice to avoid legal review requirements.

## Selection Rationale

### Architecture Diversity (5 family groups)
- **gpt2** — GPT-2 is the foundational autoregressive transformer. Tests the
  pipeline against the original architecture that influenced most modern LLMs.
  Base model (not instruction-tuned) but produces coherent, assessable text completions.
- **qwen-family (qwen2)** — Qwen2-1.5B-Instruct from Alibaba. Instruction-tuned,
  Apache-2.0 licensed. Note: Qwen2 (not Qwen2.5) avoids the `license: other`
  (qwen-research) that affects Qwen2.5 models.
- **llama** — TinyLlama-1.1B-Chat is a well-known compact instruct model.
  Provides a lower bound on model size for stress-testing the onboarding pipeline.
- **olmo-family (olmo2)** — OLMo-2 from Allen AI. Fully open research LLM with
  instruction tuning. Tests a less common but Mobius-recognized architecture.
- **stablelm** — StableLM-3b from Stability AI. The largest model in the set at
  2.8B params. Base model with strong text generation capability. CC-BY-SA-4.0 licensed.

### Size Spread
- 0.14B (GPT-2) → 1.10B (TinyLlama) → 1.48B (OLMo-2) → 1.54B (Qwen2) → 2.80B (StableLM)
- All within the 4B parameter ceiling for CPU int4 inference practicality.
- Meaningful spread from 140M to 2.8B parameters.

### Popularity/Quality Spread
- Very high: GPT-2 (14.3M downloads)
- High: TinyLlama (1.9M), Qwen2-1.5B (969K)
- Medium: OLMo-2 (74K), StableLM (47K)

## Changes from Revision 1

| Issue | Resolution |
|-------|-----------|
| Qwen2.5-3B-Instruct had `license: other` (qwen-research), reported as apache-2.0 | Replaced with Qwen2-1.5B-Instruct (apache-2.0) |
| kakaocorp/kanana-2-3b-instruct had `license: other` (kanana-open-license) | Replaced with stabilityai/stablelm-3b-4e1t (cc-by-sa-4.0) |
| ministral/Ministral-3b-instruct was community upload, not official publisher | Replaced with openai-community/gpt2 (official, MIT) |
| Catalog matching had false positives (broad substring match) | Hardened with normalized prefix matching, dot↔hyphen, confidence/reason |
| License data was incorrectly reported as apache-2.0 for all | Now uses authoritative HF Hub metadata with spdx_id, license_name, URL |

## Excluded Models

### Already in Recipe Registry
| Model ID | Recipe |
|----------|--------|
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | smollm2-1.7b-cpu-int4 (verified) |
| `distil-whisper/distil-medium.en` | distil-whisper-cpu-fp16 (blocked) |
| `ibm-granite/granite-3.3-2b-instruct` | granite-3.3-2b-cpu-int4 (verified) |

### Already in Foundry Catalog
Qwen2.5-{0.5B,1.5B,7B,14B}-Instruct, Qwen3-{0.6B–14B}, Phi-3/3.5/4 variants,
DeepSeek-R1, SmolLM3-3B, Mistral-7B/Nemo-12B, OLMo-3-7B, Whisper variants.

### Rejected for License
| Model | License | Reason |
|-------|---------|--------|
| Qwen/Qwen2.5-3B-Instruct | `other` (qwen-research) | Custom license requires review |
| kakaocorp/kanana-2-3b-instruct | `other` (kanana-open-license) | Custom license requires review |
| stabilityai/stablelm-zephyr-3b | `other` | No license name/URL documented |
| THUDM/glm-edge-{1.5b,4b}-chat | `other` (glm-4) | Custom license requires review |

### Rejected for Provenance
| Model | Reason |
|-------|--------|
| ministral/Ministral-3b-instruct | Community upload, not official mistralai publisher |
| Various community fine-tunes | Non-official orgs, questionable provenance |

## Catalog Matching

Matching uses hardened normalization:
- Organization prefixes stripped from both HF and catalog identifiers
- Dots, underscores, and hyphens treated as equivalent
- Multiple hyphens collapsed
- Matches checked against alias, id, and displayName fields
- Each match returns confidence level (exact/high/none) and reason string
- Tests cover org-prefixed IDs, lossy punctuation, and false-positive cases

## Reproducibility

```bash
python evaluation/recipe-agent-v1/select_models.py
```

## Evaluation Prompts

Four objective functional prompts per model:
1. **basic_instruction** — hash table explanation
2. **reasoning** — arithmetic word problem
3. **code_generation** — palindrome function
4. **creative_writing** — programming haiku

## Uncertainties

1. **GPT-2 and StableLM are base models** — not instruction-tuned but produce
   assessable text completions. Selected for architecture diversity with clear licenses.
2. **Catalog drift** — Foundry catalog may add these models between selection and
   evaluation. Frozen SHAs remain valid regardless.
3. **Qwen2 vs Qwen2.5** — Qwen2-1.5B-Instruct was chosen over Qwen2.5-3B-Instruct
   specifically because Qwen2 has apache-2.0 while Qwen2.5 has a custom license.
