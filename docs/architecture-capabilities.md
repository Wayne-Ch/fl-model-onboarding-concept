# Architecture capabilities registry

This registry captures **architecture-level capability knowledge** for deterministic recipe compilation.  
It is keyed by **model_type / architecture alias + task + device + requested precision**, never by exact model ID.

## What this is (and is not)

- **Architecture capability**: "For this architecture family, these tool contracts are known and this status/evidence applies."
- **Model recipe**: "For this exact model revision, we validated an end-to-end path."

A capability can be `verified`, `tool-supported-unverified`, or `source-change-required`.  
`tool-supported-unverified` means tooling/docs recognize the family, but FL end-to-end proof is still missing.  
`source-change-required` means current upstream contracts are incompatible and require owner-side fixes.

This separation prevents a frequent mistake: **Mobius model-type recognition is not treated as Foundry Local compatibility proof.**
It also prevents overclaiming multimodal variants: only aliases directly supported by evidence are marked verified.

## Data + schema

- Data: `config/architecture-capabilities.json`
- Schema: `contracts/architecture-capabilities.schema.json`
- Loader/resolver: `src/fl_model_onboarding/architecture_capabilities.py`

The loader fails closed on:

1. Missing required evidence.
2. Duplicate aliases that would produce ambiguous task/device/precision matches.
3. Invalid status transition declarations.
4. Schema-version mismatch.
5. Argument-confidence/status mismatches (for example, unverified capabilities using evidence-pinned argument confidence).

## Current seeded families

- **Verified (LLM CPU INT4)**: `llama`, `granite`
- **Tool-supported-unverified (LLM CPU INT4 candidates)**: `qwen` (`qwen2`/`qwen3`), `phi` (`phi`/`phi3`/`phi3small`/`phimoe`)
- **Source-change-required (non-text boundary reference)**: `whisper` ASR

`mllama` / `MllamaForCausalLM` is intentionally **not** in the verified llama capability. It resolves as unregistered/not-eligible for text-only recipe generation until dedicated evidence exists.

## Verified arguments vs candidate arguments

Each capability includes explicit argument confidence markers:

- `mobius_rules.dtype_confidence`
- `olive_rules.precision_confidence`

Allowed values:

- `evidence-pinned` (verified or source-change-required entries with observed evidence)
- `candidate-unverified` (tool-supported-unverified entries)

Schema and loader invariants require:

1. `verified` -> both confidence markers must be `evidence-pinned`.
2. `tool-supported-unverified` -> both confidence markers must be `candidate-unverified`.
3. `source-change-required` -> both confidence markers must be `evidence-pinned`.

## Capability boundary behavior

Resolver inputs are normalized HF metadata + `task` + `device` + `requested_precision`.

The resolver rejects at capability boundary (fail closed) when:

1. Model is gated.
2. Model requires remote code.
3. Task is not `llm`.
4. Device is not `cpu`.
5. Architecture resolves only to non-text capability families.
6. Precision is unsupported or mapping is ambiguous.
