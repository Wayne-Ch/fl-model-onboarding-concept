# Deterministic Recipe Compiler v1

`src/fl_model_onboarding/recipe_compiler.py` compiles a **candidate** `ModelRecipe` from normalized Hugging Face metadata plus an exact architecture capability resolution.

## Scope

- LLM text-generation only.
- CPU target only.
- Requested precision `auto` or `int4` only.
- Non-gated + no remote code only.
- Capability status must be `verified` or `tool-supported-unverified`.

Anything outside this boundary fails closed with `GeneratedRecipeCompileError`.

## Input contract

`RecipeCompilerInput` requires:

- Full model ID + pinned 40-char HF revision SHA.
- Model type and/or architecture aliases.
- Requested task/device/precision.
- Gated and remote-code flags.
- Config/tokenizer/available file metadata.
- Exact `CapabilityResolution`.
- Explicit toolchain versions.

The compiler uses typed capability fields for Mobius/Olive args and never accepts free-form argv from metadata.

## Output contract

`compile_generated_recipe(...)` returns `GeneratedRecipe`:

- `recipe`: full `ModelRecipe` shape (Mobius args, Olive args, ancillary rules, runtime validation profile, optimization choices, cache/model prefixes).
- `pinned_revision`: exact immutable source revision.
- `provenance`: capability id/version/status, argument-confidence provenance, evidence, normalized input metadata, toolchain versions.
- `fingerprint`: SHA-256 over canonical JSON payload.
- `canonical_json`: deterministic canonical payload bytes.

Newly generated recipes are always `experimental`, even when capability status is `verified`.

## Ancillary rule safety

Compilation rejects ancillary rules when:

1. Relative paths are unsafe (absolute, `..`, drive-qualified, control chars).
2. Rules are ambiguous after normalization (case-insensitive path collisions).
3. OGA required files are missing from required ancillary rules.

## Promotion to verified

`promote_generated_recipe(...)` creates a **new** recipe version and never mutates the candidate in place.

Promotion requires explicit `PromotionGateEvidence` showing all gates passed with evidence:

- Mobius build
- Olive optimize
- ONNX validation
- ORT validation
- OGA validation
- FL SDK inference
- Quality validation

Any missing/failed gate raises `GeneratedRecipePromotionError`.

## Schema

Compiled payloads are validated against:

- `contracts/generated-recipe.schema.json`

This schema captures candidate + promoted payload structure, provenance, and mandatory gate evidence.

## Slice 3A1: trusted candidate compilation

`compile_trusted_candidate_recipe(default_recipe, *, policy, candidate, schema_path=None)` is the
**only** boundary permitted to layer a quantization override onto a compiled recipe's actual Olive
arguments. See `docs/recipe-agent-trusted-candidate-compilation.md` for the full contract: identity
validation, the resulting `OliveRecipeArgs.block_size` field, the `TrustedCandidateProvenance`
record, and how the recipe fingerprint reflects the resolved override.

